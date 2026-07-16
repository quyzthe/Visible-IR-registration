"""
Organize a raw DJI DCIM mission folder into visible/ + thermal/ subfolders,
then run the batch visible-thermal registration pipeline.

STEP 1 -- ORGANIZE
-------------------
Scans `source_dir` (recursively) for image files, classifies each as
THERMAL or VISIBLE by RESOLUTION (thermal sensor is always much smaller
than visible, regardless of filename), then pairs them by the shared ID
embedded in the filename -- the LAST run of digits in the filename stem
(e.g. "..._0008_V.JPG" / "..._0008_T.JPG" -> id "8"), NOT by timestamp.
Pairs that don't share an exact id (imperfect naming) get a second pass:
matched to whichever leftover file shares the longest common filename
PREFIX. Matched pairs are copied into visible_dir/thermal_dir renamed as
DJI_<idx>_V<ext> / DJI_<idx>_T<ext>. A mapping CSV + an unmatched-files
list are written for traceability.

STEP 2 -- CALIBRATE ONCE, THEN APPLY TO ALL
----------------------------------------------
The visible and thermal cameras are rigidly mounted on the same drone, so
the geometric relationship between them (scale/rotation/offset) is fixed
across the whole flight -- fitting a separate affine transform per image
pair is unnecessary AND fragile (a low-texture frame can give LoFTR/RANSAC
a bad fit, warping that one thermal frame entirely out of the canvas ->
solid black output). Instead:

  2a) CALIBRATE (runs once): LoFTR-match a small SAMPLE of pairs (spread
      evenly across the organized set for scene diversity), pool every
      confident match from all of them together, and fit ONE robust
      affine with a single RANSAC/MAGSAC pass over the pooled set.
      Then GUIDED INLIER RECOVERY (POS-GIFT, ISPRS J. Photogramm. 2022):
      use that fit to rescue lower-confidence LoFTR matches that are
      geometrically consistent with it (common in texture-poor regions
      like water or bare soil) and refit on the enlarged, still-verified
      set. Optionally, a smooth LOCAL correction (thin-plate-spline, as
      in the UAV-TIRVis benchmark) is fit on top of the affine to absorb
      the residual the rigid model can't capture -- e.g. the two lenses
      having different distortion. Saved to register_results/calibration.npz;
      reused on every later run unless calibration.recalibrate=True.

  2b) APPLY (fast, no LoFTR): use the calibrated transform (+ local
      correction, if fit) to warp every pair's ORIGINAL COLOR thermal
      image directly onto the ORIGINAL COLOR visible image (no forced
      grayscale, no colormap) and composite them at full opacity: real
      thermal data where the warp actually covers, original visible
      pixels elsewhere (no black padding, no alpha blending).
      `max_pairs_per_run` / `skip_already_processed` still apply here so
      you can process the dataset incrementally.

Run with: python organize_and_register.py
"""

import os
import re
import csv
import shutil

import cv2
import torch
import kornia.feature as KF
import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Needs Pillow: pip install pillow") from e


# =====================================================================
# CONFIG -- edit paths/params here
# =====================================================================

CONFIG = dict(
    # ---- Step 1: organize ----
    source_dir=r"E:\drone_090426\Raw_images\DCIM_1\DJI_202604090901_001_zoneseuils1",
    visible_dir=r"E:\drone_090426\Raw_images\DCIM_1\DJI_202604090901_001_zoneseuils1\visible",
    thermal_dir=r"E:\drone_090426\Raw_images\DCIM_1\DJI_202604090901_001_zoneseuils1\thermal",
    register_results_dir=r"E:\drone_090426\Raw_images\DCIM_1\DJI_202604090901_001_zoneseuils1\register_results",
    copy_mode="copy",              # "copy" (safe, keeps originals) or "move"
    fresh_start=False,             # True = delete+recreate visible_dir, thermal_dir, and
                                    # register_results_dir before doing anything else -- a clean
                                    # slate instead of manually deleting those folders yourself.
                                    # Only affects the run it's True for; set back to False afterwards
                                    # or every run will wipe out what the previous run just did.
                                    # NOTE: if copy_mode="move", the ORIGINALS in source_dir were
                                    # already moved out -- fresh_start would delete the only copy.
                                    # fresh_start is refused (raises) in that combination as a safety net.
    min_prefix_fallback_len=6,     # for files with no exact id match: min shared filename prefix length to accept as a pair
    resolution_threshold_px=None,  # None = auto-detect from the folder; or set a fixed pixel-count cutoff
    dry_run=False,                 # True = only print what WOULD happen, don't copy/rename anything

    # ---- Step 2a: calibration (runs once, finds ONE shared transform) ----
    work_size=(640, 480),          # LoFTR working resolution used only during calibration
    clahe_clip=3.0,
    clahe_tile=(8, 8),
    conf_thresh=0.5,
    min_matches=20,                # min POOLED matches required (across the whole calibration sample)
    ransac_thresh=3.0,
    ransac_confidence=0.999,
    transform_model="homography",  # "homography" (default) or "affine".
                                    # The visible and thermal cameras sit SIDE BY SIDE (a real physical
                                    # baseline, not coaxial), so the exact mapping between two images of
                                    # a planar/distant scene taken from two different camera centers is a
                                    # HOMOGRAPHY, not merely an affine transform -- affine has no
                                    # perspective/keystone term and is only a valid approximation when the
                                    # baseline is negligible relative to flying altitude. Use "affine" only
                                    # if homography proves unstable (see sanity_check_homography warnings).
    allow_shear=False,             # only used when transform_model="affine"
    min_inliers=15,                # min pooled inliers, else calibration is refused (raises)
    min_inlier_ratio=0.25,         # below this: calibration proceeds but prints a warning
    expected_scale_range=(0.3, 3.0),   # only used when transform_model="affine"
    max_rotation_deg=25.0,             # only used when transform_model="affine"
    max_anisotropy=1.6,                # only used when transform_model="affine"
    calibration=dict(
        sample_size=25,            # number of pairs (spread evenly) used to fit the ONE global transform
        recalibrate=False,         # True = redo calibration even if calibration.npz already exists
        save_features=True,        # save pooled matches used for calibration, for review
        guided_recovery_conf_thresh=0.15,  # 2nd-pass pool: matches down to this confidence get a
                                            # geometric-consistency check against the initial fit
                                            # (POS-GIFT-style guided inlier recovery) instead of being
                                            # discarded outright -- recovers correct matches in
                                            # texture-poor regions that LoFTR alone under-scores.
    ),
    local_refinement=dict(         # optional non-rigid residual correction ON TOP of the affine,
                                    # for the local misalignment a rigid/affine model can't capture
                                    # (the two lenses have different distortion) -- see UAV-TIRVis.
        enabled=True,
        min_points=80,             # need at least this many affine inliers before attempting it
        grid_step_px=20,           # TPS evaluated on a grid this coarse (work_size px), then upsampled
        max_correction_px=15.0,    # clip correction magnitude -- guards against TPS extrapolation
                                    # blowing up far from training points (e.g. image corners)
        tps_smoothing=2.0,         # regularization: 0 = exact interpolation (noisy), higher = smoother
    ),

    # ---- Step 2b: apply calibrated transform to every pair (color, no LoFTR) ----
    save_warped_thermal=True,
    save_overlay=True,
    max_pairs_per_run=500,          # None = process everything found; int = only this many NEW pairs this run
    skip_already_processed=True,   # skip pairs that already have an overlay output on disk

    # ---- Step 3: densify -- fill gaps using overlap between NEIGHBORING visible
    # frames (same-modality RGB-RGB matching, far more reliable than cross-modal).
    # Reuses each neighbor's already-computed warped_thermal (Step 2b), projected
    # into the target frame via the visible(neighbor)->visible(target) homography.
    densify=dict(
        enabled=True,

        # ---- 1) detect_features ----
        feature_detector="sift",     # "sift" (default -- what OpenSfM/ODM use by default, better in
                                      # low-texture regions) or "orb" (faster, no GPU/patent concerns)
        n_features=10000,             # max keypoints per image
        save_features=True,          # cache to register_results/rgb_features/<idx>.npz -- computed
                                      # ONCE per image EVER (across the whole warped_thermal pool, not
                                      # just this run's targets), reused by every future run

        # ---- 2) GLOBAL candidate selection via Bag-of-Visual-Words retrieval ----
        # (COLMAP/ORB-SLAM/OpenSfM-style "vocabulary tree" match-pair selection --
        # replaces GPS/index proximity entirely; a candidate pair is chosen by
        # VISUAL similarity across the WHOLE image pool, not local position)
        n_words=750,                 # vocabulary size (visual words) -- a single-mission dataset with
                                      # fairly homogeneous terrain doesn't need a huge vocabulary
        vocab_sample_descriptors=200000,  # cap on descriptors sampled to build the vocabulary (kmeans)
        rebuild_vocabulary=False,    # True = rebuild register_results/bow_vocabulary.npz from scratch
        top_m_candidates=15,         # per image, how many of the most visually-similar OTHER images
                                      # become match candidates. A bad candidate (visually similar but
                                      # not actually overlapping, e.g. repeated vegetation) just fails
                                      # RANSAC verification below and gets dropped -- costs a little
                                      # compute, not correctness.

        # ---- 3) match_features ----
        match_ratio=0.75,            # Lowe's ratio test threshold
        ransac_thresh=3.0,
        ransac_confidence=0.999,
        min_matches=15,

        # ---- 4) create_tracks ----
        min_track_len=2,             # keep tracks observed in at least this many images

        # ---- 5) fill ----
        feather_px=15,               # soft-blend width (px, native resolution) at the boundary where
                                      # a new source fills in -- reduces visible seams between sources
                                      # captured at different times/positions. 0 = hard cutoff, no blend.

        max_pairs_per_run=500,       # separate cap -- this stage does its own feature matching too
        skip_already_processed=True,
    ),
)

IMG_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
_PAIR_RE_TEMPLATE = r"_(\d+)_{tag}\.(jpg|jpeg|png|tif|tiff)$"


# =====================================================================
# STEP 1: SCAN / CLASSIFY / PAIR (by id/prefix, no timestamps) / RENAME
# =====================================================================

def scan_images(source_dir):
    files = []
    skip_names = {"visible", "thermal", "register_results"}
    for root, dirs, fnames in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d not in skip_names]
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, fn))
    return files


def get_pixel_count(path):
    with Image.open(path) as im:
        w, h = im.size
    return w * h


def auto_threshold(pixel_counts):
    uniq = sorted(set(pixel_counts))
    if len(uniq) < 2:
        return None
    best_ratio, best_i = 0, 0
    for i in range(len(uniq) - 1):
        ratio = uniq[i + 1] / uniq[i]
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
    return (uniq[best_i] * uniq[best_i + 1]) ** 0.5


def extract_id(path):
    """Last run of digits in the filename stem -- e.g. '..._0008_V' -> '8'.
    This is normally the shared sequence index between a visible/thermal
    pair, independent of any timestamp difference between the two shots."""
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = re.findall(r"\d+", stem)
    if digits:
        try:
            return str(int(digits[-1]))  # normalizes away leading zeros
        except ValueError:
            pass
    return stem


def _common_prefix_len(a, b):
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def prefix_fallback_pair(thermal_files, visible_files, min_len):
    """For files that didn't get an exact id match: pair whatever's left by
    longest shared filename PREFIX (greedy), above a minimum length."""
    remaining_v = list(visible_files)
    used_v = set()
    pairs = []
    for t in thermal_files:
        t_stem = os.path.splitext(os.path.basename(t))[0]
        best_v, best_len = None, 0
        for v in remaining_v:
            if v in used_v:
                continue
            v_stem = os.path.splitext(os.path.basename(v))[0]
            L = _common_prefix_len(t_stem, v_stem)
            if L > best_len:
                best_len, best_v = L, v
        if best_v is not None and best_len >= min_len:
            pairs.append((best_v, t))
            used_v.add(best_v)
    matched_t = {t for _, t in pairs}
    unmatched_t = [t for t in thermal_files if t not in matched_t]
    unmatched_v = [v for v in visible_files if v not in used_v]
    return pairs, unmatched_t, unmatched_v


def classify_and_pair(cfg):
    files = scan_images(cfg["source_dir"])
    if not files:
        raise RuntimeError(f"No image files found under {cfg['source_dir']}")

    pixel_counts = {}
    for f in files:
        try:
            pixel_counts[f] = get_pixel_count(f)
        except Exception as e:
            print(f"[WARN] could not read size of {f}: {e}")

    threshold = cfg["resolution_threshold_px"] or auto_threshold(list(pixel_counts.values()))
    if threshold is None:
        raise RuntimeError(
            "Could not auto-detect a resolution split (only one resolution present). "
            "Set CONFIG['resolution_threshold_px'] manually."
        )
    print(f"Resolution split threshold: {threshold:,.0f} px "
          f"({'auto-detected' if not cfg['resolution_threshold_px'] else 'manual'})")

    res_summary = {}
    for f in pixel_counts:
        with Image.open(f) as im:
            wh = im.size
        res_summary[wh] = res_summary.get(wh, 0) + 1
    print("Resolutions found:", {f"{w}x{h}": n for (w, h), n in sorted(res_summary.items())})

    thermal_files = [f for f, px in pixel_counts.items() if px <= threshold]
    visible_files = [f for f, px in pixel_counts.items() if px > threshold]
    print(f"Classified: {len(thermal_files)} thermal, {len(visible_files)} visible")

    thermal_by_id, visible_by_id = {}, {}
    for f in thermal_files:
        thermal_by_id.setdefault(extract_id(f), []).append(f)
    for f in visible_files:
        visible_by_id.setdefault(extract_id(f), []).append(f)

    exact_pairs = []
    used_t, used_v = set(), set()
    for id_key in sorted(set(thermal_by_id) & set(visible_by_id), key=int):
        t_path, v_path = thermal_by_id[id_key][0], visible_by_id[id_key][0]
        exact_pairs.append((v_path, t_path))
        used_t.add(t_path)
        used_v.add(v_path)

    leftover_t = [f for f in thermal_files if f not in used_t]
    leftover_v = [f for f in visible_files if f not in used_v]

    fallback_pairs, unmatched_t, unmatched_v = prefix_fallback_pair(
        leftover_t, leftover_v, cfg["min_prefix_fallback_len"]
    )

    print(f"Paired by exact id: {len(exact_pairs)} | by prefix fallback: {len(fallback_pairs)} "
          f"| unmatched thermal: {len(unmatched_t)} | unmatched visible: {len(unmatched_v)}")

    return exact_pairs + fallback_pairs, unmatched_t, unmatched_v


def _already_organized_indices(visible_dir):
    done = set()
    if os.path.isdir(visible_dir):
        for fn in os.listdir(visible_dir):
            m = re.match(r"DJI_(\d+)_V\.", fn, re.IGNORECASE)
            if m:
                done.add(int(m.group(1)))
    return done


def reset_folders(cfg):
    """Delete + recreate visible_dir, thermal_dir, register_results_dir so
    the next run starts from a clean slate, instead of deleting them by
    hand in Explorer every time."""
    if cfg["copy_mode"] == "move":
        raise RuntimeError(
            "fresh_start=True with copy_mode='move' would permanently delete files that were "
            "moved out of source_dir with no other copy remaining -- refusing. Either set "
            "copy_mode='copy', or set fresh_start=False and clean up manually if you're sure."
        )
    targets = [cfg["visible_dir"], cfg["thermal_dir"], cfg["register_results_dir"]]
    print("[fresh_start] Wiping and recreating:")
    for d in targets:
        if os.path.isdir(d):
            print(f"  deleting {d}")
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("[fresh_start] Done -- starting from a clean slate.\n")


def organize(cfg):
    pairs, unmatched_thermal, unmatched_visible = classify_and_pair(cfg)
    total_found = len(pairs)

    already_done = _already_organized_indices(cfg["visible_dir"]) if cfg.get("skip_already_processed", True) else set()

    limit = cfg.get("max_pairs_per_run")
    to_organize = []  # (idx:int, v_path, t_path)
    for i, (v_path, t_path) in enumerate(pairs, start=1):
        if i in already_done:
            continue
        to_organize.append((i, v_path, t_path))
        if limit is not None and len(to_organize) >= limit:
            break

    n_already = len(already_done)
    n_remaining_after = total_found - n_already - len(to_organize)
    print(f"{total_found} pairs found total | {n_already} already organized | "
          f"organizing {len(to_organize)} more this run"
          + (f" | {n_remaining_after} will remain for next run" if n_remaining_after > 0 else ""))

    if cfg["dry_run"]:
        print("\n[DRY RUN] Would organize (showing up to 10):")
        for idx, v, t in to_organize[:10]:
            print(f"  {idx:04d}: {os.path.basename(v)}  <->  {os.path.basename(t)}")
        if len(to_organize) > 10:
            print(f"  ... and {len(to_organize) - 10} more")
        print("Set dry_run=False to actually copy/rename.")
        return

    if not to_organize:
        print("Nothing new to organize this run.")
        return

    os.makedirs(cfg["visible_dir"], exist_ok=True)
    os.makedirs(cfg["thermal_dir"], exist_ok=True)
    os.makedirs(cfg["register_results_dir"], exist_ok=True)

    op = shutil.move if cfg["copy_mode"] == "move" else shutil.copy2

    mapping_path = os.path.join(cfg["register_results_dir"], "file_mapping.csv")
    existing_mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, newline="") as f:
            for r in csv.DictReader(f):
                existing_mapping[r["index"]] = r

    for i, v_path, t_path in to_organize:
        idx = f"{i:04d}"
        t_ext, v_ext = os.path.splitext(t_path)[1], os.path.splitext(v_path)[1]
        new_t = os.path.join(cfg["thermal_dir"], f"DJI_{idx}_T{t_ext}")
        new_v = os.path.join(cfg["visible_dir"], f"DJI_{idx}_V{v_ext}")
        op(t_path, new_t)
        op(v_path, new_v)
        existing_mapping[idx] = {
            "index": idx,
            "original_visible": v_path,
            "original_thermal": t_path,
            "new_visible": new_v,
            "new_thermal": new_t,
        }

    if existing_mapping:
        with open(mapping_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "original_visible", "original_thermal", "new_visible", "new_thermal"])
            writer.writeheader()
            for k in sorted(existing_mapping, key=int):
                writer.writerow(existing_mapping[k])
        print(f"Wrote mapping: {mapping_path}")

    if unmatched_thermal or unmatched_visible:
        unmatched_path = os.path.join(cfg["register_results_dir"], "unmatched_files.txt")
        with open(unmatched_path, "w") as f:
            f.write("Unmatched thermal files:\n")
            f.writelines(f"  {p}\n" for p in unmatched_thermal)
            f.write("\nUnmatched visible files:\n")
            f.writelines(f"  {p}\n" for p in unmatched_visible)
        print(f"[WARN] {len(unmatched_thermal) + len(unmatched_visible)} unmatched file(s) "
              f"-> see {unmatched_path}")

    print(f"Organized {len(to_organize)} pairs this run into:\n  {cfg['visible_dir']}\n  {cfg['thermal_dir']}")


# =====================================================================
# STEP 2 SHARED HELPERS
# =====================================================================

def _index_files(folder, tag):
    pattern = re.compile(_PAIR_RE_TEMPLATE.format(tag=tag), re.IGNORECASE)
    out = {}
    for fname in os.listdir(folder):
        m = pattern.search(fname)
        if m:
            out[m.group(1)] = os.path.join(folder, fname)
    return out


def find_pairs(visible_dir, thermal_dir):
    vis = _index_files(visible_dir, "V")
    th = _index_files(thermal_dir, "T")
    common = sorted(set(vis) & set(th), key=lambda s: int(s))
    return [(idx, vis[idx], th[idx]) for idx in common]


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def preprocess(gray, size, clip, tile):
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return resized, clahe.apply(resized)


def build_matcher(device):
    return KF.LoFTR(pretrained="outdoor").to(device).eval()


def get_matches_loftr_raw(matcher, visible_p, thermal_p, device):
    img0 = torch.from_numpy(visible_p / 255.0).float()[None, None].to(device)
    img1 = torch.from_numpy(thermal_p / 255.0).float()[None, None].to(device)
    with torch.no_grad():
        out = matcher({"image0": img0, "image1": img1})
    return out["keypoints0"].cpu().numpy(), out["keypoints1"].cpu().numpy(), out["confidence"].cpu().numpy()


def estimate_affine_robust(src_pts, dst_pts, allow_shear, thresh, confidence):
    estimator = cv2.estimateAffine2D if allow_shear else cv2.estimateAffinePartial2D
    for method in (getattr(cv2, "USAC_MAGSAC", None), cv2.RANSAC):
        if method is None:
            continue
        try:
            M, mask = estimator(
                src_pts, dst_pts, method=method,
                ransacReprojThreshold=thresh, confidence=confidence, maxIters=5000,
            )
            return M, mask
        except cv2.error:
            continue
    raise RuntimeError("estimate_affine_robust: no supported robust method worked")


def estimate_homography_robust(src_pts, dst_pts, thresh, confidence):
    for method in (getattr(cv2, "USAC_MAGSAC", None), cv2.RANSAC):
        if method is None:
            continue
        try:
            H, mask = cv2.findHomography(
                src_pts, dst_pts, method=method,
                ransacReprojThreshold=thresh, confidence=confidence, maxIters=5000,
            )
            return H, mask
        except cv2.error:
            continue
    raise RuntimeError("estimate_homography_robust: no supported robust method worked")


def estimate_transform_robust(src_pts, dst_pts, cfg):
    """Returns a 3x3 matrix regardless of model -- an affine fit is padded
    with [0,0,1] so downstream code (reprojection, TPS, full_res scaling,
    warping) can treat both uniformly."""
    if cfg["transform_model"] == "homography":
        H, mask = estimate_homography_robust(src_pts, dst_pts, cfg["ransac_thresh"], cfg["ransac_confidence"])
        return (None if H is None else H.astype(np.float64)), mask
    M, mask = estimate_affine_robust(src_pts, dst_pts, cfg["allow_shear"], cfg["ransac_thresh"], cfg["ransac_confidence"])
    return (None if M is None else to_3x3(M)), mask


def apply_homogeneous(M3x3, pts):
    """Transform Nx2 points through a 3x3 matrix with a proper perspective
    divide -- works for a true homography AND for an affine padded to 3x3
    (whose bottom row [0,0,1] makes the divide a no-op)."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    out_h = pts_h @ M3x3.T
    return out_h[:, :2] / out_h[:, 2:3]


def decompose_affine(M):
    A = M[:2, :2].astype(np.float64)
    t = M[:2, 2]
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    reflection = np.linalg.det(A) < 0
    anisotropy = float(S[0] / S[1]) if S[1] > 1e-9 else float("inf")
    return dict(
        translation_px=(float(t[0]), float(t[1])),
        rotation_deg=rotation_deg,
        scale_mean=float(np.mean(S)),
        anisotropy=anisotropy,
        reflection=bool(reflection),
    )


def sanity_check(decomp, cfg):
    warnings = []
    lo, hi = cfg["expected_scale_range"]
    if not (lo <= decomp["scale_mean"] <= hi):
        warnings.append(f"scale_mean={decomp['scale_mean']:.3f} outside [{lo}, {hi}]")
    if abs(decomp["rotation_deg"]) > cfg["max_rotation_deg"]:
        warnings.append(f"rotation={decomp['rotation_deg']:.1f}deg > max {cfg['max_rotation_deg']}")
    if decomp["anisotropy"] > cfg["max_anisotropy"]:
        warnings.append(f"anisotropy={decomp['anisotropy']:.2f} > max {cfg['max_anisotropy']}")
    if decomp["reflection"]:
        warnings.append("transform includes a reflection (degenerate)")
    return warnings


def sanity_check_homography(H, work_size):
    """Lightweight degeneracy check for a homography: extreme perspective
    terms (bottom row) relative to image size indicate an unstable/overfit
    fit rather than a genuine mild-parallax effect."""
    warnings = []
    W, H_ = work_size
    if abs(np.linalg.det(H)) < 1e-9:
        warnings.append("homography is near-singular (degenerate)")
    h20, h21, h22 = H[2]
    if h22 != 0:
        if abs(h20 / h22) * W > 0.5 or abs(h21 / h22) * H_ > 0.5:
            warnings.append("homography has very strong perspective terms -- likely an unstable fit "
                             "(check inlier count/distribution, or fall back to transform_model='affine')")
    return warnings


def to_3x3(M):
    if M.shape == (3, 3):
        return M.astype(np.float64)
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = M
    return out


def full_res_matrix(M_work, thermal_native_shape, visible_native_shape, work_size):
    """M_work (3x3, affine-padded or true homography) maps thermal(work_size)
    -> visible(work_size). Returns the equivalent 3x3 matrix mapping
    thermal_native -> visible_native, for warping full-resolution/full-color
    images directly (warpPerspective, or warpAffine on the top 2 rows if the
    transform happens to be a plain affine)."""
    Ht_n, Wt_n = thermal_native_shape[:2]
    Hv_n, Wv_n = visible_native_shape[:2]
    W, H = work_size
    S_t = to_3x3(np.array([[W / Wt_n, 0, 0], [0, H / Ht_n, 0]], dtype=np.float64))
    S_v_inv = to_3x3(np.array([[Wv_n / W, 0, 0], [0, Hv_n / H, 0]], dtype=np.float64))
    return S_v_inv @ to_3x3(M_work) @ S_t


def _reprojection_error(M, src_pts, dst_pts):
    """Euclidean distance between M(src_pts) and dst_pts, for every point.
    Works for a true homography or an affine padded to 3x3."""
    pred = apply_homogeneous(to_3x3(M), src_pts)
    return np.linalg.norm(pred - dst_pts.astype(np.float64), axis=1)


def fit_tps_correction(mkpts0_inlier, mkpts1_inlier, M, work_size, cfg):
    """Smooth (thin-plate-spline) residual correction, in VISIBLE (destination)
    coordinates, layered on top of the global transform M (affine or
    homography). Neither model captures the two lenses having different
    (radial) distortion -- this fits that local residual from the pooled
    inliers, the way UAV-TIRVis uses an Rbf/TPS local correction after
    coarse rigid/projective registration."""
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:
        print("[WARN] scipy.interpolate.RBFInterpolator unavailable (pip install --upgrade scipy) "
              "-- skipping local TPS refinement.")
        return None

    predicted_vis = apply_homogeneous(to_3x3(M), mkpts1_inlier)  # where M puts each thermal inlier
    delta = predicted_vis - mkpts0_inlier                        # correction needed AT the true visible location

    smoothing = cfg["local_refinement"]["tps_smoothing"]
    rbf_dx = RBFInterpolator(mkpts0_inlier, delta[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
    rbf_dy = RBFInterpolator(mkpts0_inlier, delta[:, 1], kernel="thin_plate_spline", smoothing=smoothing)

    W, H = work_size
    step = cfg["local_refinement"]["grid_step_px"]
    xs = np.unique(np.append(np.arange(0, W, step), W - 1))
    ys = np.unique(np.append(np.arange(0, H, step), H - 1))
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

    cap = cfg["local_refinement"]["max_correction_px"]
    dx = np.clip(rbf_dx(grid_pts), -cap, cap).reshape(gy.shape)
    dy = np.clip(rbf_dy(grid_pts), -cap, cap).reshape(gy.shape)

    map_x = cv2.resize((gx + dx).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize((gy + dy).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


# =====================================================================
# STEP 2a: CALIBRATE -- fit ONE shared transform from a pooled sample,
# with guided inlier recovery + optional local TPS refinement
# =====================================================================

def calibrate(cfg):
    pairs = find_pairs(cfg["visible_dir"], cfg["thermal_dir"])
    if not pairs:
        raise RuntimeError("No organized pairs found -- run Step 1 (organize) first.")

    calib_path = os.path.join(cfg["register_results_dir"], "calibration.npz")
    if os.path.exists(calib_path) and not cfg["calibration"]["recalibrate"]:
        data = np.load(calib_path)
        has_tps = bool(data["has_tps"])
        saved_model = str(data["transform_model"]) if "transform_model" in data else "affine"
        if saved_model != cfg["transform_model"]:
            print(f"[WARN] existing calibration was fit with transform_model='{saved_model}', "
                  f"but config now says '{cfg['transform_model']}'. Using the SAVED calibration -- "
                  "set calibration.recalibrate=True to refit with the new model.")
        print(f"Using existing calibration: {calib_path} "
              f"(model={saved_model}, from {int(data['n_sample_pairs'])} pairs, "
              f"{int(data['n_inliers'])}/{int(data['n_pooled_matches'])} inliers, "
              f"RMSE={float(data['rmse_px']):.2f}px{' + local TPS' if has_tps else ''}). "
              "Set calibration.recalibrate=True to redo it.")
        tps = (data["map_x"], data["map_y"]) if has_tps else None
        return data["M_work"].astype(np.float64), tuple(int(v) for v in data["work_size"]), tps

    n = min(cfg["calibration"]["sample_size"], len(pairs))
    sample_pos = sorted(set(np.linspace(0, len(pairs) - 1, n).astype(int).tolist()))
    sample = [pairs[i] for i in sample_pos]
    print(f"Calibrating from {len(sample)} pairs sampled evenly across {len(pairs)} organized pairs...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)
    matcher = build_matcher(device)

    os.makedirs(cfg["register_results_dir"], exist_ok=True)
    feat_dir = os.path.join(cfg["register_results_dir"], "calibration_features")
    if cfg["calibration"]["save_features"]:
        os.makedirs(feat_dir, exist_ok=True)

    # two pools: a high-confidence one for the initial fit, and a much larger
    # low-confidence one used only for guided recovery (never used raw/blindly)
    all_v_hi, all_t_hi, all_v_lo, all_t_lo = [], [], [], []
    for idx, vpath, tpath in sample:
        visible_native = load_gray(vpath)
        thermal_native = load_gray(tpath)
        _, visible_p = preprocess(visible_native, cfg["work_size"], cfg["clahe_clip"], cfg["clahe_tile"])
        _, thermal_p = preprocess(thermal_native, cfg["work_size"], cfg["clahe_clip"], cfg["clahe_tile"])

        mkpts0_raw, mkpts1_raw, conf_raw = get_matches_loftr_raw(matcher, visible_p, thermal_p, device)
        hi = conf_raw >= cfg["conf_thresh"]
        lo = conf_raw >= cfg["calibration"]["guided_recovery_conf_thresh"]
        print(f"  [{idx}] {int(hi.sum())} matches >= conf_thresh, {int(lo.sum())} candidates for guided recovery")

        all_v_hi.append(mkpts0_raw[hi]); all_t_hi.append(mkpts1_raw[hi])
        all_v_lo.append(mkpts0_raw[lo]); all_t_lo.append(mkpts1_raw[lo])

        if cfg["calibration"]["save_features"]:
            np.savez(os.path.join(feat_dir, f"{idx}.npz"), mkpts0=mkpts0_raw, mkpts1=mkpts1_raw, conf=conf_raw)

    mkpts0_hi = np.concatenate(all_v_hi, axis=0)
    mkpts1_hi = np.concatenate(all_t_hi, axis=0)
    print(f"Pooled high-confidence matches from {len(sample)} pairs: {len(mkpts0_hi)} total")

    if len(mkpts0_hi) < cfg["min_matches"]:
        raise RuntimeError(
            f"Only {len(mkpts0_hi)} pooled matches (need >= {cfg['min_matches']}). "
            "Increase calibration.sample_size or lower conf_thresh."
        )

    M0, mask0 = estimate_transform_robust(mkpts1_hi, mkpts0_hi, cfg)
    if M0 is None:
        raise RuntimeError("Global calibration transform estimation failed.")
    print(f"Initial pooled RANSAC ({cfg['transform_model']}): {int(mask0.sum())}/{len(mkpts0_hi)} inliers "
          f"({mask0.sum()/len(mkpts0_hi):.1%})")

    # ---- guided inlier recovery (POS-GIFT, ISPRS J. Photogramm. 2022): use the
    # initial transform to recover matches LoFTR gave LOW confidence to (common
    # in texture-poor regions -- water, bare soil) but that are geometrically
    # consistent with it, then refit on the enlarged, still-verified set ----
    mkpts0_lo = np.concatenate(all_v_lo, axis=0)
    mkpts1_lo = np.concatenate(all_t_lo, axis=0)
    err = _reprojection_error(M0, mkpts1_lo, mkpts0_lo)
    consistent = err < cfg["ransac_thresh"]
    print(f"Guided recovery: {int(consistent.sum())}/{len(mkpts0_lo)} low-confidence candidates "
          f"geometrically consistent with the initial fit")

    M, mask = estimate_transform_robust(mkpts1_lo[consistent], mkpts0_lo[consistent], cfg)
    if M is None:
        print("[WARN] refit after guided recovery failed -- keeping the initial fit")
        M, mask, mkpts0_final, mkpts1_final = M0, mask0, mkpts0_hi, mkpts1_hi
    else:
        mkpts0_final, mkpts1_final = mkpts0_lo[consistent], mkpts1_lo[consistent]

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(mkpts0_final)
    print(f"Refined RANSAC: {inliers}/{len(mkpts0_final)} inliers ({inlier_ratio:.1%})")
    if inliers < cfg["min_inliers"]:
        raise RuntimeError(
            f"Only {inliers} inliers after refinement (need >= {cfg['min_inliers']}) -- calibration is "
            "not reliable. Increase calibration.sample_size, lower conf_thresh, or check the images."
        )
    if inlier_ratio < cfg["min_inlier_ratio"]:
        print(f"[WARN] inlier ratio {inlier_ratio:.1%} is low -- calibration may be noisy.")

    if cfg["transform_model"] == "affine":
        decomp = decompose_affine(M)
        for w in sanity_check(decomp, cfg):
            print(f"[WARN] {w}")
        print(f"Calibration transform: scale={decomp['scale_mean']:.4f} "
              f"rotation={decomp['rotation_deg']:.2f}deg translation={decomp['translation_px']}")
    else:
        for w in sanity_check_homography(M, cfg["work_size"]):
            print(f"[WARN] {w}")
        print(f"Calibration transform (homography):\n{M}")

    inlier_bool = mask.ravel().astype(bool)
    inlier_pts0 = mkpts0_final[inlier_bool]
    inlier_pts1 = mkpts1_final[inlier_bool]
    rmse = float(np.sqrt(np.mean(_reprojection_error(M, inlier_pts1, inlier_pts0) ** 2)))
    print(f"Reprojection RMSE (inliers): {rmse:.2f} px (work-resolution {cfg['work_size']})")

    map_x = map_y = None
    has_tps = False
    lr = cfg["local_refinement"]
    if lr["enabled"] and len(inlier_pts0) >= lr["min_points"]:
        tps = fit_tps_correction(inlier_pts0, inlier_pts1, M, cfg["work_size"], cfg)
        if tps is not None:
            map_x, map_y = tps
            has_tps = True
            print(f"Local TPS refinement fitted on {len(inlier_pts0)} inliers "
                  f"(grid_step={lr['grid_step_px']}px, max_correction={lr['max_correction_px']}px, "
                  f"smoothing={lr['tps_smoothing']})")
    elif lr["enabled"]:
        print(f"[WARN] Only {len(inlier_pts0)} inliers -- need >= {lr['min_points']} for local TPS "
              f"refinement, skipping it ({cfg['transform_model']}-only).")

    np.savez(
        calib_path,
        M_work=M, work_size=np.array(cfg["work_size"]), transform_model=cfg["transform_model"],
        n_sample_pairs=len(sample), n_pooled_matches=len(mkpts0_final), n_inliers=inliers,
        rmse_px=rmse, has_tps=has_tps,
        map_x=map_x if has_tps else np.zeros((1, 1), dtype=np.float32),
        map_y=map_y if has_tps else np.zeros((1, 1), dtype=np.float32),
    )
    print(f"Saved calibration: {calib_path}")
    return M, cfg["work_size"], ((map_x, map_y) if has_tps else None)


# =====================================================================
# STEP 2b: APPLY -- warp ORIGINAL COLOR thermal onto ORIGINAL COLOR
# visible using the calibrated transform (+ optional local TPS). No LoFTR,
# no per-pair fit.
# =====================================================================

def apply_calibration(cfg, M_work, work_size, tps=None):
    out_dirs = {
        "warped_thermal": os.path.join(cfg["register_results_dir"], "warped_thermal"),
        "overlays": os.path.join(cfg["register_results_dir"], "overlays"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    pairs = find_pairs(cfg["visible_dir"], cfg["thermal_dir"])
    total_found = len(pairs)
    if not pairs:
        print("No organized pairs found.")
        return []

    if cfg.get("skip_already_processed", True) and os.path.isdir(out_dirs["overlays"]):
        done_ids = {fn[:-len("_overlay.png")] for fn in os.listdir(out_dirs["overlays"]) if fn.endswith("_overlay.png")}
        pending = [(idx, v, t) for idx, v, t in pairs if idx not in done_ids]
    else:
        pending = pairs

    n_done = total_found - len(pending)
    limit = cfg.get("max_pairs_per_run")
    run_now = pending[:limit] if limit is not None else pending
    remaining_after = len(pending) - len(run_now)

    print(f"{total_found} pairs total | {n_done} already applied | "
          f"applying to {len(run_now)} this session"
          + (f" | {remaining_after} will remain for next run" if remaining_after > 0 else ""))

    n_ok, n_err = 0, 0
    for idx, vpath, tpath in run_now:
        try:
            visible_color = load_color(vpath)
            thermal_color = load_color(tpath)
            out_size = (visible_color.shape[1], visible_color.shape[0])

            M_full = full_res_matrix(M_work, thermal_color.shape, visible_color.shape, work_size)
            thermal_warped = cv2.warpPerspective(thermal_color, M_full, out_size)

            # warp a solid-255 mask with the SAME transform to know exactly which
            # output pixels are real thermal data vs. black padding from the warp
            src_mask = np.full(thermal_color.shape[:2], 255, dtype=np.uint8)
            valid_mask = cv2.warpPerspective(src_mask, M_full, out_size, flags=cv2.INTER_NEAREST, borderValue=0)

            if tps is not None:
                map_x_work, map_y_work = tps
                W, H = work_size
                Wv_n, Hv_n = out_size
                sx, sy = Wv_n / W, Hv_n / H
                map_x_native = cv2.resize(map_x_work, out_size, interpolation=cv2.INTER_LINEAR) * sx
                map_y_native = cv2.resize(map_y_work, out_size, interpolation=cv2.INTER_LINEAR) * sy
                thermal_warped = cv2.remap(thermal_warped, map_x_native, map_y_native,
                                            interpolation=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                valid_mask = cv2.remap(valid_mask, map_x_native, map_y_native,
                                        interpolation=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            if cfg["save_warped_thermal"]:
                thermal_bgra = np.dstack([thermal_warped, valid_mask])
                cv2.imwrite(os.path.join(out_dirs["warped_thermal"], f"{idx}_T_warped.png"), thermal_bgra)

            if cfg["save_overlay"]:
                # full opacity, no alpha blend: thermal where valid, original visible elsewhere
                overlay = visible_color.copy()
                overlay[valid_mask > 0] = thermal_warped[valid_mask > 0]
                cv2.imwrite(os.path.join(out_dirs["overlays"], f"{idx}_overlay.png"), overlay)

            n_ok += 1
            print(f"[{idx}] applied ok")
        except Exception as e:
            n_err += 1
            print(f"[{idx}] ERROR: {e}")

    print(f"\nThis run: {n_ok} ok, {n_err} error, {len(run_now)} total")
    return run_now

# =====================================================================
# GLOBAL CANDIDATE SELECTION: Bag-of-Visual-Words retrieval
# ("vocabulary tree" match-pair selection, as used in COLMAP/ORB-SLAM/
# OpenSfM's no-GPS fallback -- see e.g. Schonberger et al., or the UAV-
# specific "Leveraging vocabulary tree for simultaneous match pair
# selection..." ISPRS 2022). Every image's descriptors are quantized
# against a small shared vocabulary into ONE compact TF-IDF histogram;
# candidate pairs are just the images with the closest histograms --
# this is a GLOBAL search over every image, not restricted to a local
# GPS/index neighborhood, which is exactly what was failing before.
# =====================================================================

def build_or_load_vocabulary(cfg, feats_by_idx):
    dz = cfg["densify"]
    vocab_path = os.path.join(cfg["register_results_dir"], "bow_vocabulary.npz")
    if os.path.exists(vocab_path) and not dz.get("rebuild_vocabulary", False):
        return np.load(vocab_path)["centers"]

    rng = np.random.default_rng(0)
    idxs = list(feats_by_idx.keys())
    rng.shuffle(idxs)
    chunks, total = [], 0
    for idx in idxs:
        d = feats_by_idx[idx][1]
        if len(d):
            chunks.append(d)
            total += len(d)
        if total >= dz["vocab_sample_descriptors"]:
            break
    all_descs = np.concatenate(chunks, axis=0).astype(np.float32)
    if len(all_descs) > dz["vocab_sample_descriptors"]:
        sel = rng.choice(len(all_descs), dz["vocab_sample_descriptors"], replace=False)
        all_descs = all_descs[sel]

    n_words = min(dz["n_words"], max(2, len(all_descs) // 10))
    print(f"Building BoW vocabulary: {n_words} words from {len(all_descs):,} sampled descriptors...")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-4)
    _, _, centers = cv2.kmeans(all_descs, n_words, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    np.savez(vocab_path, centers=centers)
    return centers


def compute_bow_vectors(feats_by_idx, vocab):
    """TF-IDF weighted, L2-normalized BoW histogram per image. Pairwise
    cosine similarity is then just the dot product of these vectors."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    idxs = list(feats_by_idx.keys())
    n_words = len(vocab)

    raw = {}
    for idx in idxs:
        descs = feats_by_idx[idx][1]
        hist = np.zeros(n_words, dtype=np.float32)
        if len(descs):
            for m in matcher.match(descs.astype(np.float32), vocab):
                hist[m.trainIdx] += 1
        raw[idx] = hist

    doc_freq = np.zeros(n_words, dtype=np.float64)
    for idx in idxs:
        doc_freq += (raw[idx] > 0)
    idf = np.log((len(idxs) + 1.0) / (doc_freq + 1.0))

    vectors = {}
    for idx in idxs:
        v = raw[idx] * idf
        norm = np.linalg.norm(v)
        vectors[idx] = (v / norm) if norm > 0 else v
    return vectors


def bow_top_candidates(vectors, top_m):
    """idx -> list of the top_m most visually-similar OTHER images
    (cosine similarity via dot product, vectors are already L2-normalized)."""
    idxs = list(vectors.keys())
    mat = np.stack([vectors[i] for i in idxs])  # (N, n_words)
    sims = mat @ mat.T
    np.fill_diagonal(sims, -1.0)
    out = {}
    for row, idx in enumerate(idxs):
        order = np.argsort(sims[row])[::-1][:top_m]
        out[idx] = [idxs[o] for o in order if sims[row, o] > 0]
    return out


# =====================================================================
# STEP 3: DENSIFY -- ODM/OpenSfM-style pipeline
#   1) detect_features : SIFT keypoints+descriptors, ONCE per image, cached
#   2) BoW candidate selection : GLOBAL visual-similarity retrieval (not
#      GPS/index-local) picks which pairs are worth attempting to match
#   3) match_features   : FLANN/BFMatcher + Lowe ratio test + homography
#      RANSAC on those candidate pairs (false-positive BoW candidates --
#      e.g. two unrelated patches of similar-looking vegetation -- simply
#      fail RANSAC verification here and are dropped, same as any SfM
#      pipeline using retrieval-based candidates)
#   4) create_tracks     : Union-Find links verified matches across ALL
#      pairs into tracks -- a track is one physical feature seen in N
#      images, including images never directly matched to each other
#   5) fill each target using every image reachable via a SHARED TRACK,
#      fitting one direct homography per (source, target) from the pooled
#      track correspondences, composited with soft feathering at the seams
# =====================================================================

def _load_warped_thermal_bgra(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(
            f"{os.path.basename(path)} has no alpha channel (shape={img.shape}) -- it was written "
            "by an older run of apply_calibration() before warped_thermal started saving BGRA. "
            "Delete the warped_thermal/ and overlays/ folders and rerun Step 2b to regenerate them "
            "in BGRA (calibration.npz stays cached, so this is fast -- no LoFTR needed)."
        )
    return img[:, :, :3], img[:, :, 3]


class UnionFind:
    """Standard union-find with path compression, keyed on arbitrary
    hashable nodes (here: (image_idx, keypoint_index) tuples)."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_feature_detector(cfg):
    dz = cfg["densify"]
    if dz["feature_detector"] == "orb":
        return cv2.ORB_create(nfeatures=dz["n_features"]), cv2.NORM_HAMMING
    return cv2.SIFT_create(nfeatures=dz["n_features"]), cv2.NORM_L2


def extract_features(idx, path, detector, cfg):
    """Per-image keypoints (work_size px coords) + descriptors, cached to
    disk so repeated runs never recompute them -- mirrors OpenSfM's
    detect_features running once per image, independent of any pair."""
    dz = cfg["densify"]
    feat_dir = os.path.join(cfg["register_results_dir"], "rgb_features")
    cache_path = os.path.join(feat_dir, f"{idx}.npz")
    if dz["save_features"] and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["pts"], data["descs"]

    gray = load_gray(path)
    _, gray_p = preprocess(gray, cfg["work_size"], cfg["clahe_clip"], cfg["clahe_tile"])
    kps, descs = detector.detectAndCompute(gray_p, None)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
    if descs is None:
        descs = np.zeros((0, 0), np.float32)
    if dz["save_features"]:
        os.makedirs(feat_dir, exist_ok=True)
        np.savez(cache_path, pts=pts, descs=descs)
    return pts, descs


def match_pair(pts_a, descs_a, pts_b, descs_b, norm_type, cfg):
    """FLANN (SIFT) or Hamming BFMatcher (ORB) + Lowe ratio test + homography
    RANSAC verification -- mirrors OpenSfM's match_features + geometric
    verification. Returning None here is how a bad BoW candidate (visually
    similar but not actually overlapping) gets silently rejected."""
    dz = cfg["densify"]
    if len(descs_a) < 2 or len(descs_b) < 2:
        return None

    if norm_type == cv2.NORM_L2:
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        knn = matcher.knnMatch(descs_a.astype(np.float32), descs_b.astype(np.float32), k=2)
    else:
        matcher = cv2.BFMatcher(norm_type)
        knn = matcher.knnMatch(descs_a, descs_b, k=2)

    good = [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < dz["match_ratio"] * n.distance]
    if len(good) < dz["min_matches"]:
        return None

    src = np.array([pts_a[i] for i, _ in good])
    dst = np.array([pts_b[j] for _, j in good])
    H, mask = estimate_homography_robust(src, dst, dz["ransac_thresh"], dz["ransac_confidence"])
    if H is None or int(mask.sum()) < dz["min_matches"]:
        return None

    inlier_pairs = [good[k] for k in range(len(good)) if mask.ravel()[k]]
    return H, int(mask.sum()), inlier_pairs


def _feather_composite(base, base_filled, add_bgr, add_alpha, feather_px):
    """Composite add_bgr into base wherever add_alpha is valid and base isn't
    filled yet, with a soft-edged blend (distance-transform feathering) at
    the boundary instead of a hard cutoff -- reduces visible seams between
    sources captured at different times/positions, the cheap panorama-
    stitching trick (full multi-band blending would help more but is a much
    bigger lift)."""
    new_only = (add_alpha > 0) & (~base_filled)
    if not new_only.any():
        return base, base_filled
    if feather_px <= 0:
        base[new_only] = add_bgr[new_only]
        return base, (base_filled | new_only)

    dist = cv2.distanceTransform(new_only.astype(np.uint8) * 255, cv2.DIST_L2, 3)
    w = np.clip(dist / float(feather_px), 0, 1)[..., None]
    base = base.astype(np.float32)
    base[new_only] = (w[new_only] * add_bgr[new_only].astype(np.float32)
                       + (1 - w[new_only]) * base[new_only])
    return base.astype(np.uint8), (base_filled | new_only)


def densify_with_neighbors(cfg):
    dz = cfg["densify"]
    if not dz["enabled"]:
        return []

    pairs = find_pairs(cfg["visible_dir"], cfg["thermal_dir"])
    by_idx = {idx: (v, t) for idx, v, t in pairs}
    warped_dir = os.path.join(cfg["register_results_dir"], "warped_thermal")
    dense_dir = os.path.join(cfg["register_results_dir"], "overlays_dense")
    os.makedirs(dense_dir, exist_ok=True)

    have_warped = {idx for idx in by_idx if os.path.exists(os.path.join(warped_dir, f"{idx}_T_warped.png"))}
    if not have_warped:
        print("No warped_thermal outputs found yet -- run Step 2b (apply) first.")
        return []

    if dz.get("skip_already_processed", True) and os.path.isdir(dense_dir):
        done = {fn[:-len("_overlay_dense.png")] for fn in os.listdir(dense_dir) if fn.endswith("_overlay_dense.png")}
        pending = sorted((have_warped - done), key=int)
    else:
        pending = sorted(have_warped, key=int)

    limit = dz.get("max_pairs_per_run")
    run_now = pending[:limit] if limit is not None else pending
    print(f"{len(have_warped)} candidates with warped thermal | {len(have_warped) - len(pending)} already "
          f"densified | densifying {len(run_now)} this session"
          + (f" | {len(pending) - len(run_now)} will remain for next run" if len(pending) > len(run_now) else ""))
    if not run_now:
        return []

    # ---- 1) detect_features on EVERY warped-thermal image (global pool --
    # BoW retrieval needs the whole corpus, not just a local neighborhood
    # of run_now). Cached, so this is a one-time cost across all runs. ----
    detector, norm_type = build_feature_detector(cfg)
    print(f"1) detect_features: {dz['feature_detector'].upper()} on {len(have_warped)} images (cached to disk)...")
    feats = {}
    for idx in have_warped:
        vpath, _ = by_idx[idx]
        feats[idx] = extract_features(idx, vpath, detector, cfg)

    # ---- 2) GLOBAL candidate selection via Bag-of-Visual-Words retrieval ----
    print("2) BoW candidate selection: building/loading vocabulary + per-image histograms...")
    vocab = build_or_load_vocabulary(cfg, feats)
    bow_vectors = compute_bow_vectors(feats, vocab)
    candidates_by_idx = bow_top_candidates(bow_vectors, dz["top_m_candidates"])

    # ---- 3) match_features on candidate pairs (verified via RANSAC) ----
    print("3) match_features: verifying BoW-selected candidate pairs...")
    involved = set(run_now)
    for idx in run_now:
        involved.update(candidates_by_idx.get(idx, []))

    uf = UnionFind()
    pair_cache = {}
    computed_pairs = set()
    for idx in involved:
        for nidx in candidates_by_idx.get(idx, []):
            key = (min(idx, nidx), max(idx, nidx))
            if key in computed_pairs:
                continue
            computed_pairs.add(key)
            a, b = key
            result = match_pair(feats[a][0], feats[a][1], feats[b][0], feats[b][1], norm_type, cfg)
            if result is None:
                pair_cache[key] = (None, 0)
                continue
            H, inliers, inlier_pairs = result
            pair_cache[key] = (H, inliers)
            for i, j in inlier_pairs:
                uf.union((a, i), (b, j))
    n_verified = sum(1 for v in pair_cache.values() if v[0] is not None)
    print(f"   {n_verified}/{len(computed_pairs)} candidate pairs verified (passed RANSAC)")

    # ---- 4) create_tracks ----
    groups = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = [obs for obs in groups.values() if len({im for im, _ in obs}) >= dz["min_track_len"]]
    n_multi = sum(1 for t in tracks if len({im for im, _ in t}) >= 3)
    print(f"4) create_tracks: {len(tracks)} tracks ({n_multi} seen in 3+ images)")

    tracks_by_image = {}
    for tid, obs in enumerate(tracks):
        for im, kp in obs:
            tracks_by_image.setdefault(im, {})[kp] = tid

    def shared_track_correspondences(a, b):
        ta = tracks_by_image.get(a, {})
        tb_by_tid = {tid: kp for kp, tid in tracks_by_image.get(b, {}).items()}
        src, dst = [], []
        for kp_a, tid in ta.items():
            if tid in tb_by_tid:
                src.append(feats[a][0][kp_a])
                dst.append(feats[b][0][tb_by_tid[tid]])
        return np.array(src), np.array(dst)

    def homography_via_tracks(a, b):
        src, dst = shared_track_correspondences(a, b)
        if len(src) >= dz["min_matches"]:
            H, mask = estimate_homography_robust(src, dst, dz["ransac_thresh"], dz["ransac_confidence"])
            if H is not None:
                return H, int(mask.sum())
        key = (min(a, b), max(a, b))
        H, inl = pair_cache.get(key, (None, 0))
        if H is None:
            return None, 0
        return (H if a == key[0] else np.linalg.inv(H)), inl

    # ---- 5) fill each target from every track-reachable image ----
    print("5) filling targets from every track-reachable image...")
    n_ok, n_err = 0, 0
    for idx in run_now:
        try:
            vpath, _ = by_idx[idx]
            visible_target = load_color(vpath)
            own_bgr, own_alpha = _load_warped_thermal_bgra(os.path.join(warped_dir, f"{idx}_T_warped.png"))

            dense = visible_target.copy()
            filled = own_alpha > 0
            dense[filled] = own_bgr[filled]
            n_sources = 1
            base_coverage = float(filled.mean())

            reachable = set()
            for kp, tid in tracks_by_image.get(idx, {}).items():
                for im, _ in tracks[tid]:
                    if im != idx and im in have_warped:
                        reachable.add(im)

            scored = []
            for nidx in reachable:
                H, inl = homography_via_tracks(nidx, idx)
                if H is not None:
                    scored.append((nidx, H, inl))
            scored.sort(key=lambda t: -t[2])

            for nidx, H_n_to_target, inl in scored:
                n_bgr, n_alpha = _load_warped_thermal_bgra(os.path.join(warped_dir, f"{nidx}_T_warped.png"))
                nvpath, _ = by_idx[nidx]
                neighbor_shape = load_color(nvpath).shape
                H_full = full_res_matrix(H_n_to_target, neighbor_shape, visible_target.shape, cfg["work_size"])
                out_size = (visible_target.shape[1], visible_target.shape[0])

                warped_bgr = cv2.warpPerspective(n_bgr, H_full, out_size)
                warped_alpha = cv2.warpPerspective(n_alpha, H_full, out_size, flags=cv2.INTER_NEAREST, borderValue=0)

                dense, filled = _feather_composite(dense, filled, warped_bgr, warped_alpha, dz["feather_px"])
                n_sources += 1

            final_coverage = float(filled.mean())
            cv2.imwrite(os.path.join(dense_dir, f"{idx}_overlay_dense.png"), dense)
            print(f"[{idx}] {n_sources} source(s) (of {len(reachable)} track-reachable), "
                  f"coverage {base_coverage:.1%} -> {final_coverage:.1%}")
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[{idx}] ERROR: {e}")

    print(f"\nThis run: {n_ok} ok, {n_err} error, {len(run_now)} total")
    return run_now



def main(cfg=CONFIG):
    if cfg.get("fresh_start"):
        reset_folders(cfg)

    print("=== STEP 1: organize ===")
    organize(cfg)
    if cfg["dry_run"]:
        return

    print("\n=== STEP 2a: calibrate (runs once, reused after) ===")
    M_work, work_size, tps = calibrate(cfg)

    print("\n=== STEP 2b: apply calibrated transform (color, no LoFTR) ===")
    apply_calibration(cfg, M_work, work_size, tps)

    print("\n=== STEP 3: densify using RGB-RGB neighbor overlap ===")
    densify_with_neighbors(cfg)


if __name__ == "__main__":
    main()