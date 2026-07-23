"""
Organize a raw DJI DCIM mission folder into visible/ + thermal/ subfolders,
then calibrate and apply visible-thermal registration.

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

STEP 3 (densifying gaps using neighboring visible-frame overlap) now lives
in its own script, step3_densify.py -- it only READS this script's outputs
(visible_dir/thermal_dir/register_results/warped_thermal), so it never
needs Step 1/2 to run again once warped_thermal exists.

Run with: python organize_and_register.py
"""

import os
import re
import csv
import shutil
import math

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
    source_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1",
    visible_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\visible",
    thermal_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\thermal",
    register_results_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\register_results",
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
    lens_self_calibration=dict(    # OPTIONAL: self-calibrate a radial (lens) distortion coefficient
                                    # for the thermal camera directly from these same natural-scene
                                    # correspondences -- no chessboard needed (plane-based self-
                                    # calibration of radial distortion; see Truong et al. 2017 for the
                                    # chessboard-based version, adapted here for our data). Only
                                    # modeled on the thermal side: consumer/prosumer visible cameras
                                    # (DJI included) typically already correct lens distortion
                                    # in-camera on the JPEG; thermal sensors generally don't.
                                    # SAFETY: only adopted if it beats the plain homography on a
                                    # held-out validation split -- never allowed to reduce accuracy.
        enabled=True,
        min_points=120,             # need a reasonably large inlier set before attempting this
        val_fraction=0.2,           # held-out fraction used to validate before adopting
        min_improvement_px=0.05,    # must beat the baseline by at least this much RMSE to be adopted
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
    max_pairs_per_run=1,           # None = process everything found; int = only this many NEW pairs this run
    skip_already_processed=True,   # skip pairs that already have an overlay output on disk
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
    return _maybe_undistort(img, path)


def load_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return _maybe_undistort(img, path)


# =====================================================================
# DJI FACTORY LENS CALIBRATION (XMP DewarpData) -- Truong et al. (ICCVW
# 2017, "Registration of RGB and thermal point clouds generated by SfM")
# calibrate a chessboard by hand to get each camera's intrinsics + Brown-
# Conrady distortion, then undistort before registering. DJI's mapping-
# oriented cameras (P4 Multispectral, Mavic 3M, P4 RTK, and possibly the
# M4T's own cameras) embed that exact factory calibration in every photo's
# XMP "DewarpData" field -- no chessboard needed, it's already in the file.
# Undistorting BEFORE matching removes lens distortion as a source of
# error entirely, instead of the TPS step trying to absorb it empirically
# after the fact. Applied globally through load_gray/load_color (a single
# switch, not threaded through every call site) so calibration and apply
# always see identical geometry -- if that were inconsistent between the
# two stages it would silently corrupt the fit, which is exactly what
# "never reduce accuracy" means to guard against here. OFF by default
# (_DEWARP_ENABLED=False): flip it on directly in code if the thermal
# sensor's own XMP DewarpData should be applied.
# =====================================================================

_DEWARP_ENABLED = False
_DEWARP_CACHE = {}


def parse_dji_dewarp(path):
    """Read DJI XMP DewarpData: 'fx,fy,cx,cy,k1,k2,p1,p2,k3' (often prefixed
    with a version/date and a semicolon). Returns a dict or None if absent/
    unparseable -- absence is normal (e.g. many thermal sensors don't have
    this), not an error."""
    try:
        with open(path, "rb") as f:
            head = f.read(131072)  # DewarpData lives in the XMP block near the file start
        text = head.decode("latin-1", errors="ignore")
        m = (re.search(r'DewarpData[>="]+([^"<]+)', text)
             or re.search(r'drone-dji:DewarpData="([^"]+)"', text))
        if not m:
            return None
        raw = m.group(1).strip()
        nums_str = raw.split(";")[-1]
        vals = [float(x) for x in nums_str.split(",") if x.strip()]
        if len(vals) != 9:
            return None
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = vals
        return dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
    except Exception:
        return None


def _get_dewarp(path):
    if path not in _DEWARP_CACHE:
        _DEWARP_CACHE[path] = parse_dji_dewarp(path)
    return _DEWARP_CACHE[path]


def undistort_dji(img, dw):
    h, w = img.shape[:2]
    cx_abs = w / 2.0 + dw["cx"]
    cy_abs = h / 2.0 + dw["cy"]
    K = np.array([[dw["fx"], 0, cx_abs], [0, dw["fy"], cy_abs], [0, 0, 1]], dtype=np.float64)
    D = np.array([dw["k1"], dw["k2"], dw["p1"], dw["p2"], dw["k3"]], dtype=np.float64)
    return cv2.undistort(img, K, D)


def _maybe_undistort(img, path):
    if not _DEWARP_ENABLED:
        return img
    dw = _get_dewarp(path)
    if dw is None:
        return img
    return undistort_dji(img, dw)


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
# LENS SELF-CALIBRATION: radial distortion coefficient for the thermal
# camera, estimated directly from the pooled natural-scene correspondences
# (no chessboard needed) -- "plane-based self-calibration of radial
# distortion" (e.g. Thirthala & Pollefeys 2005; Fitzgibbon 2001's one-
# parameter division model). Modeled on the thermal side only: consumer/
# prosumer visible cameras (DJI included) typically apply lens correction
# in-camera to the JPEG already, thermal sensors generally don't.
# =====================================================================

def undistort_division_model(pts, center, k, scale):
    """Distorted pixel coords -> undistorted. One-parameter division model
    (closed form, no iteration): x_u = c + (x_d-c) / (1 + k*(r_d/scale)^2).
    center = distortion center (assumed = image center, the standard
    simplifying assumption when no chessboard is available). scale = a
    normalizing radius (half image diagonal) so k stays O(1) regardless of
    resolution."""
    d = pts - center
    r2 = (d[:, 0] ** 2 + d[:, 1] ** 2) / (scale ** 2)
    factor = 1.0 / (1.0 + k * r2)
    return center + d * factor[:, None]


def distort_division_model_forward(pts, center, k, scale):
    """Undistorted -> distorted, first-order approximation of the division
    model's inverse (the exact inverse has no closed form). Accurate for
    the mild distortion typical of real camera lenses; any small residual
    approximation error gets absorbed by the TPS local-refinement pass
    that runs after this."""
    d = pts - center
    r2 = (d[:, 0] ** 2 + d[:, 1] ** 2) / (scale ** 2)
    factor = 1.0 - k * r2
    return center + d * factor[:, None]


def _homog_params_to_H(params):
    h = params[:8]
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def _H_to_homog_params(H3x3):
    H = H3x3 / H3x3[2, 2]
    return np.array([H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]])


def fit_homography_with_distortion(src_pts, dst_pts, H_init, center, scale):
    """Jointly refine a homography AND a single radial-distortion
    coefficient k for the source (thermal) camera, minimizing reprojection
    error with a robust (soft_l1) loss."""
    from scipy.optimize import least_squares

    def residuals(params):
        H = _homog_params_to_H(params[:8])
        k = params[8]
        undist = undistort_division_model(src_pts, center, k, scale)
        pred = apply_homogeneous(H, undist)
        return (pred - dst_pts).ravel()

    params0 = np.concatenate([_H_to_homog_params(to_3x3(H_init)), [0.0]])
    result = least_squares(residuals, params0, method="trf", loss="soft_l1", f_scale=2.0, max_nfev=3000)
    H_fit = _homog_params_to_H(result.x[:8])
    k_fit = float(result.x[8])
    return H_fit, k_fit


def _reprojection_rmse(H, k, center, scale, src_pts, dst_pts):
    pts = undistort_division_model(src_pts, center, k, scale) if k is not None else src_pts
    pred = apply_homogeneous(to_3x3(H), pts)
    return float(np.sqrt(np.mean(np.sum((pred - dst_pts) ** 2, axis=1))))


def build_undistort_map(work_size, native_shape, center_work, k, scale_work, grid_step=20):
    """Native-resolution (dst=undistorted thermal, src=original distorted
    thermal) remap for the self-calibrated radial distortion, in the
    THERMAL image's own coordinate frame (independent of the visible frame
    entirely) -- applied as a pre-pass before the usual warpPerspective(H)."""
    W, H = work_size
    Hn, Wn = native_shape[:2]
    xs = np.unique(np.append(np.arange(0, W, grid_step), W - 1))
    ys = np.unique(np.append(np.arange(0, H, grid_step), H - 1))
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    dist_pts = distort_division_model_forward(grid_pts, center_work, k, scale_work)

    map_x_work = cv2.resize(dist_pts[:, 0].reshape(gy.shape).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    map_y_work = cv2.resize(dist_pts[:, 1].reshape(gy.shape).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    sx, sy = Wn / W, Hn / H
    map_x_native = cv2.resize(map_x_work, (Wn, Hn), interpolation=cv2.INTER_LINEAR) * sx
    map_y_native = cv2.resize(map_y_work, (Wn, Hn), interpolation=cv2.INTER_LINEAR) * sy
    return map_x_native, map_y_native


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
        k_thermal = float(data["k_thermal"]) if "k_thermal" in data else 0.0
        print(f"Using existing calibration: {calib_path} "
              f"(model={saved_model}, from {int(data['n_sample_pairs'])} pairs, "
              f"{int(data['n_inliers'])}/{int(data['n_pooled_matches'])} inliers, "
              f"RMSE={float(data['rmse_px']):.2f}px{' + local TPS' if has_tps else ''}"
              f"{f' + lens k={k_thermal:.4f}' if k_thermal != 0.0 else ''}). "
              "Set calibration.recalibrate=True to redo it.")
        tps = (data["map_x"], data["map_y"]) if has_tps else None
        dist = (k_thermal, data["dist_center"], float(data["dist_scale"])) if "dist_center" in data else None
        return data["M_work"].astype(np.float64), tuple(int(v) for v in data["work_size"]), tps, dist

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

    # ---- lens self-calibration: try a radial distortion correction for the
    # thermal camera, ONLY adopted if it demonstrably beats the plain model
    # on a held-out split -- never allowed to make accuracy worse ----
    lens = cfg.get("lens_self_calibration", {"enabled": False})
    k_thermal = 0.0
    W_, H_ = cfg["work_size"]
    dist_center = np.array([W_ / 2.0, H_ / 2.0])
    dist_scale = math.hypot(W_, H_) / 2.0
    if lens.get("enabled", False) and len(inlier_pts0) >= lens.get("min_points", 120):
        rng = np.random.default_rng(0)
        n = len(inlier_pts0)
        perm = rng.permutation(n)
        n_val = max(20, int(n * lens.get("val_fraction", 0.2)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        H_train, mask_train = estimate_transform_robust(inlier_pts1[train_idx], inlier_pts0[train_idx], cfg)
        if H_train is not None:
            rmse_baseline = _reprojection_rmse(H_train, None, dist_center, dist_scale,
                                                inlier_pts1[val_idx], inlier_pts0[val_idx])
            H_dist, k_cand = fit_homography_with_distortion(
                inlier_pts1[train_idx], inlier_pts0[train_idx], H_train, dist_center, dist_scale
            )
            rmse_distortion = _reprojection_rmse(H_dist, k_cand, dist_center, dist_scale,
                                                  inlier_pts1[val_idx], inlier_pts0[val_idx])
            print(f"Lens self-calibration: held-out RMSE without={rmse_baseline:.2f}px, "
                  f"with radial distortion (k={k_cand:.4f})={rmse_distortion:.2f}px")
            if rmse_distortion < rmse_baseline - lens.get("min_improvement_px", 0.05):
                # refit on ALL inliers (not just the train split) for the final production model
                M, k_thermal = fit_homography_with_distortion(inlier_pts1, inlier_pts0, M, dist_center, dist_scale)
                rmse = _reprojection_rmse(M, k_thermal, dist_center, dist_scale, inlier_pts1, inlier_pts0)
                print(f"  -> adopted (k={k_thermal:.4f}); reprojection RMSE now {rmse:.2f}px "
                      "(this is the base transform for TPS + apply)")
            else:
                print("  -> not adopted (no reliable improvement over held-out baseline)")

    # correspondences used for TPS from here on are pre-undistorted if a
    # distortion coefficient was adopted, so TPS fits the RESIDUAL on top of
    # the (now better) homography+distortion base, not on top of raw thermal
    tps_src_pts = (undistort_division_model(inlier_pts1, dist_center, k_thermal, dist_scale)
                   if k_thermal != 0.0 else inlier_pts1)

    map_x = map_y = None
    has_tps = False
    lr = cfg["local_refinement"]
    if lr["enabled"] and len(inlier_pts0) >= lr["min_points"]:
        tps = fit_tps_correction(inlier_pts0, tps_src_pts, M, cfg["work_size"], cfg)
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
        rmse_px=rmse, has_tps=has_tps, k_thermal=k_thermal,
        dist_center=dist_center, dist_scale=dist_scale,
        map_x=map_x if has_tps else np.zeros((1, 1), dtype=np.float32),
        map_y=map_y if has_tps else np.zeros((1, 1), dtype=np.float32),
    )
    print(f"Saved calibration: {calib_path}")
    return M, cfg["work_size"], ((map_x, map_y) if has_tps else None), (k_thermal, dist_center, dist_scale)


# =====================================================================
# STEP 2b: APPLY -- warp ORIGINAL COLOR thermal onto ORIGINAL COLOR
# visible using the calibrated transform (+ optional local TPS). No LoFTR,
# no per-pair fit.
# =====================================================================

def apply_calibration(cfg, M_work, work_size, tps=None, distortion=None):
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

    k_thermal, dist_center_work, dist_scale_work = distortion if distortion is not None else (0.0, None, None)

    n_ok, n_err = 0, 0
    for idx, vpath, tpath in run_now:
        try:
            visible_color = load_color(vpath)
            thermal_color = load_color(tpath)
            out_size = (visible_color.shape[1], visible_color.shape[0])

            if k_thermal != 0.0:
                # self-calibrated lens correction, in thermal's own native frame,
                # BEFORE the usual homography warp -- everything below is unchanged
                map_x_u, map_y_u = build_undistort_map(
                    work_size, thermal_color.shape, dist_center_work, k_thermal, dist_scale_work
                )
                thermal_color = cv2.remap(thermal_color, map_x_u, map_y_u,
                                           interpolation=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

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


def main(cfg=CONFIG):
    if cfg.get("fresh_start"):
        reset_folders(cfg)

    print("=== STEP 1: organize ===")
    organize(cfg)
    if cfg["dry_run"]:
        return

    print("\n=== STEP 2a: calibrate (runs once, reused after) ===")
    M_work, work_size, tps, distortion = calibrate(cfg)

    print("\n=== STEP 2b: apply calibrated transform (color, no LoFTR) ===")
    apply_calibration(cfg, M_work, work_size, tps, distortion)

    print("\nStep 2 done. Run step3_densify.py separately to densify using "
          "neighboring visible-frame overlap (it only reads this script's outputs).")


if __name__ == "__main__":
    main()
