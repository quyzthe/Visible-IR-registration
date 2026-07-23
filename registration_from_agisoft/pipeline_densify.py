"""
STEP 4 -- DENSIFY

Fills gaps in each visible image's thermal coverage using overlap between
<<<<<<< HEAD
NEIGHBORING visible frames. Two data sources feed this, in order:

  0) geometric_fill : images with a real Agisoft pose -- overlap and pixel
     mapping computed PURELY by geometry (ray-cast to ground/DSM), no
     feature matching at all. Validated against independent SIFT matches
     before being trusted (see validate_geometric_fill).
  import_opensfm_tracks (pipeline_opensfm_import.py) : reuses an already-
     completed OpenSfM/ODM run's own features + matches, converted into
     the same rgb_features/rgb_inlier_matches.csv cache this module reads.

No feature detection, BoW candidate selection, or feature matching is
performed IN this module anymore -- create_tracks (Union-Find) and the
fill step below consume whatever correspondences those two sources
produced. An image covered by neither source just keeps its own
single-source coverage from apply() -- it is not a failure, only a
consequence of not attempting fresh matching here.
=======
NEIGHBORING visible frames (same-modality RGB-RGB matching -- far more
reliable than cross-modal). ODM/OpenSfM-style pipeline:

  1) detect_features : SIFT keypoints+descriptors, ONCE per image, cached
  2) match candidates : GLOBAL Bag-of-Visual-Words retrieval (not GPS/index
     proximity -- a candidate pair is chosen by visual similarity across
     the WHOLE image pool)
  3) match_features   : FLANN + Lowe ratio test + homography RANSAC
  4) create_tracks     : Union-Find links verified matches across ALL pairs
     into tracks, persisted to disk (rgb_matches.csv / rgb_inlier_matches.csv)
     so later runs only match NEW pairs and tracks.csv accumulates globally
     across every run, not just the current session
  5) fill each target from every track-reachable image, feathered compositing
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da

Works on whatever warped_thermal/*.png already exists, regardless of
whether pipeline_apply.py produced it via SfM or the 2D fallback.
"""

import os
import csv

import cv2
import numpy as np

from pipeline_organize import find_pairs
<<<<<<< HEAD
from pipeline_common import load_color, estimate_homography_robust, load_warped_thermal_bgra
=======
from pipeline_common import load_gray, load_color, preprocess, full_res_matrix, estimate_homography_robust, load_warped_thermal_bgra
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da


class UnionFind:
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


<<<<<<< HEAD
=======
def build_feature_detector(dz):
    if dz["feature_detector"] == "orb":
        return cv2.ORB_create(nfeatures=dz["n_features"]), cv2.NORM_HAMMING
    return cv2.SIFT_create(nfeatures=dz["n_features"]), cv2.NORM_L2


def extract_features(idx, path, detector, cfg, dz):
    feat_dir = os.path.join(cfg["register_results_dir"], "rgb_features")
    cache_path = os.path.join(feat_dir, f"{idx}.npz")
    if dz["save_features"] and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["pts"], data["descs"]
    gray = load_gray(path)
    _, gray_p = preprocess(gray, dz.get("work_size", (640, 480)), 3.0, (8, 8))
    kps, descs = detector.detectAndCompute(gray_p, None)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
    if descs is None:
        descs = np.zeros((0, 0), np.float32)
    if dz["save_features"]:
        os.makedirs(feat_dir, exist_ok=True)
        np.savez(cache_path, pts=pts, descs=descs)
    return pts, descs


def match_pair(pts_a, descs_a, pts_b, descs_b, norm_type, dz):
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
    return H, int(mask.sum()), [good[k] for k in range(len(good)) if mask.ravel()[k]]


def build_or_load_vocabulary(cfg, dz, feats_by_idx):
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
    print(f"Building BoW vocabulary: {n_words} words from {len(all_descs):,} descriptors...")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-4)
    _, _, centers = cv2.kmeans(all_descs, n_words, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    np.savez(vocab_path, centers=centers)
    return centers


def compute_bow_vectors(feats_by_idx, vocab):
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
    idxs = list(vectors.keys())
    mat = np.stack([vectors[i] for i in idxs])
    sims = mat @ mat.T
    np.fill_diagonal(sims, -1.0)
    out = {}
    for row, idx in enumerate(idxs):
        order = np.argsort(sims[row])[::-1][:top_m]
        out[idx] = [idxs[o] for o in order if sims[row, o] > 0]
    return out


>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
def _match_status_path(cfg):
    return os.path.join(cfg["register_results_dir"], "rgb_matches.csv")


def _inlier_matches_path(cfg):
    return os.path.join(cfg["register_results_dir"], "rgb_inlier_matches.csv")


<<<<<<< HEAD
=======
def load_pair_status(cfg):
    path = _match_status_path(cfg)
    status = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                status[(row["image_a"], row["image_b"])] = (row["verified"] == "1", int(row["inliers"]))
    return status


def append_pair_status(cfg, rows):
    path = _match_status_path(cfg)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["image_a", "image_b", "verified", "inliers"])
        for a, b, v, inl in rows:
            writer.writerow([a, b, int(v), inl])


>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
def append_inlier_matches(cfg, a, b, inlier_pairs):
    path = _inlier_matches_path(cfg)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["image_a", "image_b", "kp_a", "kp_b"])
        for i, j in inlier_pairs:
            writer.writerow([a, b, i, j])


def load_all_inlier_matches(cfg):
    path = _inlier_matches_path(cfg)
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append((row["image_a"], row["image_b"], int(row["kp_a"]), int(row["kp_b"])))
    return rows


def export_tracks_csv(cfg, tracks):
    path = os.path.join(cfg["register_results_dir"], "tracks.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "image", "feature_id"])
        for tid, obs in enumerate(tracks):
            for im, kp in obs:
                writer.writerow([tid, im, kp])
    return path


def _composite(base, base_filled, add_bgr, add_alpha):
    """Hard cutoff, no blending, no mask dilation: every output pixel is
    either an exact original visible pixel or an exact original thermal
    pixel from ONE source -- never a blend of two, and never a pixel
    copied from just outside a source's true valid region. These values
    represent real temperature readings, not just colors for display, so
    nothing here is allowed to interpolate or fabricate a value."""
    new_only = (add_alpha > 0) & (~base_filled)
    base[new_only] = add_bgr[new_only]
    return base, (base_filled | new_only)


<<<<<<< HEAD
# =====================================================================
# GEOMETRIC FILL: for images with a real Agisoft visible pose, find
# overlapping neighbors and compute the exact pixel mapping PURELY by
# geometry (ray-cast to ground/DSM, reproject) -- no SIFT, no BoW, no
# feature matching at all. Falls back to nothing (caller's SIFT-based path
# picks up whatever this didn't handle) for images without a pose.
# =====================================================================

def compute_ground_footprint(pose, size, ground_z, dem=None, to_las=None):
    """Ray-casts the 4 corners + center of an image to the ground/DSM.
    Returns (5,3) ECEF points -- used only for a cheap overlap pre-check,
    not for the actual per-pixel warp (that's done separately, at full
    density, only for pairs this footprint check says might overlap)."""
    from pipeline_sfm import _ray_dem_intersect_batch
    W, H = size
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H], [W / 2, H / 2]], dtype=np.float64)
    pix = np.hstack([corners, np.ones((5, 1))]).T
    K, R, C = pose["K"], pose["R"], pose["C"]
    rays_cam = np.linalg.inv(K) @ pix
    rays_world = R.T @ rays_cam
    if dem is not None and to_las is not None:
        pts = _ray_dem_intersect_batch(C, rays_world, to_las, dem)
    else:
        up = C / np.linalg.norm(C)
        denom = up @ rays_world
        s = (up @ (ground_z - C)) / denom
        pts = C[:, None] + s[None, :] * rays_world
    return pts.T


def _footprints_overlap(fp_a, fp_b, margin=0.0):
    min_a, max_a = fp_a[:, :2].min(axis=0), fp_a[:, :2].max(axis=0)
    min_b, max_b = fp_b[:, :2].min(axis=0), fp_b[:, :2].max(axis=0)
    return not (max_a[0] + margin < min_b[0] or max_b[0] + margin < min_a[0] or
                max_a[1] + margin < min_b[1] or max_b[1] + margin < min_a[1])


def _sift_match_pair_native(path_a, path_b, work_size=(640, 480), min_matches=15):
    """Independent SIFT match between two VISIBLE images, in NATIVE pixel
    coordinates -- used only to validate the geometric model, never to
    build the actual warp (that stays pure geometry, per the design)."""
    detector = cv2.SIFT_create(nfeatures=2000)
    kps_all, descs_all, natives = [], [], []
    for p in (path_a, path_b):
        img = load_color(p)
        natives.append((img.shape[1], img.shape[0]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, work_size, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(resized)
        kp, desc = detector.detectAndCompute(clahe, None)
        kps_all.append(np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32))
        descs_all.append(desc)
    if descs_all[0] is None or descs_all[1] is None or len(descs_all[0]) < 2 or len(descs_all[1]) < 2:
        return None
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = matcher.knnMatch(descs_all[0], descs_all[1], k=2)
    good = [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return None
    W, H = work_size
    (Wa, Ha), (Wb, Hb) = natives
    src_native = np.array([kps_all[0][i] for i, _ in good]) * np.array([Wa / W, Ha / H])
    dst_native = np.array([kps_all[1][j] for _, j in good]) * np.array([Wb / W, Hb / H])
    return src_native, dst_native


def validate_geometric_fill(cfg, by_idx, visible_poses, footprints, ground_z, dem, to_las, n_samples=8):
    """Samples a handful of geometrically-overlapping pairs, SIFT-matches
    them INDEPENDENTLY (never used for the actual warp), and checks how
    well pose+ground_z/DSM reprojection agrees with those real matched
    points. This is the only thing standing between 'the math is
    self-consistent' and 'this is actually correct for this dataset' --
    do not skip it, and do not trust geometric_fill's output without it
    having passed."""
    idxs = list(footprints.keys())
    candidate_pairs = [(idxs[i], idxs[j]) for i in range(len(idxs)) for j in range(i + 1, len(idxs))
                        if _footprints_overlap(footprints[idxs[i]], footprints[idxs[j]])]
    if not candidate_pairs:
        print("[geometric_fill] No overlapping pairs available to validate against -- REJECTING "
              "the geometric path (can't confirm it's correct, so don't trust it).")
        return False

    rng = np.random.default_rng(0)
    sample_n = min(n_samples, len(candidate_pairs))
    sample = [candidate_pairs[i] for i in rng.choice(len(candidate_pairs), sample_n, replace=False)]

    errs = []
    for idx_a, idx_b in sample:
        va, vb = by_idx[idx_a][0], by_idx[idx_b][0]
        m = _sift_match_pair_native(va, vb)
        if m is None:
            continue
        src_native, dst_native = m
        pose_a, pose_b = visible_poses[os.path.basename(va)], visible_poses[os.path.basename(vb)]

        pix = np.hstack([src_native, np.ones((len(src_native), 1))]).T
        rays_cam = np.linalg.inv(pose_a["K"]) @ pix
        rays_world = pose_a["R"].T @ rays_cam
        if dem is not None and to_las is not None:
            from pipeline_sfm import _ray_dem_intersect_batch
            pts_world = _ray_dem_intersect_batch(pose_a["C"], rays_world, to_las, dem)
        else:
            up = pose_a["C"] / np.linalg.norm(pose_a["C"])
            denom = up @ rays_world
            s = (up @ (ground_z - pose_a["C"])) / denom
            pts_world = pose_a["C"][:, None] + s[None, :] * rays_world

        pts_cam_b = pose_b["R"] @ pts_world + pose_b["t"][:, None]
        pts_img_b = pose_b["K"] @ pts_cam_b
        pred_b = (pts_img_b[:2] / pts_img_b[2]).T
        errs.append(np.linalg.norm(pred_b - dst_native, axis=1))

    if not errs:
        print("[geometric_fill] Could not get enough independent SIFT matches on the sampled pairs "
              "to validate -- REJECTING the geometric path (can't confirm it's correct).")
        return False

    all_errs = np.concatenate(errs)
    rmse = float(np.sqrt(np.mean(all_errs ** 2)))
    median = float(np.median(all_errs))
    threshold = cfg.get("agisoft", {}).get("geometric_fill_max_rmse_px", 50.0)
    print(f"[geometric_fill] Validation: {len(errs)} sampled pairs, RMSE={rmse:.1f}px, "
          f"median={median:.1f}px (native resolution, threshold={threshold}px)")
    if rmse > threshold:
        print(f"[geometric_fill] REJECTED: RMSE exceeds threshold -- pose error, wrong ground_z, or "
              f"DSM misalignment is producing real geometric error. Falling back to SIFT-based "
              f"densify for every image instead of trusting this.")
        return False
    print("[geometric_fill] Validation passed -- proceeding with the geometric path.")
    return True


def geometric_fill(cfg, pairs, warped_dir, dense_dir, run_now):
    """Returns the set of indices it successfully densified. Anything not
    in that set (no pose, or geometric setup failed entirely) is left for
    the SIFT-based fallback path in densify() below."""
    import pipeline_agisoft as pag
    from pipeline_sfm import build_projective_warp_map

    try:
        visible_poses = pag.load_organized_visible_poses(cfg, pairs)
    except Exception as e:
        print(f"[geometric_fill] Could not load Agisoft poses ({e}) -- skipping geometric pass entirely.")
        return set()
    if not visible_poses:
        print("[geometric_fill] No Agisoft poses available -- skipping geometric pass entirely.")
        return set()

    C_mean = np.mean([p["C"] for p in visible_poses.values()], axis=0)
    alt_hint = cfg.get("agisoft", {}).get("altitude_hint_m", 56.3)
    ground_z = C_mean - alt_hint * (C_mean / np.linalg.norm(C_mean))

    dem = to_las = None
    dcfg = cfg.get("dem", {"enabled": False})
    if dcfg.get("enabled"):
        try:
            from pipeline_dem import load_dem_from_las, build_ecef_to_las_transform, validate_dem_alignment
            candidate_dem = load_dem_from_las(dcfg["las_path"], dcfg["resolution_m"])
            candidate_transform = build_ecef_to_las_transform(cfg["agisoft"].get("dem_epsg"))
            if candidate_transform is not None and validate_dem_alignment(candidate_transform, visible_poses, candidate_dem):
                dem, to_las = candidate_dem, candidate_transform
        except Exception as e:
            print(f"[geometric_fill] DEM unavailable ({e}) -- using flat plane.")

    by_idx = {idx: (v, t) for idx, v, t in pairs}
    posed_idx = [idx for idx, v, t in pairs if os.path.basename(v) in visible_poses]
    if not posed_idx:
        return set()
    print(f"[geometric_fill] {len(posed_idx)}/{len(pairs)} images have a real 3D pose -- "
          f"computing overlap by geometry, no feature matching.")

    sample_shape = load_color(by_idx[posed_idx[0]][0]).shape
    native_size = (sample_shape[1], sample_shape[0])
    footprints = {}
    for idx in posed_idx:
        vname = os.path.basename(by_idx[idx][0])
        footprints[idx] = compute_ground_footprint(visible_poses[vname], native_size, ground_z, dem, to_las)

    if not validate_geometric_fill(cfg, by_idx, visible_poses, footprints, ground_z, dem, to_las):
        return set()

    have_warped = {idx for idx in posed_idx if os.path.exists(os.path.join(warped_dir, f"{idx}_T_warped.png"))}
    target_list = [idx for idx in run_now if idx in posed_idx]

    handled = set()
    for idx in target_list:
        vpath, _ = by_idx[idx]
        vname = os.path.basename(vpath)
        target_pose = visible_poses[vname]
        visible_target = load_color(vpath)
        out_size = (visible_target.shape[1], visible_target.shape[0])

        if idx in have_warped:
            own_bgr, own_alpha = load_warped_thermal_bgra(os.path.join(warped_dir, f"{idx}_T_warped.png"))
            dense = visible_target.copy()
            filled = own_alpha > 0
            dense[filled] = own_bgr[filled]
            n_sources = 1
        else:
            dense = visible_target.copy()
            filled = np.zeros(visible_target.shape[:2], dtype=bool)
            n_sources = 0

        candidates = [n for n in have_warped if n != idx and _footprints_overlap(footprints[idx], footprints[n])]
        for nidx in candidates:
            nname = os.path.basename(by_idx[nidx][0])
            neighbor_pose = visible_poses[nname]
            n_bgr, n_alpha = load_warped_thermal_bgra(os.path.join(warped_dir, f"{nidx}_T_warped.png"))

            map_x, map_y = build_projective_warp_map(target_pose, neighbor_pose, ground_z, out_size, 20, dem, to_las)
            warped_bgr = cv2.remap(n_bgr, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_alpha = cv2.remap(n_alpha, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            dense, filled = _composite(dense, filled, warped_bgr, warped_alpha)
            n_sources += 1

        cv2.imwrite(os.path.join(dense_dir, f"{idx}_overlay_dense.png"), dense)
        print(f"[{idx}] geometric: {n_sources} source(s), coverage {filled.mean():.1%}"
              + (" [DSM]" if dem is not None else " [flat plane]"))
        handled.add(idx)

    return handled


=======
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
def densify(cfg):
    dz = cfg["densify"]
    if not dz["enabled"]:
        return []
    dz = dict(dz)
<<<<<<< HEAD
=======
    dz["work_size"] = cfg.get("calibrate_2d", {}).get("work_size", (640, 480))
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da

    pairs = find_pairs(cfg["visible_dir"], cfg["thermal_dir"])
    by_idx = {idx: (v, t) for idx, v, t in pairs}
    warped_dir = os.path.join(cfg["register_results_dir"], "warped_thermal")
    dense_dir = os.path.join(cfg["register_results_dir"], "overlays_dense")
    os.makedirs(dense_dir, exist_ok=True)

    have_warped = {idx for idx in by_idx if os.path.exists(os.path.join(warped_dir, f"{idx}_T_warped.png"))}
    if not have_warped:
        print("No warped_thermal outputs found yet -- run apply() first.")
        return []

    if dz.get("skip_already_processed", True) and os.path.isdir(dense_dir):
        done = {fn[:-len("_overlay_dense.png")] for fn in os.listdir(dense_dir) if fn.endswith("_overlay_dense.png")}
        pending = sorted((have_warped - done), key=int)
    else:
        pending = sorted(have_warped, key=int)

    limit = dz.get("max_pairs_per_run")
    run_now = pending[:limit] if limit is not None else pending
    print(f"{len(have_warped)} candidates | {len(have_warped) - len(pending)} already densified | "
          f"densifying {len(run_now)} this session")
    if not run_now:
        return []

<<<<<<< HEAD
    from pipeline_opensfm_import import import_opensfm_tracks
    import_opensfm_tracks(cfg)

    print("0) geometric_fill: trying real poses first (no feature matching)...")
    handled_geometrically = geometric_fill(cfg, pairs, warped_dir, dense_dir, run_now)
    run_now = [idx for idx in run_now if idx not in handled_geometrically]
    print(f"   {len(handled_geometrically)} densified geometrically | {len(run_now)} left for the "
          f"imported-track fallback below (no pose available, or geometric setup failed)")
    if not run_now:
        return list(handled_geometrically)

    # ---- everything below consumes whatever correspondences
    # import_opensfm_tracks() populated -- no feature detection or matching
    # happens in this module. An image with neither a pose nor imported
    # match data just keeps its own single-source coverage from apply(). ----
    print("1) loading feature caches (populated by opensfm_import, if any)...")
    feats = {}
    for idx in have_warped:
        cache_path = os.path.join(cfg["register_results_dir"], "rgb_features", f"{idx}.npz")
        if not os.path.exists(cache_path):
            continue
        try:
            data = np.load(cache_path)
            feats[idx] = (data["pts"], data["descs"])
        except Exception as e:
            print(f"[{idx}] cache {cache_path} is corrupted ({e}) -- skipping this image's features.")
    print(f"   {len(feats)}/{len(have_warped)} images have imported feature data")

    inlier_rows = load_all_inlier_matches(cfg)
    print(f"2) create_tracks: {len(inlier_rows)} imported correspondences...")

    # ---- graph connectivity diagnostic: is the imported match graph one
    # big connected piece, or fragmented into small isolated clusters
    # (which would limit coverage regardless of how much data exists) ----
    uf_img = UnionFind()
    all_matched_images = set()
    for a, b, _, _ in inlier_rows:
        uf_img.union(a, b)
        all_matched_images.add(a)
        all_matched_images.add(b)
    if all_matched_images:
        img_groups = {}
        for im in all_matched_images:
            img_groups.setdefault(uf_img.find(im), set()).add(im)
        component_sizes = sorted((len(g) for g in img_groups.values()), reverse=True)
=======
    detector, norm_type = build_feature_detector(dz)
    print(f"1) detect_features: {dz['feature_detector'].upper()} on {len(have_warped)} images...")
    feats = {idx: extract_features(idx, by_idx[idx][0], detector, cfg, dz) for idx in have_warped}

    print("2) BoW candidate selection...")
    vocab = build_or_load_vocabulary(cfg, dz, feats)
    bow_vectors = compute_bow_vectors(feats, vocab)
    candidates_by_idx = bow_top_candidates(bow_vectors, dz["top_m_candidates"])

    print("3) match_features (new pairs only)...")
    pair_status = load_pair_status(cfg)
    involved = set(run_now)
    for idx in run_now:
        involved.update(candidates_by_idx.get(idx, []))

    status_rows, n_new_verified = [], 0
    for idx in involved:
        for nidx in candidates_by_idx.get(idx, []):
            key = (min(idx, nidx), max(idx, nidx))
            if key in pair_status:
                continue
            a, b = key
            result = match_pair(feats[a][0], feats[a][1], feats[b][0], feats[b][1], norm_type, dz)
            if result is None:
                pair_status[key] = (False, 0)
                status_rows.append((a, b, False, 0))
                continue
            H, inliers, inlier_pairs = result
            pair_status[key] = (True, inliers)
            status_rows.append((a, b, True, inliers))
            append_inlier_matches(cfg, a, b, inlier_pairs)
            n_new_verified += 1
    if status_rows:
        append_pair_status(cfg, status_rows)
    print(f"   {len(status_rows)} new pairs attempted ({n_new_verified} verified) | "
          f"{sum(1 for v,_ in pair_status.values() if v)} verified total")

    # ---- graph connectivity diagnostic: is the match graph one big
    # connected piece, or fragmented into small isolated clusters (which
    # would explain poor coverage even with plenty of accumulated data,
    # since images in different pieces can never reach each other) ----
    uf_img = UnionFind()
    all_matched_images = set()
    for (a, b), (verified, _) in pair_status.items():
        if verified:
            uf_img.union(a, b)
            all_matched_images.add(a)
            all_matched_images.add(b)
    img_groups = {}
    for im in all_matched_images:
        img_groups.setdefault(uf_img.find(im), set()).add(im)
    component_sizes = sorted((len(g) for g in img_groups.values()), reverse=True)
    if component_sizes:
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
        shown = component_sizes[:10]
        print(f"   Match graph connectivity: {len(component_sizes)} connected component(s) "
              f"over {len(all_matched_images)} images | sizes: {shown}"
              + ("..." if len(component_sizes) > 10 else ""))
        if len(component_sizes) > 1:
            print(f"   [WARN] FRAGMENTED into {len(component_sizes)} pieces -- images in different "
<<<<<<< HEAD
                  f"pieces cannot fill each other no matter how much data has accumulated elsewhere.")

    uf = UnionFind()
    for a, b, kp_a, kp_b in inlier_rows:
=======
                  f"pieces cannot fill each other no matter how much data has accumulated elsewhere. "
                  f"Raise densify.top_m_candidates to give more images a chance at a verified edge "
                  f"into the largest component ({component_sizes[0]} images).")

    uf = UnionFind()
    for a, b, kp_a, kp_b in load_all_inlier_matches(cfg):
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
        uf.union((a, kp_a), (b, kp_b))
    groups = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = [obs for obs in groups.values() if len({im for im, _ in obs}) >= dz["min_track_len"]]
    tracks_path = export_tracks_csv(cfg, tracks)
<<<<<<< HEAD
    print(f"   {len(tracks)} tracks, exported to {tracks_path}")
=======
    print(f"4) create_tracks: {len(tracks)} tracks, exported to {tracks_path}")
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da

    tracks_by_image = {}
    for tid, obs in enumerate(tracks):
        for im, kp in obs:
            tracks_by_image.setdefault(im, {})[kp] = tid

    def shared_track_correspondences(a, b):
        ta = tracks_by_image.get(a, {})
        tb_by_tid = {tid: kp for kp, tid in tracks_by_image.get(b, {}).items()}
        src, dst = [], []
        for kp_a, tid in ta.items():
            if tid in tb_by_tid and a in feats and b in feats:
                src.append(feats[a][0][kp_a])
                dst.append(feats[b][0][tb_by_tid[tid]])
        return np.array(src), np.array(dst)

    def homography_via_tracks(a, b):
        src, dst = shared_track_correspondences(a, b)
        if len(src) < dz["min_matches"]:
            return None, 0
        H, mask = estimate_homography_robust(src, dst, dz["ransac_thresh"], dz["ransac_confidence"])
        return (None, 0) if H is None else (H, int(mask.sum()))

<<<<<<< HEAD
    print("3) filling targets...")
=======
    print("5) filling targets...")
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
    n_ok, n_err = 0, 0
    for idx in run_now:
        try:
            vpath, _ = by_idx[idx]
            visible_target = load_color(vpath)
            own_bgr, own_alpha = load_warped_thermal_bgra(os.path.join(warped_dir, f"{idx}_T_warped.png"))
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

<<<<<<< HEAD
            for nidx, H_full, inl in scored:
                # H_full already maps neighbor-native -> target-native directly:
                # feats[...] are native-resolution points (opensfm_import writes
                # denormalized pixel coords using each image's real width/height),
                # so the fitted homography needs no further rescaling here.
                n_bgr, n_alpha = load_warped_thermal_bgra(os.path.join(warped_dir, f"{nidx}_T_warped.png"))
=======
            for nidx, H_n_to_target, inl in scored:
                n_bgr, n_alpha = load_warped_thermal_bgra(os.path.join(warped_dir, f"{nidx}_T_warped.png"))
                nvpath, _ = by_idx[nidx]
                neighbor_shape = load_color(nvpath).shape
                H_full = full_res_matrix(H_n_to_target, neighbor_shape, visible_target.shape, dz["work_size"])
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
                out_size = (visible_target.shape[1], visible_target.shape[0])
                warped_bgr = cv2.warpPerspective(n_bgr, H_full, out_size, flags=cv2.INTER_NEAREST)
                warped_alpha = cv2.warpPerspective(n_alpha, H_full, out_size, flags=cv2.INTER_NEAREST, borderValue=0)
                dense, filled = _composite(dense, filled, warped_bgr, warped_alpha)
                n_sources += 1

            final_coverage = float(filled.mean())
            cv2.imwrite(os.path.join(dense_dir, f"{idx}_overlay_dense.png"), dense)
            print(f"[{idx}] {n_sources} source(s), coverage {base_coverage:.1%} -> {final_coverage:.1%}")
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[{idx}] ERROR: {e}")

<<<<<<< HEAD
    print(f"\nThis run: {len(handled_geometrically)} geometric, {n_ok} track-based ok, {n_err} error, "
          f"{len(handled_geometrically) + len(run_now)} total")
    return list(handled_geometrically) + run_now
=======
    print(f"\nThis run: {n_ok} ok, {n_err} error, {len(run_now)} total")
    return run_now
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
