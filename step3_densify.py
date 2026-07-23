"""
STEP 3 (standalone) -- densify thermal coverage using a LOCAL, TRIANGULATED
mesh warp instead of one global homography per neighbor.

PREREQUISITE: Steps 1 and 2 of organize_and_register.py have already been
run (visible_dir/thermal_dir are organized, register_results/warped_thermal
exists). This script only READS those outputs, plus ODM's opensfm/ outputs
-- it never touches Step 1/2, and Step 1/2 never need to run again.

WHY A MESH INSTEAD OF ONE HOMOGRAPHY PER NEIGHBOR
--------------------------------------------------
The old Step 3 fit ONE homography per (target, neighbor) pair from every
track-shared point between them, then warped the ENTIRE neighbor image with
that single transform. That's an approximation: it assumes the whole
overlap between the two images lies on one shared plane. Real terrain
(riverbanks, vegetation, slopes) isn't planar, so the single homography is
increasingly wrong the further a point sits from wherever the fit happened
to center its error.

This version instead:
  1) takes EVERY matched point between a target and a neighbor (via ODM's
     matches, bridged through tracks exactly as before -- no point is
     dropped in favor of "the best" subset the way a single RANSAC fit
     implicitly discards inliers/outliers),
  2) builds a Delaunay triangulation over those points in the TARGET image,
  3) warps each small triangle independently with its own affine transform,
     fit from just that triangle's 3 corners.

A triangle a few dozen pixels across is planar to an excellent approximation
almost everywhere, even where the *overall* overlap region is not -- so this
adapts to real local terrain instead of assuming one shared plane, and it
literally uses every match as a control point, not just whichever ones a
global model happened to fit best.

Coverage still only extends to the convex hull of each neighbor's matched
points (same as a homography-based warp would also be limited to its
overlap region) -- an image with very few matches to a given neighbor
simply contributes little there, same as before.

MEMORY: everything below works on a CROPPED region sized to each pair's
actual matched-point extent, never a full native-resolution (e.g.
4032x3024) canvas per neighbor. An earlier version allocated a full-frame
output canvas (~48MB) for every single neighbor, and a target image with
dozens of reachable neighbors could easily need >1GB of allocate/discard
churn just for itself -- repeated across hundreds of target images in one
run, that fragments/exhausts memory even when each individual allocation
looks modest (which is exactly the 'OutOfMemoryError ... 36MB' pattern this
produced). Working in small, actual-extent-sized crops avoids that.

Run with: python step3_densify.py
"""

import os
import re
import csv
import json
import gzip
import pickle

import cv2
import numpy as np
from scipy.spatial import Delaunay


# =====================================================================
# CONFIG
# =====================================================================

CONFIG = dict(
    # ---- must match Step 1/2's paths exactly -- this script only reads them ----
    visible_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\visible",
    thermal_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\thermal",
    register_results_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\register_results",

    densify=dict(
        enabled=True,

        # ---- 1) load ODM feature points (opensfm/features/*.npz), NATIVE
        # resolution throughout -- no work_size downscale/upscale round trip,
        # since this script never runs LoFTR/CLAHE (Step 2's job, already done) ----
        save_features=True,          # cache points to register_results/rgb_features/<idx>.npz

        # ---- 2) candidate selection + 3) match verification: both come
        # directly from ODM's opensfm/matches/*.pkl.gz -- see CONFIG['odm'] ----
        ransac_thresh=3.0,           # only used if odm.verify_planar_homography=True
        ransac_confidence=0.999,
        min_matches=15,              # min shared points a (target, neighbor) pair needs to
                                      # contribute anything -- also the practical floor for a
                                      # triangulation that's actually worth having (Delaunay
                                      # itself only needs 4, but that's too sparse to trust)

        # ---- 4) create_tracks ----
        min_track_len=2,             # keep tracks observed in at least this many images

        # ---- 5) fill: mesh-warp every reachable neighbor's matched points into
        # a Delaunay triangulation, composite triangle-warped results in order
        # of most-shared-points-first, dilate+feather at each neighbor's outer
        # boundary the same way as before -- all in a CROPPED region sized to
        # each pair's actual point extent, never a full native-resolution canvas ----
        mask_dilate_px=8,            # grow each neighbor's mesh-covered silhouette (and the
                                      # target's own warped_thermal) by this many px before
                                      # compositing -- the triangulated warp is locally accurate,
                                      # but coverage still hard-stops at the convex hull of that
                                      # neighbor's matched points, and two neighbors' hulls won't
                                      # perfectly touch at the pixel -- a small overlap margin
                                      # closes that instead of leaving a visible seam.
        feather_px=15,               # soft-blend width (px, native resolution) at the boundary
                                      # where a new source fills in. 0 = hard cutoff, no blend.

        max_pairs_per_run=100,       # None = process everything pending; int = only this many
                                      # NEW target images this run
        skip_already_processed=True,
    ),

    odm=dict(
        enabled=True,
        project_dir=r"E:\drone_090426\Raw_images\DCIM_1\feed_odm",  # ODM project root; expects
                                                       # opensfm/reconstruction.json,
                                                       # opensfm/camera_models.json,
                                                       # opensfm/features/, opensfm/matches/
                                                       # underneath it (standard ODM/OpenSfM layout)
        reconstruction_path=None,      # override for reconstruction.json's location
        camera_models_path=None,       # override for camera_models.json's location -- FALLBACK
                                        # only, has no shots, so feature/match reuse stays
                                        # disabled without reconstruction.json specifically
        features_path_template="{project_dir}/opensfm/features/{filename}.features.npz",
        matches_path_template="{project_dir}/opensfm/matches/{filename}_matches.pkl.gz",
        verify_planar_homography=False,  # False (default): trust ODM's matches directly as track
                                          # correspondences, no extra RANSAC pass. True:
                                          # additionally re-verify each ODM-matched pair with a
                                          # planar-homography RANSAC before accepting it into the
                                          # track graph -- stricter, but throws away correct
                                          # matches over non-flat terrain, and is less necessary
                                          # now that Step 5 doesn't assume planarity either; kept
                                          # as an option for diagnosing bad tracks specifically.
    ),
)

_PAIR_RE_TEMPLATE = r"_(\d+)_{tag}\.(jpg|jpeg|png|tif|tiff)$"


# =====================================================================
# SHARED HELPERS (same behavior as organize_and_register.py -- copied here
# so this script is fully standalone and never imports Step 1/2's file)
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


def load_color(path):
    """Visible images are used exactly as-shot -- never undistorted (see
    organize_and_register.py's history: undistorting them bought no
    accuracy this pipeline needs, left a black border on every output, and
    put ODM's feature points -- always in the original as-shot pixel space
    -- out of sync with the pixels)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _load_warped_thermal_bgra(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(
            f"{os.path.basename(path)} has no alpha channel (shape={img.shape}) -- it wasn't "
            "written by the current Step 2b (apply_calibration), which saves BGRA. Regenerate "
            "warped_thermal/ via Step 2b before running this script."
        )
    return img[:, :, :3], img[:, :, 3]


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


# =====================================================================
# ODM/OpenSfM REUSE -- features and feature matches computed by ODM's own
# SfM run on the visible images, reused here instead of recomputed. An
# image ODM didn't reconstruct is simply excluded, no fallback.
# =====================================================================

def load_odm_reconstruction(path):
    """Load an OpenSfM/ODM reconstruction.json -- a LIST of reconstructions
    (ODM splits the mission into several disconnected ones if parts of the
    flight didn't overlap enough to bundle-adjust together). Returns
    (cameras, shots) from the largest one; warns if there's more than one so
    a partial submodel doesn't silently look complete."""
    with open(path) as f:
        recons = json.load(f)
    if not recons:
        raise RuntimeError(f"{path}: no reconstructions found (empty list).")
    if len(recons) > 1:
        sizes = [len(r.get("shots", {})) for r in recons]
        print(f"[WARN] [ODM] reconstruction.json has {len(recons)} disconnected reconstructions "
              f"(shot counts: {sizes}) -- using the largest. Images in the smaller ones won't get "
              "an ODM camera model / features / matches here.")
        recons = sorted(recons, key=lambda r: len(r.get("shots", {})), reverse=True)
    r0 = recons[0]
    return r0.get("cameras", {}), r0.get("shots", {})


def _load_idx_to_original_filename(cfg):
    """idx ('0001') -> original visible filename basename, read from
    file_mapping.csv (written by Step 1). Bridges our own DJI_<idx>_V
    naming to the filenames ODM's reconstruction/features/matches use."""
    mapping_path = os.path.join(cfg["register_results_dir"], "file_mapping.csv")
    out = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, newline="") as f:
            for row in csv.DictReader(f):
                out[row["index"]] = os.path.basename(row["original_visible"])
    return out


_ODM_STATE_CACHE = {}


def _get_odm_state(cfg):
    """Lazy singleton: loads the ODM reconstruction + the idx<->original-
    filename bridge once, cached by project_dir."""
    odm_cfg = cfg.get("odm") or {}
    if not odm_cfg.get("enabled", False):
        return None
    project_dir = odm_cfg.get("project_dir")
    if project_dir in _ODM_STATE_CACHE:
        return _ODM_STATE_CACHE[project_dir]

    recon_path = odm_cfg.get("reconstruction_path") or os.path.join(project_dir, "opensfm", "reconstruction.json")
    cammodels_path = odm_cfg.get("camera_models_path") or os.path.join(project_dir, "opensfm", "camera_models.json")

    state = dict(cameras={}, shots={})
    if os.path.exists(recon_path):
        state["cameras"], state["shots"] = load_odm_reconstruction(recon_path)
        print(f"[ODM] loaded reconstruction.json: {len(state['cameras'])} camera(s), "
              f"{len(state['shots'])} shot(s) -- {recon_path}")
    elif os.path.exists(cammodels_path):
        with open(cammodels_path) as f:
            state["cameras"] = json.load(f)
        print(f"[WARN] [ODM] no reconstruction.json at {recon_path} -- using camera_models.json "
              f"instead ({cammodels_path}). It has no shots, so feature/match reuse (which needs "
              "reconstruction.json's shots to know each image's width/height) stays disabled.")
    else:
        print(f"[WARN] [ODM] neither reconstruction.json nor camera_models.json found under "
              f"{project_dir} -- ODM feature/match reuse disabled for this run.")

    state["idx_to_orig"] = _load_idx_to_original_filename(cfg)
    state["orig_to_idx"] = {v: k for k, v in state["idx_to_orig"].items()}
    _ODM_STATE_CACHE[project_dir] = state
    return state


def _odm_features_path(cfg, original_filename):
    odm_cfg = cfg.get("odm") or {}
    return odm_cfg["features_path_template"].format(
        project_dir=odm_cfg.get("project_dir"), filename=original_filename)


def _odm_matches_path(cfg, original_filename):
    odm_cfg = cfg.get("odm") or {}
    return odm_cfg["matches_path_template"].format(
        project_dir=odm_cfg.get("project_dir"), filename=original_filename)


def load_odm_points_for_image(cfg, idx):
    """ODM's cached SIFT keypoint pixel coordinates for the visible image at
    this idx, in NATIVE pixel resolution (converted from OpenSfM's
    normalized [-0.5,0.5]-on-the-larger-dimension coords -- OpenSfM's
    universal feature-coordinate convention). No work_size downscale here:
    this script warps directly in native resolution throughout, so keeping
    points at native resolution from the start avoids an unnecessary
    quantization round-trip. Returns None if unavailable (image not
    ODM-reconstructed, or its features.npz isn't on disk) -- that image is
    simply excluded from densify, no fallback."""
    state = _get_odm_state(cfg)
    if not state:
        return None
    orig = state["idx_to_orig"].get(idx)
    shot = state["shots"].get(orig) if orig else None
    if orig is None or shot is None:
        return None
    cam = state["cameras"].get(shot["camera"])
    if cam is None:
        return None
    path = _odm_features_path(cfg, orig)
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    pts_norm = data["points"][:, :2].astype(np.float64)  # descriptors intentionally never read

    w, h = cam["width"], cam["height"]
    size = max(w, h)
    px = pts_norm[:, 0] * size + w / 2.0
    py = pts_norm[:, 1] * size + h / 2.0
    return np.stack([px, py], axis=1).astype(np.float32)


def load_odm_matches_for_image(cfg, idx):
    """ODM's already-verified matches for the visible image at this idx,
    against every neighbor it was matched to during ODM's own SfM run --
    translated from ODM's original filenames back to OUR idx strings (only
    neighbors that are ALSO organized on our side are usable). Returns
    {neighbor_idx: Nx2 int array of (kp_index_in_this_image,
    kp_index_in_neighbor_image)}, or None if unavailable."""
    state = _get_odm_state(cfg)
    if not state:
        return None
    orig = state["idx_to_orig"].get(idx)
    if orig is None:
        return None
    path = _odm_matches_path(cfg, orig)
    if not os.path.exists(path):
        return None

    with gzip.open(path, "rb") as f:
        raw = pickle.load(f)

    out = {}
    for neighbor_orig, arr in raw.items():
        neighbor_idx = state["orig_to_idx"].get(neighbor_orig)
        if neighbor_idx is not None and len(arr):
            out[neighbor_idx] = np.asarray(arr, dtype=np.int64)
    return out


def load_points_cached(cfg, idx):
    """Points for image idx (native px coords), from the on-disk cache
    (register_results/rgb_features/<idx>.npz) if present, else loaded fresh
    from ODM's features.npz and cached. Returns None if ODM has no
    features.npz for this image -- caller excludes it from densify."""
    dz = cfg["densify"]
    feat_dir = os.path.join(cfg["register_results_dir"], "rgb_features")
    cache_path = os.path.join(feat_dir, f"{idx}.npz")
    if dz["save_features"] and os.path.exists(cache_path):
        return np.load(cache_path)["pts"]

    pts = load_odm_points_for_image(cfg, idx)
    if pts is not None and dz["save_features"]:
        os.makedirs(feat_dir, exist_ok=True)
        np.savez(cache_path, pts=pts)
    return pts


def match_pair_from_odm(pts_a, pts_b, idx_pairs, cfg):
    """OPTIONAL stricter path (CONFIG['odm']['verify_planar_homography']=True):
    homography-RANSAC re-verification on top of ODM's already-matched
    keypoint index pairs, before accepting them into the track graph. NOT
    used by default -- ODM's matches already survived a full bundle
    adjustment across the whole dataset, a much stronger check than a
    single pairwise planar-homography RANSAC, and (now that Step 5 doesn't
    assume planarity either) this would only ever throw away genuinely
    correct matches over non-flat terrain."""
    dz = cfg["densify"]
    if idx_pairs is None or len(idx_pairs) < dz["min_matches"]:
        return None
    src = pts_a[idx_pairs[:, 0]]
    dst = pts_b[idx_pairs[:, 1]]
    H, mask = estimate_homography_robust(src, dst, dz["ransac_thresh"], dz["ransac_confidence"])
    if H is None or int(mask.sum()) < dz["min_matches"]:
        return None
    inlier_pairs = [tuple(idx_pairs[k]) for k in range(len(idx_pairs)) if mask.ravel()[k]]
    return H, int(mask.sum()), inlier_pairs


def _match_status_path(cfg):
    return os.path.join(cfg["register_results_dir"], "rgb_matches.csv")


def _inlier_matches_path(cfg):
    return os.path.join(cfg["register_results_dir"], "rgb_inlier_matches.csv")


def load_pair_status(cfg):
    """(a,b) -> (verified: bool, inliers: int) for every candidate pair EVER
    attempted across all past runs -- lets later runs skip pairs already
    tried instead of redoing them."""
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


def append_inlier_matches(cfg, a, b, inlier_pairs):
    """Raw verified correspondences for ONE pair -- the ground truth tracks
    are rebuilt from every run. Appended once, read many times."""
    path = _inlier_matches_path(cfg)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["image_a", "image_b", "kp_a", "kp_b"])
        for i, j in inlier_pairs:
            writer.writerow([a, b, i, j])


def load_all_inlier_matches(cfg):
    """Every verified correspondence ever recorded, across all past runs --
    the full history the global track graph gets rebuilt from."""
    path = _inlier_matches_path(cfg)
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append((row["image_a"], row["image_b"], int(row["kp_a"]), int(row["kp_b"])))
    return rows


def export_tracks_csv(cfg, tracks):
    """track_id,image,feature_id -- one row per observation. Regenerated
    fresh each run from the full match history."""
    path = os.path.join(cfg["register_results_dir"], "tracks.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "image", "feature_id"])
        for tid, obs in enumerate(tracks):
            for im, kp in obs:
                writer.writerow([tid, im, kp])
    return path


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


# =====================================================================
# STEP 5 (fill): Delaunay-triangulated mesh warp, composited directly into
# a CROPPED region of the target -- never a full native-resolution canvas
# =====================================================================

def mesh_warp_neighbor_into(dense, filled, src_bgr, src_alpha, dst_pts, src_pts, feather_px, dilate_px):
    """Triangulate dst_pts<->src_pts (both Nx2, native pixel coords) and
    composite src_bgr/src_alpha's mesh-warped content directly into
    dense/filled (mutated in place -- numpy views), working ONLY within the
    bounding box of dst_pts (plus a small margin for dilation) -- never a
    full native-resolution canvas. A neighbor's actual overlap with the
    target is usually a small fraction of the whole frame; allocating a
    full-size buffer per neighbor wastes huge amounts of memory on mostly-
    empty canvases, and doing that across dozens of neighbors x hundreds of
    target images is what exhausts memory in one run.

    Each small triangle gets its own affine transform fit from just its 3
    corners -- a far better local approximation of real (non-planar)
    terrain than one shared plane over the whole overlap. Every matched
    point becomes a mesh vertex -- nothing is discarded the way a single
    robust-homography fit implicitly discards points that don't agree with
    one global model.

    Coverage is limited to the convex hull of dst_pts (same limitation a
    homography-based warp would have too) -- pixels outside it are left
    untouched."""
    if len(dst_pts) < 4:
        return dense, filled
    try:
        tri = Delaunay(dst_pts)
    except Exception:
        return dense, filled  # degenerate point set (e.g. all collinear) -- nothing usable

    H, W = dense.shape[:2]
    margin = dilate_px + 2
    x0 = max(int(np.floor(dst_pts[:, 0].min())) - margin, 0)
    y0 = max(int(np.floor(dst_pts[:, 1].min())) - margin, 0)
    x1 = min(int(np.ceil(dst_pts[:, 0].max())) + margin, W)
    y1 = min(int(np.ceil(dst_pts[:, 1].max())) + margin, H)
    if x1 <= x0 or y1 <= y0:
        return dense, filled
    cw, ch = x1 - x0, y1 - y0

    local_bgr = np.zeros((ch, cw, 3), np.uint8)
    local_alpha = np.zeros((ch, cw), np.uint8)
    dst_local = dst_pts - (x0, y0)

    for simplex in tri.simplices:
        t_dst = dst_local[simplex].astype(np.float32)  # triangle in the LOCAL crop's coords
        t_src = src_pts[simplex].astype(np.float32)     # same triangle, in the (already-cropped) SOURCE

        rd = cv2.boundingRect(t_dst)
        rs = cv2.boundingRect(t_src)
        if rd[2] < 1 or rd[3] < 1 or rs[2] < 1 or rs[3] < 1:
            continue

        t_dst_l = (t_dst - (rd[0], rd[1])).astype(np.float32)
        t_src_l = (t_src - (rs[0], rs[1])).astype(np.float32)

        src_patch = src_bgr[rs[1]:rs[1] + rs[3], rs[0]:rs[0] + rs[2]]
        src_alpha_patch = src_alpha[rs[1]:rs[1] + rs[3], rs[0]:rs[0] + rs[2]]

        M = cv2.getAffineTransform(t_src_l, t_dst_l)
        warped_patch = cv2.warpAffine(src_patch, M, (rd[2], rd[3]),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_alpha_patch = cv2.warpAffine(src_alpha_patch, M, (rd[2], rd[3]),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        tri_mask = np.zeros((rd[3], rd[2]), np.uint8)
        cv2.fillConvexPoly(tri_mask, np.int32(np.round(t_dst_l)), 255)

        m = (tri_mask > 0) & (warped_alpha_patch > 127)
        if not m.any():
            continue

        local_bgr[rd[1]:rd[1] + rd[3], rd[0]:rd[0] + rd[2]][m] = warped_patch[m]
        local_alpha[rd[1]:rd[1] + rd[3], rd[0]:rd[0] + rd[2]][m] = 255

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        local_alpha = cv2.dilate(local_alpha, kernel)
        local_bgr = cv2.dilate(local_bgr, kernel)  # dilate the color too, else the newly-grown
                                                     # ring would composite invalid black pixels
                                                     # (warpAffine's borderValue) instead of real color

    dense_region = dense[y0:y1, x0:x1]
    filled_region = filled[y0:y1, x0:x1]
    new_only = (local_alpha > 0) & (~filled_region)
    if not new_only.any():
        return dense, filled

    if feather_px <= 0:
        dense_region[new_only] = local_bgr[new_only]
    else:
        dist = cv2.distanceTransform(new_only.astype(np.uint8) * 255, cv2.DIST_L2, 3)
        w_sel = np.clip(dist[new_only] / float(feather_px), 0, 1)[:, None].astype(np.float32)
        blended = w_sel * local_bgr[new_only].astype(np.float32) + (1 - w_sel) * dense_region[new_only].astype(np.float32)
        dense_region[new_only] = np.clip(blended, 0, 255).astype(np.uint8)

    filled_region |= new_only  # dense/filled are mutated in place (numpy views), returned for clarity
    return dense, filled


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
        print("No warped_thermal outputs found -- run Step 2b (apply_calibration) first.")
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

    odm_cfg = cfg.get("odm") or {}
    if not odm_cfg.get("enabled"):
        print("Needs CONFIG['odm']['enabled']=True -- features and matches come only from ODM's "
              "opensfm/ outputs, there's no self-computed fallback. Skipping.")
        return []

    # ---- 1) load ODM feature points (native px) for every warped-thermal
    # image -- cached, one-time cost across all runs ----
    print(f"1) loading ODM feature points (opensfm/features/*.npz) for {len(have_warped)} images...")
    feats = {}
    for idx in have_warped:
        pts = load_points_cached(cfg, idx)
        if pts is not None:
            feats[idx] = pts
    missing_pts = have_warped - set(feats)
    if missing_pts:
        print(f"   {len(missing_pts)}/{len(have_warped)} image(s) have no ODM features.npz -- "
              "excluded from densify (no fallback).")

    # ---- 2) candidate selection: directly from ODM's own matched-pair list ----
    print("2) candidate selection: reusing ODM's own matches (opensfm/matches/*.pkl.gz)...")
    odm_pair_corr = {}  # (min_idx, max_idx) -> Nx2 array (kp_in_min_idx, kp_in_max_idx)
    odm_has_matches = set()
    for idx in feats:
        m = load_odm_matches_for_image(cfg, idx)
        if m is None:
            continue
        odm_has_matches.add(idx)
        for nidx, arr in m.items():
            if nidx not in feats:
                continue
            key = (min(idx, nidx), max(idx, nidx))
            if key in odm_pair_corr:
                continue
            odm_pair_corr[key] = arr if idx == key[0] else arr[:, ::-1]

    candidates_by_idx = {}
    for a, b in odm_pair_corr:
        candidates_by_idx.setdefault(a, []).append(b)
        candidates_by_idx.setdefault(b, []).append(a)

    no_matches = set(feats) - odm_has_matches
    print(f"   {len(odm_pair_corr)} candidate pairs from ODM matches"
          + (f" | {len(no_matches)}/{len(feats)} image(s) have no ODM matches file -- excluded "
             "(no fallback)" if no_matches else ""))

    # ---- 3) match verification: trusted directly as track correspondences
    # by default (see CONFIG['odm']['verify_planar_homography']) ----
    print("3) match_features: " + (
        "re-verifying ODM's matches with our own homography RANSAC" if odm_cfg.get("verify_planar_homography", False)
        else "trusting ODM's pre-matched correspondences directly") + " (skipping ones already attempted)...")
    pair_status = load_pair_status(cfg)
    involved = set(run_now) & set(feats)
    for idx in set(run_now) & set(feats):
        involved.update(candidates_by_idx.get(idx, []))

    status_rows, n_new_verified = [], 0
    for idx in involved:
        for nidx in candidates_by_idx.get(idx, []):
            key = (min(idx, nidx), max(idx, nidx))
            if key in pair_status:
                continue
            a, b = key

            idx_pairs = odm_pair_corr.get(key)
            if odm_cfg.get("verify_planar_homography", False):
                result = match_pair_from_odm(feats[a], feats[b], idx_pairs, cfg)
            elif idx_pairs is not None and len(idx_pairs) >= dz["min_matches"]:
                inlier_pairs = [(int(p[0]), int(p[1])) for p in idx_pairs]
                result = (len(inlier_pairs), inlier_pairs)
            else:
                result = None

            if result is None:
                pair_status[key] = (False, 0)
                status_rows.append((a, b, False, 0))
                continue
            inliers, inlier_pairs = result
            pair_status[key] = (True, inliers)
            status_rows.append((a, b, True, inliers))
            append_inlier_matches(cfg, a, b, inlier_pairs)
            n_new_verified += 1
    if status_rows:
        append_pair_status(cfg, status_rows)
    n_verified_total = sum(1 for v, _ in pair_status.values() if v)
    print(f"   {len(status_rows)} new pairs attempted this run ({n_new_verified} verified) | "
          f"{n_verified_total} verified pairs total (all runs, in {_match_status_path(cfg)})")

    # ---- 4) create_tracks from the FULL accumulated match history ----
    uf = UnionFind()
    for a, b, kp_a, kp_b in load_all_inlier_matches(cfg):
        uf.union((a, kp_a), (b, kp_b))
    groups = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = [obs for obs in groups.values() if len({im for im, _ in obs}) >= dz["min_track_len"]]
    n_multi = sum(1 for t in tracks if len({im for im, _ in t}) >= 3)
    tracks_path = export_tracks_csv(cfg, tracks)
    print(f"4) create_tracks: {len(tracks)} tracks ({n_multi} seen in 3+ images), exported to {tracks_path}")

    tracks_by_image = {}
    for tid, obs in enumerate(tracks):
        for im, kp in obs:
            tracks_by_image.setdefault(im, {})[kp] = tid

    def shared_track_correspondences(a, b):
        """EVERY point pair for (a,b) reachable via a shared track -- this
        includes their direct pairwise match (that's what created the
        track) PLUS any extra points bridged in through other images'
        tracks. Every one becomes a mesh vertex in mesh_warp_neighbor_into.
        Returns (points_in_a, points_in_b), matched 1:1 by position."""
        ta = tracks_by_image.get(a, {})
        tb_by_tid = {tid: kp for kp, tid in tracks_by_image.get(b, {}).items()}
        src, dst = [], []
        for kp_a, tid in ta.items():
            if tid in tb_by_tid and a in feats and b in feats:
                src.append(feats[a][kp_a])
                dst.append(feats[b][tb_by_tid[tid]])
        return np.array(src), np.array(dst)

    # ---- 5) fill each target from every track-reachable image, via mesh warp ----
    print("5) filling targets (Delaunay mesh warp) from every track-reachable image...")
    n_ok, n_err = 0, 0
    for idx in run_now:
        try:
            vpath, _ = by_idx[idx]
            visible_target = load_color(vpath)
            own_bgr, own_alpha = _load_warped_thermal_bgra(os.path.join(warped_dir, f"{idx}_T_warped.png"))

            if dz["mask_dilate_px"] > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * dz["mask_dilate_px"] + 1, 2 * dz["mask_dilate_px"] + 1)
                )
                own_alpha = cv2.dilate(own_alpha, kernel)
                own_bgr = cv2.dilate(own_bgr, kernel)

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

            # priority order: most shared points first (a reasonable proxy
            # for reliability/coverage quality -- unlike before, this is no
            # longer a RANSAC inlier count since no global fit happens)
            scored = []
            for nidx in reachable:
                target_pts, neighbor_pts = shared_track_correspondences(idx, nidx)
                if len(target_pts) >= dz["min_matches"]:
                    scored.append((nidx, target_pts, neighbor_pts))
            scored.sort(key=lambda t: -len(t[1]))

            for nidx, target_pts, neighbor_pts in scored:
                n_bgr_full, n_alpha_full = _load_warped_thermal_bgra(os.path.join(warped_dir, f"{nidx}_T_warped.png"))

                # crop the neighbor's source to just the region this pair's
                # points could possibly need -- avoids holding the full
                # native-resolution neighbor frame (~48MB) in memory for
                # the whole triangle loop
                nh, nw = n_bgr_full.shape[:2]
                nx0 = max(int(np.floor(neighbor_pts[:, 0].min())), 0)
                ny0 = max(int(np.floor(neighbor_pts[:, 1].min())), 0)
                nx1 = min(int(np.ceil(neighbor_pts[:, 0].max())) + 1, nw)
                ny1 = min(int(np.ceil(neighbor_pts[:, 1].max())) + 1, nh)
                n_bgr = n_bgr_full[ny0:ny1, nx0:nx1].copy()
                n_alpha = n_alpha_full[ny0:ny1, nx0:nx1].copy()
                del n_bgr_full, n_alpha_full
                neighbor_pts_local = neighbor_pts - (nx0, ny0)

                dense, filled = mesh_warp_neighbor_into(
                    dense, filled, n_bgr, n_alpha,
                    dst_pts=target_pts, src_pts=neighbor_pts_local,
                    feather_px=dz["feather_px"], dilate_px=dz["mask_dilate_px"],
                )
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
    print("=== STEP 3 (standalone): densify using triangulated mesh warp ===")
    densify_with_neighbors(cfg)


if __name__ == "__main__":
    main()
