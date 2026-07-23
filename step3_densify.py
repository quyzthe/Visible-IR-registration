"""
STEP 3 (standalone) -- densify thermal coverage.

PREREQUISITE: Steps 1 and 2 of organize_and_register.py have already been
run (visible_dir/thermal_dir are organized, register_results/warped_thermal
exists). This script only READS those outputs, plus ODM's opensfm/ outputs
-- it never touches Step 1/2, and Step 1/2 never need to run again.

DESIGN
------
Same overall structure as the algorithm that worked well originally (one
robust homography per neighbor, warped into the target and feather-
composited), with three fixes:

  1) NATIVE RESOLUTION, not work_size-then-rescale. An earlier version fit
     the homography at LoFTR's 640x480 work_size, then scaled the matrix
     up to native (~4032x3024) resolution. That rescale amplifies any
     small fit error by ~6x, worst at points far from the fit's centroid
     -- exactly why errors were concentrated at corners/edges with fewer
     matched points. Fitting directly at native resolution removes that
     amplification entirely.
  2) NO neighbor is ever excluded by point count. Every image reachable
     via a shared track is attempted; only a genuinely degenerate fit
     (near-singular matrix, too few RANSAC inliers) is skipped, and each
     neighbor is wrapped in its own try/except so a bad one can never
     abort the rest of the target image's processing.
  3) REAL-DATA PROPAGATION: each neighbor's BEST available thermal is
     used, not just its raw Step 2 output. If that neighbor was already
     densified in an earlier run, its richer, already-filled-in result
     (register_results/overlays_dense) is used instead -- but ONLY the
     portion of it that's marked "real" (a mask saved alongside every
     overlay_dense distinguishes actual matched-and-warped content from
     that image's own inpainted guesses). This lets real thermal coverage
     propagate across CHAINS of neighbors over multiple runs, not just one
     hop, without ever quietly compounding guesses as if they were data.

Whatever is STILL unfilled after every reachable neighbor has been tried
(usually small, isolated patches with no matched points at all, e.g. flat
featureless water) is filled by simple image inpainting as a last resort
-- this is what GUARANTEES exactly 100% coverage in the final image,
independent of how much real data was available. The "real" mask saved
alongside each output is what keeps this inpainted portion from ever being
mistaken for real data by a later run.

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
        min_matches=15,              # min points a raw ODM pair needs to be trusted enough to
                                      # link two images into the SHARED track graph in the first
                                      # place (other neighbor pairs' reachability depends on this
                                      # same graph, so it's worth protecting) -- NOT used to skip
                                      # a neighbor in Step 5 once it's already known reachable;
                                      # see the module docstring.

        # ---- 4) create_tracks ----
        min_track_len=2,             # keep tracks observed in at least this many images

        # ---- 5) fill: for every reachable neighbor, fit ONE robust homography
        # via RANSAC from all its shared points -- simple, no TPS, no per-
        # triangle mesh. A neighbor with too few points or a degenerate fit is
        # skipped cleanly (never crashes the whole image). All composited in a
        # CROPPED region, never a full native-resolution canvas. ----
        ransac_thresh=3.0,
        ransac_confidence=0.999,

        # ---- final gap-fill: GUARANTEES exactly 100% coverage regardless of
        # how much/little feature-based coverage was achieved above ----
        inpaint_radius=5,             # cv2.inpaint's radius (px) -- how far around each gap pixel
                                       # it looks for known thermal to extrapolate from

        mask_dilate_px=8,            # grow each neighbor's warped silhouette (and the target's
                                      # own warped_thermal) by this many px before compositing --
                                      # warpPerspective leaves ragged/gappy mask edges, and two
                                      # independently-fit transforms won't perfectly agree on a
                                      # shared boundary to the pixel -- a small overlap margin
                                      # closes that instead of a visible gap.
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


def to_3x3(M):
    if M.shape == (3, 3):
        return M.astype(np.float64)
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = M
    return out


def apply_homogeneous(M3x3, pts):
    """Transform Nx2 points through a 3x3 matrix with a proper perspective
    divide -- works for a true homography AND for an affine padded to 3x3
    (whose bottom row [0,0,1] makes the divide a no-op)."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    out_h = pts_h @ M3x3.T
    return out_h[:, :2] / out_h[:, 2:3]


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
# STEP 5 (fill): ONE robust homography per neighbor, fit at NATIVE
# resolution -- NOT downscaled to work_size then rescaled up, which was
# the actual cause of corner/edge inaccuracy in an earlier version: a
# small fit error at 640x480 gets amplified ~6x when scaled up to
# ~4032x3024, worst exactly at points far from the fit's centroid (i.e.
# the corners). Every reachable neighbor is attempted -- none excluded up
# front for having "few" shared points; only a fit that genuinely fails
# (degenerate matrix, too few RANSAC inliers) is skipped, and only for
# that ONE neighbor (wrapped in try/except so it can never abort the rest
# of the target image).
#
# Each neighbor's BEST available source is used: if that neighbor was
# ALREADY densified in an earlier run (register_results/overlays_dense has
# its output plus a "real" mask), that richer, already-filled-in result is
# used instead of just its own raw Step-2 warped_thermal -- propagating
# real thermal coverage across CHAINS of neighbors, not just one hop away.
# Only the REAL portion of a neighbor's densified result is ever used
# (never that neighbor's own inpainted/extrapolated gap-fill) -- the
# "real" mask saved alongside every overlay_dense is what makes that
# distinction possible.
# =====================================================================

def _composite_full(dense, filled, add_bgr, add_alpha_bin, dilate_px, feather_px):
    """Dilate (mask AND color together, else the newly-grown ring would
    paint invalid black pixels instead of real color) and feather-blend
    add_bgr/add_alpha_bin into dense/filled, in place, over the full frame."""
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        add_alpha_bin = cv2.dilate(add_alpha_bin, kernel)
        add_bgr = cv2.dilate(add_bgr, kernel)

    new_only = (add_alpha_bin > 0) & (~filled)
    if not new_only.any():
        return
    if feather_px <= 0:
        dense[new_only] = add_bgr[new_only]
    else:
        dist = cv2.distanceTransform(new_only.astype(np.uint8) * 255, cv2.DIST_L2, 3)
        w_sel = np.clip(dist[new_only] / float(feather_px), 0, 1)[:, None].astype(np.float32)
        blended = w_sel * add_bgr[new_only].astype(np.float32) + (1 - w_sel) * dense[new_only].astype(np.float32)
        dense[new_only] = np.clip(blended, 0, 255).astype(np.uint8)
    filled |= new_only


def _source_for_neighbor(cfg, nidx, warped_dir, dense_dir):
    """The BEST available thermal source for neighbor nidx: its own
    already-densified result (register_results/overlays_dense), if it
    exists, together with the "real" mask saved alongside it (255 only
    where that densification came from actual matched-and-warped content,
    never from THAT image's own inpainted gap-fill) -- richer coverage
    than the raw Step 2 output, propagated in from nidx's own neighbors in
    an earlier run. Falls back to the raw Step 2 warped_thermal (own-only,
    but always 100% real wherever its alpha is set) if nidx hasn't been
    densified yet."""
    dense_path = os.path.join(dense_dir, f"{nidx}_overlay_dense.png")
    mask_path = os.path.join(dense_dir, f"{nidx}_overlay_dense_realmask.png")
    if os.path.exists(dense_path) and os.path.exists(mask_path):
        bgr = cv2.imread(dense_path, cv2.IMREAD_COLOR)
        real_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if bgr is not None and real_mask is not None and bgr.shape[:2] == real_mask.shape[:2]:
            return bgr, real_mask
    return _load_warped_thermal_bgra(os.path.join(warped_dir, f"{nidx}_T_warped.png"))


def warp_neighbor_full(dense, filled, src_bgr, src_alpha, target_pts, neighbor_pts, cfg):
    """Warp src_bgr/src_alpha (a neighbor's best available thermal source,
    in the neighbor's native visible frame -- see _source_for_neighbor)
    into dense/filled (target's native frame) using ONE robust homography
    fit from EVERY point this pair shares, at native resolution. Returns
    (dense, filled, contributed: bool). Any failure (degenerate fit, too
    few inliers, numerical error) is caught and this neighbor is simply
    skipped -- never crashes the rest of the target image's processing."""
    dz = cfg["densify"]
    if len(target_pts) < 4:
        return dense, filled, False
    try:
        H, mask = estimate_homography_robust(neighbor_pts, target_pts, dz["ransac_thresh"], dz["ransac_confidence"])
        if H is None or not np.all(np.isfinite(H)):
            return dense, filled, False
        if abs(np.linalg.det(H)) < 1e-9:
            return dense, filled, False  # degenerate -- would produce garbage
        if int(mask.ravel().astype(bool).sum()) < 4:
            return dense, filled, False

        out_size = (dense.shape[1], dense.shape[0])
        warped_bgr = cv2.warpPerspective(src_bgr, H, out_size,
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_alpha = cv2.warpPerspective(src_alpha, H, out_size,
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_alpha_bin = ((warped_alpha > 127).astype(np.uint8)) * np.uint8(255)

        _composite_full(dense, filled, warped_bgr, warped_alpha_bin, dz["mask_dilate_px"], dz["feather_px"])
        return dense, filled, True
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return dense, filled, False


def fill_remaining_gaps(dense, filled, cfg):
    """GUARANTEES exactly 100% coverage in the returned image: whatever is
    still unfilled after every reachable neighbor has been tried is filled
    by simple image inpainting from the surrounding already-filled
    thermal. `filled` itself is left UNCHANGED by this function -- it
    stays the "real" mask (own + genuinely matched-and-warped neighbor
    content only), which the caller saves alongside the output so LATER
    images that use this one as a source (see _source_for_neighbor) never
    propagate this inpainted guess as if it were real data."""
    missing = ~filled
    frac_missing = float(missing.mean())
    if frac_missing == 0.0:
        return dense
    if frac_missing > 0.5:
        print(f"   [WARN] {frac_missing:.1%} of this frame had no usable neighbor coverage at all -- "
              "inpainting that much is a rough extrapolation, not real thermal data. Worth checking "
              "whether this image genuinely has few overlapping/matched neighbors.")
    mask = missing.astype(np.uint8) * 255
    radius = cfg["densify"].get("inpaint_radius", 5)
    filled_img = cv2.inpaint(dense, mask, radius, cv2.INPAINT_TELEA)
    dense = dense.copy()
    dense[missing] = filled_img[missing]
    return dense


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
        tracks. Returns (points_in_a, points_in_b), matched 1:1 by
        position, in NATIVE pixel coordinates."""
        ta = tracks_by_image.get(a, {})
        tb_by_tid = {tid: kp for kp, tid in tracks_by_image.get(b, {}).items()}
        src, dst = [], []
        for kp_a, tid in ta.items():
            if tid in tb_by_tid and a in feats and b in feats:
                src.append(feats[a][kp_a])
                dst.append(feats[b][tb_by_tid[tid]])
        return np.array(src), np.array(dst)

    # ---- 5) fill each target from every track-reachable neighbor ----
    print("5) filling targets: native-resolution homography per neighbor, using each "
          "neighbor's richest available real thermal (its own densified result if it has "
          "one already, else its raw Step 2 output)...")
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

            # EVERY reachable neighbor is a candidate -- no count-based
            # exclusion here. Priority order (most shared points first) is
            # just a reasonable proxy for reliability, not a filter.
            scored = []
            for nidx in reachable:
                target_pts, neighbor_pts = shared_track_correspondences(idx, nidx)
                if len(target_pts) >= 1:
                    scored.append((nidx, target_pts, neighbor_pts))
            scored.sort(key=lambda t: -len(t[1]))

            n_skipped = 0
            for nidx, target_pts, neighbor_pts in scored:
                n_bgr, n_alpha = _source_for_neighbor(cfg, nidx, warped_dir, dense_dir)
                dense, filled, contributed = warp_neighbor_full(
                    dense, filled, n_bgr, n_alpha, target_pts, neighbor_pts, cfg
                )
                if contributed:
                    n_sources += 1
                else:
                    n_skipped += 1

            coverage_from_neighbors = float(filled.mean())
            dense_final = fill_remaining_gaps(dense, filled, cfg)

            cv2.imwrite(os.path.join(dense_dir, f"{idx}_overlay_dense.png"), dense_final)
            cv2.imwrite(os.path.join(dense_dir, f"{idx}_overlay_dense_realmask.png"), filled.astype(np.uint8) * 255)
            print(f"[{idx}] {n_sources - 1} neighbor(s) used, {n_skipped} skipped (of {len(reachable)} "
                  f"track-reachable), coverage {base_coverage:.1%} -> {coverage_from_neighbors:.1%} real "
                  f"-> 100.0% after gap-fill")
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[{idx}] ERROR: {e}")

    print(f"\nThis run: {n_ok} ok, {n_err} error, {len(run_now)} total")
    return run_now


def main(cfg=CONFIG):
    print("=== STEP 3 (standalone): densify using per-neighbor homography + real-data propagation ===")
    densify_with_neighbors(cfg)


if __name__ == "__main__":
    main()