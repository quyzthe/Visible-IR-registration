"""
Imports an already-completed OpenSfM run's features + matches directly into
pipeline_densify.py's own cache format (rgb_features/<idx>.npz,
rgb_inlier_matches.csv, rgb_matches.csv) -- so create_tracks/fill downstream
work completely unchanged, just skipping the expensive detect_features +
BoW + match_features steps for every image this covers.

VERIFIED FORMATS (inspected from the user's real files, not guessed)
-------------------------------------------------------------------------
matches/<original_name>_matches.pkl.gz:
    gzip pickle -> dict {other_original_name: Nx2 int64 array}
    Each row = [feature_index_in_this_image, feature_index_in_other_image],
    already RANSAC-verified. Empty array (shape (0,), dtype float64) means
    that candidate was tried but had zero verified matches -- not "not
    tried yet".

features/<original_name>.features.npz: NOT YET VERIFIED against a real
file from this project -- built from OpenSfM's own documented convention
(same one already confirmed correct via reconstruction.json's focal/
principal-point normalization earlier in this pipeline): points stored in
"normalized image coordinates" -- origin at image center, x right, y down,
both normalized by max(width, height). Key name defaults to "points" (the
long-standing OpenSfM convention) but is auto-detected with a clear error
listing the real keys if that guess is wrong -- READ that error if it
fires; don't assume "points" is definitely right for this export.

Filenames throughout are OpenSfM's -- the ORIGINAL DJI filenames, not
organize()'s renamed ones -- bridged via file_mapping.csv, same as the
Agisoft XML import.
"""

import os
import csv
import gzip
import pickle

import numpy as np
from PIL import Image

from pipeline_organize import find_pairs, load_original_name_to_organized_path


# =====================================================================
# FEATURES
# =====================================================================

def load_opensfm_features_npz(path):
    """Returns Nx2 float64 array of NORMALIZED (x, y) coordinates.
    Auto-detects the coordinate array's key name rather than assuming --
    fails loudly with the real keys/shapes if none of the expected names
    match, instead of silently reading the wrong array."""
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    points = None
    for candidate in ("points", "keypoints", "features"):
        if candidate in keys:
            points = data[candidate]
            break
    if points is None:
        raise RuntimeError(
            f"{path}: none of the expected keys ('points','keypoints','features') found. "
            f"Actual keys: {keys}, shapes: {[(k, data[k].shape) for k in keys]}. "
            f"Tell me which key holds the Nx(>=2) coordinate array so this can be fixed."
        )
    if points.ndim != 2 or points.shape[1] < 2:
        raise RuntimeError(f"{path}: coordinate array has unexpected shape {points.shape} "
                            f"(expected Nx2 or wider) -- can't proceed without checking this by hand.")
    return points[:, :2].astype(np.float64)


def denormalize_opensfm_xy(norm_xy, width, height):
    """OpenSfM's documented convention (confirmed via their own docs and
    already validated once in this pipeline via reconstruction.json):
    origin at image center, x right, y down, normalized by max(w,h)."""
    scale = max(width, height)
    px = norm_xy[:, 0] * scale + width / 2.0
    py = norm_xy[:, 1] * scale + height / 2.0
    return np.stack([px, py], axis=1)


# =====================================================================
# MATCHES
# =====================================================================

def load_opensfm_matches_pkl(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# =====================================================================
# MAIN IMPORT
# =====================================================================

def import_opensfm_tracks(cfg):
    """Populates register_results/rgb_features/, rgb_inlier_matches.csv,
    rgb_matches.csv from an already-completed OpenSfM run. Idempotent --
    safe to call every run; only imports what's missing."""
    ocfg = cfg.get("opensfm_import", {})
    if not ocfg.get("enabled"):
        print("[opensfm_import] disabled (opensfm_import.enabled=False in config) -- skipping, "
              "densify() will use SIFT for everything.")
        return
    opensfm_dir = ocfg["opensfm_dir"]
    features_dir = os.path.join(opensfm_dir, "features")
    matches_dir = os.path.join(opensfm_dir, "matches")
    if not (os.path.isdir(features_dir) and os.path.isdir(matches_dir)):
        print(f"[opensfm_import] {features_dir} or {matches_dir} not found -- skipping import.")
        return

    pairs = find_pairs(cfg["visible_dir"], cfg["thermal_dir"])
    organized_path_by_idx = {idx: v for idx, v, t in pairs}
    original_to_organized = load_original_name_to_organized_path(cfg)
    organized_to_idx = {v: idx for idx, v in organized_path_by_idx.items()}
    original_to_idx = {
        orig: organized_to_idx[org_path]
        for orig, org_path in original_to_organized.items()
        if org_path in organized_to_idx
    }
    print(f"[opensfm_import] {len(original_to_idx)}/{len(pairs)} images have an OpenSfM original-name mapping")

    feat_cache_dir = os.path.join(cfg["register_results_dir"], "rgb_features")
    os.makedirs(feat_cache_dir, exist_ok=True)

    # ---- 1) features ----
    print("[opensfm_import] Importing features...")
    n_new, feature_counts = 0, {}
    suffix = ocfg.get("features_suffix", ".features.npz")
    for orig_name, idx in original_to_idx.items():
        cache_path = os.path.join(feat_cache_dir, f"{idx}.npz")
        if os.path.exists(cache_path):
            try:
                feature_counts[idx] = len(np.load(cache_path)["pts"])
                continue
            except Exception as e:
                print(f"[opensfm_import] cache {cache_path} is corrupted ({e}) -- rebuilding it.")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        feat_path = os.path.join(features_dir, orig_name + suffix)
        if not os.path.exists(feat_path):
            continue
        try:
            norm_xy = load_opensfm_features_npz(feat_path)
        except RuntimeError as e:
            print(f"[opensfm_import] FATAL format issue on first file -- stopping here:\n  {e}")
            return
        with Image.open(organized_path_by_idx[idx]) as im:
            w, h = im.size
        pts = denormalize_opensfm_xy(norm_xy, w, h).astype(np.float32)
        tmp_path = cache_path + f".tmp{os.getpid()}"
        np.savez(tmp_path, pts=pts, descs=np.zeros((0, 0), np.float32))
        os.replace(tmp_path, cache_path)
        feature_counts[idx] = len(pts)
        n_new += 1
    print(f"[opensfm_import]   {n_new} new feature caches written ({len(feature_counts)} available total)")

    # ---- 2) matches ----
    print("[opensfm_import] Importing matches...")
    inlier_path = os.path.join(cfg["register_results_dir"], "rgb_inlier_matches.csv")
    status_path = os.path.join(cfg["register_results_dir"], "rgb_matches.csv")
    already_done = set()
    if os.path.exists(status_path):
        with open(status_path, newline="") as f:
            for row in csv.DictReader(f):
                already_done.add((row["image_a"], row["image_b"]))

    inlier_is_new = not os.path.exists(inlier_path)
    status_is_new = not os.path.exists(status_path)
    match_suffix = ocfg.get("matches_suffix", "_matches.pkl.gz")
    n_pairs, n_verified = 0, 0
    with open(inlier_path, "a", newline="") as fi, open(status_path, "a", newline="") as fs:
        wi, ws = csv.writer(fi), csv.writer(fs)
        if inlier_is_new:
            wi.writerow(["image_a", "image_b", "kp_a", "kp_b"])
        if status_is_new:
            ws.writerow(["image_a", "image_b", "verified", "inliers"])

        for orig_a, idx_a in original_to_idx.items():
            if idx_a not in feature_counts:
                continue
            match_path = os.path.join(matches_dir, orig_a + match_suffix)
            if not os.path.exists(match_path):
                continue
            try:
                matches = load_opensfm_matches_pkl(match_path)
            except Exception as e:
                print(f"[opensfm_import]   could not read {match_path}: {e}")
                continue

            for orig_b, idx_pairs in matches.items():
                idx_b = original_to_idx.get(orig_b)
                if idx_b is None or idx_b not in feature_counts or idx_b == idx_a:
                    continue
                key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if key in already_done:
                    continue
                already_done.add(key)
                n_pairs += 1
                n = len(idx_pairs) if getattr(idx_pairs, "ndim", 0) == 2 else 0
                ws.writerow([key[0], key[1], int(n > 0), n])
                if n > 0:
                    n_verified += 1
                    a_is_first = (idx_a == key[0])
                    for row in idx_pairs:
                        ia, ib = (int(row[0]), int(row[1])) if a_is_first else (int(row[1]), int(row[0]))
                        wi.writerow([key[0], key[1], ia, ib])

    print(f"[opensfm_import]   {n_pairs} pairs imported ({n_verified} verified) -- "
          f"create_tracks/fill in densify() will use these directly, no re-matching needed.")