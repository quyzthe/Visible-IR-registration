"""
STEP 4 -- DENSIFY

Fills gaps in each visible image's thermal coverage using overlap between
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

Works on whatever warped_thermal/*.png already exists, regardless of
whether pipeline_apply.py produced it via SfM or the 2D fallback.
"""

import os
import csv

import cv2
import numpy as np

from pipeline_organize import find_pairs
from pipeline_common import load_gray, load_color, preprocess, full_res_matrix, estimate_homography_robust, load_warped_thermal_bgra


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


def _match_status_path(cfg):
    return os.path.join(cfg["register_results_dir"], "rgb_matches.csv")


def _inlier_matches_path(cfg):
    return os.path.join(cfg["register_results_dir"], "rgb_inlier_matches.csv")


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


def densify(cfg):
    dz = cfg["densify"]
    if not dz["enabled"]:
        return []
    dz = dict(dz)
    dz["work_size"] = cfg.get("calibrate_2d", {}).get("work_size", (640, 480))

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
        shown = component_sizes[:10]
        print(f"   Match graph connectivity: {len(component_sizes)} connected component(s) "
              f"over {len(all_matched_images)} images | sizes: {shown}"
              + ("..." if len(component_sizes) > 10 else ""))
        if len(component_sizes) > 1:
            print(f"   [WARN] FRAGMENTED into {len(component_sizes)} pieces -- images in different "
                  f"pieces cannot fill each other no matter how much data has accumulated elsewhere. "
                  f"Raise densify.top_m_candidates to give more images a chance at a verified edge "
                  f"into the largest component ({component_sizes[0]} images).")

    uf = UnionFind()
    for a, b, kp_a, kp_b in load_all_inlier_matches(cfg):
        uf.union((a, kp_a), (b, kp_b))
    groups = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = [obs for obs in groups.values() if len({im for im, _ in obs}) >= dz["min_track_len"]]
    tracks_path = export_tracks_csv(cfg, tracks)
    print(f"4) create_tracks: {len(tracks)} tracks, exported to {tracks_path}")

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

    print("5) filling targets...")
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

            for nidx, H_n_to_target, inl in scored:
                n_bgr, n_alpha = load_warped_thermal_bgra(os.path.join(warped_dir, f"{nidx}_T_warped.png"))
                nvpath, _ = by_idx[nidx]
                neighbor_shape = load_color(nvpath).shape
                H_full = full_res_matrix(H_n_to_target, neighbor_shape, visible_target.shape, dz["work_size"])
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

    print(f"\nThis run: {n_ok} ok, {n_err} error, {len(run_now)} total")
    return run_now
