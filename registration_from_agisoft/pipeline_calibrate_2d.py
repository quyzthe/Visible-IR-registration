"""
STEP 2b (FALLBACK) -- 2D SELF-CALIBRATION

Used only for pairs pipeline_sfm.py's rig reconstruction didn't produce a
pose for (or for the whole dataset if sfm.enabled=False). Same method
developed earlier in this project: pool LoFTR matches from a sample of
pairs, fit ONE global homography (guided inlier recovery extends it with
low-confidence-but-consistent matches), self-calibrate a radial lens
distortion coefficient for the thermal camera if it measurably helps
(validated on a held-out split -- never adopted if it doesn't), and fit a
thin-plate-spline local residual correction on top.

This produces ONE fixed transform applied to every pair that falls back to
it -- less accurate than a real per-image 3D pose (pipeline_sfm.py), but a
solid safety net.
"""

import os
import math

import cv2
import torch
import kornia.feature as KF
import numpy as np

from pipeline_common import (
    load_gray, preprocess, to_3x3, apply_homogeneous,
    estimate_homography_robust, full_res_matrix,
)


def build_matcher(device):
    return KF.LoFTR(pretrained="outdoor").to(device).eval()


def get_matches_loftr_raw(matcher, img0_p, img1_p, device):
    img0 = torch.from_numpy(img0_p / 255.0).float()[None, None].to(device)
    img1 = torch.from_numpy(img1_p / 255.0).float()[None, None].to(device)
    with torch.no_grad():
        out = matcher({"image0": img0, "image1": img1})
    return out["keypoints0"].cpu().numpy(), out["keypoints1"].cpu().numpy(), out["confidence"].cpu().numpy()


def _reprojection_error(M, src_pts, dst_pts):
    pred = apply_homogeneous(to_3x3(M), src_pts)
    return np.linalg.norm(pred - dst_pts.astype(np.float64), axis=1)


# ---- radial distortion self-calibration (division model, plane-based) ----

def undistort_division_model(pts, center, k, scale):
    d = pts - center
    r2 = (d[:, 0] ** 2 + d[:, 1] ** 2) / (scale ** 2)
    factor = 1.0 / (1.0 + k * r2)
    return center + d * factor[:, None]


def distort_division_model_forward(pts, center, k, scale):
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
    from scipy.optimize import least_squares

    def residuals(params):
        H = _homog_params_to_H(params[:8])
        k = params[8]
        undist = undistort_division_model(src_pts, center, k, scale)
        pred = apply_homogeneous(H, undist)
        return (pred - dst_pts).ravel()

    params0 = np.concatenate([_H_to_homog_params(to_3x3(H_init)), [0.0]])
    result = least_squares(residuals, params0, method="trf", loss="soft_l1", f_scale=2.0, max_nfev=3000)
    return _homog_params_to_H(result.x[:8]), float(result.x[8])


def _reprojection_rmse(H, k, center, scale, src_pts, dst_pts):
    pts = undistort_division_model(src_pts, center, k, scale) if k is not None else src_pts
    pred = apply_homogeneous(to_3x3(H), pts)
    return float(np.sqrt(np.mean(np.sum((pred - dst_pts) ** 2, axis=1))))


def build_undistort_map(work_size, native_shape, center_work, k, scale_work, grid_step=20):
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
    return cv2.resize(map_x_work, (Wn, Hn), interpolation=cv2.INTER_LINEAR) * sx, \
           cv2.resize(map_y_work, (Wn, Hn), interpolation=cv2.INTER_LINEAR) * sy


# ---- local (TPS) residual refinement ----

def fit_tps_correction(dst_pts, src_pts, M, work_size, lr_cfg):
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:
        print("[WARN] scipy.interpolate.RBFInterpolator unavailable -- skipping local TPS refinement.")
        return None

    predicted = apply_homogeneous(to_3x3(M), src_pts)
    delta = predicted - dst_pts

    smoothing = lr_cfg["tps_smoothing"]
    rbf_dx = RBFInterpolator(dst_pts, delta[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
    rbf_dy = RBFInterpolator(dst_pts, delta[:, 1], kernel="thin_plate_spline", smoothing=smoothing)

    W, H = work_size
    step = lr_cfg["grid_step_px"]
    xs = np.unique(np.append(np.arange(0, W, step), W - 1))
    ys = np.unique(np.append(np.arange(0, H, step), H - 1))
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

    cap = lr_cfg["max_correction_px"]
    dx = np.clip(rbf_dx(grid_pts), -cap, cap).reshape(gy.shape)
    dy = np.clip(rbf_dy(grid_pts), -cap, cap).reshape(gy.shape)

    map_x = cv2.resize((gx + dx).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize((gy + dy).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


# =====================================================================
# CALIBRATE
# =====================================================================

def calibrate_2d(cfg, pairs=None):
    """pairs: optionally restrict the pooled sample to a subset (e.g. only
    the pairs the SfM rig approach failed on) -- defaults to sampling from
    ALL organized pairs if not given."""
    from pipeline_organize import find_pairs
    c2d = cfg["calibrate_2d"]

    if pairs is None:
        pairs = find_pairs(cfg["visible_dir"], cfg["thermal_dir"])
    if not pairs:
        raise RuntimeError("No pairs available to calibrate from.")

    calib_path = os.path.join(cfg["register_results_dir"], "calibration_2d.npz")
    if os.path.exists(calib_path) and not c2d["recalibrate"]:
        data = np.load(calib_path)
        has_tps = bool(data["has_tps"])
        k_thermal = float(data["k_thermal"]) if "k_thermal" in data else 0.0
        print(f"Using existing 2D calibration: {calib_path} (RMSE={float(data['rmse_px']):.2f}px"
              f"{' + TPS' if has_tps else ''}{f' + lens k={k_thermal:.4f}' if k_thermal else ''})")
        tps = (data["map_x"], data["map_y"]) if has_tps else None
        dist = (k_thermal, data["dist_center"], float(data["dist_scale"])) if "dist_center" in data else None
        return data["M_work"].astype(np.float64), tuple(int(v) for v in data["work_size"]), tps, dist

    n = min(c2d["sample_size"], len(pairs))
    sample_pos = sorted(set(np.linspace(0, len(pairs) - 1, n).astype(int).tolist()))
    sample = [pairs[i] for i in sample_pos]
    print(f"Calibrating (2D fallback) from {len(sample)} sampled pairs...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = build_matcher(device)
    os.makedirs(cfg["register_results_dir"], exist_ok=True)
    feat_dir = os.path.join(cfg["register_results_dir"], "calibration_features")
    if c2d["save_features"]:
        os.makedirs(feat_dir, exist_ok=True)

    all_v_hi, all_t_hi, all_v_lo, all_t_lo = [], [], [], []
    for idx, vpath, tpath in sample:
        _, visible_p = preprocess(load_gray(vpath), c2d["work_size"], c2d["clahe_clip"], c2d["clahe_tile"])
        _, thermal_p = preprocess(load_gray(tpath), c2d["work_size"], c2d["clahe_clip"], c2d["clahe_tile"])
        mkpts0, mkpts1, conf = get_matches_loftr_raw(matcher, visible_p, thermal_p, device)
        hi = conf >= c2d["conf_thresh"]
        lo = conf >= c2d["guided_recovery_conf_thresh"]
        print(f"  [{idx}] {int(hi.sum())} matches >= conf_thresh, {int(lo.sum())} candidates for guided recovery")
        all_v_hi.append(mkpts0[hi]); all_t_hi.append(mkpts1[hi])
        all_v_lo.append(mkpts0[lo]); all_t_lo.append(mkpts1[lo])
        if c2d["save_features"]:
            np.savez(os.path.join(feat_dir, f"{idx}.npz"), mkpts0=mkpts0, mkpts1=mkpts1, conf=conf)

    mkpts0_hi, mkpts1_hi = np.concatenate(all_v_hi), np.concatenate(all_t_hi)
    if len(mkpts0_hi) < c2d["min_matches"]:
        raise RuntimeError(f"Only {len(mkpts0_hi)} pooled matches -- increase sample_size or lower conf_thresh.")

    M0, mask0 = estimate_homography_robust(mkpts1_hi, mkpts0_hi, c2d["ransac_thresh"], c2d["ransac_confidence"])
    if M0 is None:
        raise RuntimeError("Initial homography estimation failed.")
    print(f"Initial pooled RANSAC: {int(mask0.sum())}/{len(mkpts0_hi)} inliers")

    mkpts0_lo, mkpts1_lo = np.concatenate(all_v_lo), np.concatenate(all_t_lo)
    consistent = _reprojection_error(M0, mkpts1_lo, mkpts0_lo) < c2d["ransac_thresh"]
    print(f"Guided recovery: {int(consistent.sum())}/{len(mkpts0_lo)} consistent")

    M, mask = estimate_homography_robust(mkpts1_lo[consistent], mkpts0_lo[consistent],
                                          c2d["ransac_thresh"], c2d["ransac_confidence"])
    if M is None:
        M, mask, mkpts0_final, mkpts1_final = M0, mask0, mkpts0_hi, mkpts1_hi
    else:
        mkpts0_final, mkpts1_final = mkpts0_lo[consistent], mkpts1_lo[consistent]

    inliers = int(mask.sum())
    print(f"Refined RANSAC: {inliers}/{len(mkpts0_final)} inliers")
    if inliers < c2d["min_inliers"]:
        raise RuntimeError(f"Only {inliers} inliers -- calibration not reliable.")

    inlier_bool = mask.ravel().astype(bool)
    inlier_pts0, inlier_pts1 = mkpts0_final[inlier_bool], mkpts1_final[inlier_bool]
    rmse = float(np.sqrt(np.mean(_reprojection_error(M, inlier_pts1, inlier_pts0) ** 2)))
    print(f"Reprojection RMSE (inliers): {rmse:.2f} px")

    lens = c2d.get("lens_self_calibration", {"enabled": False})
    k_thermal = 0.0
    W_, H_ = c2d["work_size"]
    dist_center = np.array([W_ / 2.0, H_ / 2.0])
    dist_scale = math.hypot(W_, H_) / 2.0
    if lens.get("enabled") and len(inlier_pts0) >= lens.get("min_points", 120):
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(inlier_pts0))
        n_val = max(20, int(len(perm) * lens.get("val_fraction", 0.2)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        H_train, _ = estimate_homography_robust(inlier_pts1[train_idx], inlier_pts0[train_idx],
                                                  c2d["ransac_thresh"], c2d["ransac_confidence"])
        if H_train is not None:
            rmse_base = _reprojection_rmse(H_train, None, dist_center, dist_scale, inlier_pts1[val_idx], inlier_pts0[val_idx])
            H_dist, k_cand = fit_homography_with_distortion(inlier_pts1[train_idx], inlier_pts0[train_idx],
                                                              H_train, dist_center, dist_scale)
            rmse_dist = _reprojection_rmse(H_dist, k_cand, dist_center, dist_scale, inlier_pts1[val_idx], inlier_pts0[val_idx])
            print(f"Lens self-calibration: without={rmse_base:.2f}px, with k={k_cand:.4f}: {rmse_dist:.2f}px")
            if rmse_dist < rmse_base - lens.get("min_improvement_px", 0.05):
                M, k_thermal = fit_homography_with_distortion(inlier_pts1, inlier_pts0, M, dist_center, dist_scale)
                rmse = _reprojection_rmse(M, k_thermal, dist_center, dist_scale, inlier_pts1, inlier_pts0)
                print(f"  -> adopted (k={k_thermal:.4f}), RMSE now {rmse:.2f}px")
            else:
                print("  -> not adopted")

    tps_src = undistort_division_model(inlier_pts1, dist_center, k_thermal, dist_scale) if k_thermal else inlier_pts1
    map_x = map_y = None
    has_tps = False
    lr = c2d["local_refinement"]
    if lr["enabled"] and len(inlier_pts0) >= lr["min_points"]:
        tps = fit_tps_correction(inlier_pts0, tps_src, M, c2d["work_size"], lr)
        if tps is not None:
            map_x, map_y = tps
            has_tps = True
            print(f"Local TPS refinement fitted on {len(inlier_pts0)} inliers")

    np.savez(
        calib_path, M_work=M, work_size=np.array(c2d["work_size"]),
        n_inliers=inliers, rmse_px=rmse, has_tps=has_tps, k_thermal=k_thermal,
        dist_center=dist_center, dist_scale=dist_scale,
        map_x=map_x if has_tps else np.zeros((1, 1), np.float32),
        map_y=map_y if has_tps else np.zeros((1, 1), np.float32),
    )
    print(f"Saved: {calib_path}")
    return M, c2d["work_size"], ((map_x, map_y) if has_tps else None), (k_thermal, dist_center, dist_scale)


def warp_thermal_via_2d(visible_color, thermal_color, M_work, work_size, tps, distortion):
    """Same output contract as pipeline_sfm.warp_thermal_via_sfm: returns
    (thermal_warped_bgr, valid_mask_uint8) in visible's native frame."""
    out_size = (visible_color.shape[1], visible_color.shape[0])
    k_thermal, dist_center_work, dist_scale_work = distortion if distortion else (0.0, None, None)

<<<<<<< HEAD
    # Build the mask from the TRUE original image, before any processing --
    # every geometric step below (undistortion, homography, TPS) must carry
    # this mask through it too, or padding introduced by that step gets
    # silently marked "valid" and composited in as if it were real data.
    src_mask = np.full(thermal_color.shape[:2], 255, dtype=np.uint8)

=======
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
    if k_thermal:
        map_x_u, map_y_u = build_undistort_map(work_size, thermal_color.shape, dist_center_work, k_thermal, dist_scale_work)
        thermal_color = cv2.remap(thermal_color, map_x_u, map_y_u, interpolation=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
<<<<<<< HEAD
        src_mask = cv2.remap(src_mask, map_x_u, map_y_u, interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    M_full = full_res_matrix(M_work, thermal_color.shape, visible_color.shape, work_size)
    thermal_warped = cv2.warpPerspective(thermal_color, M_full, out_size, flags=cv2.INTER_NEAREST)
=======

    M_full = full_res_matrix(M_work, thermal_color.shape, visible_color.shape, work_size)
    thermal_warped = cv2.warpPerspective(thermal_color, M_full, out_size, flags=cv2.INTER_NEAREST)
    src_mask = np.full(thermal_color.shape[:2], 255, dtype=np.uint8)
>>>>>>> 44a9db3ed608ee77b9e0e1f2d8447c2372a3d5da
    valid_mask = cv2.warpPerspective(src_mask, M_full, out_size, flags=cv2.INTER_NEAREST, borderValue=0)

    if tps is not None:
        map_x_work, map_y_work = tps
        W, H = work_size
        Wv_n, Hv_n = out_size
        sx, sy = Wv_n / W, Hv_n / H
        map_x_native = cv2.resize(map_x_work, out_size, interpolation=cv2.INTER_LINEAR) * sx
        map_y_native = cv2.resize(map_y_work, out_size, interpolation=cv2.INTER_LINEAR) * sy
        thermal_warped = cv2.remap(thermal_warped, map_x_native, map_y_native, interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_mask = cv2.remap(valid_mask, map_x_native, map_y_native, interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return thermal_warped, valid_mask