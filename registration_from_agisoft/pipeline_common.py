"""
Small shared helpers used by more than one pipeline stage -- kept in one
place instead of copy-pasted, so a fix here fixes it everywhere.
"""

import cv2
import numpy as np


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


def to_3x3(M):
    if M.shape == (3, 3):
        return M.astype(np.float64)
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = M
    return out


def apply_homogeneous(M3x3, pts):
    """Nx2 points through a 3x3 matrix with a proper perspective divide --
    works for a true homography or an affine padded to 3x3."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    out_h = pts_h @ M3x3.T
    return out_h[:, :2] / out_h[:, 2:3]


def full_res_matrix(M_work, src_native_shape, dst_native_shape, work_size):
    """M_work (3x3) maps src(work_size) -> dst(work_size). Returns the
    equivalent 3x3 matrix mapping src_native -> dst_native, for warping
    full-resolution images directly with warpPerspective."""
    Hs_n, Ws_n = src_native_shape[:2]
    Hd_n, Wd_n = dst_native_shape[:2]
    W, H = work_size
    S_src = to_3x3(np.array([[W / Ws_n, 0, 0], [0, H / Hs_n, 0]], dtype=np.float64))
    S_dst_inv = to_3x3(np.array([[Wd_n / W, 0, 0], [0, Hd_n / H, 0]], dtype=np.float64))
    return S_dst_inv @ to_3x3(M_work) @ S_src


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


def load_warped_thermal_bgra(path):
    """warped_thermal/*.png are stored as BGRA: BGR = thermal data, alpha =
    exact valid-region mask (which pixels are real thermal vs. black
    padding from the warp)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"{path} has no alpha channel (shape={img.shape}) -- regenerate it via apply().")
    return img[:, :, :3], img[:, :, 3]
