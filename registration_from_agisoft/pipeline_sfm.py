"""
STEP 2a (PRIMARY) -- SfM RIG RECONSTRUCTION

Runs OpenSfM directly (not through ODM's higher-level wrapper, which
doesn't obviously expose rig configuration) on BOTH the visible AND
thermal image sequences together, declared as a rigid 2-camera rig via
OpenSfM's native rig support:

  https://opensfm.org/docs/rig.html

    - `opensfm create_rig` groups images into rig instances (one visible +
      one thermal shot, captured together) from a filename PATTERN, and
      writes rig_cameras.json + rig_assignments.json.
    - The normal OpenSfM pipeline (extract_metadata, detect_features,
      match_features, create_tracks, reconstruct) then runs as usual, but
      bundle adjustment now ALSO refines the rig's relative pose (the fixed
      transform between the visible and thermal camera) as part of the
      SAME optimization -- instead of us decomposing a 2D homography and
      guessing which way the composition goes.
    - Because the rig constraint ties a thermal shot's pose rigidly to its
      visible partner's pose, a thermal image can end up with a well-
      determined pose even if it has poor overlap with OTHER thermal
      images (the usual problem with running SfM on thermal alone) -- it
      "rides along" on the visible reconstruction.

reconstruction.json then contains REAL per-image poses for BOTH cameras
directly -- no EXIF-based guess for K_thermal, no decomposeHomographyMat,
no rig_compose_order coin flip.

THINGS THAT ARE NOT VERIFIED AGAINST A REAL RUN
---------------------------------------------------
- The exact `create_rig` pattern syntax (rig_pattern in pipeline_config.py)
  is built from OpenSfM's documented example ("(RED)"/"(GREEN)" for
  multispectral filenames) but wasn't tested against your actual OpenSfM
  version -- run `opensfm create_rig --help` inside your Docker image and
  adjust rig_pattern if it errors.
- The `opensfm_bin` path inside an ODM-based Docker image -- if
  opensfm.CONFIG["sfm"]["opensfm_bin"] is left as None, this script tries a
  few common locations and tells you how to find the right one if none work.

Everything downstream (pipeline_apply.py, pipeline_densify.py) only cares
about reconstruction.json existing and containing poses -- it doesn't care
how you produced it, so if you already have a rig reconstruction from
elsewhere, just point sfm.project_dir at it and skip straight to
`load_rig_reconstruction`.
"""

import os
import json
import shutil
import subprocess

import cv2
import numpy as np

from pipeline_organize import find_pairs


_CANDIDATE_OPENSFM_PATHS = [
    "/code/SuperBuild/install/bin/opensfm",
    "/code/SuperBuild/src/opensfm/bin/opensfm",
    "/opt/opensfm/bin/opensfm",
    "opensfm",  # on PATH
]


def _find_opensfm_bin(cfg, docker_image):
    if cfg["sfm"]["opensfm_bin"]:
        return cfg["sfm"]["opensfm_bin"]
    for candidate in _CANDIDATE_OPENSFM_PATHS:
        check = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "sh", docker_image, "-c", f"test -x {candidate} || which {candidate}"],
            capture_output=True, text=True,
        )
        if check.returncode == 0:
            return candidate
    raise RuntimeError(
        "Could not find the `opensfm` executable in the Docker image automatically. Run:\n"
        f"  docker run --rm --entrypoint find {docker_image} / -iname opensfm -type f\n"
        "and set sfm.opensfm_bin to whatever path that finds."
    )


def prepare_rig_project(cfg):
    """Copies BOTH visible and thermal images into one OpenSfM project's
    images/ folder (rig instances are grouped by filename pattern, not by
    folder), so the create_rig pattern can tell them apart."""
    sfm = cfg["sfm"]
    images_dir = os.path.join(sfm["project_dir"], "images")
    os.makedirs(images_dir, exist_ok=True)

    pairs = find_pairs(cfg["visible_dir"], cfg["thermal_dir"])
    if not pairs:
        raise RuntimeError("No organized pairs found -- run pipeline_organize.organize() first.")

    print(f"Copying {len(pairs)} visible + {len(pairs)} thermal images into the rig project...")
    for idx, vpath, tpath in pairs:
        for src in (vpath, tpath):
            dst = os.path.join(images_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    return images_dir


def run_opensfm_rig_pipeline(cfg):
    """Runs the OpenSfM commands needed to go from raw images to a rig-
    aware reconstruction.json. Each `docker run` is a fresh container
    sharing the project dir as a volume, matching how ODM itself invokes
    OpenSfM internally."""
    sfm = cfg["sfm"]
    project_dir = sfm["project_dir"]
    recon_path = os.path.join(project_dir, "reconstruction.json")

    if sfm["skip_if_reconstruction_exists"] and os.path.exists(recon_path):
        print(f"Found existing {recon_path} -- skipping the SfM run.")
        return recon_path

    prepare_rig_project(cfg)
    opensfm_bin = _find_opensfm_bin(cfg, sfm["docker_image"])

    def run(*args):
        cmd = ["docker", "run", "--rm", "-v", f"{project_dir}:/data",
               "--entrypoint", opensfm_bin, sfm["docker_image"], *args, "/data"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    rig_pattern_json = json.dumps(sfm["rig_pattern"])
    print("=== opensfm create_rig ===")
    run("create_rig", rig_pattern_json)

    for stage in ("extract_metadata", "detect_features", "match_features", "create_tracks", "reconstruct"):
        print(f"=== opensfm {stage} ===")
        run(stage)

    if not os.path.exists(recon_path):
        raise RuntimeError(f"OpenSfM finished but {recon_path} wasn't produced -- check the log above.")
    return recon_path


# =====================================================================
# PARSE reconstruction.json -- confirmed OpenSfM conventions (validated
# against a real reconstruction.json the user provided):
#   - rotation: world->camera axis-angle (Rodrigues), translation part of
#     [R|t] with X_camera = R @ X_world + t. Optical center C = -R.T @ t.
#   - camera model (projection_type "brown"): focal_x/focal_y normalized
#     by max(width,height), c_x/c_y a normalized principal-point offset
#     from image center, k1/k2/k3/p1/p2 Brown-Conrady distortion --
#     self-calibrated by bundle adjustment for BOTH cameras now (thermal
#     included), no EXIF guess needed for either.
# =====================================================================

def K_from_opensfm_camera(cam):
    w, h = cam["width"], cam["height"]
    m = max(w, h)
    focal_x = cam.get("focal_x", cam.get("focal"))
    focal_y = cam.get("focal_y", focal_x)
    cx_norm = cam.get("c_x", 0.0)
    cy_norm = cam.get("c_y", 0.0)
    K = np.array([[focal_x * m, 0, w / 2.0 + cx_norm * m],
                  [0, focal_y * m, h / 2.0 + cy_norm * m],
                  [0, 0, 1]], dtype=np.float64)
    dist = dict(k1=cam.get("k1", 0.0), k2=cam.get("k2", 0.0),
                p1=cam.get("p1", 0.0), p2=cam.get("p2", 0.0), k3=cam.get("k3", 0.0))
    return K, dist


def load_rig_reconstruction(recon_path):
    """Returns {image_filename: dict(K,R,t,C,width,height,distortion)} for
    EVERY reconstructed shot -- visible AND thermal together, since they
    were reconstructed jointly."""
    with open(recon_path) as f:
        recons = json.load(f)

    poses = {}
    for recon in recons:
        cameras = recon["cameras"]
        for shot_id, shot in recon["shots"].items():
            cam = cameras[shot["camera"]]
            K, dist = K_from_opensfm_camera(cam)
            R, _ = cv2.Rodrigues(np.array(shot["rotation"], dtype=np.float64))
            t = np.array(shot["translation"], dtype=np.float64)
            C = -R.T @ t
            poses[shot_id] = dict(K=K, R=R, t=t, C=C, width=cam["width"], height=cam["height"], distortion=dist)

    n_v = sum(1 for name in poses if "_V." in name.upper())
    n_t = sum(1 for name in poses if "_T." in name.upper())
    print(f"Loaded {len(poses)} reconstructed poses ({n_v} visible, {n_t} thermal) from {recon_path}")
    return poses


def estimate_ground_z(poses):
    """OpenSfM places the ground near Z=0 in its local metric frame when
    GPS is available -- median camera height above that is a self-
    consistent ground-plane estimate. Same planarity assumption the whole
    pipeline relies on; a real DEM would remove it."""
    zs = [p["C"][2] for p in poses.values()]
    return float(np.median(zs))


# =====================================================================
# DENSE PROJECTIVE WARP: for a pair with BOTH poses known, ray-cast every
# visible-frame pixel either to a flat ground plane (default) or, if a
# validated DEM is supplied (see pipeline_dem.py), to the REAL terrain
# surface -- then reproject into the thermal camera. Gives a proper
# per-pixel remap, not a single fixed 2D homography.
# =====================================================================

def _ray_dem_intersect_batch(origin_local, dirs_local, to_las, dem, max_dist=800.0,
                              coarse_step=2.0, refine_iters=8):
    """origin_local: (3,) camera center, LOCAL (OpenSfM) frame, shared by
    every ray. dirs_local: (3,N) ray directions, LOCAL frame. Returns (3,N)
    intersection points in LOCAL frame (so the caller keeps using local-
    frame math afterward) -- found by marching along each ray, transforming
    sample points to the LAS/UTM frame to query DSM height, until the ray
    drops below the surface, then bisecting to refine. Falls back to
    origin+max_dist*dir for any ray that never intersects within max_dist
    (e.g. pointing at the sky)."""
    from pipeline_dem import query_dem

    N = dirs_local.shape[1]
    t_lo = np.zeros(N)
    t_hi = np.full(N, np.nan)

    def height_above_dsm(t):
        pts_local = origin_local[:, None] + t[None, :] * dirs_local
        pts_las = to_las(pts_local)
        dsm_z = query_dem(dem, pts_las[0], pts_las[1])
        return pts_las[2] - dsm_z  # >0 = still above the surface

    t = np.full(N, coarse_step)
    prev_t = np.zeros(N)
    prev_above = height_above_dsm(prev_t) > 0
    found = np.zeros(N, dtype=bool)
    while t.max() <= max_dist and not found.all():
        cur_above = height_above_dsm(t) > 0
        crossed = prev_above & (~cur_above) & (~found)
        t_lo[crossed] = prev_t[crossed]
        t_hi[crossed] = t[crossed]
        found |= crossed
        prev_t, prev_above = t.copy(), cur_above
        t = np.where(found, t, t + coarse_step)

    # bisect the bracketed rays
    for _ in range(refine_iters):
        mid = (t_lo + t_hi) / 2
        above_mid = height_above_dsm(mid) > 0
        t_lo = np.where(found & above_mid, mid, t_lo)
        t_hi = np.where(found & (~above_mid), mid, t_hi)
    t_final = np.where(found, (t_lo + t_hi) / 2, max_dist)

    return origin_local[:, None] + t_final[None, :] * dirs_local


def build_projective_warp_map(visible_pose, thermal_pose, ground_z, out_size, grid_step=20,
                               dem=None, to_las=None):
    """dst=visible frame, src=thermal frame. Returns (map_x, map_y) at
    out_size resolution, ready for cv2.remap. Uses the real terrain surface
    (dem + to_las, from pipeline_dem.py) if both are given and alignment
    was validated; otherwise intersects the flat plane Z=ground_z as before."""
    W, H = out_size
    xs = np.unique(np.append(np.arange(0, W, grid_step), W - 1))
    ys = np.unique(np.append(np.arange(0, H, grid_step), H - 1))
    gx, gy = np.meshgrid(xs, ys)
    pix = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=0).astype(np.float64)

    K_v, R_v, C_v = visible_pose["K"], visible_pose["R"], visible_pose["C"]
    rays_cam = np.linalg.inv(K_v) @ pix
    rays_world = R_v.T @ rays_cam

    if dem is not None and to_las is not None:
        pts_world = _ray_dem_intersect_batch(C_v, rays_world, to_las, dem)
    else:
        s = (ground_z - C_v[2]) / rays_world[2]
        pts_world = C_v[:, None] + s[None, :] * rays_world

    K_t, R_t, t_t = thermal_pose["K"], thermal_pose["R"], thermal_pose["t"]
    pts_cam_t = R_t @ pts_world + t_t[:, None]
    pts_img_t = K_t @ pts_cam_t
    map_x = (pts_img_t[0] / pts_img_t[2]).reshape(gy.shape).astype(np.float32)
    map_y = (pts_img_t[1] / pts_img_t[2]).reshape(gy.shape).astype(np.float32)

    map_x_dense = cv2.resize(map_x, out_size, interpolation=cv2.INTER_LINEAR)
    map_y_dense = cv2.resize(map_y, out_size, interpolation=cv2.INTER_LINEAR)
    return map_x_dense, map_y_dense


def warp_thermal_via_sfm(visible_color, thermal_color, visible_pose, thermal_pose, ground_z,
                          grid_step=20, dem=None, to_las=None):
    """Returns (thermal_warped_bgr, valid_mask_uint8) in visible's native
    frame -- same output contract as the 2D fallback path, so downstream
    code (apply/densify) doesn't need to know which method produced it."""
    out_size = (visible_color.shape[1], visible_color.shape[0])
    map_x, map_y = build_projective_warp_map(visible_pose, thermal_pose, ground_z, out_size,
                                              grid_step, dem, to_las)

    thermal_warped = cv2.remap(thermal_color, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    th, tw = thermal_color.shape[:2]
    valid = ((map_x >= 0) & (map_x < tw) & (map_y >= 0) & (map_y < th)).astype(np.uint8) * 255
    return thermal_warped, valid
