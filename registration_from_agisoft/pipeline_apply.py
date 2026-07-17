"""
STEP 3 -- APPLY

For every organized pair, get a thermal-onto-visible warp from whichever
source is available, best first:

  1. A per-image 3D pose (from pose_source="agisoft" or "opensfm") -- real
     geometry, most accurate, handles genuine terrain relief if dem.enabled
     validates successfully.
  2. The 2D self-calibrated homography fallback (pipeline_calibrate_2d.py)
     -- used for any pair missing a 3D pose (unaligned in Agisoft/OpenSfM,
     or if the whole 3D path isn't available/didn't validate).

Both paths produce the exact same output contract, so pipeline_densify.py
never needs to know which one was used for a given pair:

    register_results/warped_thermal/<idx>_T_warped.png   (BGRA: BGR=thermal,
                                                            alpha=valid mask)
    register_results/overlays/<idx>_overlay.png           (full-opacity
                                                            composite: real
                                                            thermal where
                                                            valid, original
                                                            visible elsewhere)
"""

import os

import cv2
import numpy as np

from pipeline_organize import find_pairs
from pipeline_common import load_color


def _scale_K(K, native_wh, work_wh):
    sx, sy = work_wh[0] / native_wh[0], work_wh[1] / native_wh[1]
    K2 = K.copy()
    K2[0, 0] *= sx; K2[0, 2] *= sx
    K2[1, 1] *= sy; K2[1, 2] *= sy
    return K2


def _get_agisoft_thermal_poses(cfg, all_pairs, visible_poses, M_work, work_size):
    """Derives a 3D pose for thermal images by decomposing the 2D
    self-calibrated homography into a rig transform, then composing it with
    every known Agisoft visible pose. Validated (in WORK-resolution space,
    using the SAME correspondences pipeline_calibrate_2d used) against the
    2D-only baseline RMSE before being trusted -- returns {} if it doesn't
    hold up, so the caller just uses 2D for everyone (safe default)."""
    import pipeline_agisoft as pag

    sensors, _, _ = pag.parse_agisoft_xml(cfg["agisoft"]["xml_path"])
    any_sensor = next(iter(sensors.values()))
    K_visible_native = pag.K_from_sensor(any_sensor)
    native_wh_v = (any_sensor["width"], any_sensor["height"])

    sample_tpath = all_pairs[len(all_pairs) // 2][2]
    intr_t = pag.get_camera_intrinsics_from_exif(sample_tpath)
    if intr_t is None:
        print("[WARN] Could not get thermal EXIF intrinsics -- rig derivation skipped, using 2D for everyone.")
        return {}

    K_thermal_native = pag.K_from_intrinsics(intr_t)
    native_wh_t = (intr_t["width"], intr_t["height"])
    K_visible_work = _scale_K(K_visible_native, native_wh_v, work_size)
    K_thermal_work = _scale_K(K_thermal_native, native_wh_t, work_size)

    sample_vname = sorted(visible_poses.keys())[len(visible_poses) // 2]
    sample_pose = visible_poses[sample_vname]
    alt_hint = cfg["agisoft"].get("altitude_hint_m", 56.3)  # this project's own report says 56.3m --
                                                             # override for a different flight/project
    up = sample_pose["C"] / np.linalg.norm(sample_pose["C"])
    ground_z_ecef = sample_pose["C"] - alt_hint * up

    R_rel, t_rel = pag.decompose_rig_from_homography(M_work, K_visible_work, K_thermal_work, ground_z_ecef, sample_pose)

    # ---- validate in WORK-resolution space, same correspondences + same
    # metric pipeline_calibrate_2d.calibrate_2d() already used ----
    calib_path = os.path.join(cfg["register_results_dir"], "calibration_2d.npz")
    feat_dir = os.path.join(cfg["register_results_dir"], "calibration_features")
    if not (os.path.exists(calib_path) and os.path.isdir(feat_dir)):
        print("[WARN] No calibration_2d.npz/calibration_features to validate the rig against -- "
              "using 2D for everyone.")
        return {}
    baseline_rmse = float(np.load(calib_path)["rmse_px"])

    errs = []
    for fname in os.listdir(feat_dir):
        idx = fname.replace(".npz", "")
        vname = f"DJI_{idx}_V.JPG"
        if vname not in visible_poses:
            continue
        data = np.load(os.path.join(feat_dir, fname))
        mkpts0, mkpts1, conf = data["mkpts0"], data["mkpts1"], data["conf"]
        keep = conf >= cfg["calibrate_2d"]["conf_thresh"]
        mkpts0, mkpts1 = mkpts0[keep], mkpts1[keep]
        if len(mkpts0) == 0:
            continue
        vpose = visible_poses[vname]
        R_t, t_t = R_rel @ vpose["R"], R_rel @ vpose["t"] + t_rel

        pix = np.hstack([mkpts0, np.ones((len(mkpts0), 1))]).T
        rays_cam = np.linalg.inv(K_visible_work) @ pix
        rays_world = vpose["R"].T @ rays_cam
        up_v = vpose["C"] / np.linalg.norm(vpose["C"])
        denom = up_v @ rays_world
        s = (up_v @ (ground_z_ecef - vpose["C"])) / denom
        pts_world = vpose["C"][:, None] + s[None, :] * rays_world

        pts_cam_t = R_t @ pts_world + t_t[:, None]
        pts_img_t = K_thermal_work @ pts_cam_t
        pred_thermal = (pts_img_t[:2] / pts_img_t[2]).T
        errs.append(np.linalg.norm(pred_thermal - mkpts1, axis=1))

    rig_rmse = float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else float("inf")
    print(f"Rig-derived 3D pose validation: {rig_rmse:.2f}px vs 2D-only baseline {baseline_rmse:.2f}px (work-res)")
    if rig_rmse < baseline_rmse * 1.5:
        by_visible_name = pag.derive_thermal_poses(visible_poses, K_thermal_native, R_rel, t_rel)
        # re-key by THERMAL filename (derive_thermal_poses keys by visible name,
        # since it iterates visible_poses) so the apply loop -- which looks up
        # thermal poses by the thermal image's own filename, same as the
        # OpenSfM path -- finds them correctly.
        thermal_poses = {}
        for idx, vpath, tpath in all_pairs:
            vname = os.path.basename(vpath)
            if vname in by_visible_name:
                thermal_poses[os.path.basename(tpath)] = by_visible_name[vname]
        print(f"  -> ACCEPTED: using per-image 3D poses for {len(thermal_poses)} images")
        return thermal_poses
    print("  -> REJECTED (much worse than the 2D baseline): using 2D homography for everyone")
    return {}


def apply_pipeline(cfg):
    ap = cfg["apply"]
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

    if ap.get("skip_already_processed", True) and os.path.isdir(out_dirs["overlays"]):
        done_ids = {fn[:-len("_overlay.png")] for fn in os.listdir(out_dirs["overlays"]) if fn.endswith("_overlay.png")}
        pending = [(idx, v, t) for idx, v, t in pairs if idx not in done_ids]
    else:
        pending = pairs

    n_done = total_found - len(pending)
    limit = ap.get("max_pairs_per_run")
    run_now = pending[:limit] if limit is not None else pending
    print(f"{total_found} pairs total | {n_done} already applied | applying to {len(run_now)} this session"
          + (f" | {len(pending) - len(run_now)} will remain for next run" if len(pending) > len(run_now) else ""))
    if not run_now:
        return []

    # ---- Step 2a: 3D poses, from whichever pose_source is configured ----
    visible_poses = {}
    thermal_poses_3d = {}
    ground_z = None
    dem = to_las = None
    pose_source = cfg.get("pose_source", "agisoft")

    if pose_source == "agisoft" and cfg.get("agisoft", {}).get("xml_path"):
        import pipeline_agisoft as pag
        from pipeline_calibrate_2d import calibrate_2d
        try:
            visible_paths_by_label = {os.path.basename(v): v for _, v, _ in pairs}
            visible_poses = pag.load_agisoft_poses(cfg, visible_paths_by_label)
            print("\nCalibrating 2D homography (needed both as the fallback AND to derive the rig)...")
            M_work, work_size, tps, distortion = calibrate_2d(cfg)
            thermal_poses_3d = _get_agisoft_thermal_poses(cfg, pairs, visible_poses, M_work, work_size)
            if thermal_poses_3d:
                zs = [p["C"] for p in visible_poses.values()]
                C_mean = np.mean(zs, axis=0)
                alt_hint = cfg["agisoft"].get("altitude_hint_m", 56.3)
                ground_z = C_mean - alt_hint * (C_mean / np.linalg.norm(C_mean))
        except Exception as e:
            print(f"[WARN] Agisoft pose path failed ({e}) -- falling back to 2D for everyone.")
            M_work = work_size = tps = distortion = None

        dcfg = cfg.get("dem", {"enabled": False})
        if thermal_poses_3d and dcfg.get("enabled"):
            try:
                from pipeline_dem import load_dem_from_las, build_ecef_to_las_transform, validate_dem_alignment
                candidate_dem = load_dem_from_las(dcfg["las_path"], dcfg["resolution_m"])
                candidate_transform = build_ecef_to_las_transform(cfg["agisoft"].get("dem_epsg"))
                if candidate_transform is not None and validate_dem_alignment(candidate_transform, visible_poses, candidate_dem):
                    dem, to_las = candidate_dem, candidate_transform
                    print("DEM path ENABLED -- ray-casting against real terrain, not a flat plane.")
                else:
                    print("DEM path DISABLED (alignment check failed) -- using flat-plane ground_z instead.")
            except Exception as e:
                print(f"[WARN] DEM setup failed ({e}) -- using flat-plane ground_z instead.")

    elif pose_source == "opensfm" and cfg["sfm"]["enabled"]:
        from pipeline_sfm import run_opensfm_rig_pipeline, load_rig_reconstruction, estimate_ground_z
        try:
            recon_path = run_opensfm_rig_pipeline(cfg)
            all_poses = load_rig_reconstruction(recon_path)
            visible_poses = {k: v for k, v in all_poses.items() if "_V." in k.upper()}
            thermal_poses_3d = {k: v for k, v in all_poses.items() if "_T." in k.upper()}
            ground_z = cfg["sfm"]["ground_z"] or estimate_ground_z(all_poses)
            print(f"SfM ground_z estimate: {ground_z:.1f} m")
        except Exception as e:
            print(f"[WARN] SfM rig reconstruction unavailable/failed ({e}) -- falling back to 2D for everyone.")

        dcfg = cfg.get("dem", {"enabled": False})
        if thermal_poses_3d and dcfg.get("enabled"):
            try:
                from pipeline_dem import load_dem_from_las, build_local_to_las_transform, validate_dem_alignment
                candidate_dem = load_dem_from_las(dcfg["las_path"], dcfg["resolution_m"])
                candidate_transform = build_local_to_las_transform(cfg["sfm"]["project_dir"], dcfg.get("epsg"))
                if candidate_transform is not None and validate_dem_alignment(candidate_transform, visible_poses, candidate_dem):
                    dem, to_las = candidate_dem, candidate_transform
                    print("DEM path ENABLED -- ray-casting against real terrain, not a flat plane.")
                else:
                    print("DEM path DISABLED (alignment check failed) -- using flat-plane ground_z instead.")
            except Exception as e:
                print(f"[WARN] DEM setup failed ({e}) -- using flat-plane ground_z instead.")

    # ---- Step 2b: 2D fallback, calibrated from whichever pairs the 3D path
    # couldn't cover (or from everyone, if the 3D path is disabled/failed) ----
    need_2d_fallback = [
        (idx, v, t) for idx, v, t in run_now
        if not (os.path.basename(v) in visible_poses and os.path.basename(t) in thermal_poses_3d)
    ]
    M_work = work_size = tps = distortion = None
    if need_2d_fallback:
        from pipeline_calibrate_2d import calibrate_2d, warp_thermal_via_2d
        print(f"\n{len(need_2d_fallback)}/{len(run_now)} pairs need the 2D fallback "
              f"(no 3D pose for one or both images) -- calibrating it...")
        M_work, work_size, tps, distortion = calibrate_2d(cfg)

    # ---- apply ----
    from pipeline_sfm import warp_thermal_via_sfm  # reused: same {K,R,t,C} pose-dict interface
    n_3d, n_2d, n_err = 0, 0, 0
    for idx, vpath, tpath in run_now:
        try:
            visible_color = load_color(vpath)
            thermal_color = load_color(tpath)
            vname, tname = os.path.basename(vpath), os.path.basename(tpath)

            if vname in visible_poses and tname in thermal_poses_3d:
                thermal_warped, valid_mask = warp_thermal_via_sfm(
                    visible_color, thermal_color, visible_poses[vname], thermal_poses_3d[tname],
                    ground_z, cfg.get("sfm", {}).get("warp_grid_step_px", 20), dem, to_las,
                )
                n_3d += 1
                method = f"{pose_source}+dem" if dem is not None else pose_source
            else:
                thermal_warped, valid_mask = warp_thermal_via_2d(
                    visible_color, thermal_color, M_work, work_size, tps, distortion
                )
                n_2d += 1
                method = "2d"

            thermal_bgra = np.dstack([thermal_warped, valid_mask])
            if ap["save_warped_thermal"]:
                cv2.imwrite(os.path.join(out_dirs["warped_thermal"], f"{idx}_T_warped.png"), thermal_bgra)
            if ap["save_overlay"]:
                overlay = visible_color.copy()
                overlay[valid_mask > 0] = thermal_warped[valid_mask > 0]
                cv2.imwrite(os.path.join(out_dirs["overlays"], f"{idx}_overlay.png"), overlay)

            print(f"[{idx}] applied ({method})")
        except Exception as e:
            n_err += 1
            print(f"[{idx}] ERROR: {e}")

    print(f"\nThis run: {n_3d} via 3D pose, {n_2d} via 2D fallback, {n_err} error, {len(run_now)} total")
    return run_now
