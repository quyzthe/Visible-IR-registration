"""
Single shared CONFIG for the whole thermal/visible registration pipeline.
Every other module imports CONFIG from here -- edit paths/params in ONE
place, run `python main.py`.
"""

CONFIG = dict(
    # ---- Step 1: organize ----
    source_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1",
    visible_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\visible",
    thermal_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\thermal",
    register_results_dir=r"E:\drone_090426\Raw_images\DCIM_1\full_flight1_zone1\register_results",
    copy_mode="copy",               # "copy" (safe, keeps originals) or "move"
    fresh_start=True,              # True = wipe visible_dir/thermal_dir/register_results_dir first
    min_prefix_fallback_len=6,
    resolution_threshold_px=None,   # None = auto-detect
    dry_run=False,
    organize_max_per_run=None,      # None = organize everything found this run

    # ---- Step 2a (PRIMARY): where do VISIBLE camera poses come from? ----
    pose_source="agisoft",  # "agisoft" (you have this -- reuses your project's own bundle
                             # adjustment, NO Docker/OpenSfM/ODM needed at all) or "opensfm" (an
                             # ALTERNATIVE for when you DON'T have an Agisoft project -- runs OpenSfM
                             # itself via Docker; see pipeline_sfm.py. Needs its own sfm=dict config
                             # block added back in -- removed here since unused with "agisoft").
                             # Either way, thermal NEVER goes through SfM/bundle adjustment -- its
                             # pose is always derived as (visible pose) + (rig transform self-
                             # calibrated in pipeline_calibrate_2d.py from thermal-visible LoFTR
                             # matches), which is what pipeline_calibrate_2d.py is for.

    agisoft=dict(
        xml_path=r"E:\agisoft auto scripts\from agisoft\full_flight1 camera.xml",
        thermal_hfov_deg=None,   # Fallback ONLY used if the thermal image's own EXIF has no usable
                                  # focal-length tags at all (check the "[EXIF debug]" log line to see
                                  # what was actually found before assuming this is needed). Set to
                                  # your thermal camera's published horizontal field of view in
                                  # degrees (e.g. DJI's M4T spec sheet) to enable the rig-derived 3D
                                  # path when EXIF alone isn't enough.
        geometric_fill_max_rmse_px=50.0,   # pipeline_densify.py's geometric_fill validates itself
                                  # against independent SIFT matches before trusting pose+ground_z/DSM
                                  # for compositing -- if RMSE exceeds this, EVERY image falls back to
                                  # the slower SIFT-based densify path instead. Lower = stricter.
        dem_epsg=None,   # <-- STILL MISSING, DEM stays disabled until this is set. EPSG code of your
                          # LAS point cloud's CRS -- the chunk's own <reference> WKT is WGS84/EPSG:4326,
                          # that is NOT what you exported the LAS in. Check the LAS file's .prj sidecar
                          # (same folder, same name as the .las) or your export dialog's CRS setting.
    ),

    # ---- OPTIONAL: real terrain height from an Agisoft/any LAS point cloud
    # (pipeline_dem.py), replacing the flat ground_z plane above with actual
    # ray-vs-surface intersection wherever alignment validates successfully.
    # Safe to leave enabled: falls back to the flat plane automatically if
    # the coordinate-system check fails -- READ THE LOG either way.
    dem=dict(
        enabled=True,
        las_path=r"E:\agisoft auto scripts\from agisoft\full_flight1_zone1_v_Chunk_1_points.las",
        resolution_m=0.5,            # DSM grid cell size (meters) -- finer = more accurate but slower
        epsg=None,                   # NOT used with pose_source="agisoft" -- that path reads
                                      # agisoft.dem_epsg above instead. Only relevant for pose_source="opensfm".
    ),

    # ---- Step 2b (FALLBACK): 2D self-calibration ----
    # Used ONLY for pairs the SfM rig reconstruction didn't produce a pose
    # for (failed to reconstruct that image, insufficient overlap, etc.) --
    # or for the whole dataset if sfm.enabled=False. Same method as before:
    # pooled homography + guided inlier recovery + lens self-calibration +
    # local TPS refinement.
    calibrate_2d=dict(
        work_size=(640, 480),
        clahe_clip=3.0,
        clahe_tile=(8, 8),
        conf_thresh=0.5,
        min_matches=20,
        ransac_thresh=3.0,
        ransac_confidence=0.999,
        transform_model="homography",
        allow_shear=False,
        min_inliers=15,
        min_inlier_ratio=0.25,
        expected_scale_range=(0.3, 3.0),
        max_rotation_deg=25.0,
        max_anisotropy=1.6,
        sample_size=25,
        recalibrate=False,
        save_features=True,
        guided_recovery_conf_thresh=0.15,
        lens_self_calibration=dict(enabled=True, min_points=120, val_fraction=0.2, min_improvement_px=0.05),
        local_refinement=dict(enabled=True, min_points=80, grid_step_px=20,
                               max_correction_px=15.0, tps_smoothing=2.0),
    ),

    # ---- Step 3: apply (produces warped_thermal/ + overlays/ for every pair) ----
    apply=dict(
        save_warped_thermal=True,
        save_overlay=True,
        max_pairs_per_run=None,     # None = process everything found; int = only this many NEW pairs
        skip_already_processed=True,
    ),

    # ---- Step 4: densify (BoW + feature tracks, fills gaps using overlap
    # between NEIGHBORING visible frames) ----
    # ---- OPTIONAL: reuse an already-completed OpenSfM/ODM run's own
    # features + matches instead of recomputing them in densify() below.
    # Populates the SAME cache files pipeline_densify.py's SIFT path would
    # produce, so create_tracks/fill work unchanged either way.
    opensfm_import=dict(
        enabled=True,
        opensfm_dir=r"E:\drone_090426\Raw_images\DCIM_1\feed_odm\opensfm",  # contains features/ and matches/
        features_suffix=".features.npz",   # verify against your actual filenames if import fails
        matches_suffix="_matches.pkl.gz",
    ),

    densify=dict(
        enabled=True,
        # No feature detection or BoW candidate selection happens in this
        # pipeline anymore -- coverage comes from geometric_fill (real
        # poses) and pipeline_opensfm_import.py (reused ODM matches).
        # These params are only used by create_tracks/fill, which consume
        # whatever correspondences those two sources produced:
        ransac_thresh=3.0,        # inlier distance (px) when fitting a homography from track points
        ransac_confidence=0.999,
        min_matches=15,           # minimum shared track points required to trust a homography fit
        min_track_len=2,          # a track needs to span at least this many images to be kept
        max_pairs_per_run=None,
        skip_already_processed=True,
    ),
)