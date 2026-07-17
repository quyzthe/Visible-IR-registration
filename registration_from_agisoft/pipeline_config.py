"""
Single shared CONFIG for the whole thermal/visible registration pipeline.
Every other module imports CONFIG from here -- edit paths/params in ONE
place, run `python main.py`.
"""

CONFIG = dict(
    # ---- Step 1: organize ----
    source_dir=r"E:\drone_090426\Raw_images\DCIM_1\DJI_202604090901_001_zoneseuils1",
    visible_dir=r"E:\drone_090426\Raw_images\DCIM_1\DJI_202604090901_001_zoneseuils1\visible",
    thermal_dir=r"E:\drone_090426\Raw_images\DCIM_1\DJI_202604090901_001_zoneseuils1\thermal",
    register_results_dir=r"E:\drone_090426\Raw_images\DCIM_1\DJI_202604090901_001_zoneseuils1\register_results",
    copy_mode="copy",               # "copy" (safe, keeps originals) or "move"
    fresh_start=False,              # True = wipe visible_dir/thermal_dir/register_results_dir first
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
        xml_path=r"E:\drone_090426\full_flight1_camera.xml",   # <-- CHANGE THIS to your camera XML path
                                                                  # (File > Export > Export Cameras > "Agisoft XML")
        dem_epsg=None,   # EPSG code of your LAS point cloud's CRS (needed to query the DEM in
                          # pipeline_dem.py -- the chunk's own <reference> WKT is WGS84/EPSG:4326,
                          # NOT necessarily what you exported the LAS in). Check the LAS file's
                          # .prj sidecar or your export settings; None = DEM path stays disabled.
    ),

    # ---- OPTIONAL: real terrain height from an Agisoft/any LAS point cloud
    # (pipeline_dem.py), replacing the flat ground_z plane above with actual
    # ray-vs-surface intersection wherever alignment validates successfully.
    # Safe to leave enabled: falls back to the flat plane automatically if
    # the coordinate-system check fails -- READ THE LOG either way.
    dem=dict(
        enabled=False,               # set True once you have a LAS file to try
        las_path=r"PATH\TO\your_agisoft_pointcloud.las",
        resolution_m=0.5,            # DSM grid cell size (meters) -- finer = more accurate but slower
        epsg=None,                   # UTM EPSG code to convert the SfM reconstruction into. None =
                                      # auto-detect from the reconstruction's own longitude -- verify
                                      # this actually matches the LAS file's real CRS (check the printed
                                      # "Auto-detected UTM zone" line against your Agisoft project settings).
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
    densify=dict(
        enabled=True,
        feature_detector="sift",
        n_features=4000,
        save_features=True,
        n_words=750,
        vocab_sample_descriptors=200000,
        rebuild_vocabulary=False,
        top_m_candidates=40,          # per image, how many of the most visually-similar OTHER images
                                       # become match candidates. Consecutive along-track frames look
                                       # almost identical (captured 1-2s apart, huge overlap) and rank
                                       # as MOST similar by BoW -- with a small M they can crowd out the
                                       # images that would actually extend coverage, fragmenting the
                                       # match graph into small isolated clusters instead of one large
                                       # connected one. A bad BoW candidate just fails RANSAC and gets
                                       # dropped, so a larger M costs match-time, not correctness --
                                       # check the "Match graph connectivity" log line; if it still
                                       # reports more than one component, raise this further.
        match_ratio=0.75,
        ransac_thresh=3.0,
        ransac_confidence=0.999,
        min_matches=15,
        min_track_len=2,
        max_pairs_per_run=None,
        skip_already_processed=True,
    ),
)