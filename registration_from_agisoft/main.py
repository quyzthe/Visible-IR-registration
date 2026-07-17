"""
Single entry point for the whole thermal/visible registration pipeline.
Edit paths/params in pipeline_config.py, then just:

    python main.py

Pipeline:
  1) pipeline_organize.organize()   -- sort raw DCIM into visible/ + thermal/
  2) pipeline_apply.apply_pipeline()  -- for each pair:
       2a) PRIMARY: pipeline_sfm -- OpenSfM native rig reconstruction
           (both cameras jointly bundle-adjusted, real per-image 3D poses)
       2b) FALLBACK: pipeline_calibrate_2d -- for any pair SfM couldn't
           reconstruct (self-calibrated homography + lens distortion + TPS)
  3) pipeline_densify.densify()     -- fill remaining gaps using overlap
       between neighboring visible frames (BoW + feature tracks)
"""

from pipeline_config import CONFIG
from pipeline_organize import organize
from pipeline_apply import apply_pipeline
from pipeline_densify import densify


def main(cfg=CONFIG):
    print("=" * 70)
    print("STEP 1: organize")
    print("=" * 70)
    organize(cfg)
    if cfg["dry_run"]:
        return

    print("\n" + "=" * 70)
    print("STEP 2: apply (SfM rig primary, 2D self-calibration fallback)")
    print("=" * 70)
    apply_pipeline(cfg)

    print("\n" + "=" * 70)
    print("STEP 3: densify (BoW + feature tracks)")
    print("=" * 70)
    densify(cfg)


if __name__ == "__main__":
    main()
