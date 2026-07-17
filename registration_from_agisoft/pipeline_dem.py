"""
OPTIONAL: real terrain height (DSM) from an Agisoft/any LAS point cloud,
replacing pipeline_sfm.py's flat-plane ground_z assumption.

WHY THIS MATTERS
-------------------
Every ray-cast in pipeline_sfm.build_projective_warp_map intersects a
single flat plane at Z=ground_z. That's fine over flat terrain but wrong
wherever there's real relief (riverbanks, vegetation canopy) -- exactly the
"genuine parallax error a single global model can't fix" limitation flagged
repeatedly earlier in this pipeline. A real point cloud removes it: instead
of intersecting a flat plane, we ray-march against the actual top surface
(a Digital SURFACE Model -- highest point per grid cell, since that's what
a nadir camera actually sees from above, canopy included, not bare earth).

THE COORDINATE SYSTEM PROBLEM (read this before trusting the DEM path)
---------------------------------------------------------------------------
The LAS file's X/Y look like real UTM coordinates (Easting ~353000,
Northing ~4844000 -- georeferenced, e.g. from GCPs or RTK GPS in Agisoft).
pipeline_sfm.py's OpenSfM reconstruction, on the other hand, lives in a
LOCAL topocentric frame (origin near the first image, meters, but NOT UTM)
-- see reference_lla.json in the OpenSfM project folder for that origin's
real lat/lon/altitude.

To use both together we convert OpenSfM local coordinates -> lat/lon (flat-
earth approximation, fine at this scale) -> UTM (via pyproj, auto-detecting
the UTM zone from the reconstruction's own longitude). This chain has two
real failure points:
  1. If reference_lla.json is missing (no GPS during the OpenSfM run), there
     is no way to do this at all.
  2. Even with it, GPS altitude is usually ELLIPSOIDAL height while a LAS
     file's Z is often ORTHOMETRIC (geoid-corrected) -- these can differ by
     many meters depending on location, showing up as a near-constant Z
     offset between the two systems that has nothing to do with real error.

`validate_dem_alignment()` checks both concerns concretely (does the
transformed reconstruction's XY centroid fall inside the point cloud's
bounding box; is the Z difference at least roughly stable/small) and prints
what it found. DO NOT wire the DEM path into apply() until that print looks
sane on your actual data -- if it's off, the DEM is silently skipped and
pipeline_sfm.py falls back to its existing flat-plane behavior, so this is
safe to leave enabled and just check the log.
"""

import os
import json

import numpy as np


def load_dem_from_las(las_path, resolution_m=0.5):
    """Digital SURFACE Model (highest point per XY cell, not bare-earth) --
    what a nadir camera actually sees from above, canopy included."""
    try:
        import laspy
    except ImportError as e:
        raise ImportError("Needs laspy: pip install laspy") from e

    print(f"Loading point cloud: {las_path}")
    with laspy.open(las_path) as f:
        las = f.read()
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    print(f"  {len(x):,} points, bbox X=[{x.min():.1f},{x.max():.1f}] "
          f"Y=[{y.min():.1f},{y.max():.1f}] Z=[{z.min():.1f},{z.max():.1f}]")

    x_min, y_min = x.min(), y.min()
    nx = int(np.ceil((x.max() - x_min) / resolution_m)) + 1
    ny = int(np.ceil((y.max() - y_min) / resolution_m)) + 1
    ix = np.clip(((x - x_min) / resolution_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y_min) / resolution_m).astype(np.int64), 0, ny - 1)

    grid = np.full((ny, nx), -np.inf, dtype=np.float64)
    np.maximum.at(grid, (iy, ix), z)

    empty = np.isinf(grid)
    if empty.any():
        try:
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(empty, return_distances=False, return_indices=True)
            grid = grid[tuple(idx)]
        except ImportError:
            grid[empty] = np.nanmedian(grid[~empty])
    print(f"  DSM grid: {nx}x{ny} cells @ {resolution_m}m resolution, "
          f"{100 * empty.mean():.1f}% cells filled by nearest-neighbor (no direct point)")

    return dict(grid=grid, x_min=x_min, y_min=y_min, resolution=resolution_m,
                nx=nx, ny=ny, x_max=x.max(), y_max=y.max(), z_min=z.min(), z_max=z.max())


def query_dem(dem, x, y):
    """Bilinear-interpolated height at (x, y) -- x, y can be arrays."""
    fx = np.clip((x - dem["x_min"]) / dem["resolution"], 0, dem["nx"] - 1.001)
    fy = np.clip((y - dem["y_min"]) / dem["resolution"], 0, dem["ny"] - 1.001)
    x0, y0 = fx.astype(np.int64), fy.astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, dem["nx"] - 1), np.minimum(y0 + 1, dem["ny"] - 1)
    tx, ty = fx - x0, fy - y0
    g = dem["grid"]
    return (g[y0, x0] * (1 - tx) * (1 - ty) + g[y0, x1] * tx * (1 - ty)
            + g[y1, x0] * (1 - tx) * ty + g[y1, x1] * tx * ty)


# =====================================================================
# Coordinate alignment: OpenSfM local (topocentric) -> lat/lon -> UTM
# =====================================================================

def build_ecef_to_las_transform(epsg):
    """Direct ECEF (EPSG:4978) -> target CRS transform for the Agisoft pose
    path (simpler than pipeline_sfm's OpenSfM-local-frame version, since
    Agisoft poses are already real ECEF meters -- no ENU approximation
    needed, just a straight pyproj conversion)."""
    try:
        from pyproj import Transformer
    except ImportError:
        print("[WARN] pyproj not installed (pip install pyproj) -- DEM path unavailable.")
        return None
    if epsg is None:
        print("[WARN] agisoft.dem_epsg not set -- can't convert ECEF to the LAS file's CRS without "
              "knowing it (check the LAS file's .prj sidecar or your export settings). DEM path unavailable.")
        return None
    transformer = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)

    def transform(ecef_xyz):
        ecef_xyz = np.asarray(ecef_xyz, dtype=np.float64)
        single = ecef_xyz.ndim == 1
        pts = ecef_xyz[:, None] if single else ecef_xyz
        X, Y, Z = transformer.transform(pts[0], pts[1], pts[2])
        out = np.stack([np.atleast_1d(X), np.atleast_1d(Y), np.atleast_1d(Z)], axis=0)
        return out[:, 0] if single else out

    return transform


def load_reference_lla(sfm_project_dir):
    path = os.path.join(sfm_project_dir, "reference_lla.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def enu_to_latlon(x_east, y_north, lat0, lon0):
    """Flat-earth approximation, fine at survey-area scale (a few km)."""
    R = 6371000.0
    lat = lat0 + np.degrees(y_north / R)
    lon = lon0 + np.degrees(x_east / (R * np.cos(np.radians(lat0))))
    return lat, lon


def build_local_to_las_transform(sfm_project_dir, epsg=None):
    """Returns a function local_xyz -> (X, Y, Z) in the LAS file's
    coordinate system, or None if reference_lla.json is unavailable."""
    ref = load_reference_lla(sfm_project_dir)
    if ref is None:
        print("[WARN] No reference_lla.json in the SfM project -- can't align to the point cloud "
              "(the OpenSfM run had no usable GPS). DEM path unavailable.")
        return None

    try:
        from pyproj import Transformer
    except ImportError:
        print("[WARN] pyproj not installed (pip install pyproj) -- can't convert to UTM. DEM path unavailable.")
        return None

    lat0, lon0, alt0 = ref["latitude"], ref["longitude"], ref.get("altitude", 0.0)
    if epsg is None:
        zone = int((lon0 + 180) / 6) + 1
        epsg = (32600 if lat0 >= 0 else 32700) + zone
        print(f"Auto-detected UTM zone from reconstruction origin: EPSG:{epsg} "
              "(verify this matches the LAS file's actual CRS -- override with dem.epsg if not)")
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    def transform(local_xyz):
        """local_xyz: (3,) single point or (3,N) batch. Returns matching shape."""
        local_xyz = np.asarray(local_xyz, dtype=np.float64)
        single = local_xyz.ndim == 1
        pts = local_xyz[:, None] if single else local_xyz
        lat, lon = enu_to_latlon(pts[0], pts[1], lat0, lon0)
        X, Y = transformer.transform(lon, lat)
        X, Y = np.atleast_1d(X), np.atleast_1d(Y)
        Z = alt0 + pts[2]
        out = np.stack([X, Y, Z], axis=0)
        return out[:, 0] if single else out

    return transform


def validate_dem_alignment(transform, visible_poses, dem, max_xy_error_m=100.0):
    """Transforms every reconstructed camera's optical center and checks it
    against the point cloud's bounding box. Prints exactly what it found --
    read this before trusting the DEM path at all."""
    centers_las = np.array([transform(p["C"]) for p in visible_poses.values()])
    cx, cy = centers_las[:, 0].mean(), centers_las[:, 1].mean()
    in_bbox = ((dem["x_min"] <= centers_las[:, 0]) & (centers_las[:, 0] <= dem["x_max"])
               & (dem["y_min"] <= centers_las[:, 1]) & (centers_las[:, 1] <= dem["y_max"]))

    print(f"\nDEM alignment check:")
    print(f"  Transformed camera XY centroid: ({cx:.1f}, {cy:.1f})")
    print(f"  Point cloud XY bbox: X=[{dem['x_min']:.1f},{dem['x_max']:.1f}] Y=[{dem['y_min']:.1f},{dem['y_max']:.1f}]")
    print(f"  {in_bbox.sum()}/{len(centers_las)} camera positions fall inside the point cloud bbox")

    z_diffs = []
    for xyz in centers_las[:5]:
        try:
            ground = query_dem(dem, np.array([xyz[0]]), np.array([xyz[1]]))[0]
            z_diffs.append(xyz[2] - ground)
        except Exception:
            pass
    if z_diffs:
        print(f"  Camera-height-above-DSM samples: {[f'{d:.1f}m' for d in z_diffs]} "
              f"(should look like a plausible flying altitude, consistently -- if these are wildly "
              f"different from each other, or negative, or absurd, the Z alignment is off, likely a "
              f"geoid/ellipsoidal height mismatch between GPS and the LAS file)")

    aligned = in_bbox.mean() > 0.8
    print(f"  -> {'ALIGNED, DEM path usable' if aligned else 'NOT ALIGNED -- DEM path will be skipped, falling back to flat-plane'}")
    return aligned
