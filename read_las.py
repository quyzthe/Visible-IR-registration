import laspy
import numpy as np

# Đường dẫn tới file
las_path = r"E:\\agisoft auto scripts\\from agisoft\\full_flight1_zone1_v_Chunk_1_points.las"

# Đọc file
las = laspy.read(las_path)

# =========================
# HEADER
# =========================
print("=" * 60)
print("LAS HEADER")
print("=" * 60)
print(f"Version           : {las.header.version}")
print(f"Point format      : {las.header.point_format}")
print(f"Number of points  : {len(las.points)}")
print(f"Scales            : {las.header.scales}")
print(f"Offsets           : {las.header.offsets}")

# =========================
# AVAILABLE FIELDS
# =========================
print("\nAvailable dimensions:")
for dim in las.point_format.dimension_names:
    print(" -", dim)

# =========================
# BOUNDING BOX
# =========================
print("\nBounding box")
print(f"X: {las.x.min():.3f} -> {las.x.max():.3f}")
print(f"Y: {las.y.min():.3f} -> {las.y.max():.3f}")
print(f"Z: {las.z.min():.3f} -> {las.z.max():.3f}")

# =========================
# FIRST 10 POINTS
# =========================
print("\nFirst 10 points:")

dims = list(las.point_format.dimension_names)

for i in range(min(10, len(las.points))):
    point = {}
    for d in dims:
        point[d] = las[d][i]
    print(point)

# =========================
# UNIQUE CLASSIFICATIONS
# =========================
if "classification" in dims:
    classes, counts = np.unique(las.classification, return_counts=True)
    print("\nClassification counts:")
    for c, n in zip(classes, counts):
        print(f"Class {c}: {n}")

# =========================
# RGB?
# =========================
if {"red", "green", "blue"}.issubset(dims):
    print("\nThis point cloud contains RGB colors.")

# =========================
# INTENSITY?
# =========================
if "intensity" in dims:
    print("\nIntensity range:")
    print(las.intensity.min(), "->", las.intensity.max())