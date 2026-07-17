"""
ALTERNATIVE to pipeline_sfm.py: use camera poses ALREADY computed by an
existing Agisoft project (File > Export > Export Cameras > "Agisoft XML"),
instead of running SfM ourselves. Metashape already did the hard part --
bundle adjustment on the VISIBLE sequence, which is the only sequence
that's feasible to run SfM on anyway (thermal has too little inter-frame
overlap for SfM to track on its own -- the whole reason this registration
pipeline exists). No Docker, no OpenSfM, no create_rig needed. Thermal
never participates in SfM at any point -- its pose is derived purely from
(known visible pose) + (rig transform self-calibrated in
pipeline_calibrate_2d.py from thermal-visible LoFTR matches, which handles
cross-modal matching far better than classical SfM feature matching would).

FORMAT -- verified against a real export from this project
------------------------------------------------------------
    <chunk>
      <sensors>
        <sensor id="0" ...>
          <calibration>
            <fx>..</fx> <fy>..</fy> <cx>..</cx> <cy>..</cy>
            <k1>..</k1> <k2>..</k2> <k3>..</k3> <p1>..</p1> <p2>..</p2>
          </calibration>
        </sensor>
      </sensors>
      <cameras>
        <camera id="0" label="DJI_..._V.JPG" sensor_id="0">
          <transform>16 numbers, row-major 4x4, CAMERA-TO-CHUNK-LOCAL</transform>
        </camera>            <!-- cameras that failed to align have NO <transform>
                                  at all, only a raw GPS <reference> -- skipped here,
                                  falls back to pipeline_calibrate_2d.py for that pair -->
      </cameras>
      <transform>            <!-- chunk-level: LOCAL -> ECEF similarity transform -->
        <rotation>9 numbers, row-major 3x3</rotation>
        <translation>3 numbers, ECEF meters</translation>
        <scale>1 number</scale>
      </transform>
      <reference>WKT of the chunk's reference CRS (WGS 84 / EPSG:4326 in this
                  project's report) -- NOTE this is the CRS the chunk's own
                  internal reference/GPS uses, not necessarily the CRS you
                  exported the LAS point cloud in (Metashape lets you pick a
                  different CRS specifically at export time).</reference>
    </chunk>

Full chain per camera: camera-local -> (camera transform) -> chunk-local ->
(chunk transform: scale*R + T) -> ECEF (EPSG:4978) -> (pyproj) -> whatever
CRS the LAS file actually uses.

This is a much more direct/less ambiguous format than an Omega-Phi-Kappa
text export (direct matrices, not angles needing a guessed composition
convention) -- but `validate_agisoft_poses()` still does a lightweight
sanity check (reprojects real SIFT matches between two nearby images) since
row-major vs column-major matrix reading is still a place a mistake could
hide. Read that printed RMSE before trusting the result.
"""

import xml.etree.ElementTree as ET

import cv2
import numpy as np


# =====================================================================
# PARSE
# =====================================================================

def _parse_matrix(text, n):
    vals = [float(v) for v in text.split()]
    if len(vals) != n * n:
        raise ValueError(f"Expected {n * n} values, got {len(vals)}")
    return np.array(vals).reshape(n, n)


def parse_agisoft_xml(path):
    """Returns (sensors, cameras, chunk) where:
    sensors: {sensor_id(str): dict(width,height,fx,fy,cx,cy,k1,k2,k3,p1,p2)}
    cameras: {label: dict(sensor_id, R_local (3x3 cam->chunk-local), C_local (3,))}
              -- only cameras with a <transform> (i.e. successfully aligned)
    chunk:   dict(R (3x3), T (3,) ECEF meters, scale (float))
    """
    tree = ET.parse(path)
    root = tree.getroot()
    chunk_el = root.find("chunk")
    if chunk_el is None:
        raise RuntimeError(f"No <chunk> element in {path} -- is this really an Agisoft camera XML export?")

    sensors = {}
    for sensor_el in chunk_el.findall("./sensors/sensor"):
        sid = sensor_el.get("id")
        res = sensor_el.find("resolution")
        calib = sensor_el.find("calibration")
        if calib is None:
            continue

        def g(tag, default=0.0):
            el = calib.find(tag)
            return float(el.text) if el is not None else default

        sensors[sid] = dict(
            width=int(res.get("width")), height=int(res.get("height")),
            fx=g("fx"), fy=g("fy", g("fx")), cx=g("cx"), cy=g("cy"),
            k1=g("k1"), k2=g("k2"), k3=g("k3"), p1=g("p1"), p2=g("p2"),
        )
    print(f"Parsed {len(sensors)} sensor(s): "
          + ", ".join(f"id={sid} fx={s['fx']:.1f} {s['width']}x{s['height']}" for sid, s in sensors.items()))

    cameras = {}
    n_total, n_aligned = 0, 0
    for cam_el in chunk_el.findall("./cameras/camera"):
        n_total += 1
        label = cam_el.get("label")
        sensor_id = cam_el.get("sensor_id")
        transform_el = cam_el.find("transform")
        if transform_el is None or transform_el.text is None:
            continue  # this camera failed to align -- no pose available
        M = _parse_matrix(transform_el.text, 4)
        cameras[label] = dict(sensor_id=sensor_id, R_local=M[:3, :3], C_local=M[:3, 3])
        n_aligned += 1
    print(f"Cameras: {n_aligned}/{n_total} aligned (have a <transform>)")

    chunk_transform_el = chunk_el.find("transform")
    if chunk_transform_el is None:
        raise RuntimeError("No chunk-level <transform> found -- can't convert to a real-world CRS.")
    R_chunk = _parse_matrix(chunk_transform_el.find("rotation").text, 3)
    T_chunk = np.array([float(v) for v in chunk_transform_el.find("translation").text.split()])
    scale_chunk = float(chunk_transform_el.find("scale").text)
    chunk = dict(R=R_chunk, T=T_chunk, scale=scale_chunk)

    ref_el = chunk_el.find("reference")
    if ref_el is not None and ref_el.text:
        print(f"Chunk reference CRS (WKT, truncated): {ref_el.text[:80]}...")

    return sensors, cameras, chunk


def K_from_sensor(sensor):
    return np.array([[sensor["fx"], 0, sensor["cx"]],
                      [0, sensor["fy"], sensor["cy"]],
                      [0, 0, 1]], dtype=np.float64)


# =====================================================================
# BUILD POSES: camera-local -> chunk-local -> ECEF. Kept in ECEF (a valid
# non-degenerate 3D Cartesian frame) for all ray-casting math; only
# converted to UTM/whatever the LAS uses when actually querying the DEM.
# =====================================================================

def build_agisoft_poses(sensors, cameras, chunk):
    """Returns {label: dict(K,R,t,C)} in the SAME shape pipeline_sfm.py's
    poses use (R,t: world(ECEF)->camera; C: camera center in ECEF meters),
    so warp_thermal_via_sfm()/build_projective_warp_map() work unchanged."""
    poses = {}
    for label, cam in cameras.items():
        sensor = sensors.get(cam["sensor_id"])
        if sensor is None:
            continue
        K = K_from_sensor(sensor)
        R_c2w_ecef = chunk["R"] @ cam["R_local"]
        C_ecef = chunk["scale"] * (chunk["R"] @ cam["C_local"]) + chunk["T"]
        R = R_c2w_ecef.T
        t = -R @ C_ecef
        poses[label] = dict(K=K, R=R, t=t, C=C_ecef)
    return poses


# =====================================================================
# Lightweight sanity check: match SIFT points between a couple of nearby
# visible images, reproject with the derived poses, report RMSE. With a
# direct-matrix format there's much less room for a convention mistake
# than with OPK angles, but a row/column-major mixup could still hide here.
# =====================================================================

def _sift_match_pair(path_a, path_b, work_size=(640, 480)):
    detector = cv2.SIFT_create(nfeatures=3000)
    kps_all, descs_all = [], []
    for p in (path_a, path_b):
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(gray, work_size, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(resized)
        kps, descs = detector.detectAndCompute(clahe, None)
        kps_all.append(np.array([k.pt for k in kps], dtype=np.float32))
        descs_all.append(descs)
    if descs_all[0] is None or descs_all[1] is None:
        return None
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = matcher.knnMatch(descs_all[0], descs_all[1], k=2)
    good = [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None
    return (np.array([kps_all[0][i] for i, _ in good]),
            np.array([kps_all[1][j] for _, j in good]), work_size)


def validate_agisoft_poses(poses, visible_paths_by_label, ground_z_ecef_hint=None, n_pairs=6):
    """Reprojects real SIFT matches between nearby-index visible images
    using the derived ECEF poses; prints RMSE. Does NOT raise on a bad
    result -- prints a loud warning instead, since you should look at an
    actual warped image either way before trusting this in production."""
    labels = sorted(l for l in poses if l in visible_paths_by_label)
    if len(labels) < 2:
        print("[WARN] Not enough aligned+available images to validate poses.")
        return None

    if ground_z_ecef_hint is None:
        # rough guess: nudge the mean camera position ~150m "inward" (toward
        # Earth's center) as a crude ground-height stand-in -- this only
        # needs to be roughly right to sanity-check the ROTATION convention,
        # not to be an accurate ground height.
        C_mean = np.mean([poses[l]["C"] for l in labels], axis=0)
        up_mean = C_mean / np.linalg.norm(C_mean)
        ground_z_ecef_hint = C_mean - 150.0 * up_mean

    step = max(1, len(labels) // (n_pairs + 1))
    test_pairs = [(labels[i], labels[min(i + 1, len(labels) - 1)]) for i in range(0, len(labels) - 1, step)][:n_pairs]

    errs = []
    for a, b in test_pairs:
        if a == b:
            continue
        m = _sift_match_pair(visible_paths_by_label[a], visible_paths_by_label[b])
        if m is None:
            continue
        src, dst, work_size = m
        native_a = cv2.imread(visible_paths_by_label[a]).shape
        native_b = cv2.imread(visible_paths_by_label[b]).shape
        Wn, Hn = native_a[1], native_a[0]
        W, H = work_size
        pt_native = src * np.array([Wn / W, Hn / H])
        pix = np.hstack([pt_native, np.ones((len(pt_native), 1))]).T

        K_a, R_a, C_a = poses[a]["K"], poses[a]["R"], poses[a]["C"]
        rays_cam = np.linalg.inv(K_a) @ pix
        rays_world = R_a.T @ rays_cam  # (3, N)

        up = C_a / np.linalg.norm(C_a)
        denom = up @ rays_world                       # (N,)
        s = (up @ (ground_z_ecef_hint - C_a)) / denom  # (N,)
        pts_world = C_a[:, None] + s[None, :] * rays_world

        K_b, R_b, t_b = poses[b]["K"], poses[b]["R"], poses[b]["t"]
        pts_cam_b = R_b @ pts_world + t_b[:, None]
        pts_img_b = K_b @ pts_cam_b
        pred_native_b = (pts_img_b[:2] / pts_img_b[2]).T

        Wn_b, Hn_b = native_b[1], native_b[0]
        dst_native = dst * np.array([Wn_b / W, Hn_b / H])
        errs.append(np.linalg.norm(pred_native_b - dst_native, axis=1))

    if not errs:
        print("[WARN] Could not get any SIFT matches to validate against.")
        return None

    rmse = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    print(f"Agisoft pose validation RMSE (native px, {len(errs)} pairs): {rmse:.1f}")
    if rmse > 500:
        print("  [WARN] very high -- inspect a warped result visually before trusting this. "
              "The rough ground-height guess used here could account for SOME error, but not "
              "thousands of pixels -- that would point to an actual convention mistake.")
    else:
        print("  -> looks reasonable.")
    return rmse


def load_agisoft_poses(cfg, visible_paths_by_label):
    """Main entry point: parse the XML, build ECEF poses, sanity-check,
    return {label: pose_dict} -- ready for pipeline_sfm.warp_thermal_via_sfm
    (same {K,R,t,C} shape, ECEF meters instead of OpenSfM's local frame --
    doesn't matter, both are just self-consistent 3D Cartesian frames)."""
    ag = cfg["agisoft"]
    sensors, cameras, chunk = parse_agisoft_xml(ag["xml_path"])
    poses = build_agisoft_poses(sensors, cameras, chunk)
    validate_agisoft_poses(poses, visible_paths_by_label)
    return poses


# =====================================================================
# THERMAL POSES: Agisoft only ever processed visible, so thermal has no
# pose of its own -- derive it as (visible pose) + (rig transform),
# decomposing the EXISTING 2D self-calibrated homography now that we have
# real (not EXIF-guessed) K_visible and real ground height to work with.
# =====================================================================

def decompose_rig_from_homography(H_work, K_visible_work, K_thermal_work, ground_z_ecef, sample_visible_pose):
    """Decompose the thermal(work)->visible(work) homography into a metric
    rig rotation + translation (thermal relative to visible)."""
    H_norm = np.linalg.inv(K_visible_work) @ H_work @ K_thermal_work
    n_sol, Rs, Ts, Ns = cv2.decomposeHomographyMat(H_norm, np.eye(3))

    # disambiguate using a REAL prior this time: thermal and visible are
    # rigidly mounted side by side, pointing the same general direction --
    # the rig rotation should be small, so pick whichever solution is
    # closest to the identity rotation (much stronger than guessing from
    # the plane normal alone, since we now know these are genuinely
    # near-parallel cameras, not an arbitrary two-view pair).
    best = min(range(n_sol), key=lambda i: np.linalg.norm(Rs[i] - np.eye(3)))
    R_rel = Rs[best]
    t_rel_normalized = Ts[best].ravel()

    # scale: perpendicular distance from the sample camera to the ground,
    # approximated as the straight-line distance (fine for a near-nadir shot)
    dist = float(np.linalg.norm(sample_visible_pose["C"] - ground_z_ecef))
    t_rel = t_rel_normalized * dist
    return R_rel, t_rel


def derive_thermal_poses(visible_poses, K_thermal_native, R_rel, t_rel):
    """thermal_pose(image) = rig ∘ visible_pose(image), for every visible
    image with a known Agisoft pose. K_thermal_native: that image's thermal
    camera K in NATIVE resolution (same for every image in practice, since
    it's the same physical sensor)."""
    out = {}
    for name, pose in visible_poses.items():
        R_t = R_rel @ pose["R"]
        t_t = R_rel @ pose["t"] + t_rel
        C_t = -R_t.T @ t_t
        out[name] = dict(K=K_thermal_native, R=R_t, t=t_t, C=C_t)
    return out
