"""
STEP 1 -- ORGANIZE

Scans `source_dir` (recursively) for image files, classifies each as
THERMAL or VISIBLE by RESOLUTION (thermal sensor is always much smaller
than visible, regardless of filename), then pairs them by the shared ID
embedded in the filename -- the LAST run of digits in the filename stem
(e.g. "..._0008_V.JPG" / "..._0008_T.JPG" -> id "8"), NOT by timestamp.
Pairs that don't share an exact id (imperfect naming) get a second pass:
matched to whichever leftover file shares the longest common filename
PREFIX. Matched pairs are copied into visible_dir/thermal_dir renamed as
DJI_<idx>_V<ext> / DJI_<idx>_T<ext> -- this exact naming is also what
pipeline_sfm.py's rig_pattern regex expects to distinguish the two cameras.
"""

import os
import re
import csv
import shutil

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def scan_images(source_dir):
    files = []
    skip_names = {"visible", "thermal", "register_results", "sfm_rig"}
    for root, dirs, fnames in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d not in skip_names]
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, fn))
    return files


def get_pixel_count(path):
    with Image.open(path) as im:
        w, h = im.size
    return w * h


def auto_threshold(pixel_counts):
    uniq = sorted(set(pixel_counts))
    if len(uniq) < 2:
        return None
    best_ratio, best_i = 0, 0
    for i in range(len(uniq) - 1):
        ratio = uniq[i + 1] / uniq[i]
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
    return (uniq[best_i] * uniq[best_i + 1]) ** 0.5


def extract_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = re.findall(r"\d+", stem)
    if digits:
        try:
            return str(int(digits[-1]))
        except ValueError:
            pass
    return stem


def _common_prefix_len(a, b):
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def prefix_fallback_pair(thermal_files, visible_files, min_len):
    remaining_v = list(visible_files)
    used_v = set()
    pairs = []
    for t in thermal_files:
        t_stem = os.path.splitext(os.path.basename(t))[0]
        best_v, best_len = None, 0
        for v in remaining_v:
            if v in used_v:
                continue
            v_stem = os.path.splitext(os.path.basename(v))[0]
            L = _common_prefix_len(t_stem, v_stem)
            if L > best_len:
                best_len, best_v = L, v
        if best_v is not None and best_len >= min_len:
            pairs.append((best_v, t))
            used_v.add(best_v)
    matched_t = {t for _, t in pairs}
    unmatched_t = [t for t in thermal_files if t not in matched_t]
    unmatched_v = [v for v in visible_files if v not in used_v]
    return pairs, unmatched_t, unmatched_v


def classify_and_pair(cfg):
    files = scan_images(cfg["source_dir"])
    if not files:
        raise RuntimeError(f"No image files found under {cfg['source_dir']}")

    pixel_counts = {}
    for f in files:
        try:
            pixel_counts[f] = get_pixel_count(f)
        except Exception as e:
            print(f"[WARN] could not read size of {f}: {e}")

    threshold = cfg["resolution_threshold_px"] or auto_threshold(list(pixel_counts.values()))
    if threshold is None:
        raise RuntimeError("Could not auto-detect a resolution split -- set resolution_threshold_px manually.")
    print(f"Resolution split threshold: {threshold:,.0f} px")

    res_summary = {}
    for f in pixel_counts:
        with Image.open(f) as im:
            wh = im.size
        res_summary[wh] = res_summary.get(wh, 0) + 1
    print("Resolutions found:", {f"{w}x{h}": n for (w, h), n in sorted(res_summary.items())})

    thermal_files = [f for f, px in pixel_counts.items() if px <= threshold]
    visible_files = [f for f, px in pixel_counts.items() if px > threshold]
    print(f"Classified: {len(thermal_files)} thermal, {len(visible_files)} visible")

    thermal_by_id, visible_by_id = {}, {}
    for f in thermal_files:
        thermal_by_id.setdefault(extract_id(f), []).append(f)
    for f in visible_files:
        visible_by_id.setdefault(extract_id(f), []).append(f)

    exact_pairs = []
    used_t, used_v = set(), set()
    for id_key in sorted(set(thermal_by_id) & set(visible_by_id), key=int):
        t_path, v_path = thermal_by_id[id_key][0], visible_by_id[id_key][0]
        exact_pairs.append((v_path, t_path))
        used_t.add(t_path)
        used_v.add(v_path)

    leftover_t = [f for f in thermal_files if f not in used_t]
    leftover_v = [f for f in visible_files if f not in used_v]
    fallback_pairs, unmatched_t, unmatched_v = prefix_fallback_pair(
        leftover_t, leftover_v, cfg["min_prefix_fallback_len"]
    )
    print(f"Paired by exact id: {len(exact_pairs)} | by prefix fallback: {len(fallback_pairs)} "
          f"| unmatched thermal: {len(unmatched_t)} | unmatched visible: {len(unmatched_v)}")
    return exact_pairs + fallback_pairs, unmatched_t, unmatched_v


def _already_organized_indices(visible_dir):
    done = set()
    if os.path.isdir(visible_dir):
        for fn in os.listdir(visible_dir):
            m = re.match(r"DJI_(\d+)_V\.", fn, re.IGNORECASE)
            if m:
                done.add(int(m.group(1)))
    return done


def reset_folders(cfg):
    if cfg["copy_mode"] == "move":
        raise RuntimeError(
            "fresh_start=True with copy_mode='move' would permanently delete files that were "
            "moved out of source_dir with no other copy remaining -- refusing."
        )
    targets = [cfg["visible_dir"], cfg["thermal_dir"], cfg["register_results_dir"]]
    print("[fresh_start] Wiping and recreating:")
    for d in targets:
        if os.path.isdir(d):
            print(f"  deleting {d}")
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("[fresh_start] Done.\n")


def organize(cfg):
    if cfg.get("fresh_start"):
        reset_folders(cfg)

    pairs, unmatched_thermal, unmatched_visible = classify_and_pair(cfg)
    total_found = len(pairs)

    already_done = _already_organized_indices(cfg["visible_dir"])
    limit = cfg.get("organize_max_per_run")
    to_organize = []
    for i, (v_path, t_path) in enumerate(pairs, start=1):
        if i in already_done:
            continue
        to_organize.append((i, v_path, t_path))
        if limit is not None and len(to_organize) >= limit:
            break

    n_already = len(already_done)
    n_remaining_after = total_found - n_already - len(to_organize)
    print(f"{total_found} pairs found total | {n_already} already organized | "
          f"organizing {len(to_organize)} more this run"
          + (f" | {n_remaining_after} will remain for next run" if n_remaining_after > 0 else ""))

    if cfg["dry_run"]:
        print("\n[DRY RUN] Would organize (showing up to 10):")
        for idx, v, t in to_organize[:10]:
            print(f"  {idx:04d}: {os.path.basename(v)}  <->  {os.path.basename(t)}")
        print("Set dry_run=False to actually copy/rename.")
        return

    if not to_organize:
        print("Nothing new to organize this run.")
        return

    os.makedirs(cfg["visible_dir"], exist_ok=True)
    os.makedirs(cfg["thermal_dir"], exist_ok=True)
    os.makedirs(cfg["register_results_dir"], exist_ok=True)

    op = shutil.move if cfg["copy_mode"] == "move" else shutil.copy2

    mapping_path = os.path.join(cfg["register_results_dir"], "file_mapping.csv")
    existing_mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, newline="") as f:
            for r in csv.DictReader(f):
                existing_mapping[r["index"]] = r

    for i, v_path, t_path in to_organize:
        idx = f"{i:04d}"
        t_ext, v_ext = os.path.splitext(t_path)[1], os.path.splitext(v_path)[1]
        new_t = os.path.join(cfg["thermal_dir"], f"DJI_{idx}_T{t_ext}")
        new_v = os.path.join(cfg["visible_dir"], f"DJI_{idx}_V{v_ext}")
        op(t_path, new_t)
        op(v_path, new_v)
        existing_mapping[idx] = {
            "index": idx, "original_visible": v_path, "original_thermal": t_path,
            "new_visible": new_v, "new_thermal": new_t,
        }

    if existing_mapping:
        with open(mapping_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "original_visible", "original_thermal", "new_visible", "new_thermal"])
            writer.writeheader()
            for k in sorted(existing_mapping, key=int):
                writer.writerow(existing_mapping[k])
        print(f"Wrote mapping: {mapping_path}")

    if unmatched_thermal or unmatched_visible:
        unmatched_path = os.path.join(cfg["register_results_dir"], "unmatched_files.txt")
        with open(unmatched_path, "w") as f:
            f.write("Unmatched thermal files:\n")
            f.writelines(f"  {p}\n" for p in unmatched_thermal)
            f.write("\nUnmatched visible files:\n")
            f.writelines(f"  {p}\n" for p in unmatched_visible)
        print(f"[WARN] {len(unmatched_thermal) + len(unmatched_visible)} unmatched file(s) -> see {unmatched_path}")

    print(f"Organized {len(to_organize)} pairs this run into:\n  {cfg['visible_dir']}\n  {cfg['thermal_dir']}")


# =====================================================================
# Shared by every other module: find already-organized DJI_<idx>_V/T pairs
# =====================================================================

_PAIR_RE_TEMPLATE = r"_(\d+)_{tag}\.(jpg|jpeg|png|tif|tiff)$"


def _index_files(folder, tag):
    pattern = re.compile(_PAIR_RE_TEMPLATE.format(tag=tag), re.IGNORECASE)
    out = {}
    for fname in os.listdir(folder):
        m = pattern.search(fname)
        if m:
            out[m.group(1)] = os.path.join(folder, fname)
    return out


def find_pairs(visible_dir, thermal_dir):
    vis = _index_files(visible_dir, "V")
    th = _index_files(thermal_dir, "T")
    common = sorted(set(vis) & set(th), key=lambda s: int(s))
    return [(idx, vis[idx], th[idx]) for idx in common]


def load_original_name_to_organized_path(cfg):
    """{original_visible_basename: organized_visible_path} -- bridges
    Agisoft XML camera labels (which are the ORIGINAL DJI filenames, e.g.
    'DJI_20260409091358_0001_V.JPG') to the renamed files organize()
    actually produced ('DJI_0001_V.JPG'), via file_mapping.csv."""
    import csv
    mapping_path = os.path.join(cfg["register_results_dir"], "file_mapping.csv")
    if not os.path.exists(mapping_path):
        raise RuntimeError(f"{mapping_path} not found -- run organize() first.")
    out = {}
    with open(mapping_path, newline="") as f:
        for row in csv.DictReader(f):
            out[os.path.basename(row["original_visible"])] = row["new_visible"]
    return out


if __name__ == "__main__":
    from pipeline_config import CONFIG
    organize(CONFIG)