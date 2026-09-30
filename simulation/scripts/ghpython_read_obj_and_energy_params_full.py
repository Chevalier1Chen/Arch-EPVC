# -*- coding: utf-8 -*-
# GhPython full helper: read one campus OBJ and join energy/PV parameters.
#
# Recommended GhPython inputs:
#   base_dir   (str)  Package root. If empty, default path below is used.
#   sample_uid (str)  Example: "SD0010". Used when obj_path is empty.
#   obj_path   (str)  Optional absolute OBJ path. Leave empty to use sample_uid.
#   csv_path   (str)  Optional absolute mapping CSV path. Leave empty to use default.
#   only_teaching (bool) True = skip campus boundary and return teaching buildings only.
#
# Recommended outputs:
#   M               Mesh list
#   N               OBJ object names
#   P               parameter dictionaries matched from CSV
#   teaching_ids    teaching_building_id list
#   floors          estimated_floor_count list
#   years           construction_year_numeric list
#   roof_u          roof_u_w_m2k list
#   wall_u          wall_u_w_m2k list
#   ground_u        ground_u_w_m2k list
#   window_u        window_u_w_m2k list
#   shgc            window_shgc list
#   roof_pv_area    roof_pv_area_60pct_m2 list
#   facade_pv_area  south_facade_pv_area_65pct_m2 list
#   status          envelope_param_status list
#   log             diagnostic text

import os
import csv
import Rhino.Geometry as rg


DEFAULT_BASE_DIR = r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package"


def _as_text(value):
    if value is None:
        return ""
    return str(value)


def _norm_path(path):
    return os.path.normpath(_as_text(path).strip().strip('"'))


def _default_if_empty(value, default_value):
    value = _as_text(value).strip()
    return value if value else default_value


def _read_csv_dicts(path):
    rows = []
    # utf-8-sig removes BOM from the first header in Rhino 8 Python.
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _build_param_indexes(rows):
    by_object_name = {}
    by_prefix = {}
    for row in rows:
        obj_name = row.get("obj_object_name", "")
        prefix = row.get("obj_object_name_prefix", "")
        if obj_name:
            by_object_name[obj_name] = row
        if prefix:
            by_prefix[prefix] = row
    return by_object_name, by_prefix


def _lookup_params(name, by_object_name, by_prefix):
    if name in by_object_name:
        return by_object_name[name]
    # Multi-part merged buildings may be named like *_seg02.
    for prefix, row in by_prefix.items():
        if name == prefix or name.startswith(prefix + "_seg"):
            return row
    return {}


def _parse_obj(path):
    vertices = []
    objects = []
    current_name = None
    current_faces = []

    def save_current():
        if current_name and current_faces:
            objects.append((current_name, list(current_faces)))

    with open(path, "rb") as handle:
        for raw_bytes in handle:
            raw = raw_bytes.decode("utf-8", "ignore")
            line = raw.strip()
            if not line:
                continue

            if line.startswith("o "):
                save_current()
                current_name = line[2:].strip()
                current_faces = []

            elif line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append(
                        rg.Point3d(
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        )
                    )

            elif line.startswith("f "):
                face = []
                for token in line.split()[1:]:
                    idx_text = token.split("/")[0]
                    if not idx_text:
                        continue
                    idx = int(idx_text)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    face.append(idx)
                if len(face) >= 3:
                    current_faces.append(face)

    save_current()
    return vertices, objects


def _object_to_mesh(name, faces, vertices):
    local_index = {}
    mesh = rg.Mesh()

    def get_local(global_index):
        if global_index not in local_index:
            local_index[global_index] = mesh.Vertices.Add(vertices[global_index])
        return local_index[global_index]

    for face in faces:
        ids = [get_local(i) for i in face if 0 <= i < len(vertices)]
        if len(ids) == 3:
            mesh.Faces.AddFace(ids[0], ids[1], ids[2])
        elif len(ids) == 4:
            mesh.Faces.AddFace(ids[0], ids[1], ids[2], ids[3])
        elif len(ids) > 4:
            # Fan triangulation for roof/base polygons with more than 4 vertices.
            for i in range(1, len(ids) - 1):
                mesh.Faces.AddFace(ids[0], ids[i], ids[i + 1])

    mesh.Normals.ComputeNormals()
    mesh.Compact()
    return mesh


def _get(row, key):
    return row.get(key, "") if row else ""


# ------------------------------
# Main GhPython execution
# ------------------------------

messages = []

base_dir = _norm_path(_default_if_empty(globals().get("base_dir", ""), DEFAULT_BASE_DIR))
sample_uid = _default_if_empty(globals().get("sample_uid", ""), "SD0010")

obj_path_in = _as_text(globals().get("obj_path", "")).strip()
if obj_path_in:
    obj_file = _norm_path(obj_path_in)
else:
    obj_file = os.path.join(base_dir, "models", "per_campus_obj", sample_uid + ".obj")

csv_path_in = _as_text(globals().get("csv_path", "")).strip()
if csv_path_in:
    map_csv = _norm_path(csv_path_in)
else:
    map_csv = os.path.join(base_dir, "data", "gh_obj_object_to_building_map.csv")

only_teaching_value = globals().get("only_teaching", True)
only_teaching = bool(only_teaching_value)

M = []
N = []
P = []

if not os.path.exists(obj_file):
    messages.append("OBJ not found: " + obj_file)
elif not os.path.exists(map_csv):
    messages.append("CSV not found: " + map_csv)
else:
    param_rows = _read_csv_dicts(map_csv)
    by_object_name, by_prefix = _build_param_indexes(param_rows)
    vertices, objects = _parse_obj(obj_file)
    messages.append("OBJ: " + obj_file)
    messages.append("CSV: " + map_csv)
    messages.append("OBJ objects parsed: " + str(len(objects)))

    for name, faces in objects:
        if only_teaching and "_T" not in name:
            continue
        mesh = _object_to_mesh(name, faces, vertices)
        if mesh.Faces.Count == 0:
            continue
        row = _lookup_params(name, by_object_name, by_prefix)
        M.append(mesh)
        N.append(name)
        P.append(row)

    messages.append("Meshes returned: " + str(len(M)))
    missing = [name for name, row in zip(N, P) if not row]
    if missing:
        messages.append("Missing parameter rows: " + str(len(missing)))
        messages.append("First missing: " + ", ".join(missing[:5]))
    else:
        messages.append("All returned meshes matched CSV parameters.")


teaching_ids = [_get(row, "teaching_building_id") for row in P]
floors = [_get(row, "estimated_floor_count") for row in P]
years = [_get(row, "construction_year_numeric") for row in P]
roof_u = [_get(row, "roof_u_w_m2k") for row in P]
wall_u = [_get(row, "wall_u_w_m2k") for row in P]
ground_u = [_get(row, "ground_u_w_m2k") for row in P]
window_u = [_get(row, "window_u_w_m2k") for row in P]
shgc = [_get(row, "window_shgc") for row in P]
roof_pv_area = [_get(row, "roof_pv_area_60pct_m2") for row in P]
facade_pv_area = [_get(row, "south_facade_pv_area_65pct_m2") for row in P]
status = [_get(row, "envelope_param_status") for row in P]

log = "\n".join(messages)
