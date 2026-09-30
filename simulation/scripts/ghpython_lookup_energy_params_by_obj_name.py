# GhPython helper: lookup energy/PV parameters by OBJ object name.
#
# Inputs:
#   csv_path: path to data/gh_obj_object_to_building_map.csv
#   obj_names: one object name string or a list of object name strings
#
# Outputs:
#   rows: matched rows as dictionaries
#   teaching_ids: matched teaching_building_id values
#   floors: estimated_floor_count values
#   years: construction_year_numeric values
#   roof_u: roof_u_w_m2k values
#   wall_u: wall_u_w_m2k values
#   window_u: window_u_w_m2k values
#   shgc: window_shgc values
#   roof_pv_area: roof_pv_area_60pct_m2 values
#   facade_pv_area: south_facade_pv_area_65pct_m2 values
#   status: envelope_param_status values

import csv


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def read_table(path):
    with open(path, "r") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_index(table):
    exact = {}
    prefix = {}
    for row in table:
        exact[row.get("obj_object_name", "")] = row
        prefix[row.get("obj_object_name_prefix", "")] = row
    return exact, prefix


def lookup_one(name, exact, prefix):
    if name in exact:
        return exact[name]
    # Segment names may look like SDxxxx_T001_1hao_seg02.
    for key, row in prefix.items():
        if key and (name == key or name.startswith(key + "_seg")):
            return row
    return None


table = read_table(csv_path)
exact_index, prefix_index = build_index(table)

rows = []
for name in as_list(obj_names):
    row = lookup_one(str(name), exact_index, prefix_index)
    if row:
        rows.append(row)

teaching_ids = [r.get("teaching_building_id", "") for r in rows]
floors = [r.get("estimated_floor_count", "") for r in rows]
years = [r.get("construction_year_numeric", "") for r in rows]
roof_u = [r.get("roof_u_w_m2k", "") for r in rows]
wall_u = [r.get("wall_u_w_m2k", "") for r in rows]
window_u = [r.get("window_u_w_m2k", "") for r in rows]
shgc = [r.get("window_shgc", "") for r in rows]
roof_pv_area = [r.get("roof_pv_area_60pct_m2", "") for r in rows]
facade_pv_area = [r.get("south_facade_pv_area_65pct_m2", "") for r in rows]
status = [r.get("envelope_param_status", "") for r in rows]
