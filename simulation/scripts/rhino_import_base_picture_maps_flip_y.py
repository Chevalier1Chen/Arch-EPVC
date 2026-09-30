# -*- coding: utf-8 -*-
# Rhino PictureFrame orientation tester.
# Default loads only SD0001. Change SAMPLE_UID_FILTER to "" after choosing the right orientation.

import csv
import os
import rhinoscriptsyntax as rs

BASE_DIR = r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package"
IMPORT_OBJ = True
ADD_BASE_MAPS = True

# Try: "normal", "flip_x", "flip_y", or "rot180".
ORIENTATION = "flip_y"

# Keep SD0001 for quick testing. Set to "" to load all campuses.
SAMPLE_UID_FILTER = ""

# Optional city block filter. 0 means all.
CITY_ORDER_FILTER = 0

OBJ_PATH = os.path.join(BASE_DIR, "models", "combined", "shandong_all_schools_teaching_buildings_table_by_city_campus_no.obj")
CSV_BY_ORIENTATION = {
    "normal": "data/rhino_base_picture_maps_ascii.csv",
    "flip_x": "data/rhino_base_picture_maps_rhino_corrected_ascii.csv",
    "flip_y": "data/rhino_base_picture_maps_flip_y_ascii.csv",
    "rot180": "data/rhino_base_picture_maps_rot180_ascii.csv",
}

def ensure_layer(name, color):
    if not rs.IsLayer(name):
        rs.AddLayer(name, color)
    else:
        rs.LayerColor(name, color)
    return name

def add_picture(row, layer):
    image_path = os.path.join(BASE_DIR, row["image_path"])
    if not os.path.exists(image_path):
        print("Missing image:", image_path)
        return None
    x = float(row["cell_min_x"])
    y = float(row["cell_max_y"])
    z = float(row["z"])
    w = float(row["width_m"])
    h = float(row["height_m"])
    plane = rs.PlaneFromFrame((x, y, z), (1, 0, 0), (0, -1, 0))
    try:
        obj = rs.AddPictureFrame(plane, image_path, w, h, True, False, False, True)
    except TypeError:
        obj = rs.AddPictureFrame(plane, image_path, w, h, True, False, False)
    if obj:
        rs.ObjectLayer(obj, layer)
    return obj

base_layer = ensure_layer("DATASET_BASE_picture_" + ORIENTATION, (120, 120, 120))

rs.EnableRedraw(False)
if IMPORT_OBJ and os.path.exists(OBJ_PATH):
    rs.Command('_-Import "{}" _Enter'.format(OBJ_PATH), False)

csv_rel = CSV_BY_ORIENTATION.get(ORIENTATION)
if not csv_rel:
    raise Exception("Unknown ORIENTATION: " + ORIENTATION)
map_csv = os.path.join(BASE_DIR, csv_rel)

count = 0
if ADD_BASE_MAPS:
    with open(map_csv, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if CITY_ORDER_FILTER and int(row["city_order"]) != CITY_ORDER_FILTER:
                continue
            if SAMPLE_UID_FILTER and row["sample_uid"] != SAMPLE_UID_FILTER:
                continue
            if add_picture(row, base_layer):
                count += 1
rs.EnableRedraw(True)
rs.ZoomExtents()
print("Orientation {}: added {} campus base maps.".format(ORIENTATION, count))
