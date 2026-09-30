# -*- coding: utf-8 -*-
# FINAL Rhino importer for all Shandong campus base maps.
# Orientation has been calibrated by SD0001: use flip_y.
# Run in a new Rhino file with RunPythonScript.

import csv
import os
import rhinoscriptsyntax as rs

BASE_DIR = r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package"

# Set IMPORT_OBJ = False only if the table-layout OBJ is already imported.
IMPORT_OBJ = True
ADD_BASE_MAPS = True

# 0 means all city blocks. Use 1 for Jinan only, 2 for Qingdao only, etc.
CITY_ORDER_FILTER = 0

# Empty string means all campuses. Use "SD0001" for one-campus testing.
SAMPLE_UID_FILTER = ""

OBJ_PATH = os.path.join(
    BASE_DIR,
    "models",
    "combined",
    "shandong_all_schools_teaching_buildings_table_by_city_campus_no.obj",
)
MAP_CSV = os.path.join(BASE_DIR, "data", "rhino_base_picture_maps_flip_y_ascii.csv")


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


base_layer = ensure_layer("DATASET_BASE_campus_picture_maps_FINAL_flip_y", (120, 120, 120))

rs.EnableRedraw(False)

if IMPORT_OBJ and os.path.exists(OBJ_PATH):
    rs.Command('_-Import "{}" _Enter'.format(OBJ_PATH), False)

count = 0
if ADD_BASE_MAPS:
    with open(MAP_CSV, "r") as handle:
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
print("FINAL flip_y: added {} campus base maps.".format(count))
