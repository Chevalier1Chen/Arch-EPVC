# -*- coding: utf-8 -*-
# Rhino PictureFrame campus base maps, horizontal-mirror corrected.
# Use this version if the campus image appears mirrored in Rhino.

import csv
import os
import rhinoscriptsyntax as rs

BASE_DIR = r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package"
IMPORT_OBJ = True
ADD_BASE_MAPS = True

# Optional: load one city block only. 0 means all cities; 1 means Jinan block.
CITY_ORDER_FILTER = 0
# Optional: load one campus only, e.g. "SD0001". Empty string means all campuses.
SAMPLE_UID_FILTER = ""

OBJ_PATH = os.path.join(BASE_DIR, "models", "combined", "shandong_all_schools_teaching_buildings_table_by_city_campus_no.obj")
MAP_CSV = os.path.join(BASE_DIR, "data", "rhino_base_picture_maps_rhino_corrected_ascii.csv")

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

base_layer = ensure_layer("DATASET_BASE_campus_picture_maps_corrected", (120, 120, 120))

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
print("Added {} corrected campus base picture maps.".format(count))
