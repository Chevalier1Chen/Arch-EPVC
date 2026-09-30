# -*- coding: utf-8 -*-
# Rhino fast label loader for Shandong school dataset.
# Run this inside Rhino: Tools > PythonScript > Run, then choose this file.
# It imports the lightweight no-texture OBJ and adds fast native TextDot labels.

import csv
import codecs
import os
import rhinoscriptsyntax as rs

BASE_DIR = r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package"
IMPORT_OBJ = True
ADD_CITY_LABELS = True
ADD_SCHOOL_LABELS = True
ADD_BUILDING_LABELS = True

# Optional: set to a city name such as u"济南市" to add labels for one city only.
CITY_FILTER = u""

OBJ_PATH = os.path.join(BASE_DIR, "models", "combined", "shandong_all_schools_teaching_buildings_table_by_city_campus_no.obj")
LABEL_CSV = os.path.join(BASE_DIR, "data", "rhino_fast_textdot_labels.csv")

def ensure_layer(name, color):
    if not rs.IsLayer(name):
        rs.AddLayer(name, color)
    else:
        rs.LayerColor(name, color)
    return name

def add_dot(text, x, y, z, layer, color):
    dot = rs.AddTextDot(text, (float(x), float(y), float(z)))
    if dot:
        rs.ObjectLayer(dot, layer)
        rs.ObjectColor(dot, color)
    return dot

city_layer = ensure_layer("DATASET_LABELS_city", (180, 40, 40))
school_layer = ensure_layer("DATASET_LABELS_school", (20, 70, 140))
building_layer = ensure_layer("DATASET_LABELS_building_no", (220, 100, 20))

rs.EnableRedraw(False)

if IMPORT_OBJ and os.path.exists(OBJ_PATH):
    rs.Command('_-Import "{}" _Enter'.format(OBJ_PATH), False)

count = 0
with codecs.open(LABEL_CSV, "r", "utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if CITY_FILTER and row["city"] != CITY_FILTER:
            continue
        label_type = row["label_type"]
        if label_type == "city" and ADD_CITY_LABELS:
            add_dot(row["label_text"], row["x"], row["y"], row["z"], city_layer, (180, 40, 40))
            count += 1
        elif label_type == "school" and ADD_SCHOOL_LABELS:
            add_dot(row["label_text"], row["x"], row["y"], row["z"], school_layer, (20, 70, 140))
            count += 1
        elif label_type == "building" and ADD_BUILDING_LABELS:
            add_dot(row["label_text"], row["x"], row["y"], row["z"], building_layer, (220, 100, 20))
            count += 1

rs.EnableRedraw(True)
rs.ZoomExtents()
print("Added {} fast TextDot labels.".format(count))
