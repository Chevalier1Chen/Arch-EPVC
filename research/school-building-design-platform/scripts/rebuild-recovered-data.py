"""Rebuild browser data from the preserved research package after source recovery.

The original trained first-route checkpoints are stored only in the last online
site archive. Until those binaries are downloaded, the preserved measured /
simulated targets are retained as the annual reference so every research
building remains selectable and produces building-specific outputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source-data"
PUBLIC_DATA = ROOT / "public" / "data"


def value(row: pd.Series, key: str, default=None):
    item = row.get(key, default)
    return default if pd.isna(item) else item


def number(row: pd.Series, key: str, default=None):
    item = value(row, key, default)
    if item is None or item == "":
        return default
    try:
        return float(item)
    except (TypeError, ValueError):
        return default


def integer(row: pd.Series, key: str, default=None):
    item = number(row, key, default)
    return default if item is None else int(round(item))


def main() -> None:
    buildings_source = pd.read_csv(SOURCE / "gh_building_energy_inputs.csv")
    campuses_source = pd.read_csv(SOURCE / "schools_teaching_campus.csv")
    targets_source = pd.read_excel(
        SOURCE / "numerical data.xlsx", sheet_name="selected_variables"
    )

    campuses = campuses_source.set_index("sample_uid", drop=False)
    targets = targets_source.set_index("Number", drop=False)
    buildings: list[dict] = []

    for _, row in buildings_source.iterrows():
        campus = campuses.loc[row["sample_uid"]]
        building_id = str(row["obj_object_name_prefix"])
        target = targets.loc[building_id] if building_id in targets.index else None

        prediction = None
        if target is not None:
            eui = number(target, "EUI", 0.0)
            epv = number(target, "Epv", 0.0)
            cei = number(target, "CEI-no PV system", 0.0)
            prediction = {
                "trueEui": eui,
                "annualEui": eui,
                "trueEpv": epv,
                "annualEpv": epv,
                "trueCei": cei,
                "annualCei": cei,
                "firstEui": eui,
                "firstEpv": epv,
                "firstCei": cei,
                "source": "recovered-dataset-target-pending-online-model-binaries",
            }

        roof_u = number(row, "roof_u_w_m2k")
        wall_u = number(row, "wall_u_w_m2k")
        ground_u = number(row, "ground_u_w_m2k")
        window_u = number(row, "window_u_w_m2k")
        if target is not None:
            roof_u = roof_u or number(target, "K.Roof thermal coefficient")
            wall_u = wall_u or number(target, "L.Wall thermal coefficient")
            ground_u = ground_u or number(target, "M.Ground thermal coefficient")
            window_u = window_u or number(target, "N.Window U-value")

        obj_path = str(value(row, "obj_path", "")).replace("\\", "/")
        if obj_path.startswith("models/"):
            obj_path = "/" + obj_path

        buildings.append(
            {
                "id": building_id,
                "teachingId": str(row["teaching_building_id"]),
                "campusId": str(row["sample_uid"]),
                "campusBuildingNo": integer(row, "campus_building_no", 1),
                "sourceBuildingIds": value(row, "source_building_ids"),
                "sourceBuildingCount": integer(row, "source_building_count", 1),
                "city": str(row["city"]),
                "school": str(row["school_name"]),
                "lat": number(campus, "lat"),
                "lon": number(campus, "lon"),
                "obj": obj_path,
                "length": number(row, "length_m", 0.0),
                "width": number(row, "width_m", 0.0),
                "height": number(row, "height_m", 0.0),
                "floors": integer(row, "estimated_floor_count", 1),
                "year": integer(row, "construction_year_numeric"),
                "constructionYearRaw": value(row, "construction_year_raw"),
                "shapeFactor": number(row, "shape_factor", 0.0),
                "orientation": number(row, "orientation_deg", 0.0),
                "footprintArea": number(row, "footprint_area_m2", 0.0),
                "buildingArea": number(row, "building_area_m2", 0.0),
                "roofArea": number(row, "roof_area_m2", 0.0),
                "southFacadeArea": number(row, "south_facade_area_net_m2", 0.0),
                "wwr": number(row, "wwr_assumption", 0.3),
                "roofPvArea": number(row, "roof_pv_area_60pct_m2", 0.0),
                "facadePvArea": number(row, "south_facade_pv_area_65pct_m2", 0.0),
                "enclosureType": value(row, "A8_enclosure_type"),
                "confidence": value(row, "confidence"),
                "filterReason": value(row, "filter_reason"),
                "schoolType": value(campus, "A2.学校类型"),
                "averageFloorHeight": number(campus, "A6.建筑层高"),
                "buildingSpacing": number(campus, "A12.教学楼间距"),
                "campusArea": number(campus, "学校场地面积"),
                "campusDensity": number(campus, "A14.建筑密度"),
                "campusFar": number(campus, "容积率"),
                "envelopeBucket": value(row, "envelope_standard_bucket"),
                "shapeBin": value(row, "shape_factor_bin"),
                "thermal": {
                    "roofU": roof_u,
                    "wallU": wall_u,
                    "groundU": ground_u,
                    "windowU": window_u,
                    "shgc": number(row, "window_shgc"),
                    "status": str(
                        value(row, "envelope_param_status", "recovered_from_numerical_data")
                    ),
                },
                "prediction": prediction,
            }
        )

    # A stable 8,760-point shape used only when the archived browser LSTM is
    # unavailable. DesignPlatform rescales it to each building's annual values.
    hourly = []
    for index in range(8760):
        day = index // 24 + 1
        hour = index % 24
        occupied = 1.0 if 7 <= hour <= 18 else 0.28
        weekday = 1.0 if (day - 1) % 7 < 5 else 0.52
        seasonal = 0.82 + 0.30 * abs(math.sin(2 * math.pi * (day - 16) / 365))
        daylight = max(0.0, math.sin(math.pi * (hour - 6) / 12))
        energy = 0.22 + occupied * weekday * seasonal
        pv = daylight * (0.78 + 0.22 * math.sin(2 * math.pi * (day - 80) / 365))
        hourly.append(
            {"hour": index + 1, "energy": energy, "pv": max(pv, 0.0), "carbon": energy}
        )

    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DATA / "buildings.json").write_text(
        json.dumps(buildings, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (PUBLIC_DATA / "demo-hourly.json").write_text(
        json.dumps(hourly, separators=(",", ":")), encoding="utf-8"
    )
    summary = {
        "buildings": len(buildings),
        "campuses": len({b["campusId"] for b in buildings}),
        "cities": len({b["city"] for b in buildings}),
        "annualReferences": sum(b["prediction"] is not None for b in buildings),
        "coordinateLevel": "campus",
        "recoveryStatus": "online-model-binaries-pending",
    }
    (PUBLIC_DATA / "dataset-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
