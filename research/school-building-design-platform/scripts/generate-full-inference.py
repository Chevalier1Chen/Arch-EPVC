"""Generate full-building annual predictions and second-route static features.

This is an offline preparation step.  It uses the archived trained checkpoints;
the browser never invents fallback performance values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "model-service" / "platform_model_package"
MODELS = PACKAGE / "models"
CODE = PACKAGE / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PACKAGE))

import reduced_data_multimodal_vs_tabular as networks  # noqa: E402
from platform_inference import HourlyMultimodalModel  # noqa: E402


def thermal_prior(year: float | None, shape: float) -> tuple[float, float, float, float]:
    if year is None or not np.isfinite(year):
        return np.nan, np.nan, np.nan, np.nan
    base = (1, 1.5, .65, 5.7) if year < 1986 else ((.7, 1, .55, 3.5) if year < 2006 else ((.45, .55, .4, 2.5) if year < 2016 else (.3, .4, .3, 1.8)))
    factor = .92 if shape > .35 else (.96 if shape > .25 else 1)
    return base[0] * factor, base[1] * factor, base[2], base[3] * factor


def enclosure_group(value: object) -> int:
    text = str(value)
    if "四面" in text:
        return 4
    if "三面" in text or "S形" in text or "E形" in text:
        return 3
    if "双面" in text or "双向" in text or "L形" in text:
        return 2
    return 1


def feature_frame(buildings: list[dict], source: dict[str, dict]) -> pd.DataFrame:
    rows = []
    cities = ["东营市", "临沂市", "威海市", "德州市", "日照市", "枣庄市", "泰安市", "济南市", "济宁市", "淄博市", "滨州市", "潍坊市", "烟台市", "聊城市", "菏泽市", "青岛市"]
    for b in buildings:
        s = source.get(b["id"], {})
        area = max(float(s.get("A.Building area", b.get("buildingArea")) or 0), 1e-6)
        footprint = max(float(s.get("B.Building footprint", b.get("footprintArea")) or 0), 1e-6)
        pv_area = max(float(s.get("屋顶光伏发电总面积", b.get("roofPvArea")) or 0), 0)
        orientation = float(s.get("H.Orientation", b.get("orientation")) or 0)
        year = s.get("construction_year_user", b.get("year"))
        year = float(year) if year is not None else np.nan
        thermal = b.get("thermal") or {}
        roof_u, wall_u, ground_u, window_u = thermal_prior(year if np.isfinite(year) else None, float(b.get("shapeFactor") or 0))
        row = {
            "City": s.get("City", b.get("city")),
            "A.Building area": area,
            "B.Building footprint": footprint,
            "C.Building height": s.get("C.Building height", b.get("height")),
            "D.Layer": s.get("D.Layer", b.get("floors")),
            "E.Height": s.get("E.Height", float(b.get("height") or 0) / max(float(b.get("floors") or 1), 1)),
            "F.Building length": s.get("F.Building length", b.get("length")),
            "G.Building width": s.get("G.Building width", b.get("width")),
            "J.Shape coefficient": s.get("J.Shape coefficient", b.get("shapeFactor")),
            "K.Roof thermal coefficient": s.get("K.Roof thermal coefficient", thermal.get("roofU") if thermal.get("roofU") is not None else roof_u),
            "L.Wall thermal coefficient": s.get("L.Wall thermal coefficient", thermal.get("wallU") if thermal.get("wallU") is not None else wall_u),
            "M.Ground thermal coefficient": s.get("M.Ground thermal coefficient", thermal.get("groundU") if thermal.get("groundU") is not None else ground_u),
            "N.Window U-value": s.get("N.Window U-value", thermal.get("windowU") if thermal.get("windowU") is not None else window_u),
            "construction_year_user": year,
            "屋顶光伏发电总面积": pv_area,
            "enclosure_group": enclosure_group(s.get("I.Enclosure method", b.get("enclosureType"))),
            "orientation_sin": np.sin(np.deg2rad(orientation)),
            "orientation_cos": np.cos(np.deg2rad(orientation)),
            "construction_age_2026": 2026 - year,
            "pv_to_floor_area": pv_area / area,
            "footprint_to_floor_area": footprint / area,
            "log1p__A.Building area": np.log1p(area),
            "log1p__B.Building footprint": np.log1p(footprint),
            "log1p__屋顶光伏发电总面积": np.log1p(pv_area),
            "log1p__F.Building length": np.log1p(max(float(s.get("F.Building length", b.get("length")) or 0), 0)),
            "log1p__G.Building width": np.log1p(max(float(s.get("G.Building width", b.get("width")) or 0), 0)),
        }
        for city in cities:
            row[f"pv_area_x_city__{city}"] = pv_area if row["City"] == city else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def main() -> None:
    buildings_path = ROOT / "public" / "data" / "buildings.json"
    buildings = json.loads(buildings_path.read_text(encoding="utf-8"))
    workbook = ROOT.parent / "outputs" / "eui_restored" / "numerical_data_restored_aligned.xlsx"
    source_frame = pd.read_excel(workbook, sheet_name="selected_variables")
    source = source_frame.set_index("教学楼序号").to_dict(orient="index")
    source_present = np.asarray([building["id"] in source for building in buildings])
    processor = joblib.load(MODELS / "tabular_preprocessor.joblib")
    tabular = np.asarray(processor.transform(feature_frame(buildings, source)), dtype=np.float32)

    geometry_csv = pd.read_csv(PACKAGE / "data" / "geometry_3dcnn_features_resnet_gated.csv")
    id_column = geometry_csv.columns[0]
    geometry_csv[id_column] = geometry_csv[id_column].astype(str).str.strip()
    geometry_csv = pd.DataFrame({id_column: [b["id"] for b in buildings]}).merge(geometry_csv, on=id_column, how="left", validate="one_to_one")
    geo_values = geometry_csv.drop(columns=[id_column]).to_numpy(dtype=np.float32)
    geometry_missing = ~np.isfinite(geo_values).all(axis=1)
    scaler = joblib.load(MODELS / "geometry_feature_scaler.joblib")
    geo_values[geometry_missing] = scaler.mean_
    geometry = scaler.transform(geo_values).astype(np.float32)

    target_scaler = np.load(MODELS / "target_scaler.npz")
    target_mean, target_std = target_scaler["mean"], target_scaler["std"]
    zero_geometry = np.zeros((len(buildings), geometry.shape[1]), dtype=np.float32)

    def predict(model, checkpoint: str, geo: np.ndarray) -> np.ndarray:
        model.load_state_dict(torch.load(MODELS / checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        result = []
        for start in range(0, len(buildings), 256):
            result.append(model(torch.from_numpy(geo[start:start + 256]), torch.from_numpy(tabular[start:start + 256])).numpy())
        return np.concatenate(result) * target_std + target_mean

    mlp = predict(networks.TabularOnlyModel("mlp", tabular.shape[1], 3), "best_tabular_mlp.pt", zero_geometry)
    resnet = predict(networks.TabularOnlyModel("resnet", tabular.shape[1], 3), "best_tabular_resnet.pt", zero_geometry)
    multimodal_model = networks.FixedGeometryFusionModel("tabtransformer", tabular.shape[1], geometry.shape[1], 3)
    multimodal = predict(multimodal_model, "best_multimodal_tabtransformer.pt", geometry)
    annual = np.maximum((mlp + resnet + multimodal) / 3, 0)
    annual[geometry_missing] = np.maximum((mlp[geometry_missing] + resnet[geometry_missing]) / 2, 0)

    # Recreate the 259-D static input used by the hourly LSTM model.
    multimodal_model.eval()
    shared = []
    for start in range(0, len(buildings), 256):
        geo = torch.from_numpy(geometry[start:start + 256])
        tab = torch.from_numpy(tabular[start:start + 256])
        g = multimodal_model.geometry_encoder(geo)
        t = multimodal_model.tabular_encoder(tab)
        gate = multimodal_model.gate(torch.cat([g, t], dim=1))
        fused = torch.cat([gate * g, (1 - gate) * t, g * t, torch.abs(g - t)], dim=1)
        shared.append(multimodal_model.shared(fused).numpy())
    shared = np.concatenate(shared).astype(np.float32)
    training_mean = annual[source_present].mean(axis=0, keepdims=True)
    training_std = annual[source_present].std(axis=0, keepdims=True)
    normalized_annual = ((annual - training_mean) / (training_std + 1e-6)).astype(np.float32)
    static = np.concatenate([shared, normalized_annual], axis=1).astype(np.float32)
    (ROOT / "public" / "models").mkdir(parents=True, exist_ok=True)
    static.tofile(ROOT / "public" / "models" / "second-route-static.f32")

    for index, (building, values) in enumerate(zip(buildings, annual)):
        previous = building.get("prediction") or {}
        building["prediction"] = {
            **{key: previous[key] for key in ("trueEui", "trueEpv", "trueCei") if key in previous},
            "annualEui": round(float(values[0]), 6),
            "annualEpv": round(float(values[1]), 4),
            "annualCei": round(float(values[2]), 6),
            "firstEui": round(float(values[0]), 6),
            "firstEpv": round(float(values[1]), 4),
            "firstCei": round(float(values[2]), 6),
            "source": "full-first-route-ensemble" if not geometry_missing[index] else "full-tabular-ensemble-geometry-missing",
        }
        building["staticFeatureIndex"] = index
    buildings_path.write_text(json.dumps(buildings, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Export the exact trained LSTM hourly model for browser inference.
    hourly = HourlyMultimodalModel(259)
    hourly.load_state_dict(torch.load(MODELS / "best_hourly_lstm.pt", map_location="cpu", weights_only=True))
    hourly.eval()
    torch.onnx.export(
        hourly,
        (torch.zeros(1, 259), torch.zeros(1, 8760, 10)),
        ROOT / "public" / "models" / "second-route-hourly-lstm.onnx",
        input_names=["static", "weather"], output_names=["normalized_hourly"],
        opset_version=17,
        dynamo=False,
    )
    hourly_scalers = np.load(MODELS / "hourly_scalers.npz")
    scaler_json = {key: hourly_scalers[key].astype(float).tolist() for key in hourly_scalers.files}
    (ROOT / "public" / "models" / "second-route-scalers.json").write_text(json.dumps(scaler_json, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"buildings": len(buildings), "geometry_missing": int(geometry_missing.sum()), "static_shape": list(static.shape), "annual_min": annual.min(axis=0).tolist(), "annual_max": annual.max(axis=0).tolist()}))


if __name__ == "__main__":
    main()
