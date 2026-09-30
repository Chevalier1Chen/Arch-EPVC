from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\pt251")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\pt251\torch\lib")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import retrain_leakage_free_routes as core

OUT = core.OUT / "route1_eui_optimization"
OUT.mkdir(parents=True, exist_ok=True)


def engineered(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[core.CONT].copy()
    for column in ["campus_area", "campus_density", "campus_far", "building_spacing",
                   "campus_building_count", "campus_total_building_area",
                   "campus_total_footprint", "campus_mean_height"]:
        x[column] = frame[column]
    area = frame["A.Building area"].clip(lower=1)
    footprint = frame["B.Building footprint"].clip(lower=1)
    height = frame["C.Building height"].clip(lower=.1)
    length = frame["F.Building length"].clip(lower=.1)
    width = frame["G.Building width"].clip(lower=.1)
    wall = 2 * (length + width) * height
    roof = footprint
    angle = np.deg2rad(frame["H.Orientation"].astype(float))
    x["floor_area_to_footprint"] = area / footprint
    x["floor_consistency"] = area / (footprint * frame["D.Layer"].clip(lower=1))
    x["volume_proxy"] = footprint * height
    x["surface_to_floor"] = (wall + 2 * footprint) / area
    x["wall_to_floor"] = wall / area
    x["roof_to_floor"] = roof / area
    x["aspect_ratio"] = np.maximum(length, width) / np.minimum(length, width)
    x["orientation_sin"] = np.sin(angle); x["orientation_cos"] = np.cos(angle)
    x["wall_heat_loss"] = wall * (.7 * frame["L.Wall thermal coefficient"] + .3 * frame["N.Window U-value"]) / area
    x["roof_heat_loss"] = footprint * frame["K.Roof thermal coefficient"] / area
    x["ground_heat_loss"] = footprint * frame["M.Ground thermal coefficient"] / area
    x["total_heat_loss_index"] = x["wall_heat_loss"] + x["roof_heat_loss"] + x["ground_heat_loss"]
    x["solar_climate_index"] = frame["GHI_kWh_m2"] * footprint / area
    x["heating_exposure"] = frame["HDD18"] * x["total_heat_loss_index"]
    x["cooling_exposure"] = frame["CDD26"] * x["surface_to_floor"]
    x["building_age"] = 2026 - frame["construction_year_user"].astype(float)
    x["city"] = frame["city"].astype(str)
    x["enclosure"] = frame["enclosure_class"].astype(str)
    x["school_type"] = frame["school_type"].astype(str)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0)


def add_campus_features(frame: pd.DataFrame) -> pd.DataFrame:
    records = json.loads(core.BUILDINGS_JSON.read_text(encoding="utf-8"))
    lookup = {str(item["id"]): item for item in records}
    result = frame.copy()
    result["campus_area"] = [lookup[n].get("campusArea") for n in result["Number"].astype(str)]
    result["campus_density"] = [lookup[n].get("campusDensity") for n in result["Number"].astype(str)]
    result["campus_far"] = [lookup[n].get("campusFar") for n in result["Number"].astype(str)]
    result["building_spacing"] = [lookup[n].get("buildingSpacing") for n in result["Number"].astype(str)]
    result["school_type"] = [lookup[n].get("schoolType", "unknown") for n in result["Number"].astype(str)]
    result["campus_id"] = result["Number"].map(core.campus_id)
    grouped = result.groupby("campus_id")
    result["campus_building_count"] = grouped["Number"].transform("size")
    result["campus_total_building_area"] = grouped["A.Building area"].transform("sum")
    result["campus_total_footprint"] = grouped["B.Building footprint"].transform("sum")
    result["campus_mean_height"] = grouped["C.Building height"].transform("mean")
    for column in ["campus_area", "campus_density", "campus_far", "building_spacing"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result[column].fillna(result[column].median())
    return result


def score(y, p):
    return r2_score(y, p), mean_squared_error(y, p) ** .5, mean_absolute_error(y, p)


def main():
    frame = core.load_training_frame(); core.prepare_route2_cache(frame); frame = core.add_annual_climate(frame)
    frame = add_campus_features(frame)
    split = json.loads((core.R1_OUT / "split.json").read_text(encoding="utf-8"))
    train, val, test = (np.asarray(split[k], int) for k in ("train", "val", "test"))
    base = engineered(frame)
    cat_cols = ["city", "enclosure", "school_type"]
    y = frame["EUI"].to_numpy(float)
    embedding = np.load(core.R1_OUT / "static_shared_embedding_256.npy")
    annual_pred_scaled = np.load(core.R1_OUT / "annual_prediction_scaled_3.npy")
    target_scale = np.load(core.R1_OUT / "target_scaler.npz")
    annual_pred = annual_pred_scaled * target_scale["std"] + target_scale["mean"]
    annual_pred[:, 1] = np.expm1(annual_pred[:, 1])
    pca = PCA(n_components=24, random_state=42).fit(embedding[train])
    epca = pca.transform(embedding)

    variants = {}
    variants["physics_tabular"] = base.copy()
    variants["physics_plus_neural_prediction"] = base.assign(
        neural_EUI=annual_pred[:, 0], neural_log_Epv=np.log1p(np.maximum(annual_pred[:, 1], 0)), neural_CEI=annual_pred[:, 2])
    rich = variants["physics_plus_neural_prediction"].copy()
    for j in range(epca.shape[1]): rich[f"embedding_pc_{j+1:02d}"] = epca[:, j]
    variants["physics_plus_multimodal_embedding"] = rich

    records, predictions = [], {}
    for variant_name, x in variants.items():
        for depth in (5, 6, 7, 8):
            model = CatBoostRegressor(iterations=1800, depth=depth, learning_rate=.03, loss_function="RMSE",
                                      l2_leaf_reg=6, random_seed=42, random_strength=.25,
                                      verbose=False, allow_writing_files=False)
            model.fit(x.iloc[train], y[train], cat_features=cat_cols,
                      eval_set=(x.iloc[val], y[val]), early_stopping_rounds=120)
            pv = model.predict(x.iloc[val]); pt = model.predict(x.iloc[test])
            vr2, vrmse, vmae = score(y[val], pv); tr2, trmse, tmae = score(y[test], pt)
            records.append({"model":"CatBoost","variant":variant_name,"depth":depth,"best_iteration":model.get_best_iteration(),
                            "val_R2":vr2,"val_RMSE":vrmse,"val_MAE":vmae,
                            "test_R2":tr2,"test_RMSE":trmse,"test_MAE":tmae})
            predictions[(variant_name, depth)] = pt

    numeric = pd.get_dummies(rich, columns=cat_cols, dtype=float)
    for depth in (4, 6, 8):
        model = XGBRegressor(n_estimators=2200, max_depth=depth, learning_rate=.025, subsample=.82,
                             colsample_bytree=.82, reg_lambda=8, reg_alpha=.05,
                             objective="reg:squarederror", random_state=42, n_jobs=12)
        model.fit(numeric.iloc[train], y[train], eval_set=[(numeric.iloc[val], y[val])], verbose=False)
        pv=model.predict(numeric.iloc[val]);pt=model.predict(numeric.iloc[test])
        vr2,vrmse,vmae=score(y[val],pv);tr2,trmse,tmae=score(y[test],pt)
        records.append({"model":"XGBoost","variant":"physics_plus_multimodal_embedding","depth":depth,"best_iteration":getattr(model,"best_iteration",None),
                        "val_R2":vr2,"val_RMSE":vrmse,"val_MAE":vmae,"test_R2":tr2,"test_RMSE":trmse,"test_MAE":tmae})

    result = pd.DataFrame(records).sort_values(["val_RMSE", "test_RMSE"])
    result.to_csv(OUT / "fixed_split_model_comparison.csv", index=False, encoding="utf-8-sig")
    best = result.iloc[0]
    if best["model"] == "CatBoost":
        key=(best["variant"],int(best["depth"])); best_pred=predictions[key]
    else:
        best_pred=np.full(len(test),np.nan)
    pd.DataFrame({"row_index":test,"building_id":frame.iloc[test]["Number"].to_numpy(),"true_EUI":y[test],"pred_EUI":best_pred}).to_csv(OUT/"best_validation_selected_test_predictions.csv",index=False,encoding="utf-8-sig")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__": main()
