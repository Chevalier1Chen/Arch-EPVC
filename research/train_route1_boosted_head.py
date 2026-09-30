from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\pt251")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\pt251\torch\lib")

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import retrain_leakage_free_routes as core

OUT = core.OUT / "route1_boosted_multimodal_head"
OUT.mkdir(parents=True, exist_ok=True)


def metrics(scope, truth, prediction, names):
    rows = []
    for i, name in enumerate(names):
        y, p = truth[:, i], prediction[:, i]
        rows.append({
            "scope": scope, "target": name, "n": len(y),
            "R2": float(r2_score(y, p)),
            "RMSE": float(mean_squared_error(y, p) ** 0.5),
            "MAE": float(mean_absolute_error(y, p)),
        })
    return rows


@torch.no_grad()
def external_features(frame, scaler, category_map, model):
    external = pd.read_csv(core.EXTERNAL_61)
    overlap = pd.read_csv(core.EXTERNAL_OVERLAP)[["Validation ID", "training_id"]]
    external = external.merge(overlap, on="Validation ID", how="left", validate="one_to_one")
    external["construction_year_user"] = external["Construction_year"]
    external["roof_area_m2"] = external["B.Building footprint"]
    external["enclosure_class"] = external["I.Enclosure method"].map(core.enclosure_code)
    external["enclosure_class_model"] = external["enclosure_class"].replace({5: 3})
    external[core.original_r1.CAT] = external["enclosure_class_model"].astype(str)
    cont = scaler.transform(external[core.CONT]).astype(np.float32)
    cat = external[core.original_r1.CAT].map(category_map).to_numpy(np.int64)
    all_vox = np.load(core.EXTERNAL_VOXELS, mmap_mode="r")
    vox = np.array(all_vox[external["Number"].astype(int).to_numpy() - 1], dtype=np.float32, copy=True)
    _, embedding = model(torch.from_numpy(vox), torch.from_numpy(cont), torch.from_numpy(cat), True)
    x = np.concatenate([embedding.cpu().numpy(), cont, cat[:, None].astype(np.float32)], axis=1)
    return external, x


def main():
    core.seed_all(core.SEED)
    frame = core.load_training_frame()
    core.prepare_route2_cache(frame)
    frame = core.add_annual_climate(frame)
    split = json.loads((core.R1_OUT / "split.json").read_text(encoding="utf-8"))
    train = np.asarray(split["train"], dtype=int)
    val = np.asarray(split["val"], dtype=int)
    test = np.asarray(split["test"], dtype=int)
    scaler = joblib.load(core.R1_OUT / "continuous_scaler.joblib")
    category_map = joblib.load(core.R1_OUT / "category_mapping.joblib")
    cont = scaler.transform(frame[core.CONT]).astype(np.float32)
    cat = frame[core.original_r1.CAT].map(category_map).to_numpy(np.int64)
    embedding = np.load(core.R1_OUT / "static_shared_embedding_256.npy")
    x = np.concatenate([embedding, cont, cat[:, None].astype(np.float32)], axis=1)
    y = frame[core.TARGETS].to_numpy(np.float64)

    model = core.original_r1.Route1(len(core.CONT), len(category_map))
    model.load_state_dict(torch.load(core.R1_OUT / "best_route1_3dcnn_tabtransformer.pt", weights_only=True, map_location="cpu"))
    model.eval()
    external, x_external = external_features(frame, scaler, category_map, model)

    pred_test = np.zeros((len(test), 3), dtype=np.float64)
    pred_external = np.zeros((len(external), 3), dtype=np.float64)
    for target_index, target_name in enumerate(core.TARGET_LABELS):
        target_train = np.log1p(np.maximum(y[train, target_index], 0)) if target_name == "Epv" else y[train, target_index]
        target_val = np.log1p(np.maximum(y[val, target_index], 0)) if target_name == "Epv" else y[val, target_index]
        regressor = CatBoostRegressor(
            iterations=1600, depth=7, learning_rate=0.035, loss_function="RMSE",
            l2_leaf_reg=5.0, random_seed=core.SEED + target_index,
            random_strength=0.35, verbose=100, allow_writing_files=False,
        )
        regressor.fit(x[train], target_train, eval_set=(x[val], target_val), early_stopping_rounds=100)
        test_prediction = regressor.predict(x[test])
        external_prediction = regressor.predict(x_external)
        if target_name == "Epv":
            test_prediction = np.expm1(test_prediction)
            external_prediction = np.expm1(external_prediction)
        pred_test[:, target_index] = test_prediction
        pred_external[:, target_index] = external_prediction
        regressor.save_model(str(OUT / f"catboost_{target_name}.cbm"))

    internal = pd.DataFrame({"row_index": test, "building_id": frame.iloc[test]["Number"].to_numpy()})
    for i, name in enumerate(core.TARGET_LABELS):
        internal[f"true_{name}"] = y[test, i]
        internal[f"pred_{name}"] = pred_test[:, i]
    internal.to_csv(OUT / "internal_test_predictions.csv", index=False, encoding="utf-8-sig")
    internal_metrics = pd.DataFrame(metrics("Internal campus-grouped test", y[test], pred_test, core.TARGET_LABELS))
    internal_metrics.to_csv(OUT / "internal_test_metrics.csv", index=False)

    external_result = external.copy()
    external_result["Model predicted EUI"] = pred_external[:, 0]
    external_result["Model predicted Epv"] = pred_external[:, 1]
    external_result["Model predicted CEI direct head"] = pred_external[:, 2]
    external_result["Observed CEI"] = external_result["Observed EUI"] * 0.5942
    external_result["Model predicted CEI"] = external_result["Model predicted EUI"] * 0.5942
    external_result.to_csv(OUT / "external_61_predictions.csv", index=False, encoding="utf-8-sig")
    truth_external = np.column_stack([external_result["Observed EUI"], external_result["Observed CEI"]])
    prediction_external = np.column_stack([external_result["Model predicted EUI"], external_result["Model predicted CEI"]])
    external_metrics = pd.DataFrame(metrics("External 61 excluded campuses", truth_external, prediction_external, ["EUI", "CEI"]))
    external_metrics.to_csv(OUT / "external_61_metrics.csv", index=False)
    print(internal_metrics.to_string(index=False), flush=True)
    print(external_metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
