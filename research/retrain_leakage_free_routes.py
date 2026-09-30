from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, r"C:\pt251")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\pt251\torch\lib")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(r"C:\Users\DELL\Documents\论文")
DATA = ROOT / "leakage_free_dataset_excluding_61_campuses"
OUT = ROOT / "leakage_free_retraining"
R1_OUT = OUT / "route1"
R2_OUT = OUT / "route2_lstm"
CACHE_OUT = OUT / "route2_cache"
for path in (OUT, R1_OUT, R2_OUT, CACHE_OUT):
    path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "internal_two_route_rebuild"))
import train_route1 as original_r1
import train_route2_lstm as original_r2


SEED = 42
SPLIT_POLICY = "campus-grouped"
TARGETS = ["EUI", "Epv", "CEI-PV system"]
TARGET_LABELS = ["EUI", "Epv", "CEI"]
CLIMATE_FEATURES = [
    "Temp_mean", "Humidity_mean", "Wind_mean", "HDD18", "CDD26",
    "DNI_kWh_m2", "DHI_kWh_m2", "GHI_kWh_m2",
]
CONT = original_r1.CONT + CLIMATE_FEATURES
TABLE = DATA / "source-data" / "numerical data.xlsx"
VOXELS = DATA / "geometry" / "training_voxels_32.npy"
GH = DATA / "metadata" / "gh_building_energy_inputs.csv"
ORIGINAL_YEAR = Path(r"C:\Users\DELL\Desktop\SCI8\Simulation\gh_building_simulation_min_inputs.xlsx")
ORIGINAL_R2_CACHE = ROOT / "internal_two_route_rebuild" / "route2_cache"
EXTERNAL_61 = ROOT / "external_validation_61_final" / "external_validation_input_prediction_61.csv"
EXTERNAL_OVERLAP = DATA / "excluded_61_campuses.csv"
EXTERNAL_VOXELS = ROOT / "external_validation_150" / "model_run_restored_inputs" / "external_voxels_32.npy"
BUILDINGS_JSON = ROOT / "school-building-design-platform" / "public" / "data" / "buildings.json"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def campus_id(value: str) -> str:
    return str(value).split("_T", 1)[0]


def enclosure_code(value) -> int:
    """Collapse detailed C/E/S/H subtypes to the five thesis-level classes."""
    if pd.isna(value):
        return 3
    text = str(value).strip()
    try:
        number = int(float(text))
        if 1 <= number <= 5:
            return number
    except ValueError:
        pass
    if "行列" in text:
        return 1
    if "双面" in text:
        return 2
    if "三面" in text or text.startswith("3"):
        return 3
    if "四面" in text:
        return 4
    if "综合" in text:
        return 5
    raise ValueError(f"Unknown enclosure label: {value!r}")


def group_split(groups: np.ndarray):
    all_indices = np.arange(len(groups))
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val, test = next(first.split(all_indices, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=SEED + 1)
    train_local, val_local = next(second.split(train_val, groups=groups[train_val]))
    train = train_val[train_local]
    val = train_val[val_local]
    return train.astype(int), val.astype(int), test.astype(int)


def load_training_frame():
    frame = pd.read_excel(TABLE, sheet_name="selected_variables")
    frame["Number"] = frame["Number"].astype(str).str.strip()
    gh = pd.read_csv(GH, encoding="utf-8-sig")
    gh = gh.rename(columns={"obj_object_name_prefix": "Number"}).drop_duplicates("Number")
    years = pd.read_excel(ORIGINAL_YEAR, sheet_name=0).iloc[:, :3]
    years.columns = ["Number", "story_height_raw", "construction_year_user"]
    years["Number"] = years["Number"].astype(str).str.strip()
    frame = frame.merge(gh[["Number", "city", "roof_area_m2"]], on="Number", how="left", validate="one_to_one")
    frame = frame.merge(years[["Number", "construction_year_user"]], on="Number", how="left", validate="one_to_one")
    if frame[["city", "roof_area_m2", "construction_year_user"]].isna().any().any():
        raise ValueError(frame[["city", "roof_area_m2", "construction_year_user"]].isna().sum())
    frame["enclosure_class"] = frame["I.Enclosure method"].map(enclosure_code)
    # City was intentionally removed from Route I; climate is represented in Route II.
    frame[original_r1.CAT] = frame["enclosure_class"].astype(str)
    return frame


def add_annual_climate(frame):
    """Add annual EPW summaries without exposing hourly targets to Route I."""
    weather = np.load(CACHE_OUT / "weather.npy", mmap_mode="r")
    temp = np.asarray(weather[:, :, 4], dtype=np.float32)
    frame = frame.copy()
    frame["Temp_mean"] = np.nanmean(temp, axis=1)
    frame["Humidity_mean"] = np.nanmean(np.asarray(weather[:, :, 5]), axis=1)
    frame["Wind_mean"] = np.nanmean(np.asarray(weather[:, :, 6]), axis=1)
    frame["HDD18"] = np.nansum(np.maximum(18.0 - temp, 0.0), axis=1) / 24.0
    frame["CDD26"] = np.nansum(np.maximum(temp - 26.0, 0.0), axis=1) / 24.0
    frame["DNI_kWh_m2"] = np.nansum(np.asarray(weather[:, :, 7]), axis=1) / 1000.0
    frame["DHI_kWh_m2"] = np.nansum(np.asarray(weather[:, :, 8]), axis=1) / 1000.0
    frame["GHI_kWh_m2"] = np.nansum(np.asarray(weather[:, :, 9]), axis=1) / 1000.0
    for column in CLIMATE_FEATURES:
        values = frame[column].replace([np.inf, -np.inf], np.nan)
        frame[column] = values.fillna(values.median())
    return frame


@torch.no_grad()
def predict_route1(model, loader):
    model.eval()
    predictions, embeddings, indices = [], [], []
    for voxel, continuous, category, _, row_index in loader:
        prediction, embedding = model(voxel, continuous, category, True)
        predictions.append(prediction.cpu().numpy())
        embeddings.append(embedding.cpu().numpy())
        indices.append(row_index.numpy())
    return np.concatenate(indices).astype(int), np.concatenate(predictions), np.concatenate(embeddings)


def inverse_targets(values, mean, std):
    result = values * std + mean
    result[:, 1] = np.expm1(result[:, 1])
    return result


def metric_rows(route, truth, prediction, names):
    rows = []
    for index, name in enumerate(names):
        y = truth[..., index].reshape(-1)
        p = prediction[..., index].reshape(-1)
        rows.append(
            {
                "route": route,
                "target": name,
                "n": len(y),
                "R2": r2_score(y, p),
                "RMSE": mean_squared_error(y, p) ** 0.5,
                "MAE": mean_absolute_error(y, p),
            }
        )
    return rows


def train_route1(frame):
    voxels = np.load(VOXELS, mmap_mode="r")
    groups = frame["Number"].map(campus_id).to_numpy()
    train, val, test = group_split(groups)
    if SPLIT_POLICY == "campus-grouped":
        assert not set(groups[train]) & set(groups[val])
        assert not set(groups[train]) & set(groups[test])
        assert not set(groups[val]) & set(groups[test])

    scaler = StandardScaler().fit(frame.loc[train, CONT])
    continuous = scaler.transform(frame[CONT]).astype(np.float32)
    categories = sorted(frame[original_r1.CAT].astype(str).unique())
    category_map = {value: index for index, value in enumerate(categories)}
    category = frame[original_r1.CAT].astype(str).map(category_map).to_numpy(np.int64)
    raw_targets = frame[TARGETS].to_numpy(np.float64)
    transformed = raw_targets.copy()
    transformed[:, 1] = np.log1p(np.maximum(transformed[:, 1], 0))
    target_mean = transformed[train].mean(axis=0)
    target_std = transformed[train].std(axis=0)
    target_std[target_std < 1e-8] = 1.0
    targets = ((transformed - target_mean) / target_std).astype(np.float32)

    loaders = {
        name: DataLoader(
            original_r1.Samples(voxels, continuous, category, targets, indices),
            batch_size=48,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, indices in (("train", train), ("val", val), ("test", test), ("all", np.arange(len(frame))))
    }
    model = original_r1.Route1(len(CONT), len(categories))
    checkpoint = R1_OUT / "best_route1_3dcnn_tabtransformer.pt"
    if not checkpoint.exists():
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
        loss_fn = nn.SmoothL1Loss(beta=0.5)
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, 81):
            model.train()
            running, count = 0.0, 0
            for voxel, cont, cat, target, _ in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(voxel, cont, cat)
                loss = loss_fn(prediction, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                running += loss.item() * len(voxel)
                count += len(voxel)
            _, val_prediction, _ = predict_route1(model, loaders["val"])
            val_loss = float(np.mean((val_prediction - targets[val]) ** 2))
            history.append((epoch, running / count, val_loss))
            print(f"route1 epoch={epoch:03d} train={running/count:.6f} val_mse={val_loss:.6f}", flush=True)
            if val_loss < best - 1e-5:
                best, stale = val_loss, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= 12:
                    break
        pd.DataFrame(history, columns=["epoch", "train_loss", "val_mse"]).to_csv(R1_OUT / "history.csv", index=False)

    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    test_order, scaled_prediction, _ = predict_route1(model, loaders["test"])
    test_prediction = inverse_targets(scaled_prediction, target_mean, target_std)
    test_truth = raw_targets[test_order]
    metrics = pd.DataFrame(metric_rows("Route I", test_truth, test_prediction, TARGET_LABELS))
    metrics.to_csv(R1_OUT / "internal_test_metrics.csv", index=False)
    prediction_frame = pd.DataFrame({"row_index": test_order, "building_id": frame.iloc[test_order]["Number"].to_numpy()})
    for index, target in enumerate(TARGET_LABELS):
        prediction_frame[f"true_{target}"] = test_truth[:, index]
        prediction_frame[f"pred_{target}"] = test_prediction[:, index]
    prediction_frame.to_csv(R1_OUT / "internal_test_predictions.csv", index=False, encoding="utf-8-sig")

    all_order, all_scaled_prediction, all_embedding = predict_route1(model, loaders["all"])
    order = np.argsort(all_order)
    all_embedding = all_embedding[order].astype(np.float32)
    all_scaled_prediction = all_scaled_prediction[order].astype(np.float32)
    np.save(R1_OUT / "static_shared_embedding_256.npy", all_embedding)
    np.save(R1_OUT / "annual_prediction_scaled_3.npy", all_scaled_prediction)
    joblib.dump(scaler, R1_OUT / "continuous_scaler.joblib")
    joblib.dump(category_map, R1_OUT / "category_mapping.joblib")
    np.savez(R1_OUT / "target_scaler.npz", mean=target_mean, std=target_std, epv_log=np.asarray([1]))
    split = {
        "seed": SEED,
        "split_policy": SPLIT_POLICY,
        "split_unit": "campus",
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
        "train_campuses": sorted(set(groups[train])),
        "val_campuses": sorted(set(groups[val])),
        "test_campuses": sorted(set(groups[test])),
        "categories": categories,
        "features": CONT,
    }
    (R1_OUT / "split.json").write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    return model, scaler, category_map, target_mean, target_std, split, continuous, category, all_embedding, all_scaled_prediction


def prepare_route2_cache(frame):
    weather_path = CACHE_OUT / "weather.npy"
    targets_path = CACHE_OUT / "targets.npy"
    valid_path = CACHE_OUT / "valid.npy"
    if weather_path.exists() and targets_path.exists() and valid_path.exists():
        return
    original_index = pd.read_csv(ORIGINAL_R2_CACHE / "index.csv")
    original_map = dict(zip(original_index["building_id"].astype(str), original_index["row_index"].astype(int)))
    source_rows = np.asarray([original_map[value] for value in frame["Number"].astype(str)], dtype=int)
    source_weather = np.load(ORIGINAL_R2_CACHE / "weather.npy", mmap_mode="r")
    source_targets = np.load(ORIGINAL_R2_CACHE / "targets.npy", mmap_mode="r")
    source_valid = np.load(ORIGINAL_R2_CACHE / "valid.npy")
    weather = open_memmap(weather_path, mode="w+", dtype="float32", shape=(len(frame), 8760, 10))
    targets = open_memmap(targets_path, mode="w+", dtype="float32", shape=(len(frame), 8760, 3))
    for start in range(0, len(frame), 32):
        rows = source_rows[start : start + 32]
        weather[start : start + len(rows)] = source_weather[rows]
        targets[start : start + len(rows)] = source_targets[rows]
        if start % 320 == 0:
            print(f"route2 cache {min(start+len(rows),len(frame))}/{len(frame)}", flush=True)
    weather.flush()
    targets.flush()
    np.save(valid_path, source_valid[source_rows])
    pd.DataFrame({"row_index": np.arange(len(frame)), "building_id": frame["Number"]}).to_csv(CACHE_OUT / "index.csv", index=False)


def train_route2(frame, split, embeddings, annual_scaled):
    prepare_route2_cache(frame)
    weather = np.load(CACHE_OUT / "weather.npy", mmap_mode="r")
    targets = np.load(CACHE_OUT / "targets.npy", mmap_mode="r")
    valid = np.load(CACHE_OUT / "valid.npy")
    static = np.concatenate([embeddings, annual_scaled], axis=1).astype(np.float32)
    np.save(R2_OUT / "static_condition_259.npy", static)
    train = np.asarray(split["train"], dtype=int)
    val = np.asarray(split["val"], dtype=int)
    test = np.asarray(split["test"], dtype=int)
    train, val, test = train[valid[train]], val[valid[val]], test[valid[test]]
    input_mean, input_std = original_r2.stream_stats(weather, train)
    label_mean, label_std = original_r2.stream_stats(targets, train, log=True)
    np.savez(R2_OUT / "scalers.npz", input_mean=input_mean, input_std=input_std, label_mean=label_mean, label_std=label_std)
    train_dataset = original_r2.Windows(weather, targets, static, train, input_mean, input_std, label_mean, label_std, n=7000, length=72, seed=SEED)
    val_dataset = original_r2.Windows(weather, targets, static, val, input_mean, input_std, label_mean, label_std, n=1200, length=168, seed=SEED + 999)
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=48, num_workers=0)
    model = original_r2.HourlyLSTM()
    checkpoint = R2_OUT / "best_route2_lstm.pt"
    if not checkpoint.exists():
        optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, 31):
            model.train()
            running, count = 0.0, 0
            for static_batch, weather_batch, target_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(static_batch, weather_batch)
                loss = nn.functional.smooth_l1_loss(prediction, target_batch, beta=0.5)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                running += loss.item() * target_batch.numel()
                count += target_batch.numel()
            val_mse = original_r2.validate_windows(model, val_loader)
            history.append((epoch, running / count, val_mse))
            print(f"route2 epoch={epoch:02d} train={running/count:.6f} val_mse={val_mse:.6f}", flush=True)
            if val_mse < best - 1e-5:
                best, stale = val_mse, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= 6:
                    break
        pd.DataFrame(history, columns=["epoch", "train_loss", "val_mse"]).to_csv(R2_OUT / "history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    truth, prediction, _, _ = original_r2.infer_test(model, weather, targets, static, test, input_mean, input_std, label_mean, label_std)
    rows = metric_rows("Route II", truth, prediction, ["Hourly energy", "Hourly PV generation", "Hourly carbon emission"])
    pd.DataFrame(rows).to_csv(R2_OUT / "internal_test_metrics.csv", index=False)
    np.savez_compressed(
        R2_OUT / "internal_test_predictions_8760x3.npz",
        row_indices=test.astype(np.int32),
        hour=np.arange(1, 8761, dtype=np.int16),
        true=truth.astype(np.float32),
        prediction=prediction.astype(np.float32),
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


@torch.no_grad()
def external_validation(model, scaler, category_map, target_mean, target_std):
    external = pd.read_csv(EXTERNAL_61)
    overlap = pd.read_csv(EXTERNAL_OVERLAP)[["Validation ID", "training_id"]]
    external = external.merge(overlap, on="Validation ID", how="left", validate="one_to_one")
    with BUILDINGS_JSON.open("r", encoding="utf-8") as handle:
        building_records = json.load(handle)
    city_by_id = {str(record["id"]): str(record["city"]) for record in building_records}
    external["city"] = external["training_id"].map(city_by_id)
    external["construction_year_user"] = external["Construction_year"]
    # The trained variable is available roof area, not installed PV area.
    # For the external OBJ cases, the cleaned footprint is the flat-roof area proxy.
    external["roof_area_m2"] = external["B.Building footprint"]
    external["enclosure_class"] = external["I.Enclosure method"].map(enclosure_code)
    # The retained training set has no comprehensive-layout (class 5) example;
    # use the nearest available multi-wing class (three-sided, class 3).
    external["enclosure_class_model"] = external["enclosure_class"].replace({5: 3})
    external[original_r1.CAT] = external["enclosure_class_model"].astype(str)
    unknown = sorted(set(external[original_r1.CAT]) - set(category_map))
    if unknown:
        raise ValueError(f"Unseen external categories after campus removal: {unknown}")
    continuous = scaler.transform(external[CONT]).astype(np.float32)
    category = external[original_r1.CAT].map(category_map).to_numpy(np.int64)
    all_voxels = np.load(EXTERNAL_VOXELS, mmap_mode="r")
    rows = external["Number"].astype(int).to_numpy() - 1
    voxels = np.asarray(all_voxels[rows], dtype=np.float32)
    prediction_scaled = model(torch.from_numpy(voxels), torch.from_numpy(continuous), torch.from_numpy(category)).cpu().numpy()
    prediction = inverse_targets(prediction_scaled, target_mean, target_std)
    result = external.copy()
    result["Model predicted EUI"] = prediction[:, 0]
    result["Model predicted Epv"] = prediction[:, 1]
    # The measured validation buildings have no independently observed PV output;
    # validate operational carbon consistently from EUI and the stated grid factor.
    result["Model predicted CEI direct head"] = prediction[:, 2]
    result["Model predicted CEI"] = result["Model predicted EUI"] * 0.5942
    result["Observed CEI"] = result["Observed EUI"] * 0.5942
    rows_metric = []
    for target, observed_column, predicted_column in (
        ("EUI", "Observed EUI", "Model predicted EUI"),
        ("CEI", "Observed CEI", "Model predicted CEI"),
    ):
        y = result[observed_column].to_numpy(float)
        p = result[predicted_column].to_numpy(float)
        rows_metric.append(
            {
                "scope": "External 61 excluded campuses",
                "target": target,
                "n": len(y),
                "R2": r2_score(y, p),
                "RMSE": mean_squared_error(y, p) ** 0.5,
                "MAE": mean_absolute_error(y, p),
            }
        )
    result.to_csv(OUT / "external_61_true_model_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_metric).to_csv(OUT / "external_61_true_model_metrics.csv", index=False)
    print(pd.DataFrame(rows_metric).to_string(index=False), flush=True)


def main():
    seed_all(SEED)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    frame = load_training_frame()
    prepare_route2_cache(frame)
    frame = add_annual_climate(frame)
    model, scaler, category_map, target_mean, target_std, split, _, _, embeddings, annual_scaled = train_route1(frame)
    external_validation(model, scaler, category_map, target_mean, target_std)
    train_route2(frame, split, embeddings, annual_scaled)


if __name__ == "__main__":
    main()
