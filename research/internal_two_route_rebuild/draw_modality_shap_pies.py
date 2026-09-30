from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"C:\pt251")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild")
OUT = ROOT / "shap_modality"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import train_route1 as r1
import train_route2_lstm as r2


SEED = 42
MODALITIES = ("Geometry / image features", "Numerical features", "Time-series features")
COLORS = ("#2f6fb3", "#4c8f35", "#e7922c")


def load_inputs():
    df = pd.read_excel(r1.TABLE, sheet_name="selected_variables")
    gh = pd.read_csv(
        r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package\data\gh_building_energy_inputs.csv",
        encoding="utf-8-sig",
    )
    gh = gh.rename(columns={"obj_object_name_prefix": "Number"}).drop_duplicates("Number")
    gh = gh[["Number", "city", "roof_area_m2"]]
    years = pd.read_excel(
        r"C:\Users\DELL\Desktop\SCI8\Simulation\gh_building_simulation_min_inputs.xlsx",
        sheet_name=0,
    ).iloc[:, :3]
    years.columns = ["Number", "story_height_raw", "construction_year_user"]
    df = df.merge(gh, on="Number", how="left", validate="one_to_one")
    df = df.merge(years[["Number", "construction_year_user"]], on="Number", how="left", validate="one_to_one")
    df[r1.CAT] = df["city"].astype(str) + " | " + df["I.Enclosure method"].astype(str)

    split = json.loads((r1.OUT / "split.json").read_text(encoding="utf-8"))
    import joblib

    scaler = joblib.load(r1.OUT / "continuous_scaler.joblib")
    continuous = scaler.transform(df[r1.CONT]).astype(np.float32)
    category_map = {value: index for index, value in enumerate(split["categories"])}
    category = df[r1.CAT].astype(str).map(category_map).fillna(0).to_numpy(np.int64)
    train = np.asarray(split["train"], dtype=int)
    test = np.asarray(split["test"], dtype=int)
    category_baseline = int(pd.Series(category[train]).mode().iloc[0])
    return df, split, continuous, category, category_baseline, train, test


@torch.no_grad()
def route1_coalitions(model, voxels, continuous, category, category_baseline, indices):
    actual_vox = torch.from_numpy(np.asarray(voxels[indices], dtype=np.float32))
    base_vox = torch.zeros_like(actual_vox)
    actual_cont = torch.from_numpy(continuous[indices])
    base_cont = torch.zeros_like(actual_cont)
    actual_cat = torch.from_numpy(category[indices])
    base_cat = torch.full_like(actual_cat, category_baseline)

    outputs = {}
    statics = {}
    for geometry_on, numerical_on in itertools.product((0, 1), repeat=2):
        vox = actual_vox if geometry_on else base_vox
        cont = actual_cont if numerical_on else base_cont
        cat = actual_cat if numerical_on else base_cat
        prediction, shared = model(vox, cont, cat, True)
        key = (geometry_on, numerical_on)
        outputs[key] = prediction.cpu().numpy()
        statics[key] = torch.cat([shared, prediction], dim=1).cpu().numpy().astype(np.float32)
    return outputs, statics


def exact_two_group_shap(values):
    v00, v10, v01, v11 = values[(0, 0)], values[(1, 0)], values[(0, 1)], values[(1, 1)]
    geometry = 0.5 * ((v10 - v00) + (v11 - v01))
    numerical = 0.5 * ((v01 - v00) + (v11 - v10))
    return np.stack([geometry, numerical], axis=-1)


@torch.no_grad()
def route2_value(model, static, weather, input_mean, input_std, indices, temporal_on):
    static_tensor = torch.from_numpy(static)
    pieces = []
    # Four representative weeks provide seasonal coverage without treating
    # discontinuous months as one recurrent sequence.
    starts = (336, 2520, 4704, 6888)
    for start in starts:
        x = np.asarray(weather[indices, start : start + 168], dtype=np.float32)
        x = (x - input_mean) / input_std
        if not temporal_on:
            x = np.zeros_like(x)
        prediction, _ = model(static_tensor, torch.from_numpy(x.astype(np.float32)))
        pieces.append(prediction.cpu().numpy())
    return np.concatenate(pieces, axis=1)


def exact_three_group_shap(values):
    # values maps a three-bit coalition (geometry, numerical, temporal) to f(S).
    n = 3
    result = []
    for player in range(n):
        phi = np.zeros_like(next(iter(values.values())))
        others = [j for j in range(n) if j != player]
        for size in range(n):
            for subset in itertools.combinations(others, size):
                coalition = [0, 0, 0]
                for member in subset:
                    coalition[member] = 1
                without = tuple(coalition)
                coalition[player] = 1
                with_player = tuple(coalition)
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                phi += weight * (values[with_player] - values[without])
        result.append(phi)
    return np.stack(result, axis=-1)


def summarize(shap_values, route, targets, modalities):
    # Shape is [..., target, modality]. Importance is mean absolute SHAP.
    axes = tuple(range(shap_values.ndim - 2))
    importance = np.mean(np.abs(shap_values), axis=axes)
    rows = []
    for target_index, target in enumerate(targets):
        total = float(importance[target_index].sum())
        for modality_index, modality in enumerate(modalities):
            value = float(importance[target_index, modality_index])
            rows.append(
                {
                    "route": route,
                    "target": target,
                    "modality": modality,
                    "mean_abs_group_SHAP": value,
                    "share_percent": 100.0 * value / total if total else 0.0,
                }
            )
    return rows


def draw_pies(summary):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.23, wspace=0.30)
    panels = [
        ("Route I", "(a) Route I: annual prediction", MODALITIES[:2]),
        ("Route II", "(b) Route II: hourly prediction", MODALITIES),
    ]
    for ax, (route, title, modality_order) in zip(axes, panels):
        subset = summary[summary.route == route].set_index("modality")
        values = [float(subset.loc[m, "average_share_percent"]) for m in modality_order]
        colors = COLORS[: len(values)]
        wedges, _, autotexts = ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            pctdistance=0.70,
            wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
            textprops={"color": "#172b4d", "fontsize": 10},
        )
        for text in autotexts:
            text.set_fontweight("bold")
        ax.set_title(title, loc="left", fontweight="bold", color="#172b4d")
        ax.set_aspect("equal")
    fig.suptitle(
        "Modality-level SHAP contribution of the dual-route prediction framework",
        fontsize=14,
        fontweight="bold",
        color="#172b4d",
    )
    legend_handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(COLORS, MODALITIES)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        frameon=False,
        ncol=3,
    )
    fig.text(
        0.5,
        0.035,
        "Percentages are target-normalized mean absolute group SHAP values, averaged across outputs.",
        ha="center",
        color="#667085",
        fontsize=9,
    )
    fig.savefig(OUT / "SHAP_Modality_Contribution_Pies.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "SHAP_Modality_Contribution_Pies.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(16, torch.get_num_threads()))

    _, split, continuous, category, category_baseline, train, test = load_inputs()
    voxels = np.load(r1.VOXELS, mmap_mode="r")
    valid = np.load(r2.CACHE / "valid.npy")
    eligible = test[valid[test]]
    if len(eligible) > 32:
        positions = np.linspace(0, len(eligible) - 1, 32).round().astype(int)
        selected = eligible[positions]
    else:
        selected = eligible

    route1_model = r1.Route1(len(r1.CONT), len(split["categories"]))
    route1_model.load_state_dict(
        torch.load(r1.OUT / "best_route1_3dcnn_tabtransformer.pt", weights_only=True, map_location="cpu")
    )
    route1_model.eval()
    annual_values, static_values = route1_coalitions(
        route1_model, voxels, continuous, category, category_baseline, selected
    )
    route1_shap = exact_two_group_shap(annual_values)

    route2_model = r2.HourlyLSTM()
    route2_model.load_state_dict(
        torch.load(r2.OUT / "best_route2_lstm.pt", weights_only=True, map_location="cpu")
    )
    route2_model.eval()
    scalers = np.load(r2.OUT / "scalers.npz")
    weather = np.load(r2.CACHE / "weather.npy", mmap_mode="r")

    hourly_values = {}
    for geometry_on, numerical_on, temporal_on in itertools.product((0, 1), repeat=3):
        hourly_values[(geometry_on, numerical_on, temporal_on)] = route2_value(
            route2_model,
            static_values[(geometry_on, numerical_on)],
            weather,
            scalers["input_mean"],
            scalers["input_std"],
            selected,
            bool(temporal_on),
        )
    route2_shap = exact_three_group_shap(hourly_values)

    rows = []
    rows += summarize(route1_shap, "Route I", ("EUI", "Epv", "CEI"), MODALITIES[:2])
    rows += summarize(
        route2_shap,
        "Route II",
        ("Hourly energy", "Hourly PV generation", "Hourly carbon emission"),
        MODALITIES,
    )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "shap_modality_contribution_by_target.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby(["route", "modality"], as_index=False)["share_percent"]
        .mean()
        .rename(columns={"share_percent": "average_share_percent"})
    )
    summary.to_csv(OUT / "shap_modality_contribution_summary.csv", index=False, encoding="utf-8-sig")
    draw_pies(summary)
    print(summary.to_string(index=False))
    print("selected_test_buildings", len(selected))


if __name__ == "__main__":
    main()
