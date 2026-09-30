from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, r"C:\pt251")

import joblib
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild")
OUT = ROOT / "shap_corrected"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import train_route1 as r1
import train_route2_lstm as r2


SEED = 42
ROUTE1_TARGETS = ("EUI", "Epv", "CEI")
ROUTE2_TARGETS = ("Hourly energy", "Hourly PV generation", "Hourly carbon emission")
STATIC_LABELS = (
    "3D geometry representation",
    "Building area",
    "Building footprint",
    "Building height",
    "Number of floors",
    "Floor height",
    "Building length",
    "Building width",
    "Orientation",
    "Shape coefficient",
    "Roof thermal coefficient",
    "Wall thermal coefficient",
    "Ground thermal coefficient",
    "Window U-value",
    "Construction year",
    "Roof PV area",
    "City × enclosure category",
)
TEMPORAL_LABELS = (
    "Day sine",
    "Day cosine",
    "Hour sine",
    "Hour cosine",
    "Temperature",
    "Humidity",
    "Wind speed",
    "DNI",
    "DHI",
    "GHI",
)
BLUE = "#2f6fb3"
TEAL = "#16877a"
ORANGE = "#e7922c"
INK = "#172b4d"
GRID = "#d9e2e8"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_table_inputs():
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
    scaler = joblib.load(r1.OUT / "continuous_scaler.joblib")
    continuous = scaler.transform(df[r1.CONT]).astype(np.float32)
    category_map = {value: index for index, value in enumerate(split["categories"])}
    category = df[r1.CAT].astype(str).map(category_map).fillna(0).to_numpy(np.int64)
    return df, split, continuous, category


@torch.no_grad()
def encode_geometry(model: r1.Route1, voxels, indices, batch_size=32):
    chunks = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start : start + batch_size]
        tensor = torch.from_numpy(np.asarray(voxels[ids], dtype=np.float32))
        chunks.append(model.geometry(tensor).cpu())
    return torch.cat(chunks, dim=0)


def tabular_from_embedding(model: r1.Route1, continuous, category_embedding):
    tab = model.tabular
    tokens = continuous.unsqueeze(-1) * tab.cont_weight.unsqueeze(0) + tab.cont_bias.unsqueeze(0)
    tokens = torch.cat([tokens, category_embedding.unsqueeze(1)], dim=1) + tab.position
    return tab.proj(tab.encoder(tokens).mean(1))


def static_forward(model: r1.Route1, geometry, continuous, category_embedding):
    tabular = tabular_from_embedding(model, continuous, category_embedding)
    gate = model.gate(torch.cat([geometry, tabular], dim=1))
    shared = model.shared(
        torch.cat(
            [gate * geometry, (1.0 - gate) * tabular, geometry * tabular, torch.abs(geometry - tabular)],
            dim=1,
        )
    )
    prediction = model.head(shared)
    return prediction, shared


class AnnualWrapper(nn.Module):
    def __init__(self, model, target_index, target_mean, target_std):
        super().__init__()
        self.model = model
        self.target_index = target_index
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def forward(self, geometry, continuous, category_embedding):
        prediction, _ = static_forward(self.model, geometry, continuous, category_embedding)
        value = prediction[:, self.target_index] * self.target_std + self.target_mean
        if self.target_index == 1:
            value = torch.expm1(value)
        return value


class HourlyWrapper(nn.Module):
    def __init__(self, route1_model, route2_model, target_index, label_mean, label_std):
        super().__init__()
        self.route1_model = route1_model
        self.route2_model = route2_model
        self.target_index = target_index
        self.label_mean = float(label_mean)
        self.label_std = float(label_std)

    def forward(self, geometry, continuous, category_embedding, weather):
        annual_prediction, shared = static_forward(
            self.route1_model, geometry, continuous, category_embedding
        )
        static = torch.cat([shared, annual_prediction], dim=1)
        hourly_prediction, _ = self.route2_model(static, weather)
        value = hourly_prediction[:, -1, self.target_index] * self.label_std + self.label_mean
        return torch.expm1(value)


def expected_gradients(model, samples, backgrounds, steps=16, seed=42):
    """Expected gradients: a GradientSHAP estimator for several continuous inputs."""
    generator = torch.Generator().manual_seed(seed)
    n = samples[0].shape[0]
    m = backgrounds[0].shape[0]
    accum = [torch.zeros_like(value) for value in samples]
    model.eval()
    for step in range(steps):
        background_ids = torch.randint(0, m, (n,), generator=generator)
        interpolated = []
        deltas = []
        alpha_base = torch.rand((n,), generator=generator)
        for sample, background in zip(samples, backgrounds):
            selected_background = background[background_ids]
            delta = sample - selected_background
            shape = (n,) + (1,) * (sample.ndim - 1)
            alpha = alpha_base.reshape(shape)
            point = (selected_background + alpha * delta).detach().requires_grad_(True)
            interpolated.append(point)
            deltas.append(delta)
        output = model(*interpolated)
        gradients = torch.autograd.grad(output.sum(), interpolated, retain_graph=False)
        for index, (gradient, delta) in enumerate(zip(gradients, deltas)):
            accum[index] += gradient.detach() * delta
    return [value.cpu().numpy() / steps for value in accum]


def group_static_shap(attributions):
    geometry, continuous, category_embedding = attributions
    return np.column_stack(
        [geometry.sum(axis=1), continuous, category_embedding.sum(axis=1)]
    )


def group_static_values(geometry, continuous, category_embedding):
    geo_value = np.linalg.norm(geometry, axis=1)
    geo_value = (geo_value - geo_value.mean()) / max(geo_value.std(), 1e-8)
    cat_value = np.linalg.norm(category_embedding, axis=1)
    cat_value = (cat_value - cat_value.mean()) / max(cat_value.std(), 1e-8)
    return np.column_stack([geo_value, continuous, cat_value])


def group_hourly_shap(attributions):
    geometry, continuous, category_embedding, temporal = attributions
    return np.column_stack(
        [
            geometry.sum(axis=1),
            continuous,
            category_embedding.sum(axis=1),
            temporal.sum(axis=1),
        ]
    )


def group_hourly_values(geometry, continuous, category_embedding, temporal):
    static = group_static_values(geometry, continuous, category_embedding)
    return np.column_stack([static, temporal.mean(axis=1)])


def beeswarm(ax, shap_matrix, feature_values, labels, title):
    importance = np.mean(np.abs(shap_matrix), axis=0)
    order = np.argsort(importance)[::-1]
    rng = np.random.default_rng(SEED)
    cmap = mpl.colormaps["coolwarm"]
    for row, feature_index in enumerate(order):
        shap_value = shap_matrix[:, feature_index]
        feature_value = feature_values[:, feature_index]
        lo, hi = np.nanpercentile(feature_value, [5, 95])
        normalized = np.clip((feature_value - lo) / max(hi - lo, 1e-8), 0, 1)
        ranks = pd.Series(shap_value).rank(method="first").to_numpy()
        jitter = ((ranks % 9) - 4) * 0.045 + rng.normal(0, 0.012, size=len(ranks))
        ax.scatter(
            shap_value,
            row + jitter,
            c=cmap(normalized),
            s=19,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.3,
        )
    ax.axvline(0, color="#667085", linewidth=0.9)
    ax.set_yticks(np.arange(len(order)), [labels[index] for index in order])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.grid(True, axis="x", color=GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#98a2b3")
    return importance


def save_shap_figure(results, labels, targets, stem, heading):
    height = 8.2 if len(labels) <= 18 else 11.5
    fig, axes = plt.subplots(1, 3, figsize=(18.2, height), constrained_layout=True)
    rows = []
    for panel, (ax, target) in enumerate(zip(axes, targets)):
        shap_matrix, feature_values = results[target]
        importance = beeswarm(ax, shap_matrix, feature_values, labels, f"({chr(97 + panel)}) {target}")
        for label, value in zip(labels, importance):
            rows.append(
                {
                    "target": target,
                    "feature": label,
                    "mean_abs_SHAP": float(value),
                    "importance_percent": float(value / importance.sum() * 100.0),
                }
            )
    fig.suptitle(heading, fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.5,
        -0.005,
        "Color represents the relative feature value (blue: low; red: high). Geometry latent dimensions and category-embedding dimensions are each aggregated as one feature group.",
        ha="center",
        fontsize=9,
        color="#667085",
    )
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pd.DataFrame(rows)


def short_label(label):
    aliases = {
        "3D geometry representation": "3D geometry",
        "City × enclosure category": "City × enclosure",
        "Roof thermal coefficient": "Roof U-value",
        "Wall thermal coefficient": "Wall U-value",
        "Ground thermal coefficient": "Ground U-value",
        "Construction year": "Construction year",
        "Number of floors": "Floors",
        "Roof PV area": "Roof PV area",
        "Hourly PV generation": "Hourly PV",
        "Hourly carbon emission": "Hourly carbon",
    }
    return aliases.get(label, label)


def pie_code(label):
    aliases = {
        "3D geometry representation": "3D",
        "City × enclosure category": "C×E",
        "Building area": "Area",
        "Building footprint": "Footprint",
        "Building height": "Height",
        "Number of floors": "Floors",
        "Floor height": "Floor h.",
        "Building length": "Length",
        "Building width": "Width",
        "Orientation": "Orient.",
        "Shape coefficient": "Shape",
        "Roof thermal coefficient": "Roof U",
        "Wall thermal coefficient": "Wall U",
        "Ground thermal coefficient": "Ground U",
        "Window U-value": "Window U",
        "Construction year": "Year",
        "Roof PV area": "PV area",
        "Day sine": "Day sin",
        "Day cosine": "Day cos",
        "Hour sine": "Hour sin",
        "Hour cosine": "Hour cos",
        "Temperature": "Temp.",
        "Humidity": "RH",
        "Wind speed": "Wind",
    }
    return aliases.get(label, label)


def save_reference_style_figure(results, labels, targets, stem, heading, top_n=9):
    """Reference layout: importance bars + inset pie + SHAP beeswarm for each output."""
    fig = plt.figure(figsize=(22.5, 7.4), facecolor="white")
    outer = fig.add_gridspec(1, 3, left=0.035, right=0.985, top=0.88, bottom=0.12, wspace=0.20)
    # A restrained, consistent blue system is used for global importance.
    # The red–blue SHAP gradient is retained only where color encodes feature value.
    palette = ["#2f6fb3", "#5f8fc3", "#86a9cf", "#aec5de"]
    cmap = mpl.colormaps["coolwarm"]

    for panel, target in enumerate(targets):
        inner = outer[panel].subgridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.12)
        ax_bar = fig.add_subplot(inner[0])
        ax_swarm = fig.add_subplot(inner[1])
        shap_matrix, feature_values = results[target]
        importance = np.mean(np.abs(shap_matrix), axis=0)
        top = np.argsort(importance)[::-1][: min(top_n, len(labels))]
        display = top[::-1]
        y = np.arange(len(display))
        ax_bar.barh(y, importance[display], color=BLUE, height=0.72)
        ax_bar.set_yticks(y, [short_label(labels[index]) for index in display])
        ax_bar.set_xlabel("Mean |SHAP value|")
        ax_bar.set_title(f"Feature importance · {target}", fontsize=11, fontweight="bold", color=INK)
        ax_bar.grid(True, axis="x", color=GRID, linewidth=0.65, alpha=0.8)
        ax_bar.set_axisbelow(True)
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.spines[["left", "bottom"]].set_color("#98a2b3")
        xmax = max(float(importance[top].max()), 1e-8)
        ax_bar.set_xlim(0, xmax * 1.22)
        for row, index in enumerate(display):
            ax_bar.text(
                importance[index] + xmax * 0.018,
                row,
                f"{importance[index]:.2f}",
                va="center",
                fontsize=8,
                color=INK,
            )

        # The four leading features are shown separately; all remaining features form "Other".
        pie_top = top[:4]
        pie_values = [float(importance[index]) for index in pie_top]
        other = float(importance.sum() - importance[pie_top].sum())
        pie_values.append(max(other, 0.0))
        pie_labels = [pie_code(labels[index]) for index in pie_top] + ["Other"]
        pie_colors = palette + ["#d3d8df"]
        inset = ax_bar.inset_axes([0.46, 0.015, 0.50, 0.43])
        inset.pie(
            pie_values,
            labels=pie_labels,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            autopct=lambda p: f"{p:.0f}%" if p >= 7 else "",
            pctdistance=0.62,
            labeldistance=0.92,
            textprops={"fontsize": 6.3, "color": INK},
            wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
        )
        inset.set_aspect("equal")

        rng = np.random.default_rng(SEED + panel)
        for row, index in enumerate(top):
            values = shap_matrix[:, index]
            feature = feature_values[:, index]
            lo, hi = np.nanpercentile(feature, [5, 95])
            normalized = np.clip((feature - lo) / max(hi - lo, 1e-8), 0, 1)
            ranks = pd.Series(values).rank(method="first").to_numpy()
            jitter = ((ranks % 9) - 4) * 0.052 + rng.normal(0, 0.012, size=len(ranks))
            ax_swarm.scatter(
                values,
                row + jitter,
                c=cmap(normalized),
                s=18,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.25,
            )
        ax_swarm.axvline(0, color="#667085", linewidth=0.85)
        ax_swarm.set_yticks(np.arange(len(top)))
        ax_swarm.set_yticklabels([])
        ax_swarm.invert_yaxis()
        ax_swarm.set_xlabel("SHAP value")
        ax_swarm.set_title(f"SHAP analysis · {target}", fontsize=11, fontweight="bold", color=INK)
        ax_swarm.grid(True, axis="x", color=GRID, linewidth=0.65, alpha=0.8)
        ax_swarm.set_axisbelow(True)
        ax_swarm.spines[["top", "right"]].set_visible(False)
        ax_swarm.spines[["left", "bottom"]].set_color("#98a2b3")
        color_axis = ax_swarm.inset_axes([0.92, 0.05, 0.025, 0.28])
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap), cax=color_axis)
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.ax.tick_params(labelsize=7, length=0)
        colorbar.set_label("Feature value", fontsize=7)

        ax_bar.text(-0.04, 1.08, f"({chr(97 + panel)})", transform=ax_bar.transAxes, fontsize=12, fontweight="bold", color=INK)

    fig.suptitle(heading, fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.5,
        0.035,
        "Bars and inset pies show global importance; beeswarm points show the direction and distribution of feature effects.",
        ha="center",
        fontsize=9,
        color="#667085",
    )
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def modality_summary(route1_results, route2_results):
    rows = []
    for route, results, targets in (
        ("Route I", route1_results, ROUTE1_TARGETS),
        ("Route II", route2_results, ROUTE2_TARGETS),
    ):
        for target in targets:
            matrix = results[target][0]
            if route == "Route I":
                grouped = np.column_stack([matrix[:, 0], matrix[:, 1:].sum(axis=1)])
                names = ("Geometry / image features", "Numerical features")
            else:
                temporal_start = len(STATIC_LABELS)
                grouped = np.column_stack(
                    [matrix[:, 0], matrix[:, 1:temporal_start].sum(axis=1), matrix[:, temporal_start:].sum(axis=1)]
                )
                names = ("Geometry / image features", "Numerical features", "Time-series features")
            importance = np.mean(np.abs(grouped), axis=0)
            for name, value in zip(names, importance):
                rows.append(
                    {
                        "route": route,
                        "target": target,
                        "modality": name,
                        "mean_abs_group_SHAP": float(value),
                        "share_percent": float(value / importance.sum() * 100.0),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["route", "modality"], as_index=False)["share_percent"]
        .mean()
        .rename(columns={"share_percent": "average_share_percent"})
    )
    detail.to_csv(OUT / "corrected_modality_SHAP_by_target.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "corrected_modality_SHAP_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def draw_modality_pies(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.23, wspace=0.30)
    panels = (
        ("Route I", "(a) Route I: annual prediction", ("Geometry / image features", "Numerical features")),
        ("Route II", "(b) Route II: hourly prediction", ("Geometry / image features", "Numerical features", "Time-series features")),
    )
    colors = (BLUE, "#4c8f35", ORANGE)
    for ax, (route, title, order) in zip(axes, panels):
        subset = summary[summary.route == route].set_index("modality")
        values = [float(subset.loc[name, "average_share_percent"]) for name in order]
        ax.pie(
            values,
            colors=colors[: len(values)],
            startangle=90,
            counterclock=False,
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            pctdistance=0.70,
            wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
            textprops={"color": INK, "fontsize": 10, "fontweight": "bold"},
        )
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.set_aspect("equal")
    handles = [mpl.patches.Patch(facecolor=color, edgecolor="none", label=label) for color, label in zip(colors, panels[1][2])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085), frameon=False, ncol=3)
    fig.suptitle("Corrected modality-level SHAP contribution", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.5, 0.035, "Target-normalized mean absolute grouped SHAP values, averaged across the three outputs.", ha="center", color="#667085", fontsize=9)
    fig.savefig(OUT / "Corrected_SHAP_Modality_Contribution_Pies.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Corrected_SHAP_Modality_Contribution_Pies.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    seed_all(SEED)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )

    _, split, continuous, category = load_table_inputs()
    train = np.asarray(split["train"], dtype=int)
    test = np.asarray(split["test"], dtype=int)
    voxels = np.load(r1.VOXELS, mmap_mode="r")

    route1_model = r1.Route1(len(r1.CONT), len(split["categories"]))
    route1_model.load_state_dict(torch.load(r1.OUT / "best_route1_3dcnn_tabtransformer.pt", weights_only=True, map_location="cpu"))
    route1_model.eval()
    target_scaler = np.load(r1.OUT / "target_scaler.npz")

    rng = np.random.default_rng(SEED)
    explain_indices = rng.choice(test, size=min(96, len(test)), replace=False)
    background_indices = rng.choice(train, size=24, replace=False)
    geometry_samples = encode_geometry(route1_model, voxels, explain_indices)
    geometry_background = encode_geometry(route1_model, voxels, background_indices)
    continuous_samples = torch.from_numpy(continuous[explain_indices])
    continuous_background = torch.from_numpy(continuous[background_indices])
    with torch.no_grad():
        category_samples = route1_model.tabular.cat_embed(torch.from_numpy(category[explain_indices]))
        category_background = route1_model.tabular.cat_embed(torch.from_numpy(category[background_indices]))

    route1_results = {}
    static_values = group_static_values(
        geometry_samples.numpy(), continuous_samples.numpy(), category_samples.numpy()
    )
    for target_index, target in enumerate(ROUTE1_TARGETS):
        wrapper = AnnualWrapper(route1_model, target_index, target_scaler["mean"][target_index], target_scaler["std"][target_index])
        attribution = expected_gradients(
            wrapper,
            [geometry_samples, continuous_samples, category_samples],
            [geometry_background, continuous_background, category_background],
            steps=20,
            seed=SEED + target_index,
        )
        route1_results[target] = (group_static_shap(attribution), static_values)

    route1_importance = save_shap_figure(
        route1_results,
        STATIC_LABELS,
        ROUTE1_TARGETS,
        "Corrected_SHAP_Route_I_EUI_Epv_CEI",
        "Route I SHAP analysis: 3D-CNN–TabTransformer annual prediction",
    )
    route1_importance.insert(0, "route", "Route I")
    route1_importance.to_csv(OUT / "corrected_SHAP_importance_route1.csv", index=False, encoding="utf-8-sig")
    save_reference_style_figure(
        route1_results,
        STATIC_LABELS,
        ROUTE1_TARGETS,
        "Corrected_SHAP_Route_I_Reference_Style",
        "Route I feature importance and SHAP analysis",
        top_n=9,
    )

    valid = np.load(r2.CACHE / "valid.npy")
    valid_train = train[valid[train]]
    valid_test = test[valid[test]]
    route2_model = r2.HourlyLSTM()
    route2_model.load_state_dict(torch.load(r2.OUT / "best_route2_lstm.pt", weights_only=True, map_location="cpu"))
    route2_model.eval()
    scalers = np.load(r2.OUT / "scalers.npz")
    weather = np.load(r2.CACHE / "weather.npy", mmap_mode="r")

    # Sixteen buildings × four seasons; each sample uses the preceding 72 h and predicts the final hour.
    building_samples = rng.choice(valid_test, size=min(16, len(valid_test)), replace=False)
    seasonal_ends = np.asarray([720, 2904, 5088, 7272], dtype=int)
    sample_buildings = np.repeat(building_samples, len(seasonal_ends))
    sample_ends = np.tile(seasonal_ends, len(building_samples))
    background_buildings = rng.choice(valid_train, size=24, replace=False)
    background_ends = rng.integers(72, 8760, size=len(background_buildings))

    def weather_windows(buildings, ends):
        output = []
        for building, end in zip(buildings, ends):
            value = np.asarray(weather[building, end - 72 : end], dtype=np.float32)
            output.append((value - scalers["input_mean"]) / scalers["input_std"])
        return torch.from_numpy(np.stack(output).astype(np.float32))

    hourly_geometry_samples = encode_geometry(route1_model, voxels, sample_buildings)
    hourly_geometry_background = encode_geometry(route1_model, voxels, background_buildings)
    hourly_continuous_samples = torch.from_numpy(continuous[sample_buildings])
    hourly_continuous_background = torch.from_numpy(continuous[background_buildings])
    with torch.no_grad():
        hourly_category_samples = route1_model.tabular.cat_embed(torch.from_numpy(category[sample_buildings]))
        hourly_category_background = route1_model.tabular.cat_embed(torch.from_numpy(category[background_buildings]))
    weather_samples = weather_windows(sample_buildings, sample_ends)
    weather_background = weather_windows(background_buildings, background_ends)

    hourly_values = group_hourly_values(
        hourly_geometry_samples.numpy(),
        hourly_continuous_samples.numpy(),
        hourly_category_samples.numpy(),
        weather_samples.numpy(),
    )
    route2_results = {}
    for target_index, target in enumerate(ROUTE2_TARGETS):
        wrapper = HourlyWrapper(
            route1_model,
            route2_model,
            target_index,
            scalers["label_mean"][target_index],
            scalers["label_std"][target_index],
        )
        attribution = expected_gradients(
            wrapper,
            [hourly_geometry_samples, hourly_continuous_samples, hourly_category_samples, weather_samples],
            [hourly_geometry_background, hourly_continuous_background, hourly_category_background, weather_background],
            steps=20,
            seed=SEED + 100 + target_index,
        )
        route2_results[target] = (group_hourly_shap(attribution), hourly_values)

    route2_labels = STATIC_LABELS + TEMPORAL_LABELS
    route2_importance = save_shap_figure(
        route2_results,
        route2_labels,
        ROUTE2_TARGETS,
        "Corrected_SHAP_Route_II_Energy_PV_Carbon",
        "Route II SHAP analysis: 3D-CNN–TabTransformer–LSTM hourly prediction",
    )
    route2_importance.insert(0, "route", "Route II")
    route2_importance.to_csv(OUT / "corrected_SHAP_importance_route2.csv", index=False, encoding="utf-8-sig")
    save_reference_style_figure(
        route2_results,
        route2_labels,
        ROUTE2_TARGETS,
        "Corrected_SHAP_Route_II_Reference_Style",
        "Route II feature importance and SHAP analysis",
        top_n=9,
    )

    summary = modality_summary(route1_results, route2_results)
    draw_modality_pies(summary)
    print(summary.to_string(index=False))
    print("outputs", OUT)


if __name__ == "__main__":
    main()
