from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "leakage_free_retraining"
OUT = BASE / "validation_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})


def panel(ax, y, p, label, unit, n_label=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    pad = max((hi - lo) * 0.04, 1e-9)
    ax.scatter(y, p, s=10, c="#74add1", alpha=0.62, edgecolors="none")
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], ls="--", lw=1.4, c="#e6550d")
    ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
    ax.set_xlabel(f"Actual {label} ({unit})")
    ax.set_ylabel(f"Predicted {label} ({unit})")
    r2 = r2_score(y, p); rmse = mean_squared_error(y, p) ** 0.5; mae = mean_absolute_error(y, p)
    n_text = n_label if n_label is not None else len(y)
    ax.text(0.035, 0.965, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:,.2f}\nMAE = {mae:,.2f}\nn = {n_text:,}",
            transform=ax.transAxes, va="top", ha="left")
    ax.grid(True, color="#dddddd", lw=0.55, alpha=0.7)
    for spine in ax.spines.values(): spine.set_linewidth(0.8)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def route1():
    d = pd.read_csv(BASE / "route1" / "internal_test_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
    specs = [("EUI", "kWh/m²·year"), ("Epv", "kWh/year"), ("CEI", "kgCO₂/m²·year")]
    for ax, (name, unit) in zip(axes, specs): panel(ax, d[f"true_{name}"], d[f"pred_{name}"], name, unit)
    fig.suptitle("Route I: 3D-CNN–TabTransformer Annual Prediction (Campus-Grouped Internal Test Set)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "Route_I_internal_test_scatter")


def route2():
    p = BASE / "route2_second_attempt_no_city_no_climate" / "internal_test_predictions_8760x3.npz"
    a = np.load(p); true, pred = a["true"], a["prediction"]
    rng = np.random.default_rng(42); total = true.shape[0] * true.shape[1]
    sample = rng.choice(total, size=min(30000, total), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
    specs = [("Energy", "kWh"), ("PV generation", "kWh"), ("Carbon emission", "kgCO₂")]
    for j, (ax, (name, unit)) in enumerate(zip(axes, specs)):
        panel(ax, true[:, :, j].reshape(-1)[sample], pred[:, :, j].reshape(-1)[sample], name, unit, n_label=total)
    fig.suptitle("Route II: 3D-CNN–TabTransformer–LSTM Hourly Prediction (Campus-Grouped Internal Test Set)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "Route_II_internal_test_scatter")


def external():
    d = pd.read_csv(BASE / "external_61_true_model_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.65))
    panel(axes[0], d["Observed EUI"], d["Model predicted EUI"], "EUI", "kWh/m²·year")
    panel(axes[1], d["Observed CEI"], d["Model predicted CEI"], "CEI", "kgCO₂/m²·year")
    fig.suptitle("External Validation: 61 Fully Excluded School Campuses", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .93)); save(fig, "External_61_scatter_true_model")

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.45))
    for ax, name, unit in [(axes[0], "EUI", "kWh/m²·year"), (axes[1], "CEI", "kgCO₂/m²·year")]:
        observed = d[f"Observed {name}"].to_numpy(float); predicted = d[f"Model predicted {name}"].to_numpy(float)
        residual = predicted - observed
        ax.axhline(0, ls="--", lw=1.2, c="#e6550d")
        ax.scatter(observed, residual, s=18, c="#74add1", alpha=.72, edgecolors="none")
        ax.set_xlabel(f"Observed {name} ({unit})"); ax.set_ylabel(f"Residual ({unit})")
        ax.grid(True, color="#dddddd", lw=.55, alpha=.7)
    fig.suptitle("External Validation Residual Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .92)); save(fig, "External_61_residuals_true_model")


def summary():
    tables = []
    for path, variant in [
        (BASE / "route1" / "internal_test_metrics.csv", "Route I + annual climate summaries"),
        (BASE / "route1_second_attempt_no_city_no_climate" / "internal_test_metrics.csv", "Route I without climate summaries"),
        (BASE / "route2_second_attempt_no_city_no_climate" / "internal_test_metrics.csv", "Route II selected LSTM"),
        (BASE / "route2_lstm" / "internal_test_metrics.csv", "Route II climate-static rerun"),
        (BASE / "external_61_true_model_metrics.csv", "External 61; direct Route I model"),
        (BASE / "route1_boosted_multimodal_head" / "internal_test_metrics.csv", "Route I boosted-head comparison"),
        (BASE / "route1_boosted_multimodal_head" / "external_61_metrics.csv", "External 61; boosted-head comparison"),
    ]:
        d = pd.read_csv(path); d.insert(0, "variant", variant); tables.append(d)
    pd.concat(tables, ignore_index=True).to_csv(BASE / "all_validation_metrics.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "split_policy": "Campus-grouped 70/15/15; no campus overlap",
        "remaining_campuses": 931, "remaining_buildings": 3207,
        "external_campuses": 61,
        "selected_route1_checkpoint": str(BASE / "route1" / "best_route1_3dcnn_tabtransformer.pt"),
        "selected_route2_checkpoint": str(BASE / "route2_second_attempt_no_city_no_climate" / "best_route2_lstm.pt"),
        "warning": "External 61 EUI/CEI results are genuine model predictions and reveal substantial simulation-to-measurement domain shift.",
    }
    (BASE / "selected_model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    route1(); route2(); external(); summary()
