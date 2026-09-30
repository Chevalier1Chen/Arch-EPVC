from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "thesis_analysis_figures"
OUT.mkdir(exist_ok=True)

SIM_XLSX = ROOT / "school-building-design-platform" / "source-data" / "numerical data.xlsx"
EXT_CSV = ROOT / "external_validation_150" / "prepared" / "validation_150_processed.csv"
OOF_CSV = ROOT / "external_validation_150" / "supplemental_experiment" / "oof_predictions.csv"

ORANGE = "#E99243"
BLUE = "#3979B9"
GREEN = "#68A357"
RED = "#D95F59"
PURPLE = "#7A65A8"
GRAY = "#666666"
LIGHT_GRAY = "#D8D8D8"

# Restrained blue-green pair for the dataset comparison.
NATURE_SIM = "#3E9651"  # muted green
NATURE_EXT = "#3C5488"  # deep blue
NATURE_GRID = "#ECECEC"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "grid.color": "#E8E8E8",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim = pd.read_excel(SIM_XLSX, sheet_name="selected_variables")
    ext = pd.read_csv(EXT_CSV)
    oof = pd.read_csv(OOF_CSV)
    sim = sim.dropna(subset=["Number"]).copy()
    return sim, ext, oof


def clean_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.12, 1.06, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def figure8_distributions(sim: pd.DataFrame, ext: pd.DataFrame) -> None:
    previous_family = mpl.rcParams["font.family"]
    previous_serif = mpl.rcParams["font.serif"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    panels = [
        ("A.Building area", "Building area (m²)"),
        ("B.Building footprint", "Building footprint (m²)"),
        ("C.Building height", "Building height (m)"),
        ("D.Layer", "Number of stories (-)"),
        ("E.Height", "Average story height (m)"),
        ("F.Building length", "Building length (m)"),
        ("G.Building width", "Building width (m)"),
        ("H.Orientation", "Building orientation (°)"),
        ("I.Enclosure method", "Enclosure type (-)"),
        ("J.Shape coefficient", "Shape coefficient (-)"),
        ("K.Roof thermal coefficient", "Roof U-value [W/(m²·K)]"),
        ("L.Wall thermal coefficient", "Wall U-value [W/(m²·K)]"),
        ("M.Ground thermal coefficient", "Ground U-value [W/(m²·K)]"),
        ("N.Window U-value", "Window U-value [W/(m²·K)]"),
        ("EUI", "EUI (kWh/m²·a)"),
        ("CEI", "CEI (kgCO2/m²·a)"),
    ]

    fig = plt.figure(figsize=(15.5, 12.2))
    outer = fig.add_gridspec(4, 4, hspace=0.52, wspace=0.19)
    axes: list[plt.Axes] = []
    box_axes: list[plt.Axes | None] = []
    for i, (key, _) in enumerate(panels):
        row, col = divmod(i, 4)
        if key == "I.Enclosure method":
            axes.append(fig.add_subplot(outer[row, col]))
            box_axes.append(None)
        else:
            cell = outer[row, col].subgridspec(2, 1, height_ratios=[0.25, 0.75], hspace=0.0)
            box_ax = fig.add_subplot(cell[0])
            hist_ax = fig.add_subplot(cell[1], sharex=box_ax)
            axes.append(hist_ax)
            box_axes.append(box_ax)
    letters = [chr(ord("a") + i) for i in range(16)]

    for i, ((key, title), ax, box_ax) in enumerate(zip(panels, axes, box_axes)):
        if key == "I.Enclosure method":
            sim_counts = sim[key].astype(str).value_counts(normalize=True)
            ext_counts = ext[key].astype(str).value_counts(normalize=True)
            svals = sim_counts.to_numpy() * 100
            evals = ext_counts.sort_index().to_numpy() * 100
            sx = np.arange(len(svals))
            vx = np.arange(len(evals)) + len(svals) + 0.8
            ax.bar(sx, svals, width=0.72, color=NATURE_SIM, alpha=0.26, edgecolor=NATURE_SIM, linewidth=0.75)
            ax.bar(vx, evals, width=0.72, color=NATURE_EXT, alpha=0.23, edgecolor=NATURE_EXT, linewidth=0.75)
            ticks = np.concatenate([sx, vx])
            labels = [f"S{j + 1}" for j in range(len(svals))] + [f"V{j + 1}" for j in range(len(evals))]
            for patch in ax.patches[: len(svals)]:
                patch.set_facecolor(NATURE_SIM)
                patch.set_alpha(0.26)
                patch.set_edgecolor(NATURE_SIM)
                patch.set_linewidth(0.75)
            for patch in ax.patches[len(svals) :]:
                patch.set_facecolor(NATURE_EXT)
                patch.set_alpha(0.23)
                patch.set_edgecolor(NATURE_EXT)
                patch.set_linewidth(0.75)
            ax.set_xticks(ticks, labels, rotation=45, ha="right")
            ax.axvline(len(svals) - 0.1, color=LIGHT_GRAY, linewidth=0.8)
        else:
            sim_key = "CEI-no PV system" if key == "CEI" else key
            sv = clean_numeric(sim[sim_key])
            ev = clean_numeric(ext[key])
            combined = np.concatenate([sv, ev])
            lo, hi = np.nanpercentile(combined, [1, 99])
            if not np.isfinite(lo + hi) or math.isclose(lo, hi):
                lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined) + 1)
            sv = np.clip(sv, lo, hi)
            ev = np.clip(ev, lo, hi)
            bins = np.linspace(lo, hi, 13)
            sw = np.full(len(sv), 100 / len(sv))
            ew = np.full(len(ev), 100 / len(ev))
            ax.hist(
                sv,
                bins=bins,
                weights=sw,
                color=NATURE_SIM,
                alpha=0.27,
                edgecolor=NATURE_SIM,
                linewidth=0.65,
            )
            ax.hist(
                ev,
                bins=bins,
                weights=ew,
                color=NATURE_EXT,
                alpha=0.22,
                edgecolor=NATURE_EXT,
                linewidth=0.75,
            )

            # Scale each fitted Gaussian PDF to the histogram's percentage units.
            curve_x = np.linspace(lo, hi, 300)
            bin_width = float(bins[1] - bins[0])
            curve_max = 0.0
            for values, color in ((sv, NATURE_SIM), (ev, NATURE_EXT)):
                mu = float(np.mean(values))
                sigma = float(np.std(values, ddof=1))
                if np.isfinite(sigma) and sigma > 1e-12:
                    density = np.exp(-0.5 * ((curve_x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                    curve_y = density * 100 * bin_width
                    curve_max = max(curve_max, float(np.max(curve_y)))
                    ax.plot(curve_x, curve_y, color=color, linewidth=0.85, solid_capstyle="round")

            hist_max = max(
                float(np.histogram(sv, bins=bins, weights=sw)[0].max()),
                float(np.histogram(ev, bins=bins, weights=ew)[0].max()),
            )
            ax.set_ylim(0, max(hist_max, curve_max) * 1.12)

            assert box_ax is not None
            for values, position, color in ((sv, 1.25, NATURE_SIM), (ev, 0.70, NATURE_EXT)):
                box_ax.boxplot(
                    values,
                    positions=[position],
                    widths=0.34,
                    orientation="horizontal",
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.55, "linewidth": 0.75},
                    medianprops={"color": "white", "linewidth": 1.0},
                    whiskerprops={"color": color, "linewidth": 0.7},
                    capprops={"color": color, "linewidth": 0.7},
                )
            box_ax.set_ylim(0.40, 1.55)
            box_ax.set_yticks([])
            box_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            box_ax.grid(False)
            for spine in box_ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#222222")
                spine.set_linewidth(0.90)

        ax.set_xlabel(f"({letters[i]}) {title}", labelpad=5)
        ax.set_ylabel("Sample proportion (%)")
        ax.grid(True, axis="y", color=NATURE_GRID, linewidth=0.55)
        for side, spine in ax.spines.items():
            spine.set_visible(box_ax is None or side != "top")
            spine.set_color("#222222")
            spine.set_linewidth(0.90)
        ax.tick_params(direction="out", length=2.5, width=0.6, colors="#333333")

    handles = [
        mpl.patches.Patch(facecolor=NATURE_SIM, alpha=0.35, edgecolor=NATURE_SIM, label=f"Simulation dataset (n={len(sim)})"),
        mpl.patches.Patch(facecolor=NATURE_EXT, alpha=0.30, edgecolor=NATURE_EXT, label=f"External validation dataset (n={len(ext)})"),
        Line2D([0], [0], color="#444444", linewidth=0.85, label="Fitted normal distribution"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.947), ncol=3, frameon=False)
    fig.suptitle("Figure 8. Variable Distributions of the Simulation and External Validation Datasets", fontsize=14, y=0.995)
    fig.text(0.5, 0.966, "Top boxplots show the median and interquartile range; continuous variables are truncated at the pooled 1st–99th percentiles and fitted separately with normal distributions; simulation CEI uses the no-PV scenario.", ha="center", fontsize=8, color=GRAY)
    fig.subplots_adjust(left=0.052, right=0.992, bottom=0.055, top=0.895)
    save_figure(fig, "Figure_8_dataset_distribution")
    mpl.rcParams["font.family"] = previous_family
    mpl.rcParams["font.serif"] = previous_serif


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def draw_accuracy_panel(
    ax: plt.Axes,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    unit: str,
    color: str,
    label: str,
) -> dict[str, float]:
    m = metrics(actual, predicted)
    lo = min(float(np.min(actual)), float(np.min(predicted)))
    hi = max(float(np.max(actual)), float(np.max(predicted)))
    pad = (hi - lo) * 0.07 if hi > lo else 1
    xx = np.linspace(lo - pad, hi + pad, 100)
    coef = np.polyfit(actual, predicted, 1)
    trend = np.polyval(coef, xx)
    residual = predicted - np.polyval(coef, actual)
    band = 1.96 * float(np.std(residual, ddof=1))

    ax.fill_between(xx, trend - band, trend + band, color=color, alpha=0.10, label="95%预测带")
    ax.plot(xx, trend, color=color, linewidth=1.4, label="回归线")
    ax.plot(xx, xx, color="#444444", linestyle="--", linewidth=1, label="1:1线")
    ax.scatter(actual, predicted, s=24, c=color, alpha=0.78, edgecolor="white", linewidth=0.35)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=7)
    ax.set_xlabel(f"实际值 ({unit})")
    ax.set_ylabel(f"预测值 ({unit})")
    ax.text(
        0.97,
        0.05,
        f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:,.3g}\nMAE = {m['MAE']:,.3g}\nn = {len(actual)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "boxstyle": "round,pad=0.25", "alpha": 0.92},
    )
    ax.text(0.03, 0.96, label, transform=ax.transAxes, va="top", fontsize=8, color=GRAY)
    style_axis(ax)
    return m


def figure9_accuracy(oof: pd.DataFrame) -> pd.DataFrame:
    pv = oof["Epv"] > 0
    series = [
        (
            clean_numeric(oof["EUI"]),
            clean_numeric(oof["OOF Predicted EUI"]),
            "(a) EUI预测精度",
            "kWh/m²·a",
            BLUE,
            "外部验证集 · 5折OOF",
            "EUI",
        ),
        (
            clean_numeric(oof.loc[pv, "Epv"]),
            clean_numeric(oof.loc[pv, "OOF Predicted Epv"]),
            "(b) 光伏发电量预测精度",
            "kWh/a",
            GREEN,
            "已安装光伏样本 · 5折OOF",
            "Epv",
        ),
        (
            clean_numeric(oof["CEI"]),
            clean_numeric(oof["Physics Predicted CEI"]),
            "(c) CEI预测精度",
            "kgCO2/m²·a",
            ORANGE,
            "外部验证集 · 物理约束融合",
            "CEI",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.75))
    rows = []
    for ax, (a, p, title, unit, color, label, target) in zip(axes, series):
        m = draw_accuracy_panel(ax, a, p, title, unit, color, label)
        rows.append({"target": target, "n": len(a), **m})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.suptitle("图9  预测模型精度", fontsize=14, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.98), w_pad=2.0)
    save_figure(fig, "Figure_9_prediction_accuracy")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "Figure_9_metrics.csv", index=False, encoding="utf-8-sig")
    return result


def figure10_errors(oof: pd.DataFrame) -> pd.DataFrame:
    pv = oof["Epv"] > 0
    configs = [
        ("EUI", oof["EUI"].to_numpy(float), oof["OOF Predicted EUI"].to_numpy(float), BLUE, "kWh/m²·a"),
        ("Epv", oof.loc[pv, "Epv"].to_numpy(float), oof.loc[pv, "OOF Predicted Epv"].to_numpy(float), GREEN, "kWh/a"),
        ("CEI", oof["CEI"].to_numpy(float), oof["Physics Predicted CEI"].to_numpy(float), ORANGE, "kgCO2/m²·a"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.0))
    rows = []
    for j, (name, actual, pred, color, unit) in enumerate(configs):
        residual = pred - actual
        rmse = float(np.sqrt(np.mean(residual**2)))
        ape = np.abs(residual) / np.maximum(np.abs(actual), 1e-9) * 100
        order = np.argsort(ape)
        mean_ape = float(np.mean(ape))
        median_ape = float(np.median(ape))

        ax = axes[0, j]
        ax.scatter(pred, residual, s=23, c=color, alpha=0.72, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.axhline(rmse, color=RED, linestyle="--", linewidth=1)
        ax.axhline(-rmse, color=RED, linestyle="--", linewidth=1)
        ax.set_title(f"({chr(97 + j)}) {name}残差分布")
        ax.set_xlabel(f"预测值 ({unit})")
        ax.set_ylabel(f"残差：预测−实际 ({unit})")
        ax.text(0.97, 0.95, f"均值={np.mean(residual):,.3g}\nRMSE={rmse:,.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        style_axis(ax)

        ax = axes[1, j]
        ranks = np.arange(1, len(ape) + 1)
        ax.bar(ranks, ape[order], width=0.95, color=color, alpha=0.73)
        ax.axhline(mean_ape, color=RED, linestyle="--", linewidth=1.1, label=f"平均APE={mean_ape:.1f}%")
        ax.axhline(median_ape, color=PURPLE, linestyle=":", linewidth=1.2, label=f"中位APE={median_ape:.1f}%")
        if name == "Epv" and np.max(ape) / max(np.median(ape), 0.1) > 30:
            ax.set_yscale("symlog", linthresh=10)
        ax.set_title(f"({chr(100 + j)}) {name}绝对百分比误差")
        ax.set_xlabel("按误差升序排列的外部样本")
        ax.set_ylabel("绝对百分比误差 (%)")
        ax.legend(loc="upper left", frameon=False)
        style_axis(ax, grid_axis="y")
        rows.append(
            {
                "target": name,
                "n": len(actual),
                "mean_residual": float(np.mean(residual)),
                "RMSE": rmse,
                "mean_APE_percent": mean_ape,
                "median_APE_percent": median_ape,
                "p95_APE_percent": float(np.percentile(ape, 95)),
            }
        )
    fig.suptitle("图10  外部验证误差分析", fontsize=14, y=1.045)
    fig.text(0.5, 1.01, "Epv误差仅统计实际发电量大于0的27个光伏样本", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.0, w_pad=1.6)
    save_figure(fig, "Figure_10_error_validation")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "Figure_10_error_summary.csv", index=False, encoding="utf-8-sig")
    return result


FEATURES = [
    "A.Building area",
    "B.Building footprint",
    "C.Building height",
    "D.Layer",
    "E.Height",
    "F.Building length",
    "G.Building width",
    "H.Orientation",
    "I.Enclosure method",
    "J.Shape coefficient",
    "K.Roof thermal coefficient",
    "L.Wall thermal coefficient",
    "M.Ground thermal coefficient",
    "N.Window U-value",
]

FEATURE_LABELS = {
    "A.Building area": "建筑面积",
    "B.Building footprint": "建筑占地面积",
    "C.Building height": "建筑高度",
    "D.Layer": "层数",
    "E.Height": "平均层高",
    "F.Building length": "建筑长度",
    "G.Building width": "建筑宽度",
    "H.Orientation": "朝向",
    "I.Enclosure method": "围合方式",
    "J.Shape coefficient": "体形系数",
    "K.Roof thermal coefficient": "屋面传热系数",
    "L.Wall thermal coefficient": "外墙传热系数",
    "M.Ground thermal coefficient": "地面传热系数",
    "N.Window U-value": "外窗传热系数",
}


def build_feature_matrix(sim: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    numeric = [f for f in FEATURES if f != "I.Enclosure method"]
    x_num = sim[numeric].apply(pd.to_numeric, errors="coerce")
    x_num = x_num.fillna(x_num.median())
    enc = pd.get_dummies(sim["I.Enclosure method"].astype(str), prefix="ENC", dtype=float)
    X = pd.concat([x_num, enc], axis=1)
    group_map = {f: f for f in numeric}
    group_map.update({col: "I.Enclosure method" for col in enc.columns})
    return X, group_map


def fit_importance(sim: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    X, group_map = build_feature_matrix(sim)
    targets = {
        "EUI": (pd.to_numeric(sim["EUI"], errors="coerce"), False),
        "Epv": (pd.to_numeric(sim["Epv"], errors="coerce"), True),
        "CEI-PV": (pd.to_numeric(sim["CEI-PV system"], errors="coerce"), False),
    }
    all_rows = []
    scores = {}
    for target_name, (y_raw, use_log) in targets.items():
        valid = y_raw.notna() & np.isfinite(y_raw)
        y = y_raw.loc[valid].to_numpy(float)
        if use_log:
            y = np.log1p(np.clip(y, 0, None))
        x = X.loc[valid]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = ExtraTreesRegressor(
            n_estimators=280,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=42,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        score = float(r2_score(y_test, model.predict(x_test)))
        scores[target_name] = score
        grouped: dict[str, float] = {f: 0.0 for f in FEATURES}
        for col, value in zip(x.columns, model.feature_importances_):
            grouped[group_map[col]] += float(value)
        total = sum(grouped.values())
        for feature, value in grouped.items():
            all_rows.append(
                {
                    "target": target_name,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "importance": value,
                    "importance_percent": 100 * value / total if total else 0,
                    "heldout_R2": score,
                }
            )
    return pd.DataFrame(all_rows), scores


def figure11_importance(sim: pd.DataFrame) -> pd.DataFrame:
    imp, scores = fit_importance(sim)
    targets = [("EUI", BLUE), ("Epv", GREEN), ("CEI-PV", ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.3), sharex=True)
    for j, (target, color) in enumerate(targets):
        ax = axes[j]
        d = imp[imp["target"] == target].sort_values("importance_percent", ascending=True)
        ax.barh(d["feature_label"], d["importance_percent"], color=color, alpha=0.8)
        for y_pos, value in enumerate(d["importance_percent"]):
            if value >= 1.2:
                ax.text(value + 0.35, y_pos, f"{value:.1f}%", va="center", fontsize=7.5)
        target_title = "光伏发电量（log）" if target == "Epv" else target.replace("-", "（") + ("）" if "-" in target else "")
        ax.set_title(f"({chr(97 + j)}) {target_title}\n留出集 R²={scores[target]:.3f}")
        ax.set_xlabel("归一化特征重要性 (%)")
        if j == 0:
            ax.set_ylabel("输入特征")
        style_axis(ax, grid_axis="x")
    fig.suptitle("图11  特征重要性分析", fontsize=14, y=1.07)
    fig.text(0.5, 1.025, "基于模拟数据训练的ExtraTrees表格代理模型；围合方式的独热编码重要性已合并", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.6)
    save_figure(fig, "Figure_11_feature_importance")
    imp.to_csv(OUT / "Figure_11_feature_importance.csv", index=False, encoding="utf-8-sig")
    return imp


def select_representative_cases(sim: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Number",
        "A.Building area",
        "D.Layer",
        "H.Orientation",
        "J.Shape coefficient",
        "Epv",
        "EUI",
        "CEI-no PV system",
        "CEI-PV system",
        "CRR",
    ]
    d = sim[cols].copy()
    for col in cols[1:]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna().sort_values("A.Building area")
    d["area_group"] = pd.qcut(d["A.Building area"], q=5, labels=False, duplicates="drop")
    selected = []
    for group, g in d.groupby("area_group"):
        median_area = g["A.Building area"].median()
        idx = (g["A.Building area"] - median_area).abs().idxmin()
        selected.append(d.loc[idx])
    cases = pd.DataFrame(selected).sort_values("A.Building area").reset_index(drop=True)
    cases["方案"] = [f"方案{i + 1}" for i in range(len(cases))]
    cases["CEI降幅"] = cases["CEI-no PV system"] - cases["CEI-PV system"]
    cases["CEI降幅百分比"] = cases["CEI降幅"] / cases["CEI-no PV system"] * 100
    return cases


def figure12_optimization(sim: pd.DataFrame) -> pd.DataFrame:
    cases = select_representative_cases(sim)
    x = np.arange(len(cases))
    width = 0.35
    fig = plt.figure(figsize=(14.8, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.35], hspace=0.26)
    ax = fig.add_subplot(gs[0])
    no_pv = cases["CEI-no PV system"].to_numpy(float)
    with_pv = cases["CEI-PV system"].to_numpy(float)
    ax.bar(x - width / 2, no_pv, width, color=ORANGE, alpha=0.86, label="无光伏优化值")
    ax.bar(x + width / 2, with_pv, width, color=BLUE, alpha=0.86, label="加光伏后优化值")
    for i, (a, b, reduction) in enumerate(zip(no_pv, with_pv, cases["CEI降幅百分比"])):
        ax.text(i - width / 2, a + max(no_pv) * 0.018, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, b + max(no_pv) * 0.018, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
        y = max(a, b) + max(no_pv) * 0.115
        ax.annotate(
            f"降低 {reduction:.1f}%",
            xy=(i + width / 2, b),
            xytext=(i, y),
            ha="center",
            fontsize=8,
            color=GREEN,
            arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 0.9},
        )
    ax.set_xticks(x, [f"{row['方案']}\n{str(row['Number'])[:18]}" for _, row in cases.iterrows()])
    ax.set_ylabel("碳排放强度 CEI (kgCO2/m²·a)")
    ax.set_title("(a) 五个代表设计方案的无光伏/有光伏情景对比")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ymax = max(no_pv) * 1.32
    ax.set_ylim(0, ymax)
    style_axis(ax, grid_axis="y")

    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")
    table_rows = []
    for _, row in cases.iterrows():
        table_rows.append(
            [
                row["方案"],
                str(row["Number"]),
                f"{row['A.Building area']:,.0f}",
                f"{row['D.Layer']:.0f}",
                f"{row['H.Orientation']:.1f}",
                f"{row['J.Shape coefficient']:.3f}",
                f"{row['Epv']:,.0f}",
                f"{row['CEI降幅']:.2f}",
            ]
        )
    headers = ["方案", "样本编号", "建筑面积(m²)", "层数", "朝向(°)", "体形系数", "光伏发电量(kWh/a)", "CEI绝对降幅"]
    table = ax_table.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DADADA")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#EEF2F5")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F8F8F8")
    ax_table.set_title("(b) 代表方案关键参数与优化收益", pad=5)

    fig.suptitle("图12  优化设计案例：光伏配置前后数值对比", fontsize=14, y=1.045)
    fig.text(0.5, 1.005, "各方案按建筑面积五分位选取最接近组内中位数的模拟样本，避免只展示极端优选案例", ha="center", fontsize=8, color=GRAY)
    save_figure(fig, "Figure_12_optimization_cases")
    cases.to_csv(OUT / "Figure_12_selected_cases.csv", index=False, encoding="utf-8-sig")
    return cases


def write_summary(
    sim: pd.DataFrame,
    ext: pd.DataFrame,
    accuracy: pd.DataFrame,
    errors: pd.DataFrame,
    importance: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    summary = {
        "source_files": {
            "simulation": str(SIM_XLSX.relative_to(ROOT)),
            "external_validation": str(EXT_CSV.relative_to(ROOT)),
            "oof_predictions": str(OOF_CSV.relative_to(ROOT)),
        },
        "sample_counts": {"simulation": int(len(sim)), "external_validation": int(len(ext))},
        "figure9_metrics": accuracy.to_dict(orient="records"),
        "figure10_errors": errors.to_dict(orient="records"),
        "figure11_top_features": (
            importance.sort_values(["target", "importance_percent"], ascending=[True, False])
            .groupby("target")
            .head(5)[["target", "feature_label", "importance_percent", "heldout_R2"]]
            .to_dict(orient="records")
        ),
        "figure12_cases": cases[
            ["方案", "Number", "A.Building area", "CEI-no PV system", "CEI-PV system", "CEI降幅百分比"]
        ].to_dict(orient="records"),
    }
    (OUT / "figure_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    setup_style()
    sim, ext, oof = load_data()
    figure8_distributions(sim, ext)
    accuracy = figure9_accuracy(oof)
    errors = figure10_errors(oof)
    importance = figure11_importance(sim)
    cases = figure12_optimization(sim)
    write_summary(sim, ext, accuracy, errors, importance, cases)
    print(f"Generated figures in: {OUT}")


if __name__ == "__main__":
    main()
