"""Export the two trained Arch-EPVC routes for in-browser inference.

The exported Route I returns both the normalized annual prediction and the
256-dimensional fused representation.  Route II consumes that representation,
the three normalized annual anchors and an 8,760 x 10 EPW sequence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
TORCH_ROOT = Path(r"C:\pt251")
sys.path.insert(0, str(TORCH_ROOT))
sys.path.insert(0, str(ROOT / ".onnx_deps"))
sys.path.insert(0, str(ROOT / "external_validation_61_final" / ".python_deps"))
sys.path.insert(0, str(ROOT / "internal_two_route_rebuild"))
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(TORCH_ROOT / "torch" / "lib"))

import joblib
import numpy as np
import torch
import torch.nn as nn

import train_route1 as route1_module
import train_route2_lstm as route2_module


ROUTE1_DIR = ROOT / "leakage_free_retraining" / "route1"
ROUTE2_DIR = ROOT / "leakage_free_retraining" / "route2_second_attempt_no_city_no_climate"
OUT = PROJECT / "public" / "models" / "browser"


class Route1Browser(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, voxel, continuous, category):
        prediction, embedding = self.model(voxel, continuous, category, True)
        return prediction, embedding


class Route2Browser(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, static, weather):
        prediction, _ = self.model(static, weather)
        return prediction


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROUTE1_DIR / "split.json").read_text(encoding="utf-8"))
    features = split["features"]
    categories = split["categories"]

    annual = route1_module.Route1(len(features), len(categories))
    annual.load_state_dict(torch.load(ROUTE1_DIR / "best_route1_3dcnn_tabtransformer.pt", weights_only=True, map_location="cpu"))
    annual.eval()
    annual_browser = Route1Browser(annual).eval()

    torch.onnx.export(
        annual_browser,
        (torch.zeros(1, 1, 32, 32, 32), torch.zeros(1, len(features)), torch.zeros(1, dtype=torch.long)),
        OUT / "route1-annual.onnx",
        input_names=["voxel", "continuous", "category"],
        output_names=["annual_scaled", "shared_embedding"],
        opset_version=17,
        do_constant_folding=True,
    )

    hourly = route2_module.HourlyLSTM()
    hourly.load_state_dict(torch.load(ROUTE2_DIR / "best_route2_lstm.pt", weights_only=True, map_location="cpu"))
    hourly.eval()
    hourly_browser = Route2Browser(hourly).eval()
    torch.onnx.export(
        hourly_browser,
        (torch.zeros(1, 259), torch.zeros(1, 8760, 10)),
        OUT / "route2-hourly.onnx",
        input_names=["static", "weather"],
        output_names=["hourly_scaled"],
        dynamic_axes={"weather": {1: "hours"}, "hourly_scaled": {1: "hours"}},
        opset_version=17,
        do_constant_folding=True,
    )

    continuous_scaler = joblib.load(ROUTE1_DIR / "continuous_scaler.joblib")
    target_scaler = np.load(ROUTE1_DIR / "target_scaler.npz")
    hourly_scaler = np.load(ROUTE2_DIR / "scalers.npz")
    payload = {
        "voxel_size": 32,
        "voxel_margin": 0.08,
        "features": features,
        "categories": categories,
        "continuous_mean": continuous_scaler.mean_.tolist(),
        "continuous_std": continuous_scaler.scale_.tolist(),
        "target_mean": target_scaler["mean"].tolist(),
        "target_std": target_scaler["std"].tolist(),
        "target_epv_log": True,
        "hourly_input_mean": hourly_scaler["input_mean"].tolist(),
        "hourly_input_std": hourly_scaler["input_std"].tolist(),
        "hourly_label_mean": hourly_scaler["label_mean"].tolist(),
        "hourly_label_std": hourly_scaler["label_std"].tolist(),
        "route1_checkpoint": "leakage_free_retraining/route1",
        "route2_checkpoint": "leakage_free_retraining/route2_second_attempt_no_city_no_climate",
    }
    (OUT / "preprocessing.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported browser model package to {OUT}")


if __name__ == "__main__":
    main()
