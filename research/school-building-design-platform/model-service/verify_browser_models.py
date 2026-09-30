"""Numerically compare exported ONNX models with their PyTorch checkpoints."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, r"C:\pt251")
sys.path.insert(0, str(ROOT / ".onnx_deps"))
sys.path.insert(0, str(ROOT / "internal_two_route_rebuild"))
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\pt251\torch\lib")

import numpy as np
import onnxruntime as ort
import torch

import train_route1 as r1
import train_route2_lstm as r2


def main() -> None:
    browser = PROJECT / "public" / "models" / "browser"
    metadata = json.loads((browser / "preprocessing.json").read_text(encoding="utf-8"))
    generator = np.random.default_rng(42)

    voxel = generator.integers(0, 2, size=(1, 1, 32, 32, 32)).astype(np.float32)
    continuous = generator.normal(size=(1, len(metadata["features"]))).astype(np.float32)
    category = np.asarray([0], dtype=np.int64)
    annual = r1.Route1(len(metadata["features"]), len(metadata["categories"]))
    annual.load_state_dict(torch.load(ROOT / "leakage_free_retraining" / "route1" / "best_route1_3dcnn_tabtransformer.pt", weights_only=True, map_location="cpu"))
    annual.eval()
    with torch.no_grad():
        torch_prediction, torch_embedding = annual(torch.from_numpy(voxel), torch.from_numpy(continuous), torch.from_numpy(category), True)
    session1 = ort.InferenceSession(str(browser / "route1-annual.onnx"), providers=["CPUExecutionProvider"])
    onnx_prediction, onnx_embedding = session1.run(None, {"voxel": voxel, "continuous": continuous, "category": category})

    static = np.concatenate([onnx_embedding, onnx_prediction], axis=1).astype(np.float32)
    weather = generator.normal(size=(1, 8760, 10)).astype(np.float32)
    hourly = r2.HourlyLSTM()
    hourly.load_state_dict(torch.load(ROOT / "leakage_free_retraining" / "route2_second_attempt_no_city_no_climate" / "best_route2_lstm.pt", weights_only=True, map_location="cpu"))
    hourly.eval()
    with torch.no_grad():
        torch_hourly, _ = hourly(torch.from_numpy(static), torch.from_numpy(weather))
    session2 = ort.InferenceSession(str(browser / "route2-hourly.onnx"), providers=["CPUExecutionProvider"])
    onnx_hourly = session2.run(None, {"static": static, "weather": weather})[0]

    errors = {
        "route1_prediction_max_abs": float(np.max(np.abs(torch_prediction.numpy() - onnx_prediction))),
        "route1_embedding_max_abs": float(np.max(np.abs(torch_embedding.numpy() - onnx_embedding))),
        "route2_hourly_max_abs": float(np.max(np.abs(torch_hourly.numpy() - onnx_hourly))),
    }
    print(json.dumps(errors, indent=2))
    if max(errors.values()) > 1e-4:
        raise SystemExit("ONNX export differs from PyTorch beyond tolerance")


if __name__ == "__main__":
    main()
