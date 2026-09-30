from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\pt251")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\pt251\torch\lib")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import retrain_leakage_free_routes as core


def main():
    core.seed_all(core.SEED)
    core.torch.set_num_threads(min(16, core.torch.get_num_threads()))
    r1 = core.OUT / "route1_building_split"
    core.R2_OUT = core.OUT / "route2_lstm_building_split"
    core.R2_OUT.mkdir(parents=True, exist_ok=True)
    frame = core.load_training_frame()
    core.prepare_route2_cache(frame)
    split = json.loads((r1 / "split.json").read_text(encoding="utf-8"))
    embedding = np.load(r1 / "static_shared_embedding_256.npy")
    annual_scaled = np.load(r1 / "annual_prediction_scaled_3.npy")
    core.train_route2(frame, split, embedding, annual_scaled)


if __name__ == "__main__":
    main()
