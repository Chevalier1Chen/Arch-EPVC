from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\pt251")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\pt251\torch\lib")

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import retrain_leakage_free_routes as core


def building_split(groups):
    indices = np.arange(len(groups))
    train_val, test = train_test_split(indices, test_size=.15, random_state=core.SEED)
    train, val = train_test_split(train_val, test_size=.1764705882, random_state=core.SEED + 1)
    return np.asarray(train, int), np.asarray(val, int), np.asarray(test, int)


def main():
    core.seed_all(core.SEED)
    core.torch.set_num_threads(min(16, core.torch.get_num_threads()))
    core.SPLIT_POLICY = "building-random; external 61 campuses fully excluded"
    core.R1_OUT = core.OUT / "route1_building_split"
    core.R1_OUT.mkdir(parents=True, exist_ok=True)
    core.group_split = building_split
    frame = core.load_training_frame()
    core.prepare_route2_cache(frame)
    frame = core.add_annual_climate(frame)
    core.train_route1(frame)


if __name__ == "__main__":
    main()
