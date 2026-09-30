"""Extract the Arch-EPVC tabular model inputs from a cleaned building OBJ.

The cleaned OBJ must use metres, with X/Y on the ground plane and Z vertical.
Geometry-derived fields are calculated from the convex footprint. Construction
year, city and enclosure class remain explicit project inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        raise ValueError("OBJ footprint has fewer than three unique XY vertices")

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_metrics(hull: np.ndarray) -> tuple[float, float]:
    shifted = np.roll(hull, -1, axis=0)
    area = 0.5 * abs(float(np.sum(hull[:, 0] * shifted[:, 1] - shifted[:, 0] * hull[:, 1])))
    perimeter = float(np.linalg.norm(shifted - hull, axis=1).sum())
    return area, perimeter


def thermal_prior(year: int, shape: float) -> dict[str, float]:
    if year < 1986:
        roof, wall, ground, window = 1.0, 1.5, 0.65, 5.7
    elif year < 2006:
        roof, wall, ground, window = 0.7, 1.0, 0.55, 3.5
    elif year < 2016:
        roof, wall, ground, window = 0.45, 0.55, 0.4, 2.5
    else:
        roof, wall, ground, window = 0.3, 0.4, 0.3, 1.8
    factor = 0.92 if shape > 0.35 else 0.96 if shape > 0.25 else 1.0
    return {"roof": roof * factor, "wall": wall * factor, "ground": ground, "window": window * factor}


def read_vertices(path: Path) -> np.ndarray:
    vertices = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            values = line.split()[1:4]
            if len(values) == 3:
                vertices.append([float(value) for value in values])
    if len(vertices) < 6:
        raise ValueError("OBJ does not contain enough vertices for a closed building mass")
    return np.asarray(vertices, dtype=np.float64)


def extract(path: Path, city: str, year: int, floors: int, enclosure: int, pv_ratio: float, wwr: float) -> dict[str, object]:
    vertices = read_vertices(path)
    hull = convex_hull(vertices[:, :2])
    footprint, perimeter = polygon_metrics(hull)
    height = float(vertices[:, 2].max() - vertices[:, 2].min())
    if footprint <= 1 or height <= 2:
        raise ValueError("OBJ footprint area or height is outside the supported range")

    centered = hull - hull.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    minor = np.asarray([-major[1], major[0]])
    major_extent = float(np.ptp(centered @ major))
    minor_extent = float(np.ptp(centered @ minor))
    length, width = max(major_extent, minor_extent), min(major_extent, minor_extent)
    orientation = math.degrees(math.atan2(float(major[1]), float(major[0])))
    orientation = (orientation + 90) % 180 - 90
    floors = max(1, int(floors))
    floor_height = height / floors
    floor_area = footprint * floors
    shape = (2 * footprint + perimeter * height) / (footprint * height)
    roof_pv_area = footprint * max(0.0, min(float(pv_ratio), 1.0))
    thermal = thermal_prior(year, shape)

    return {
        "building_id": path.stem,
        "City": city,
        "A.Building area": floor_area,
        "B.Building footprint": footprint,
        "C.Building height": height,
        "D.Layer": floors,
        "E.Height": floor_height,
        "F.Building length": length,
        "G.Building width": width,
        "H.Orientation": orientation,
        "I.Enclosure method": enclosure,
        "J.Shape coefficient": shape,
        "K.Roof thermal coefficient": thermal["roof"],
        "L.Wall thermal coefficient": thermal["wall"],
        "M.Ground thermal coefficient": thermal["ground"],
        "N.Window U-value": thermal["window"],
        "construction_year_user": year,
        "屋顶光伏发电总面积": roof_pv_area,
        "roof_pv_area": roof_pv_area,
        "window_to_wall_ratio": wwr,
        "footprint_perimeter": perimeter,
        "feature_source": "cleaned_OBJ_geometry_plus_explicit_project_inputs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj", type=Path)
    parser.add_argument("--city", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--floors", required=True, type=int)
    parser.add_argument("--enclosure", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--pv-ratio", type=float, default=0.60)
    parser.add_argument("--wwr", type=float, default=0.30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.obj, args.city, args.year, args.floors, args.enclosure, args.pv_ratio, args.wwr)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
