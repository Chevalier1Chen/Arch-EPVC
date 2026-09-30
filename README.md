# Arch-EPVC: School Building Energy, PV Generation and Carbon Prediction Dataset

[Open Arch-EPVC Platform](https://arch-epvc.vercel.app)

This release package provides a project page, research data, original training scripts, trained models and platform source for a school-building performance study in Shandong Province, China. Training scripts retain machine-specific paths; full independent reproduction has not been verified.

**Project scope.** The dataset and workflow support both early-stage design assessment for new school buildings and low-carbon retrofit evaluation for existing teaching buildings. The study integrates 3D geometry, structured building attributes, envelope thermal properties, weather time series and simulated performance indicators for annual and hourly prediction.

## What is included

- `data/processed/`: structured numerical data and annual performance indicators.
- `data/geometry/`: summarized OBJ geometry package for teaching buildings.
- `data/samples/time_series/`: sample hourly performance files. The full hourly data collection is indexed in `data/metadata/time_series_manifest.csv` and should be distributed through GitHub Releases, Git LFS, Zenodo, OSF or another large-file archive.
- `simulation/`: Grasshopper definitions and Rhino/GH helper scripts used for OBJ import, simulation mapping and geometry processing.
- `research/`: original training scripts, network definitions, the platform's paired PyTorch checkpoints, preprocessing parameters, and implemented platform source with ONNX models.
- `models/`: model release notes; earlier configuration templates are illustrative, not the authoritative training configuration.
- `platform/`: Arch-EPVC platform overview materials.
- `docs/`: GitHub Pages project page.

## Research tasks

1. Annual-scale prediction: fuse 3D geometry and structured numerical features to predict EUI, annual PV generation and carbon-emission indicators.
2. Hourly-scale prediction: fuse geometry, numerical features and weather/performance time series to predict 8760 h energy demand, PV generation and operational carbon emissions.
3. Low-carbon assessment: estimate PV self-consumption, grid purchase, carbon reduction and retrofit/design scenario performance.

## Data release note

The full hourly dataset is larger than a normal GitHub repository. This repository therefore includes sample hourly files and a complete manifest. For formal publication, attach the full time-series archive as GitHub Release assets or publish it on Zenodo/OSF and paste the DOI/link here.

Satellite basemap images are not included in the default public release unless their redistribution license is confirmed.

## Citation

If you use this repository, please cite the associated paper and this repository. A `CITATION.cff` file is provided and can be updated after the final title, authors and DOI are confirmed.
