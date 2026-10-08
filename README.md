# Arch-EPVC School Building Dataset

This repository distributes data only: numerical building attributes, annual simulation indicators, OBJ geometry and building-level hourly simulation data. Platform source, training source, trained weights, figures, presentation slides and model-evaluation result packages are not included in the current version.

## Download

- [Dataset download page](https://chevalier1chen.github.io/Arch-EPVC/)
- [Full hourly data: 3,356 workbooks in seven independent ZIP archives](https://github.com/Chevalier1Chen/Arch-EPVC/releases/tag/v0.1.0)
- [Numerical data](data/processed/)
- [OBJ geometry data](data/geometry/teaching_building_OBJ_summary_3356.zip)
- [Hourly workbook examples](data/samples/time_series/)
- [Building-to-archive index and workbook checksums](data/metadata/hourly_archive_manifest.csv)
- [Hourly archive checksums](data/metadata/archive_checksums.csv)
- [Variable descriptions](data/metadata/variable_dictionary.csv)

Each hourly ZIP can be extracted separately. To assemble the full collection, extract all seven into the same directory. Numerical workbooks and geometry index tables are retained as dataset records, not as paper presentation tables.

This publication update changes the distributed materials, not the underlying data values. It is not a new numerical-quality audit or an independent verification of simulation accuracy or 8,760-hour completeness. No new reuse license is granted by this update.
