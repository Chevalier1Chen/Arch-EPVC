# Research source and deployed-model provenance

This snapshot preserves the directory relationships used by the platform export script.

## Model entry points

- `internal_two_route_rebuild/train_route1.py`: Geometry3DCNN, TabTransformer and annual Route1 network.
- `internal_two_route_rebuild/train_route2_lstm.py`: HourlyLSTM network and window-based training utilities.
- `retrain_leakage_free_routes.py`: retraining and external annual evaluation entry point.
- `school-building-design-platform/model-service/export_browser_models.py`: authoritative checkpoint paths for the bundled browser models.
- `school-building-design-platform/model-service/verify_browser_models.py`: original PyTorch/ONNX comparison procedure.

The annual route produces three scaled outputs and a 256-dimensional shared representation. The hourly route consumes 259 static conditions and hourly weather/time features. Read the code and included preprocessing metadata for exact feature order and units.

## Included checkpoints

The `leakage_free_retraining/route1` and `leakage_free_retraining/route2_second_attempt_no_city_no_climate` directories contain the paired checkpoint files and preprocessing artifacts selected by the platform export script. Their original split records and metrics accompany them. Other local building-split and boosted-head experiments were not substituted for these models.

## Reproduction status

This is a source and artifact release, not a newly executed experiment. Original scripts contain Windows-specific paths, local PyTorch initialization and references to caches/external validation files outside this package. Configure those dependencies before retraining. Local metric files are historical experiment outputs; they do not demonstrate fresh validation of this release. Hourly labels are simulation data. External measured validation described in the source documentation covers annual indicators.

The platform includes source, 994 campus OBJ files, browser ONNX models and its data tables. Its record count can differ from the 3356 indexed hourly workbooks; do not assume every platform building has an hourly workbook.

Environment secrets, deployment account configuration, dependency installations and generated build directories are excluded.
