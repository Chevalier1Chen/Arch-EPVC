# Model release

This folder documents the planned model release for the dual-scale prediction framework.

## Annual branch

- Inputs: voxelized/mesh-derived 3D geometry features and structured numerical building features.
- Encoders: 3D CNN for geometry and MLP/TabTransformer for tabular features.
- Outputs: annual EUI, annual PV generation, carbon emission intensity and optional economic/carbon indicators.

## Hourly branch

- Inputs: geometry features, numerical building features and weather/time-series features.
- Encoders: 3D CNN, tabular encoder and LSTM/temporal model.
- Outputs: hourly energy demand, hourly PV generation, grid purchase/export and operational carbon emissions.

## Checkpoints

The platform export script identifies the following paired trained models, now included under `research/`:

- `leakage_free_retraining/route1/best_route1_3dcnn_tabtransformer.pt`, with category mapping, continuous scaler, target scaler and split metadata.
- `leakage_free_retraining/route2_second_attempt_no_city_no_climate/best_route2_lstm.pt`, with hourly scalers.
- `school-building-design-platform/public/models/browser/`: both ONNX models and preprocessing metadata.

These are the versions referenced by the platform's export script. Other local experiments are not interchangeable with these weights. Existing metrics are retained with their original experiment; packaging does not constitute a new model evaluation. Training scripts retain local data paths and need configuration before independent reproduction. See `research/README.md`.
