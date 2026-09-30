# Source code placeholder

This folder is reserved for training, inference and preprocessing code.

Recommended structure:

```text
src/
  data/
    build_dataset.py
    voxelize_obj.py
    load_hourly_series.py
  models/
    annual_multimodal.py
    hourly_multimodal.py
  train_annual.py
  train_hourly.py
  evaluate.py
```

The current public package focuses on data, simulation assets and model documentation. Add the final training scripts after the model configuration is fixed.
