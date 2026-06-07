# Copilot Files - Quantum IDS Benchmarking

This folder contains all new implementation work requested for fair cross-dataset benchmarking.

## What is included

- `utils/quantum_ids_pipeline.py`
  - Unified data loading and preprocessing for:
    - `Datasets/train_test_network.csv`
    - `Datasets/CICIDS2017/`
  - Fair classical baselines:
    - Logistic Regression
    - Random Forest
    - XGBoost (fallback to LightGBM)
  - Quantum techniques:
    - QSVM (precomputed quantum kernel)
    - QNN base (StronglyEntanglingLayers)
    - QNN re-uploading variant
    - VQC alternate ansatz
    - Quantum-kNN distance-kernel variant
  - Fusion re-check:
    - Hard, margin, probability schemes with explicit distinctness assertions
  - Unified metrics and summary rows including:
    - AUPRC
    - TPR@0.1%FPR
    - TPR@1%FPR

- `01_train_test_network_experiments.ipynb`
  - Runs full pipeline on `Datasets/train_test_network.csv`.

- `02_cicids2017_experiments.ipynb`
  - Runs full pipeline on `Datasets/CICIDS2017/`.

- `03_summary_tables.ipynb`
  - Builds cross-dataset publication summary table from generated CSV outputs.

- `results/`
  - Per-dataset and combined metric summaries saved as CSV.

## Plot export standard

Each retained plot is exported in all formats:

- PDF: `Plots PDF/` (vector-friendly)
- PNG + JPG: `Images/` (high-resolution via configured DPI)

Style is set globally for paper visibility:

- Consistent font sizing (`paper_fontsize` in config)
- High-resolution raster export (`export_dpi` in config)
- Tight layout and bounding boxes

## Notes

- No files in `Backup/` or `Datasets/` are modified.
- Runtime controls are configurable in `ExperimentConfig` (`max_workers`, subset sizes, epochs).
- Use subset controls first for sanity runs, then increase for final runs.
