# SA-GreenNAS

Implementation of **SA-GreenNAS: Robust, Calibrated, and Device-Aware Model Selection for Resource-Constrained Industrial Vision Systems**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate grouped synthetic RGB classes with clean/brightness/blur conditions; low contrast is reserved for test.

```bash
sa-greennas demo --output data/generated.npz --count 96 --config config.json --seed 13
sa-greennas prepare --input data/generated.npz --output data/prepared
```

`x[N,3,H,W]`, class `y[N]`, source `group[N]`, and validation `condition[N]` define industrial images and shift conditions. Search-visible corruptions belong to validation; unseen shifts remain exclusively test. Baseline and ablation runs use the same prepared images and optimizer budget.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
sa-greennas train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

Search the compact width/kernel/depth space on validation data:

```bash
python -m sa_greennas.search --data data/prepared --output runs/search --population 12 --generations 5 --epochs 10 --device cpu --latency-cap 15 --memory-cap-mib 1024 --seed 13
```

## Evaluation

```bash
sa-greennas evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

Optional confidence calibration uses its independent split:

```bash
sa-greennas calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
sa-greennas evaluate --input data/prepared/test.npz --model runs/full/calibrated.pt --output runs/full/calibrated_evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_attention`, `no_fuzzy`. Reuse the prepared data and training seed:

```bash
sa-greennas train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
sa-greennas evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
sa-greennas plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
`runs/search/evaluations.json` records candidates; `selection.json` stores the feasible Pareto set and deployment choice.
