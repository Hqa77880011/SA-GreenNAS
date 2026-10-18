# SA-GreenNAS

Implementation of **SA-GreenNAS: Robust, Calibrated, and Device-Aware Model Selection for Resource-Constrained Industrial Vision Systems**.

The mobile candidate encoder has configurable stage depth, width, and depthwise kernel size. Architecture search ranks clean recognition, worst-condition F1, confidence ECE, measured P95, and parameters with a hard latency cap. Parameter count is a model-size objective; the optional deployment memory constraint uses an explicitly recorded CPU RSS observation or CUDA allocator peak. The included compact mobile search space is declared in search.py; expand it only using operators supported by the target backend.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,3,H,W]`, class `y[N]`, source `group[N]`, and validation `condition[N]` define industrial images and shift conditions. Search-visible corruptions belong to validation; unseen shifts remain exclusively test. Baseline and ablation runs use the same prepared images and optimizer budget.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
sa-greennas demo --output data/example.npz --count 96 --config config.json
sa-greennas prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
sa-greennas train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
sa-greennas evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
sa-greennas plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
sa-greennas calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_attention`, `no_fuzzy`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
sa-greennas train --input data/prepared --output runs/baseline --config config.json --variant baseline
sa-greennas evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Architecture search

```bash
python -m sa_greennas.search --data data/prepared --output runs/search --population 24 --generations 10 --epochs 12 --latency-cap 15
```

Search trains each uncached candidate, evaluates only validation conditions, and measures batch-one forward latency on the executing device. NSGA-II uses nondominated sorting, crowding, tournament selection, gene crossover, and mutation. `selection.json` contains the feasible Pareto set and the deterministic deployment choice. The compact width/kernel/depth space is explicit in `search.py`. This search does not infer physical energy; supply instrumented energy results when comparing the paper’s energy objective.

## Selected reproduction setting

Use the compact width/kernel/depth search on the current CPU and FP32 batch-one inference. The generated 96-example task keeps source groups disjoint, uses clean/brightness/blur for search-visible conditions and adds low contrast only to test. Search supports an explicit device and optional memory cap. CPU memory means maximum process RSS observed between batches; it is not an exact OS high-water mark. CUDA uses allocator peak allocated memory. Feasibility combines the measured latency cap and the selected memory field; parameter count stays a separate objective.

```bash
python -m sa_greennas.search --data data/prepared --output runs/search --population 12 --generations 5 --epochs 10 --device cpu --latency-cap 15 --memory-cap-mib 1024
```
