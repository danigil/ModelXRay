"""Canonical CSV schemas for results/.

Every result CSV produced by `scripts/experiments/*.py` and consumed by
`scripts/plots/*.py` follows one of the schemas declared below. This is the
single source of truth: runners write these schemas, plot scripts read them,
and the cached-paper CSVs under `results/` were one-shot converted to match.

If you add a new result type, declare its schema here first, then update
both the runner and the plot script. Do not ship a runner that writes a
schema the plot script can't read.

## Schemas

### FSL — per-(repeat, X_hat, X[, eval_set]) eval (covers both OML and AL)
Filename pattern: `fsl_{osl,srnet}*.csv`
Required cols: `repeat:int, X_hat:int, X:int, centroid:float, nn:float`
Optional col: `eval_set:str` — present in Exp 2 where one trained model is
evaluated against multiple eval sets (small in-domain, large OOD,
MaleficNet benigns, MaleficNet attacks). Plot scripts filter by `eval_set`.
Absent for Exp 1 (single eval set).
- X_hat = training anchor X; X = evaluation X.
- OML plot uses the diagonal: `df[df.X_hat == df.X]`.
- AL plot uses all rows.

### Classifier baseline — per-(repeat, X) eval
Filename pattern: `b{1,3}_*.csv`, `gf_{xgboost,1nn}.csv`
Columns: `repeat:int, baseline:str, X:int, accuracy:float`
Optional cols: `fold:int` for k-fold CV (Exp 4); `train_archs:str` for Exp 2.

### Classifier baseline — per-(repeat, X_hat, X) cross-X eval (AL)
Filename pattern: `b3_*_crossx.csv`
Columns: `repeat:int, baseline:str, X_hat:int, X:int, accuracy:float`
Optional cols: `train_archs:str` for Exp 2.

### Threshold detectors — per-(repeat, baseline, X) eval
Filename pattern: `b{4,5,6,7}_*.csv`, `b4_b7_threshold*.csv`, `b5_b7_threshold*.csv`
Columns: `repeat:int, baseline:str, X:int, accuracy:float`
Optional cols: `fold:int` for Exp 4; `train_archs:str` for Exp 2;
`acc_benign:float, acc_malicious:float, threshold:float` to keep diagnostic
fields available (plot scripts only need `accuracy`).

### Exp 2.5 MaleficNet OOD — per-(repeat, X_anchor, arch, payload)
Filename: `maleficnet_ood_results.csv`
Columns: `repeat:int, X_anchor:int, model_arch:str, arch:str, payload:str,
          centroid:float, nn:float`
`model_arch` ∈ {"osl_siamese_cnn", "srnet"} (the FSL detector); `arch` is the
attacked CNN architecture (e.g. "densenet121"); `payload` is the malware id.
The headline single-cell paper number averages across all (arch, payload).

### Runtime/memory — per-(image_rep, n_weights, run_i)
Filename: `measure_time.csv`
Columns: `image_rep:str, time:float, peak_memory:float, n_models:int,
          n_weights:int, run_i:int`
Already canonical (no migration needed).

## Naming conventions

- `repeat` (not `seed`, `run num`, `run`, `split_idx`): the outer repeat index.
- `X` (not `lsb`, `x`): bit count of the LSB attack.
- `X_hat` (not `model_lsb`): the training anchor X for cross-X evals.
- `accuracy` (not `acc_mean_test`, `acc`, `test_acc`): scalar mean accuracy.
- `centroid`, `nn`: FSL classifier outputs as separate columns.
- `baseline`: baseline detector name (e.g. "B1", "B3", "B4_byte_autocorrelation").
"""
