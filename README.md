# m_star_predictor

A NN-based parameterization for mechanical mixing in the ePBL scheme: predicting
the mixing energetics term **M** from ocean boundary-layer state and forcing,
with fully-connected networks in two modes (instantaneous / autoregressive).

## Structure

Three tasks, one module (plus one notebook) each:

| Task | Module | Notebook |
|------|--------|----------|
| 1. Dataset compilation | `data.py` — `DatasetMaker` (→ netCDF), `FeatureSelector` | `make_datasets.ipynb` |
| 2. Train / test / save | `model.py`, `experiment.py` | `train_predictor.ipynb` |
| 3. Analysis / viz | `experiment.py` (`Prediction`, baselines), `plots.py` | `analyze.ipynb` |

Shared primitives live in `metrics.py`; the algebraic ePBL baseline in
`epbl_basic.py`.

### Key design points

- **One canonical tensor shape `(case, time, feature)`.** Datasets are stored as
  xarray/netCDF with cases and time preserved (profiles add a `depth` dim);
  flattening to `(sample, feature)` happens only when preparing *parallel*
  training. Depth profiles expand to feature columns at selection time, so the
  network always sees `(…, n_features)`.
- **Column convention:** the first `n_ar` feature columns are the autoregressive
  state channels. `model._step` is the single forward pass shared by both
  predictors; `predict_tendency` is a residual flag on it (net predicts an
  increment added to the previous state). So:
  - `ParallelPredictor` = teacher forcing (true previous state fed in),
  - `AutoregressivePredictor` = rollout (predicted state fed back).
- **Lagged quantities are derived, not stored.** `FeatureSpec(name, lag, transform)`
  lags along the time axis: `lag=+1` → x_i-1, `lag=-1` → x_i+1.
- **One `PredictorConfig`** describes any run (`mode`, `predict_tendency`, `n_ar`,
  `k` are fields). `Registry` lists runs; `restore()` returns a `Prediction`.
- **`Prediction` is the analysis object.** Neural predictors, `epbl_prediction`,
  `persistence_prediction`, and a restored *mean* predictor are all the same
  type; plots take one or a list of them and a `quantity="state"|"tendency"`.

## Running

New runs are written under `predictors/`. Datasets are read from
`../m_star_dataset` (outside the repo).

```bash
conda activate L96M2lines
python -m pytest tests/ -q        # 51 tests, CPU-only, uses mock data
```

## Restoring pre-refactor runs

The old code, notebooks, and trained runs (`m_star_predictors/`,
`m_star_autoregressive_predictors/`) are preserved in git history. To restore
and re-analyze a pre-refactor experiment, `git checkout` the relevant commit.
