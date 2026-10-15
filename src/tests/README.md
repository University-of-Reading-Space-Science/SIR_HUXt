# SIR_HUXt `src` test suite

Run from the `src` folder (needs `pytest`):

```
pytest                    # fast tests: unit tests + fast regression test (a few seconds)
pytest --run-slow         # also the regression OSSE with the real compressible SURF model (minutes)
pytest --update-golden    # re-record the regression baselines after an INTENDED change in results
pytest -k aux_pf          # just one area
```

Tests needing a package that isn't installed (astropy, xarray, surf, huxt, sunpy, ...) are skipped, not failed.

| File | Covers |
|---|---|
| `test_likelihood_resampling.py` | Gaussian likelihood, log-sum-exp, systematic resampling (log and linear weights) |
| `test_structure_and_catalogue.py` | `cme_par_array` layout, `event_catalogue` consistency, `osse_stats` |
| `test_state_vector_and_priors.py` | dict <-> array <-> state-vector conversions, uncorrelated / uniform / Blair priors, Met Office cone file selection |
| `test_setup_and_operators.py` | `init_sir`, `da_surf_setup.Setup` (no WSA download), observation-operator helpers, bias correction, arrival-time helpers, WSA URLs |
| `test_aux_pf.py` | Liu & West shrinkage, kernel covariance, ESS, weight normalisation, convergence towards the observation, NaN handling, ancestor bookkeeping, determinism |
| `test_data_assimilation.py` | `RunDataAssimilationRoutine` helpers and the whole pipeline, plus the **regression tests** |

## Regression tests

`test_data_assimilation.py` builds a small synthetic OSSE (truth CME 495 km/s, 37.4 deg, lon 0; 400 km/s wind;
observer at 300 deg) and runs the real `run_data_assimilation()`, writing `cme_pars.nc` to a temporary folder:

* `test_regression_fast_pipeline`: 10 members, 4 observations, 2 runs, with an analytic stand-in for SURF
  (`da_test_helpers.AnalyticObservationOperator`). Checks synthetic observations, prior sampling, every Aux-PF
  step and the saved output. Always runs.
* `test_regression_full_surf_osse` (`--run-slow`): 4 members, 2 observations, 1 run, with the real compressible
  SURF model and synthetic heliospheric imager.

The first run records the results in `tests/data/<name>.json` and reports the test as *skipped*; every later run must
reproduce them (relative tolerance 1e-9, or 1e-7 for the SURF run). **Commit the `tests/data/*.json` files.**
If a change is meant to alter the results, re-run with `--update-golden` and commit the new baselines.

Known data issue marked `xfail`: event 02A's first-guess CME reaches 21.5 rS before its model start time
(`event_catalogue.SURF_INIT_TIMES["02A"]`). The test will start failing (as XPASS) once that is fixed: then remove the
xfail.
