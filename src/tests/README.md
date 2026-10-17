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
| `test_da_output.py` | `elongations.nc` / `cme_arrival_times.nc`: profile sampling, Aux-PF bookkeeping, arrival statistics, the whole pipeline with analytic stand-ins (outputs consistent with `cme_pars.nc`, skipped steps, switches, results unchanged); `--run-slow`: real SURF/HUXt arrivals and a small SURF pipeline |
| `test_da_diagnostics.py` | `da_diagnostics` statistics with known answers (weighted quantiles, circular longitude, CRPS, PIT, ESS, KS), checks on calibrated / over-confident / broken / collapsed synthetic runs, and `da_report.py` on the analytic pipeline's output |
| `test_lookahead_reuse.py` | look-ahead from stored tracks (`reuse_lookahead_tracks`): first stage from tracks, second stage divides by the same likelihood, shrink/jitter unchanged, per-member fallback, unusable tracks, run counts in the pipeline, skipped observations, `compare_da_schemes.py`; `--run-slow`: a stored track equals a fresh SURF run |
| `test_optimised_runs.py` | A (moving craft observer: observer ids, reuse for a real craft) and B (arrivals from the DA runs: run lengths, imaging window, `truncated_model_view`, exact reproduction of the original scheme, one run per member per step, `compare_da_schemes.py`); `--run-slow`: with SURF/SURF3d the longer runs give identical elongations, the combined arrival equals a separate run, a STEREO-A track is valid at later times |
| `test_memory_budget.py` | the SURF3d out-of-memory fix: model-size estimate reproduces the failing (1241, 147, 50) grid, parallel members limited by available memory and `max_workers`, a MemoryError in the workers halves them down to running in the main process with identical results; a requested `n_parallel_members` (e.g. 15) is used when it fits and reduced with a message when not, the memory report and CLI, measured worker peak memory |
| `test_fov_latitudes.py` | SURF3d only on the imager field of view: 1-deg grid over the FoV (`_compute_fov_geometry(0)`) +/- 2 deg, extra plane for the arrival body, SURF3d built on exactly those latitudes, twin and real-craft operators, memory estimate per plane; slow: real SURF3d/STEREO-A images the grid, and an extra plane outside the FoV changes no elongation |
| `test_synthetic_observer.py` | the synthetic (twin) observer of `ObserverSynthHI`: its position track covers the whole model run (a 5-day run imaged over 3 days no longer fails SURF's "model times are not within the span of the synth_heeq_pos times" check), it is requested at the output times (`time_out`), and it is fixed (a moving synthetic observer raises a ValueError) |
| `test_arrival_speed.py` | the CME arrival speed: compressible SURF gives no tracer speed (NaN), so a hit's speed is the model solar-wind speed at the body at the arrival time (interpolated in time, radius and longitude, NaN outside the domain); tracer speeds are kept where SURF gives them; both arrival paths use it |
| `test_lean_surf3d.py` | low-memory SURF3d solve (`init_sir.solve_surf3d`): planes built and solved one at a time (never more than one with full arrays), density and temperature kept over the imaging window, the Earth plane's speed and full-run temperature kept, values identical to `SURF3d.solve`, arrival runs solve only the plane nearest Earth, lean memory estimate, operator uses it; slow: identical density and arrival with SURF |
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
