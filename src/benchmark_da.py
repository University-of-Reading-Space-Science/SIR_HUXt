"""
benchmark_da.py

Time the data assimilation on a small synthetic OSSE and report where the time goes.

    python benchmark_da.py                                    # compress_surf, 8 members, 3 obs, serial
    python benchmark_da.py --model compress_surf3d --members 8 --obs 2 --parallel
    python benchmark_da.py --model analytic --members 50 --obs 10   # DA overhead only (no SURF)
    python benchmark_da.py --profile                          # also a cProfile listing of the hottest calls
    python benchmark_da.py --no-arrivals                      # without the per-step arrival-time runs

The OSSE (no WSA download or .env paths needed): uniform 400 km/s solar wind, truth CME 495 km/s, 37.4 deg
wide at Earth's longitude, observed by a synthetic imager at 300 deg; prior drawn uniformly around the truth.
Output files go to a temporary directory (or --out). The timing table (from timing.py) breaks the run into
stages: model setup / solve per member, synthetic-HI j-maps and tracking, Aux-PF steps, prior draw and saving.

--model analytic replaces SURF with the cheap analytic operator from tests/da_test_helpers.py, which isolates
the Python overhead of the DA itself.
"""
import argparse
import cProfile
import datetime
import io
import os
import pstats
import sys
import tempfile
import time
import types

import numpy as np

import astropy.units as u

import timing

T0 = datetime.datetime(2012, 1, 1, 0, 0, 0)
TRUTH_T_INIT = T0 + datetime.timedelta(hours=1)


def make_osse(out_dir: str, use_model: str, n_members: int, n_obs: int, n_runs: int, use_parallel: bool,
              n_lat_surf: int = 9, fixed_compressible_sim_time: bool = True):
    """A RunDataAssimilationRoutine configured as a small synthetic OSSE (bypasses Setup / WSA)."""
    import data_assimilation_routine as dar
    from init_sir import initialise_cme_parameter_ensemble_dict

    os.environ["OUT_BASE_DIR"] = out_dir
    r = object.__new__(dar.RunDataAssimilationRoutine)
    r.setup_class = types.SimpleNamespace(get_use_synthetic_obs=lambda: True,
                                          allowed_cov_types=lambda: {"uncorr", "donki", "mo_cone"})
    r.use_model = use_model
    r.surf_init_time = T0
    if use_model == "compress_surf3d":
        r.vr_longs = np.deg2rad(np.arange(0, 360, 5.0)) * u.rad
        r.vr_lats = np.deg2rad(np.arange(-60, 61, 5.0)) * u.rad
        r.vr_in = np.ones((len(r.vr_lats), len(r.vr_longs))) * 400 * u.km / u.s
        r.n_lon_in, r.n_lat_in = len(r.vr_longs), len(r.vr_lats)
    else:
        r.vr_in = np.ones(128) * 400 * u.km / u.s
        r.vr_longs, r.vr_lats = np.nan * u.rad, np.nan * u.rad
        r.n_lon_in, r.n_lat_in = 128, 1
    r.br_in, r.br_longs, r.br_lats = np.nan * u.nT, np.nan * u.rad, np.nan * u.rad
    r.lon_start, r.lon_stop = -70 * u.deg, 70 * u.deg
    r.latitude_min, r.latitude_max, r.n_lat_surf = -30 * u.deg, 30 * u.deg, n_lat_surf
    r.sim_time, r.dt_scale = 2 * u.day, 1
    r.r_min, r.cme_init_rad = 21.5 * u.solRad, 12 * u.solRad
    r.cme_fixed_duration, r.fixed_duration = True, 12 * 3600 * u.s
    r.n_obs = n_obs
    r.obs_radius, r.obs_lon, r.obs_lat = 1.0 * u.AU, 300 * u.deg, 0 * u.deg
    r.obs_cov = [0.5]
    r.obs_times = [TRUTH_T_INIT + datetime.timedelta(hours=10 + 3 * i) for i in range(1, 1 + n_obs)]
    r.elong_min, r.elong_max = 4.0, 30.0
    r.use_synthetic_obs, r.obs_filenames, r.obs_rng_seed = True, None, 4096
    r.ssw_event, r.craft, r.img = None, "twin", None
    r.bias_term_bool, r.bias_term_5rs, r.bias_term_21rs = False, None, None
    r.use_parallel = use_parallel
    truth = initialise_cme_parameter_ensemble_dict(1, T0)
    truth.update(t_init=TRUTH_T_INIT, v=495 * u.km / u.s, width=37.4 * u.deg, lon=0 * u.deg,
                 lat=0 * u.deg, thick=0 * u.solRad)
    r.true_cme_par_dict = truth
    r.fg_mean_cme_t_init, r.fg_mean_cme_speed, r.fg_mean_cme_width = TRUTH_T_INIT, 495, 37.4
    r.fg_mean_cme_lon, r.fg_mean_cme_lat, r.fg_mean_cme_thick = 0.0, 0.0, 0.0
    r.fg_sd_cme_t_init, r.fg_sd_cme_speed, r.fg_sd_cme_width = 0, 50, 5
    r.fg_sd_cme_lon, r.fg_sd_cme_lat, r.fg_sd_cme_thick = 5, 0, 0
    r.cme_cov_type, r.scale_corr, r.use_log_v = "uncorr", True, True
    r.prior_sd_t_init, r.prior_sd_speed, r.prior_sd_width = 0, 0.1, 5
    r.prior_sd_lon, r.prior_sd_lat, r.prior_sd_thick = 5, 0, 0
    r.n_members, r.n_runs, r.delta_aux_pf = n_members, n_runs, 0.98
    r.pars_in_state, r.time_tolerance = ["v", "width", "lon"], 0.5 * u.day
    r.fixed_compressible_sim_time = fixed_compressible_sim_time
    with timing.timed("da.get_observations"):
        r.observations = r.get_observations()
    r.obs_radius, r.obs_lon, r.obs_lat = [r.obs_radius] * n_obs, [r.obs_lon] * n_obs, [r.obs_lat] * n_obs
    return r


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="compress_surf",
                   choices=["compress_surf", "compress_surf3d", "surf", "huxt", "analytic"])
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--obs", type=int, default=3)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--n-lat-surf", type=int, default=9, help="latitudes for compress_surf3d")
    p.add_argument("--parallel", action="store_true", help="run the members on the worker pool")
    p.add_argument("--short-runs", action="store_true",
                   help="fixed_compressible_sim_time=False: run compressible models only to each obs time")
    p.add_argument("--no-arrivals", action="store_true",
                   help="skip the per-step CME arrival-time runs (compute_arrival_times=False)")
    p.add_argument("--profile", action="store_true", help="also print a cProfile listing")
    p.add_argument("--out", default=None, help="output directory (default: a temporary directory)")
    p.add_argument("--json", default=None, help="write the timing table to this JSON file")
    a = p.parse_args(argv)

    use_model = a.model
    if a.model == "analytic":
        # Replace SURF by the analytic stand-in used by the regression tests
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
        import aux_pf
        import sir_observations
        import data_assimilation_routine
        from da_test_helpers import AnalyticArrivalTimes, AnalyticObservationOperator
        aux_pf.ObservationOperator = AnalyticObservationOperator
        sir_observations.ObservationOperator = AnalyticObservationOperator
        data_assimilation_routine.EnsembleArrivalTimes = AnalyticArrivalTimes
        use_model = "compress_surf"

    out_dir = a.out or tempfile.mkdtemp(prefix="sir_benchmark_")
    print(f"Benchmark: model={a.model}, members={a.members}, obs={a.obs}, runs={a.runs}, "
          f"parallel={a.parallel}, short_runs={a.short_runs}, output in {out_dir}")

    profiler = cProfile.Profile() if a.profile else None
    t0 = time.perf_counter()
    if profiler:
        profiler.enable()
    routine = make_osse(out_dir, use_model, a.members, a.obs, a.runs, a.parallel, a.n_lat_surf,
                        fixed_compressible_sim_time=not a.short_runs)
    routine.compute_arrival_times = not a.no_arrivals
    routine.run_data_assimilation(run_start=0)
    if profiler:
        profiler.disable()
    wall = time.perf_counter() - t0

    print()
    print(timing.timing_report(f"Benchmark timing ({a.model}, {a.members} members, {a.obs} obs, "
                               f"{'parallel' if a.parallel else 'serial'}): wall time {wall:.1f} s"))
    if a.json:
        timing.save_timing_report(a.json)
    if profiler:
        buf = io.StringIO()
        pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(30)
        print("\ncProfile (main process only; worker processes are not included):")
        print(buf.getvalue())

    from sir_observation_operator import close_worker_pool
    close_worker_pool()


if __name__ == "__main__":
    main()
