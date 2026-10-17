"""
ensemble_arrival.py

CME arrival times at a body (Earth by default) for every member of a CME parameter ensemble, computed with
the same forward model, ambient solar wind, model domain and cone-CME construction as the data assimilation.
The data assimilation (data_assimilation_routine.py) uses this after every assimilation step, so the
forecast arrival time of each member can be followed as observations are assimilated.

    calc = EnsembleArrivalTimes(use_model="compress_surf", surf_init_time=..., vr_in=..., ...)
    result = calc.ensemble_arrival(cme_par_dict)
    result["arrival_time"]   # list of datetimes (None if the member misses the body)
    result["arrival_s"]      # seconds after surf_init_time (NaN for a miss)
    result["arrival_speed"]  # km/s (NaN for a miss; compressible solver: model wind speed at the body on arrival)
    result["hit"]            # bool array

Each member is a separate model run of ``sim_time`` (default 5 days, so that slow CMEs reach 1 AU), run in
parallel on the observation operator's shared worker pool if ``use_parallel``.
"""
import datetime
import time

import numpy as np
import numpy.typing as npt

import astropy.units as u
from astropy.units import Quantity

import huxt.huxt as H
import surf.surf as S

from cme_par_ens import CmeParEns
from init_sir import setup_huxt, setup_surf, setup_surf3d, solve_surf3d
from memory_budget import estimate_run_bytes, process_peak_bytes, record_worker_peak, worker_count
from sir_observation_operator import ObservationOperator, arrival_at_body, run_on_pool
from sir_observation_operator import solved_cmes  # noqa: F401  (re-exported; previously defined here)
from timing import add_timing, collect_timings, merge_timings, timed


def _arrival_task(calculator: "EnsembleArrivalTimes", cme, ens_no: int):
    """Worker task: one member's arrival, plus the timings recorded while computing it."""
    with collect_timings() as worker_timings:
        result = calculator.member_arrival(cme, ens_no=ens_no)
    return result, worker_timings, process_peak_bytes()


def arrivals_to_result(arrivals: list, surf_init_time: datetime.datetime) -> dict:
    """
    The ensemble_arrival() result format from a list of per-member (arrival datetime or None, speed, hit),
    e.g. ObservationOperator.member_arrivals.
    """
    arrival_time = [a[0] for a in arrivals]
    return {
        "arrival_time": arrival_time,
        "arrival_s": np.array([(t - surf_init_time).total_seconds() if t is not None else np.nan
                               for t in arrival_time], dtype=float),
        "arrival_speed": np.array([a[1] for a in arrivals], dtype=float),
        "hit": np.array([bool(a[2]) for a in arrivals], dtype=bool),
    }


class EnsembleArrivalTimes:
    """Arrival time and speed of every ensemble member at ``body`` (see the module docstring)."""

    def __init__(
            self,
            use_model: str,
            surf_init_time: datetime.datetime,
            vr_in: npt.NDArray[Quantity[u.km / u.s]],
            vr_longs: npt.NDArray[Quantity[u.rad]] | Quantity[u.rad] = np.nan * u.rad,
            vr_lats: npt.NDArray[Quantity[u.rad]] | Quantity[u.rad] = np.nan * u.rad,
            lon_start: Quantity[u.deg] = -70 * u.deg,
            lon_stop: Quantity[u.deg] = 70 * u.deg,
            lat_min: Quantity[u.deg] = -30 * u.deg,
            lat_max: Quantity[u.deg] = 30 * u.deg,
            n_lat_surf: int = 45,
            n_lon_surf: int = 128,
            r_min: Quantity[u.solRad] = 21.5 * u.solRad,
            cme_init_rad: Quantity[u.solRad] = 12 * u.solRad,
            cme_fixed_duration: bool = True,
            fixed_duration: Quantity[u.s] = 12 * 60 * 60 * u.s,
            sim_time: Quantity[u.day] = 5 * u.day,
            dt_scale: int | float = 1,
            body: str = "EARTH",
            use_parallel: bool = False,
            n_workers: int | None = None,
            max_workers: int | None = None,
            memory_fraction: float = 0.7,
            latitudes: Quantity[u.deg] | None = None,
    ) -> None:
        """
        :param use_model: "surf", "compress_surf", "compress_surf3d" or "huxt" (as in the data assimilation)
        :param surf_init_time: Model start time (arrival times are also given in seconds after it)
        :param vr_in, vr_longs, vr_lats: Ambient solar wind at the inner boundary (as for the DA)
        :param lon_start, lon_stop: Longitude range of the model domain (must contain the body)
        :param lat_min, lat_max, n_lat_surf, n_lon_surf: SURF3d grid (compress_surf3d only; NaN -> -30/+30 deg;
            SURF3d's sine-latitude grid, n_lat_surf bins over the whole sphere)
        :param latitudes: Explicit SURF3d latitudes instead of lat_min / lat_max / n_lat_surf (the data
            assimilation passes the observation operator's field-of-view grid, so both use the same model)
        :param r_min: Inner boundary radius of the model
        :param cme_init_rad: Radius at which t_init is defined (CMEs are launched at r_min as in the DA)
        :param cme_fixed_duration, fixed_duration: Cone-CME duration settings
        :param sim_time: Model run length; members that have not arrived by then count as misses
        :param dt_scale: Model output cadence
        :param body: Target body, e.g. "EARTH"
        :param use_parallel: Run the members on the shared worker pool
        :param n_workers: Members to run at the same time (default: limited by CPUs and memory, memory_budget.py)
        :param max_workers, memory_fraction: Limits for that default
        """
        self.use_model = use_model.lower()
        assert self.use_model in ["surf", "compress_surf", "compress_surf3d", "huxt"], self.use_model
        self.solver = "hydro" if self.use_model in ["compress_surf", "compress_surf3d"] else "huxt"
        self.surf_init_time = surf_init_time
        self.vr_in = vr_in
        self.vr_longs = vr_longs
        self.vr_lats = vr_lats
        self.lon_start = lon_start
        self.lon_stop = lon_stop
        # The SURF3d latitude range needs finite values
        lat_min_ok = lat_min is not None and np.isfinite(Quantity(lat_min).value)
        lat_max_ok = lat_max is not None and np.isfinite(Quantity(lat_max).value)
        self.lat_min = lat_min if lat_min_ok else -30 * u.deg
        self.lat_max = lat_max if lat_max_ok else 30 * u.deg
        self.n_lat_surf = n_lat_surf
        self.n_lon_surf = n_lon_surf
        self.r_min = r_min
        self.cme_init_rad = cme_init_rad
        self.cme_fixed_duration = cme_fixed_duration
        self.fixed_duration = fixed_duration
        self.sim_time = sim_time
        self.dt_scale = dt_scale
        self.body = body
        self.use_parallel = use_parallel
        self.n_workers = n_workers
        self.max_workers = max_workers
        self.memory_fraction = memory_fraction
        self.latitudes = latitudes

    @property
    def n_surf_latitudes(self) -> int:
        """
        SURF planes solved per member: 1, except for compress_surf3d without explicit latitudes (SURF3d's sine
        grid, all planes solved). With explicit latitudes only the plane nearest Earth is solved.
        """
        if self.use_model != "compress_surf3d":
            return 1
        if self.latitudes is not None:
            return 1
        n = self.n_lat_surf
        sinlat = np.linspace(-1 + 1 / n, 1 - 1 / n, n)   # SURF's latitude_grid
        lat = np.rad2deg(np.arcsin(sinlat))
        lo, hi = Quantity(self.lat_min).to(u.deg).value, Quantity(self.lat_max).to(u.deg).value
        return max(int(np.sum((lat >= lo) & (lat <= hi))), 1)

    # ------------------------------------------------------------------------------------------
    def cme_objects(self, cme_par_dict: CmeParEns) -> list:
        """
        Cone CMEs for every member, built exactly as the observation operator builds them (t_init is at
        cme_init_rad; the CME is launched when it reaches r_min, assuming constant speed).
        """
        operator = ObservationOperator(
            use_model=self.use_model,
            cme_par_dict=cme_par_dict,
            obs_time_in_datetime=self.surf_init_time,  # not used to build the CMEs
            obs_cov=0.0,
            craft=None,
            surf_init_time=self.surf_init_time,
            r_min=self.r_min,
            cme_init_rad=self.cme_init_rad,
            cme_fixed_duration=self.cme_fixed_duration,
            fixed_duration=self.fixed_duration,
            use_parallel=False,
            surf_latitudes=np.array([0.0]) * u.deg,   # only the CMEs are needed: skip the imager geometry
        )
        return operator.make_cme_objects()

    def make_model(self) -> S.SURF | S.SURF3d | H.HUXt:
        """A fresh (unsolved) model with the DA's ambient wind and domain, running for sim_time."""
        if self.use_model in ["surf", "compress_surf"]:
            return setup_surf(
                start_datetime=self.surf_init_time, vr_in=self.vr_in,
                lon_start=self.lon_start, lon_stop=self.lon_stop,
                sim_time=self.sim_time, dt_scale=self.dt_scale, r_min=self.r_min, solver=self.solver,
            )
        if self.use_model == "compress_surf3d":
            return setup_surf3d(
                start_datetime=self.surf_init_time, vr_in=self.vr_in,
                vr_lons=self.vr_longs, vr_lats=self.vr_lats,
                lat_min=self.lat_min, lat_max=self.lat_max, n_lat_surf=self.n_lat_surf,
                lon_start=self.lon_start, lon_stop=self.lon_stop, n_lon_surf=self.n_lon_surf,
                sim_time=self.sim_time, dt_scale=self.dt_scale, r_min=self.r_min, solver=self.solver,
                latitudes=self.latitudes, lazy=self.latitudes is not None,
            )
        return setup_huxt(
            start_datetime=self.surf_init_time, vr_in=self.vr_in,
            lon_start=self.lon_start, lon_stop=self.lon_stop,
            sim_time=self.sim_time, dt_scale=self.dt_scale, r_min=self.r_min,
        )

    def member_arrival(self, cme, ens_no: int | None = None) -> tuple[datetime.datetime | None, float, bool]:
        """
        Solve the model for one CME and compute its arrival at self.body.
        :return: (arrival datetime or None, arrival speed in km/s or NaN, hit)
        """
        t_start = time.perf_counter()
        model = self.make_model()
        add_timing(f"arrival.model_setup.{self.use_model}", time.perf_counter() - t_start)
        with timed(f"arrival.model_solve.{self.use_model}"):
            if self.use_model == "compress_surf3d":
                # Only the plane nearest Earth gives the arrival (solved_run), so only that one is solved
                solve_surf3d(model, [cme], only_nearest_earth=True)
            else:
                model.solve([cme])

        try:
            # (speed: the CME tracer speed, or for the compressible solver the model wind speed at the body)
            return arrival_at_body(model, self.body)
        except Exception as err:  # e.g. the CME never enters the model domain: count it as a miss
            print(f"Arrival time not computed for member {ens_no}: {type(err).__name__}: {err}")
            return None, np.nan, False

    @timed("arrival.ensemble")
    def ensemble_arrival(self, cme_par_dict: CmeParEns) -> dict:
        """
        Arrival of every member of cme_par_dict (in member order).
        :return: dict with arrival_time (list of datetime/None), arrival_s (s after surf_init_time, NaN for a
            miss), arrival_speed (km/s, NaN for a miss) and hit (bool), each of length n_members
        """
        cme_objects = self.cme_objects(cme_par_dict)

        if self.use_parallel and len(cme_objects) > 1:
            if self.n_workers is not None:
                n_proc = max(int(self.n_workers), 1)
            else:
                n_proc = worker_count(
                    estimate_run_bytes(self.use_model, self.sim_time, dt_scale=self.dt_scale, n_lat=self.n_surf_latitudes,
                                       lon_start=self.lon_start, lon_stop=self.lon_stop, r_min=self.r_min,
                                       lean=self.latitudes is not None, keep_wind=True),
                    self.max_workers, self.memory_fraction)
            results, in_process = run_on_pool(_arrival_task, [(self, cme, i) for i, cme in enumerate(cme_objects)],
                                              min(n_proc, len(cme_objects)))
            members = []
            for result, worker_timings, peak in results:
                if not in_process:
                    merge_timings(worker_timings)
                    record_worker_peak(peak)
                members.append(result)
        else:
            members = [self.member_arrival(cme, ens_no=i) for i, cme in enumerate(cme_objects)]

        arrival_time = [m[0] for m in members]
        out = {
            "arrival_time": arrival_time,
            "arrival_s": np.array([
                (t - self.surf_init_time).total_seconds() if t is not None else np.nan for t in arrival_time
            ], dtype=float),
            "arrival_speed": np.array([m[1] for m in members], dtype=float),
            "hit": np.array([bool(m[2]) for m in members], dtype=bool),
        }
        print(f"Arrival at {self.body}: {int(out['hit'].sum())}/{len(members)} members hit")
        return out
