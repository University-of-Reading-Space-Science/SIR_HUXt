"""
da_output.py

Diagnostic output of the data assimilation, written next to cme_pars.nc in every run directory and
updated after every assimilation step (so a running or interrupted run can be inspected):

elongations.nc  - observed elongation profile and the modelled elongations from the observation operator
    dims: step (assimilation of observation k, k = 0 .. n_obs-1), obs (observation index), n_ens
    obs_time(obs), obs_elongation(obs), obs_true_elongation(obs) [OSSE, noise-free], observer_lon(obs),
    obs_error_variance
    assimilated(step)                1 if observation k was assimilated, 0 if skipped (then all below are NaN)
    hx_shrunk(step, n_ens)           modelled elongation at obs_time[k] of the shrunk (first-stage) particles
    hx_posterior(step, n_ens)        modelled elongation at obs_time[k] of the new (posterior) particles
    profile_shrunk(step, n_ens, obs) the same particles' modelled elongation at EVERY observation time
    profile_posterior(step, n_ens, obs)  (NaN where the model profile is not available: before the CME is
                                     tracked / after the end of the model run; see ObservationOperator)
    ancestor(step, n_ens)            resampled (first-stage) index of each new particle (-1 if skipped)
    log_weight(step, n_ens)          posterior log-weights after step k
    ess_prior(step), ess_posterior(step)  effective sample size before / after step k
    lookahead_reused(step, n_ens)    1 if the member's first-stage value came from its stored track
                                     (reuse_lookahead_tracks), 0 if it was run at its shrunk point
    model_runs(step)                 forward-model runs made in the step (first + second stage)

cme_arrival_times.nc - arrival of every member at the target body (Earth) after every step
    dims: n_obs (0 = prior, k + 1 = after assimilating observation k, as in cme_pars.nc), n_ens
    arrival_time(n_obs, n_ens)       datetime (NaT if the member misses or has not arrived by the end of the run)
    arrival_time_s(n_obs, n_ens)     seconds after model_init_time (NaN for a miss)
    arrival_speed(n_obs, n_ens)      km/s (NaN for a miss; compressible models: model wind speed at Earth on arrival)
    hit(n_obs, n_ens), log_weight(n_obs, n_ens)
    hit_fraction / weighted_hit_fraction(n_obs), weighted_mean_arrival_time(n_obs),
    weighted_sd_arrival_hours(n_obs), median_arrival_time(n_obs)   (weights renormalised over the hits)
    true_arrival_time / true_arrival_speed (OSSEs: arrival of the truth CME with the same model)
"""
import datetime
import os

import numpy as np
import numpy.typing as npt
import xarray as xr

import astropy.units as u


def to_datetime64(times: list) -> npt.NDArray[np.datetime64]:
    """datetime64[ns] array from a list of datetimes (None / NaN -> NaT)."""
    return np.array([
        np.datetime64(t, "ns") if isinstance(t, datetime.datetime) else np.datetime64("NaT", "ns")
        for t in times
    ], dtype="datetime64[ns]")


def _as_degrees(values: list) -> npt.NDArray[float]:
    """Observer longitudes (Quantities, or plain floats in degrees) as a float array in degrees."""
    out = []
    for v in values:
        if isinstance(v, u.Quantity):
            out.append(float(np.ravel(v.to(u.deg).value)[0]))
        else:
            out.append(float(np.ravel(np.asarray(v, dtype=float))[0]))
    return np.array(out, dtype=float)


def write_dataset(ds: xr.Dataset, path: str, final: bool = True) -> None:
    """
    Write ds to a NetCDF file. Intermediate (per-step) writes only warn if the file cannot be written (e.g. it
    is open in another program), so the assimilation carries on; the final write raises.
    """
    try:
        ds.to_netcdf(path, mode="w")
    except OSError as err:
        if final:
            raise
        print(f"WARNING: could not update {path} ({err}); will retry after the next step")


class ElongationOutput:
    """Collects the observed and modelled elongations of one run (elongations.nc; see the module docstring)."""

    def __init__(
            self,
            obs_times: list[datetime.datetime],
            observations: list[float],
            obs_cov: float | list[float],
            observer_lon: list,
            n_members: int,
            use_model: str,
            model_init_time: datetime.datetime,
            true_observations: list[float] | None = None,
            reuse_lookahead_tracks: bool = False,
    ) -> None:
        self.reuse_lookahead_tracks = bool(reuse_lookahead_tracks)
        self.obs_times = list(obs_times)
        self.observations = np.asarray(observations, dtype=float)
        self.n_obs = len(self.obs_times)
        self.n_members = n_members
        self.use_model = use_model
        self.model_init_time = model_init_time
        self.obs_cov = float(np.ravel(np.asarray(obs_cov, dtype=float))[0])
        self.observer_lon = _as_degrees(observer_lon)
        self.true_observations = (
            None if true_observations is None else np.asarray(true_observations, dtype=float)
        )

        n, m = self.n_obs, n_members
        self.assimilated = np.zeros(n, dtype=np.int8)
        self.hx_shrunk = np.full((n, m), np.nan)
        self.hx_posterior = np.full((n, m), np.nan)
        self.profile_shrunk = np.full((n, m, n), np.nan)
        self.profile_posterior = np.full((n, m, n), np.nan)
        self.ancestor = np.full((n, m), -1, dtype=np.int32)
        self.log_weight = np.full((n, m), np.nan)
        self.ess_prior = np.full(n, np.nan)
        self.ess_posterior = np.full(n, np.nan)
        self.lookahead_reused = np.zeros((n, m), dtype=np.int8)
        self.model_runs = np.zeros(n, dtype=np.int32)
        self.hx_at_shrunk_points = np.full((n, m), np.nan)
        # forward-model runs of the prior particles before the first step (combine_arrival_runs)
        self.prior_model_runs = 0

    def record_step(self, step: int, aux, log_weight: npt.ArrayLike) -> None:
        """
        Store the result of assimilating observation ``step``.
        :param step: Observation index
        :param aux: The AuxPF instance after aux_pf() was called, or None if the observation was skipped
        :param log_weight: Ensemble log-weights after this step
        """
        self.log_weight[step] = np.asarray(log_weight, dtype=float)
        if aux is None:
            return
        self.assimilated[step] = 1
        self.hx_shrunk[step] = aux.hx_shrunk
        self.hx_posterior[step] = aux.hx_post
        if aux.profile_shrunk is not None:
            self.profile_shrunk[step] = aux.profile_shrunk
        if aux.profile_post is not None:
            self.profile_posterior[step] = aux.profile_post
        self.ancestor[step] = aux.ancestors
        self.ess_prior[step] = aux.ess_prior
        self.ess_posterior[step] = aux.ess_post
        self.lookahead_reused[step] = np.asarray(getattr(aux, "lookahead_reused", np.zeros(self.n_members)),
                                                 dtype=np.int8)
        self.model_runs[step] = int(getattr(aux, "n_model_runs", 0))
        if getattr(aux, "hx_at_shrunk_points", None) is not None:
            self.hx_at_shrunk_points[step] = aux.hx_at_shrunk_points

    def to_dataset(self) -> xr.Dataset:
        """The collected values as an xarray Dataset."""
        deg = {"units": "deg"}
        data_vars = dict(
            model_init_time=self.model_init_time,
            obs_error_variance=((), self.obs_cov, {"units": "deg^2"}),
            obs_elongation=(["obs"], self.observations, {**deg, "long_name": "observed elongation"}),
            observer_lon=(["obs"], self.observer_lon, {**deg, "long_name": "observer longitude"}),
            assimilated=(["step"], self.assimilated,
                         {"long_name": "1 if observation 'step' was assimilated, 0 if skipped"}),
            hx_shrunk=(["step", "n_ens"], self.hx_shrunk,
                       {**deg, "long_name": "modelled elongation at obs_time[step], shrunk (first-stage) particles"}),
            hx_posterior=(["step", "n_ens"], self.hx_posterior,
                          {**deg, "long_name": "modelled elongation at obs_time[step], posterior particles"}),
            profile_shrunk=(["step", "n_ens", "obs"], self.profile_shrunk,
                            {**deg, "long_name": "modelled elongation at every observation time, shrunk particles"}),
            profile_posterior=(["step", "n_ens", "obs"], self.profile_posterior,
                               {**deg, "long_name": "modelled elongation at every observation time, "
                                                    "posterior particles"}),
            ancestor=(["step", "n_ens"], self.ancestor,
                      {"long_name": "resampled first-stage index of each new particle (-1 if skipped)"}),
            log_weight=(["step", "n_ens"], self.log_weight, {"long_name": "posterior log-weight after step"}),
            ess_prior=(["step"], self.ess_prior, {"long_name": "effective sample size before the step"}),
            ess_posterior=(["step"], self.ess_posterior, {"long_name": "effective sample size after the step"}),
            lookahead_reused=(["step", "n_ens"], self.lookahead_reused,
                              {"long_name": "1 if the first-stage value came from the member's stored track"}),
            model_runs=(["step"], self.model_runs, {"long_name": "forward-model runs in the step (both stages)"}),
            hx_at_shrunk_points=(["step", "n_ens"], self.hx_at_shrunk_points,
                                 {"units": "deg", "long_name": "check_lookahead_tracks only: modelled elongation at "
                                                              "the shrunk points (what the original scheme uses)"}),
        )
        if self.true_observations is not None:
            data_vars["obs_true_elongation"] = (
                ["obs"], self.true_observations, {**deg, "long_name": "noise-free (truth) elongation (OSSE)"}
            )
        return xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                step=("step", np.arange(self.n_obs)),
                obs=("obs", np.arange(self.n_obs)),
                ens_no=("n_ens", np.arange(self.n_members)),
                obs_time=("obs", to_datetime64(self.obs_times)),
            ),
            attrs={"use_model": self.use_model, "reuse_lookahead_tracks": int(self.reuse_lookahead_tracks),
                   "total_model_runs": int(self.model_runs.sum()) + int(self.prior_model_runs),
                   "prior_model_runs": int(self.prior_model_runs),
                   "description": "Observed and modelled (observation-operator) elongations; see da_output.py"},
        )

    def save(self, output_dir: str, final: bool = True) -> str:
        path = os.path.join(output_dir, "elongations.nc")
        write_dataset(self.to_dataset(), path, final=final)
        return path


class ArrivalOutput:
    """Collects every member's arrival at the target body after every step (cme_arrival_times.nc)."""

    def __init__(
            self,
            obs_times: list[datetime.datetime],
            n_members: int,
            model_init_time: datetime.datetime,
            use_model: str,
            body: str = "EARTH",
            sim_time: u.Quantity | None = None,
    ) -> None:
        self.obs_times = list(obs_times)
        self.n_steps = len(self.obs_times) + 1
        self.n_members = n_members
        self.model_init_time = model_init_time
        self.use_model = use_model
        self.body = body
        self.sim_time_days = np.nan if sim_time is None else float(u.Quantity(sim_time).to(u.day).value)

        n, m = self.n_steps, n_members
        self.computed = np.zeros(n, dtype=bool)
        self.arrival_time: list[list] = [[None] * m for _ in range(n)]
        self.arrival_s = np.full((n, m), np.nan)
        self.arrival_speed = np.full((n, m), np.nan)
        self.hit = np.zeros((n, m), dtype=np.int8)
        self.log_weight = np.full((n, m), np.nan)
        self.true_arrival_time: datetime.datetime | None = None
        self.true_arrival_speed: float = np.nan
        # "separate": own model runs (ensemble_arrival); "combined": from the data assimilation's runs
        self.source = "separate"
        self.separate_runs = 0

    def record(self, step: int, result: dict | None, log_weight: npt.ArrayLike) -> None:
        """
        Store the arrivals after ``step`` (0 = prior). result=None means the ensemble did not change in this
        step (observation skipped): the previous step's arrivals are copied.
        """
        self.log_weight[step] = np.asarray(log_weight, dtype=float)
        if result is None:
            if step == 0:
                return
            prev = step - 1
            self.arrival_time[step] = list(self.arrival_time[prev])
            self.arrival_s[step] = self.arrival_s[prev]
            self.arrival_speed[step] = self.arrival_speed[prev]
            self.hit[step] = self.hit[prev]
            self.computed[step] = self.computed[prev]
            return
        self.arrival_time[step] = list(result["arrival_time"])
        self.arrival_s[step] = result["arrival_s"]
        self.arrival_speed[step] = result["arrival_speed"]
        self.hit[step] = np.asarray(result["hit"], dtype=np.int8)
        self.computed[step] = True

    def set_truth(self, result: dict) -> None:
        """Arrival of the truth CME (a one-member result of EnsembleArrivalTimes.ensemble_arrival)."""
        self.true_arrival_time = result["arrival_time"][0]
        self.true_arrival_speed = float(result["arrival_speed"][0])

    def summary(self) -> dict:
        """Hit fractions and (weighted) arrival statistics for every step (NaN where nothing arrives)."""
        n = self.n_steps
        out = {k: np.full(n, np.nan) for k in (
            "hit_fraction", "weighted_hit_fraction", "weighted_mean_arrival_s", "weighted_sd_arrival_hours",
            "median_arrival_s", "weighted_mean_arrival_speed")}
        for k in range(n):
            if not self.computed[k]:
                continue
            w = np.exp(self.log_weight[k])
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                w = np.ones(self.n_members)
            w = w / w.sum()
            hit = self.hit[k].astype(bool) & np.isfinite(self.arrival_s[k])
            out["hit_fraction"][k] = hit.mean()
            out["weighted_hit_fraction"][k] = w[hit].sum()
            if hit.any() and w[hit].sum() > 0:
                wh = w[hit] / w[hit].sum()
                mean_s = np.sum(wh * self.arrival_s[k, hit])
                out["weighted_mean_arrival_s"][k] = mean_s
                out["weighted_sd_arrival_hours"][k] = np.sqrt(np.sum(wh * (self.arrival_s[k, hit] - mean_s) ** 2)) / 3600
                out["median_arrival_s"][k] = np.median(self.arrival_s[k, hit])
                out["weighted_mean_arrival_speed"][k] = np.sum(wh * self.arrival_speed[k, hit])
        return out

    def _s_to_datetime64(self, seconds: npt.NDArray[float]) -> npt.NDArray[np.datetime64]:
        return to_datetime64([
            self.model_init_time + datetime.timedelta(seconds=float(s)) if np.isfinite(s) else None
            for s in seconds
        ])

    def to_dataset(self) -> xr.Dataset:
        """The collected values as an xarray Dataset."""
        s = self.summary()
        arrival_time = np.stack([to_datetime64(row) for row in self.arrival_time])
        data_vars = dict(
            model_init_time=self.model_init_time,
            arrival_sim_time_days=((), self.sim_time_days, {"units": "day",
                                                             "long_name": "run length of the arrival-time model runs"}),
            arrival_time=(["n_obs", "n_ens"], arrival_time,
                          {"long_name": f"CME arrival time at {self.body} (NaT: miss / not arrived)"}),
            arrival_time_s=(["n_obs", "n_ens"], self.arrival_s,
                            {"units": "s", "long_name": "arrival time in seconds after model_init_time"}),
            arrival_speed=(["n_obs", "n_ens"], self.arrival_speed,
                           {"units": "km/s", "long_name": f"CME speed at {self.body} on arrival"}),
            hit=(["n_obs", "n_ens"], self.hit, {"long_name": f"1 if the member's CME arrives at {self.body}"}),
            log_weight=(["n_obs", "n_ens"], self.log_weight, {"long_name": "ensemble log-weight"}),
            arrival_computed=(["n_obs"], self.computed.astype(np.int8),
                              {"long_name": "1 if arrivals are available for this step"}),
            hit_fraction=(["n_obs"], s["hit_fraction"]),
            weighted_hit_fraction=(["n_obs"], s["weighted_hit_fraction"]),
            weighted_mean_arrival_time=(["n_obs"], self._s_to_datetime64(s["weighted_mean_arrival_s"]),
                                        {"long_name": "weighted mean arrival time of the members that hit"}),
            weighted_sd_arrival_hours=(["n_obs"], s["weighted_sd_arrival_hours"], {"units": "hour"}),
            median_arrival_time=(["n_obs"], self._s_to_datetime64(s["median_arrival_s"]),
                                 {"long_name": "(unweighted) median arrival time of the members that hit"}),
            weighted_mean_arrival_speed=(["n_obs"], s["weighted_mean_arrival_speed"], {"units": "km/s"}),
            true_arrival_time=((), to_datetime64([self.true_arrival_time])[0],
                               {"long_name": "arrival time of the truth CME (OSSEs; NaT otherwise)"}),
            true_arrival_speed=((), self.true_arrival_speed, {"units": "km/s"}),
        )
        return xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                obs_no=("n_obs", np.arange(self.n_steps)),
                ens_no=("n_ens", np.arange(self.n_members)),
                # time of the observation assimilated in each step (NaT for the prior)
                obs_time=("n_obs", to_datetime64([None] + self.obs_times)),
            ),
            attrs={"use_model": self.use_model, "body": self.body, "arrival_source": self.source,
                   "separate_arrival_runs": int(self.separate_runs),
                   "description": "CME arrival times of every ensemble member after every assimilation step; "
                                  "step 0 is the prior (see da_output.py)"},
        )

    def save(self, output_dir: str, final: bool = True) -> str:
        path = os.path.join(output_dir, "cme_arrival_times.nc")
        write_dataset(self.to_dataset(), path, final=final)
        return path
