import numpy as np
import numpy.typing as npt
import datetime
import os
import sys

import surf.surf as S
import surf.surf_inputs as Sin

import huxt.huxt_inputs as Hin

import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

import functools
from typing import Any

import yaml
from dotenv import load_dotenv
from pathlib import Path


from init_sir import initialise_cme_parameter_ensemble_dict
from cme_par_ens import CmeParEns
from aux_pf import PFSettings
from sir_observation_operator import ObsOpSettings


def events_file() -> Path:
    """events.yaml next to this file (data_assimilation_routine.py copies it into each experiment)"""
    return Path(__file__).with_name("events.yaml")


@functools.cache
def load_events(path: str | None = None) -> dict[str, dict[str, Any]]:
    """
    The CME events in events.yaml: model start time, observation file, first guesses and (for
    osse_ana.py) arrival time of each event, and the truth and first-guess CMEs of twin experiments,
    with the times as datetimes
    :param path: events.yaml to read, e.g. the copy in an experiment folder; None = the one next to
        this file
    """
    with open(events_file() if path is None else path) as f:
        events: dict[str, dict[str, Any]] = {
            str(code): event for code, event in (yaml.safe_load(f) or {}).items()
        }
    for event in events.values():
        event["model_start"] = _to_datetime(event["model_start"])
        if "arrival_time" in event:
            event["arrival_time"] = _to_datetime(event["arrival_time"])
        for key in ("obs_times_sta", "obs_times_stb"):
            if key in event:
                event[key] = [_to_datetime(t) for t in event[key]]
        for key in ("first_guess", "first_guess_other", "truth"):
            for time_key in ("time_at_21rs", "launch_time"):
                if key in event and time_key in event[key]:
                    event[key][time_key] = _to_datetime(event[key][time_key])
    return events


def get_event(code: str, path: str | None = None) -> dict[str, Any]:
    """The entry of event `code` in events.yaml (path: see load_events)"""
    events: dict[str, dict[str, Any]] = load_events(path)
    if code not in events:
        raise KeyError(
            f"Unknown event {code!r}: add it to events.yaml (events there: {', '.join(events)})"
        )
    return events[code]


def get_twin_cme(code: str, which: str, path: str | None = None) -> dict[str, Any]:
    """
    The truth or first-guess CME of twin experiment `code` in events.yaml, with lon or lat "earth"
    replaced by Earth's direction (HEEQ) at the model start
    :param code: Name of the twin experiment in events.yaml (e.g. "twin")
    :param which: "truth" or "first_guess"
    :param path: events.yaml to read (see load_events)
    :return: {"t_init": time at the model's inner boundary, "speed": km/s, "width": deg, "lon": deg,
        "lat": deg, "thick": solar radii}
    """
    event: dict[str, Any] = get_event(code, path)
    entry: str = (
        "truth"
        if which == "first_guess" and event.get("sample_first_guess", False)
        else which
    )
    if which not in ("truth", "first_guess") or "launch_time" not in event.get(
        entry, {}
    ):
        raise KeyError(
            f"Twin experiment {code!r} in events.yaml needs a {which} entry with a launch_time"
        )
    # Runs that draw their first guess from the prior around the truth: the first guess (the centre
    #  of those draws) is the truth
    if which == "first_guess" and event.get("sample_first_guess", False):
        which = "truth"
    cme: dict[str, Any] = event[which]
    twin_cme: dict[str, Any] = {"t_init": cme["launch_time"]}
    for key in ("speed", "width", "lon", "lat", "thick"):
        value: Any = cme[key]
        if (
            key in ("lon", "lat")
            and isinstance(value, str)
            and value.strip().lower() == "earth"
        ):
            earth: S.Observer = S.Observer(
                "EARTH", Time(event["model_start"], format="datetime")
            )
            twin_cme[key] = getattr(earth, key).to(u.deg).value
        else:
            twin_cme[key] = float(value)
    return twin_cme


def is_twin_experiment(code: str, path: str | None = None) -> bool:
    """
    Whether `code` is a twin experiment in events.yaml (an entry with a truth CME, e.g. "twin");
    path: see load_events
    """
    return "truth" in load_events(path).get(code, {})


def _to_datetime(value: Any) -> datetime.datetime:
    """A time from events.yaml (YYYY-MM-DD HH:MM, read as text or as a YAML timestamp)"""
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None)
    return datetime.datetime.fromisoformat(str(value))


@functools.cache
def load_environment() -> None:
    """
    Load the environment variables (OBS_DIR, OUT_BASE_DIR, WSA_DIR, ...) from ./.env, overriding any
    already set. Only the first call in a process loads the file, so a variable set by a script after
    that (e.g. OUT_BASE_DIR in twin_bias_test.py) is not reset by a later Setup().
    """
    load_dotenv(dotenv_path=Path(".", ".env"), override=True)
    return None


class Setup:
    def __init__(self) -> None:
        load_environment()
        self.surf_par_dict: dict[str, Any] = self.initialise_surf_parameters()
        self.obs_par_dict: dict[str, Any] = self.initialise_observation_parameters()
        self.true_cme_par_dict: CmeParEns = self.initialise_true_parameters()
        self.fg_cme_par_dict: dict[str, Any] = self.initialise_fg_cme_parameters()
        self.prior_cme_cov: dict[str, Any] = self.initialise_prior_cme_cov()
        self.da_par_dict: dict[str, Any] = self.initialise_da_parameters()

    def allowed_cov_types(self) -> set[str]:
        return {"uncorr", "donki", "mo_cone"}

    def get_use_parallel(self) -> bool:
        use_parallel: bool = True

        return use_parallel

    def get_mas_wsa_const(self) -> str:
        mas_wsa_const: str = "wsa"

        return mas_wsa_const

    def get_use_synthetic_obs(self) -> bool:
        use_synthetic_obs: bool = False

        # A twin experiment chosen as the event in get_ssw_event_craft (e.g. "twin") always uses
        #  synthetic observations
        if is_twin_experiment(self.get_ssw_event_craft()[0]):
            use_synthetic_obs = True

        return use_synthetic_obs

    def get_twin_experiment(self) -> str:
        """
        The twin experiment (entry of events.yaml with the truth and first-guess CMEs) used when
        get_use_synthetic_obs() is True
        """
        # The event chosen in get_ssw_event_craft if it is a twin experiment, otherwise "twin" (for
        #  synthetic observations with a real event's code)
        ssw_event: str = self.get_ssw_event_craft()[0]
        twin_experiment: str = ssw_event if is_twin_experiment(ssw_event) else "twin"

        return twin_experiment

    def get_twin_run_sampling(self) -> dict[str, bool]:
        """
        How the runs of a twin experiment differ (events.yaml, the twin's entry):
        sample_first_guess: each run's first guess is a draw from the prior around the truth (run
            numbers >= 0; negative run numbers start from the truth itself)
        new_obs_noise_per_run: each run gets its own observation noise (seeded by the run number)
        Both False for real observations, or when the entry does not set them.
        """
        sampling: dict[str, bool] = {
            "sample_first_guess": False,
            "new_obs_noise_per_run": False,
        }
        if self.get_use_synthetic_obs():
            event: dict[str, Any] = get_event(self.get_twin_experiment())
            for key in sampling:
                sampling[key] = bool(event.get(key, False))

        return sampling

    def get_bias_settings(self) -> dict[str, Any]:
        """
        Settings for the single observation-bias term h' = h + beta * p(h) (see obs_bias.py).
        mode:
            "off"      - no bias term (default; identical to runs without bias handling)
            "fixed"    - every member uses beta = prior_mean (a fixed offline correction, not estimated)
            "estimate" - beta is added to the state vector with prior N(prior_mean, prior_sd^2)
        prior_source:
            "zero"                  - prior_mean as given below (0 by default)
            "innovation_statistics" - prior_mean fitted to the bias in innovation_statistics_file
                                      (output of innovation_statistics.py), using
                                      innovation_statistics_source ("openloop" recommended)
        prior_sd: prior standard deviation of beta (deg at 25 deg modelled elongation for the
            "quadratic"/"linear" predictors; deg at every elongation for "constant")
        true_bias: bias injected into synthetic (twin) observations (deg); only used when
            get_use_synthetic_obs() is True
        :return: Dictionary of bias settings
        """
        bias_settings: dict[str, Any] = {
            "mode": "estimate",
            "prior_source": "zero",
            "prior_mean": 0.0,
            "prior_sd": 2.0,
            "innovation_statistics_file": None,
            "innovation_statistics_source": "openloop",
            "true_bias": 0.0,
        }

        return bias_settings

    def get_pf_settings(self) -> PFSettings:
        """
        Settings of the auxiliary particle filter (adaptive resampling and its ESS threshold, what to
        do when a member produces no elongation, the first-stage weights and the reuse of model runs);
        see aux_pf.PFSettings. For example, to read the elongations of unchanged particles from
        earlier model runs (same results, fewer model runs) and to use the current particles'
        likelihoods as the first-stage weights (no model runs of the shrunk particles):
            return PFSettings(reuse_model_runs=True, first_stage="current")
        :return: PFSettings
        """
        return PFSettings(reuse_model_runs=True)

    def get_obs_op_settings(self) -> ObsOpSettings:
        """
        Settings of the observation operator (synthetic imager elongation range, the tracker's search
        band, extrapolation before the tracked feature and diagnostics); see
        sir_observation_operator.ObsOpSettings. Its bias predictor is replaced by the one in
        get_bias_settings.
        :return: ObsOpSettings
        """
        return ObsOpSettings()

    def get_correlation_settings(self) -> dict[str, Any]:
        """
        Settings for handling correlated CME parameters in the Liu-West step of the auxiliary PF
        (see state_transform.py, and the "Correlated parameters" section of the error-propagation notes).
        enabled: master switch. False (default) leaves the filter exactly as before.
        reparam:
            "none"    - Liu-West step in the natural parameters
            "log_v"   - v -> ln v
            "tref_logv" - v -> ln v and t_init -> t_ref = t_init + (t_ref_rs - cme_init_rad) / v, the time
                        the CME reaches t_ref_rs. The elongations constrain the time the CME reaches
                        the distances where it is observed, so with t_ref_rs near those distances the
                        posterior ridge in (t_ref, ln v) is nearly straight and weakly correlated
        t_ref_rs: reference radius (solar radii) for "tref_logv"; None = the inner boundary r_min.
            ~50-60 Rs suits HI elongations of ~8-30 deg (front distance at the early/middle observations)
        whiten: do the Liu-West step in z = L^-1 (u - mu), with mu, L L^T the prior ensemble's mean and
            covariance (fixed for the run)
        prior_cov_shrinkage: shrinkage of the prior ensemble covariance towards its diagonal before the
            Cholesky factorisation (the prior has only n_members particles)
        shrinkage_alpha: Liu-West covariance used for the jitter is (1 - a) Sigma_ens + a diag(Sigma_ens),
            in the working coordinates (0 = plain weighted ensemble covariance)
        min_jitter_sd: floor on the jitter sd on each working axis (prior sds when whitened); 0 = off
        carry_from_prior: parameters of pars_in_state that the filter does not update; each particle
            keeps its prior value (inherited on resampling), e.g. ["width", "lat"]
        :return: Dictionary of correlation settings
        """
        correlation_settings: dict[str, Any] = {
            "enabled": False,
            "reparam": "tref_logv",
            "t_ref_rs": 50.0,
            "whiten": True,
            "prior_cov_shrinkage": 0.1,
            "shrinkage_alpha": 0.2,
            "min_jitter_sd": 0.02,
            "carry_from_prior": [],
        }

        return correlation_settings

    def get_ssw_event_craft(self) -> tuple[str, str, str]:
        ssw_event: str = "twin_base"

        # Observing spacecraft: from the event code (09A -> STEREO-A, 13B -> STEREO-B, ssw_ events ->
        #  STEREO-A); "twin" for a twin experiment (synthetic observations)
        craft: str
        if is_twin_experiment(ssw_event):
            craft = "twin"
        elif ssw_event[0:3] == "ssw":
            craft = "sta"
        elif ssw_event[2:3] == "A":
            craft = "sta"
        elif ssw_event[2:3] == "B":
            craft = "stb"
        else:
            raise ValueError(
                f"Event {ssw_event!r}: not a twin experiment in events.yaml (an entry with a truth) and"
                f" its spacecraft cannot be told from the code (e.g. 09A -> sta, 13B -> stb)"
            )

        img: str = "diff"

        return ssw_event, craft, img

    def get_surf_init_time(self) -> datetime.datetime:
        ssw_event: str
        ssw_event, _, _ = self.get_ssw_event_craft()
        if self.get_use_synthetic_obs():
            surf_init_time: datetime.datetime = get_event(self.get_twin_experiment())[
                "model_start"
            ]
        else:
            surf_init_time = get_event(ssw_event)["model_start"]

        return surf_init_time

    def get_use_model(self) -> str:
        """
        Function to get model we require to use
        :return: use_model: String containing either "surf", "surf_compress", "huxt"
        """

        poss_models: list[str] = ["surf", "compress_surf", "compress_surf3d", "huxt"]
        use_model: str = "compress_surf3d"

        assert use_model.lower() in poss_models

        return use_model.lower()

    def get_twin_latitude_range(self) -> tuple[Quantity, Quantity]:
        """
        Latitude range of the SURF3d grid: lat_min/lat_max of the twin experiment in events.yaml
        (default -30 and 30 deg) for synthetic observations, NaN (set from the imager's field of view)
        for real observations. Warns if the truth or first-guess CME is not inside the range
        :return: latitude_min, latitude_max
        """
        if not self.get_use_synthetic_obs():
            return np.nan * u.deg, np.nan * u.deg

        twin_experiment: str = self.get_twin_experiment()
        event: dict[str, Any] = get_event(twin_experiment)
        lat_min: float = float(event.get("lat_min", -30))
        lat_max: float = float(event.get("lat_max", 30))
        if not lat_min < lat_max:
            raise ValueError(
                f"Twin experiment {twin_experiment!r} in events.yaml: lat_min ({lat_min}) must be"
                f" less than lat_max ({lat_max})"
            )
        for which in ("truth", "first_guess"):
            cme: dict[str, Any] = get_twin_cme(twin_experiment, which)
            cme_lat_min: float = cme["lat"] - cme["width"] / 2
            cme_lat_max: float = cme["lat"] + cme["width"] / 2
            if cme_lat_min < lat_min or cme_lat_max > lat_max:
                print(
                    f"WARNING: twin experiment {twin_experiment!r}: the {which} CME spans latitudes"
                    f" {cme_lat_min:.1f} to {cme_lat_max:.1f} deg, outside the SURF3d grid"
                    f" ({lat_min:g} to {lat_max:g} deg); widen lat_min/lat_max in events.yaml"
                )

        return lat_min * u.deg, lat_max * u.deg

    def initialise_surf_parameters(self) -> dict[str, Any]:
        """
        Function to initialise the SURF parameters for simulation
        :return:
        """
        mas_or_wsa: str = self.get_mas_wsa_const()

        surf_init_time: datetime.datetime = self.get_surf_init_time()
        use_model: str = self.get_use_model().lower()

        vr_map: Quantity | float = np.nan * (u.km / u.s)
        vr_longs: Quantity = np.nan * u.rad
        vr_lats: Quantity = np.nan * u.rad
        br_in: Quantity = np.nan * u.nanotesla
        br_longs: Quantity = np.nan * u.rad
        br_lats: Quantity = np.nan * u.rad
        cr_num: float = np.nan

        surf_init_time_astro: Time = Time(surf_init_time, format="datetime")
        ert: S.Observer = S.Observer("EARTH", surf_init_time_astro)
        earth_lat: Quantity = ert.lat.to(u.deg)
        earth_lon: Quantity = ert.lon.to(u.deg)
        print(use_model)
        lon_start: Quantity[u.deg] = -70 * u.deg
        lon_stop: Quantity[u.deg] = 70 * u.deg
        # Latitude range of the SURF3d grid (compress_surf3d): for twin experiments lat_min/lat_max of
        #  the twin entry in events.yaml; for real observations NaN, i.e. the range is set from the
        #  imager's field of view in sir_observation_operator.ObservationOperator
        latitude_min: Quantity[u.deg]
        latitude_max: Quantity[u.deg]
        latitude_min, latitude_max = self.get_twin_latitude_range()
        n_lat_surf: int = 45
        sim_time: Quantity[u.day] = 3 * u.day
        dt_scale: int = 10
        cme_init_rad: Quantity[u.solRad] = 12 * u.solRad
        r_min: Quantity[u.solRad] = 21.5 * u.solRad
        cme_fixed_duration: bool = True
        fixed_duration: Quantity[u.s] = 12 * 3600 * u.s

        if mas_or_wsa.upper() == "MAS":
            cr_num = int(np.trunc(sn.carrington_rotation_number(surf_init_time)))

            if use_model in ["surf", "compress_surf"]:
                vr_in = Sin.get_MAS_long_profile(cr_num, lat=0.0 * u.deg)
            elif use_model in ["huxt"]:
                vr_in = Hin.get_MAS_long_profile(cr_num, lat=0.0 * u.deg)
            else:
                sys.exit(
                    "Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'"
                )

        elif mas_or_wsa.upper() == "WSA":
            wsa_dir: str | None = os.getenv("WSA_DIR")
            output_path: str = Sin.get_WSA_from_ISWA(
                timestamp=surf_init_time,
                datadir=wsa_dir,
            )

            # Get input speeds for all longitudes for input into surf
            # Decelerate the WSA map from 1-AU calibrated speeds to expected 21.5 rS values
            # Load in map and Earth lon cut, decaccelerate both from 215 to 21.5.
            surf_init_time_astro = Time(surf_init_time, format="datetime")
            ert = S.Observer("EARTH", surf_init_time_astro)
            earth_lat = ert.lat.to(u.deg)

            # Can't use hin.map_vmap_inwards as that shifts the longitudes and interpolates, which we do not want for
            # WSA
            if use_model in ["surf", "compress_surf"]:
                vr_in = Sin.get_WSA_long_profile(output_path, lat=earth_lat)
                n_lon: int = len(vr_in)
                v_lon_grid_step: float = 2 * np.pi / n_lon
                v_longs: Quantity = (
                    np.array([i * v_lon_grid_step for i in range(n_lon)]) * u.rad
                )

                vr_in, lon_temp = Sin.map_v_inwards(
                    vr_in, 215 * u.solRad, v_longs, r_min
                )
            elif use_model in ["surf3d", "compress_surf3d"]:
                print("Hello")
                vr_in, vr_longs, vr_lats, br_in, br_longs, br_lats, cr_num = (
                    Sin.get_WSA_maps(output_path)
                )
                print(
                    f"np.shape(vr_lats) = {np.shape(vr_lats)}, np.shape(vr_longs) = {np.shape(vr_longs)}"
                )
                if np.isfinite(br_in).any():
                    vr_in, br_in = Sin.map_vmap_inwards(
                        v_map=vr_in,
                        v_map_lat=vr_lats,
                        v_map_long=vr_longs,
                        b_map=br_in,
                        r_orig=215 * u.solRad,
                        r_new=r_min,
                        gamma=1.5,
                    )
                else:
                    vr_in = Sin.map_vmap_inwards(
                        v_map=vr_in,
                        v_map_lat=vr_lats,
                        v_map_long=vr_longs,
                        r_orig=215 * u.solRad,
                        r_new=r_min,
                        gamma=1.5,
                    )
                print(np.shape(vr_in))

            elif use_model in ["huxt"]:
                vr_in = Hin.get_WSA_long_profile(output_path, lat=earth_lat)
                n_lon = len(vr_in)
                v_lon_grid_step = 2 * np.pi / n_lon
                v_longs = np.array([i * v_lon_grid_step for i in range(n_lon)]) * u.rad

                lon_temp: Quantity
                vr_in, lon_temp = Hin.map_v_inwards(
                    vr_in, 215 * u.solRad, v_longs, r_min
                )
            else:
                sys.exit(
                    "Unknown model name, expected either 'surf', 'compress_surf' or 'huxt'"
                )

        elif mas_or_wsa.upper() == "CONST":
            vr_in: Quantity[u.km / u.s] = np.zeros(128) + 400 * u.km / u.s
        else:
            print("Defaulting to constant ambient solar wind of 400 km/s")
            vr_in: Quantity[u.km / u.s] = np.zeros(128) + 400 * u.km / u.s

        out_surf_par: dict[str, Any] = {
            "use_model": use_model,
            "surf_init_time": surf_init_time,
            "vr_in": vr_in,
            "vr_longs": vr_longs,
            "vr_lats": vr_lats,
            "br_in": br_in,
            "br_longs": br_longs,
            "br_lats": br_lats,
            "lon_start": lon_start,
            "lon_stop": lon_stop,
            "latitude_min": latitude_min,
            "latitude_max": latitude_max,
            "n_lat_surf": n_lat_surf,
            "sim_time": sim_time,
            "dt_scale": dt_scale,
            "r_min": r_min,
            "cme_init_rad": cme_init_rad,
            "cme_fixed_duration": cme_fixed_duration,
            "fixed_duration": fixed_duration,
        }
        print(np.shape(vr_in))
        return out_surf_par

    def initialise_true_parameters(self) -> CmeParEns:
        surf_init_time: datetime.datetime = self.get_surf_init_time()
        true_cme_par_dict: CmeParEns = initialise_cme_parameter_ensemble_dict(
            1, surf_init_time
        )

        # Initialise true CME parameters: the truth of the twin experiment in events.yaml. The
        #  launch time is kept relative to the model start (so it is the truth's launch_time in twin
        #  runs; runs with real observations have no truth CME)
        twin_experiment: str = self.get_twin_experiment()
        truth: dict[str, Any] = get_twin_cme(twin_experiment, "truth")
        true_cme_t_init: datetime.datetime = surf_init_time + (
            truth["t_init"] - get_event(twin_experiment)["model_start"]
        )
        true_cme_speed: float = truth["speed"]
        true_cme_width: float = truth["width"]
        true_cme_lon: float = truth["lon"]
        true_cme_lat: float = truth["lat"]
        true_cme_thick: float = truth["thick"]

        true_cme_par_dict["t_init"]: datetime.datetime = true_cme_t_init
        true_t_init_str: str = true_cme_par_dict["t_init"].strftime("%Y%m%d-%H%M")

        true_cme_par_dict["v"]: Quantity[u.km / u.s] = true_cme_speed * u.km / u.s
        true_cme_par_dict["width"]: Quantity[u.deg] = true_cme_width * u.deg
        true_cme_par_dict["lon"]: Quantity[u.deg] = true_cme_lon * u.deg
        true_cme_par_dict["lat"]: Quantity[u.deg] = true_cme_lat * u.deg
        true_cme_par_dict["thick"]: Quantity[u.solRad] = true_cme_thick * u.solRad

        # Observation bias injected into synthetic (twin) observations (see get_bias_settings)
        true_cme_par_dict["bias"]: float = (
            float(self.get_bias_settings().get("true_bias", 0.0))
            if self.get_use_synthetic_obs()
            else 0.0
        )

        return true_cme_par_dict

    def initialise_fg_cme_parameters(self) -> dict[str, Any]:
        # Initialise CME parameters
        ssw_event: str
        ssw_event, _, _ = self.get_ssw_event_craft()
        donki_cme: bool = True

        if self.get_use_synthetic_obs():
            # First guess of the twin experiment in events.yaml
            first_guess: dict[str, Any] = get_twin_cme(
                self.get_twin_experiment(), "first_guess"
            )
            cme_at_12rs: datetime.datetime = first_guess["t_init"]
            fg_cme_speed: float = first_guess["speed"]
            fg_cme_width: float = first_guess["width"]
            fg_cme_lon: float = first_guess["lon"]
            fg_cme_lat: float = first_guess["lat"]
            fg_cme_thick: float = first_guess["thick"]
        else:
            # First guess from events.yaml: DONKI, or the other first guess if donki_cme is False
            #  (events without one use the DONKI values)
            event: dict[str, Any] = get_event(ssw_event)
            first_guess: dict[str, Any] = (
                event["first_guess"]
                if donki_cme
                else event.get("first_guess_other", event["first_guess"])
            )
            cme_at_21rs: datetime.datetime = first_guess["time_at_21rs"]
            fg_cme_speed: float = first_guess["speed"]

            # Launch time at 12 Rs (cme_init_rad), 9.5 Rs before 21.5 Rs at the first-guess speed
            dist_from_12_to_21rs_km: float = (9.5 * u.solRad).to(u.km).value
            time_from_12_to_21rs: float = dist_from_12_to_21rs_km / fg_cme_speed
            cme_at_12rs: datetime.datetime = cme_at_21rs - datetime.timedelta(
                seconds=time_from_12_to_21rs
            )

            fg_cme_width: float = first_guess["width"]
            fg_cme_lon: float = first_guess["lon"]
            fg_cme_lat: float = first_guess["lat"]
            fg_cme_thick: float = first_guess["thick"]

        # Time taken to get from cme_init_rad to r_min
        #  (reuse self.surf_par_dict if already built: initialise_surf_parameters re-reads the WSA map)
        if hasattr(self, "surf_par_dict"):
            surf_dict: dict[str, Any] = self.surf_par_dict
        else:
            surf_dict = self.initialise_surf_parameters()
        surf_cme_ir: Quantity[u.solRad] = surf_dict["cme_init_rad"]
        surf_r_min: Quantity[u.solRad] = surf_dict["r_min"]

        dist_in_km: Quantity[u.km] = (surf_r_min - surf_cme_ir).to(u.km)
        seconds_to_r_min: float = dist_in_km.to(u.km).value / fg_cme_speed

        fg_cme_t_init: datetime.datetime = cme_at_12rs  # (
        # )

        if self.get_use_synthetic_obs():
            sd_cme_t_init: float = 0  # In seconds
            sd_cme_speed: float = 50  # In km/s
            sd_cme_width: float = 5  # In deg
            sd_cme_lon: float = 5  # In deg
            sd_cme_lat: float = 0  # In deg
            sd_cme_thick: float = 0  # In solRad
        else:
            sd_cme_t_init: float = 3600  # In seconds
            sd_cme_speed: float = 50  # In km/s
            sd_cme_width: float = 5  # In deg
            sd_cme_lon: float = 5  # In deg
            sd_cme_lat: float = 5  # In deg
            sd_cme_thick: float = 0  # In solRad

        fg_cme_par_dict: dict[str, Any] = {
            "fg_cme_t_init": fg_cme_t_init,
            "fg_cme_speed": fg_cme_speed,
            "fg_cme_width": fg_cme_width,
            "fg_cme_lon": fg_cme_lon,
            "fg_cme_lat": fg_cme_lat,
            "fg_cme_thick": fg_cme_thick,
            "sd_cme_t_init": sd_cme_t_init,
            "sd_cme_speed": sd_cme_speed,
            "sd_cme_width": sd_cme_width,
            "sd_cme_lon": sd_cme_lon,
            "sd_cme_lat": sd_cme_lat,
            "sd_cme_thick": sd_cme_thick,
        }

        return fg_cme_par_dict

    def initialise_prior_cme_cov(self) -> dict[str, Any]:
        if self.get_use_synthetic_obs():
            cme_cov_type: str = "uncorr"
        else:
            cme_cov_type = "mo_cone"

        # Scale correlation matrix to make covariance matrix
        scale_corr: bool = True
        use_log_v: bool = True

        if self.get_use_synthetic_obs():
            # Initialise CME parameters for uniform distribution
            sd_t_init: float = 0  # In seconds

            if use_log_v:
                sd_speed: float = 0.3  # Multiplicative variance
            else:
                sd_speed: float = 50  # In km/s
            sd_width: float = 5  # In deg
            sd_lon: float = 5  # In deg
            sd_lat: float = 0  # In deg
            sd_thick: float = 0  # In solRad
        else:
            # Initialise CME parameters for uniform distribution
            sd_t_init: float = 3600  # In seconds

            if use_log_v:
                sd_speed: float = 0.3  # Multiplicative variance
            else:
                sd_speed: float = 50  # In km/s
            sd_width: float = 5  # In deg
            sd_lon: float = 5  # In deg
            sd_lat: float = 5  # In deg
            sd_thick: float = 0  # In solRad

        prior_cme_cov_dict: dict[str, Any] = {
            "cme_cov_type": cme_cov_type,
            "scale_corr": scale_corr,
            "use_log_v": use_log_v,
            "sd_t_init": sd_t_init,
            "sd_speed": sd_speed,
            "sd_width": sd_width,
            "sd_lon": sd_lon,
            "sd_lat": sd_lat,
            "sd_thick": sd_thick,
        }

        return prior_cme_cov_dict

    def initialise_observation_parameters(self) -> dict[str, Any]:
        n_obs: int = 8
        obs_cadence_hr: int = 3  # Observations cadence in hours
        true_cme_dict: CmeParEns = self.initialise_true_parameters()
        true_cme_t_init: datetime.datetime = true_cme_dict["t_init"]

        surf_init_time: datetime.datetime = self.get_surf_init_time()
        obs_radius: Quantity[u.AU] = 1.0 * u.AU
        obs_lon: Quantity[u.deg] = 300 * u.deg
        obs_lat: Quantity[u.deg] = 0 * u.deg

        obs_cov: list[float] = [0.5]  # * np.eye(len(obs))

        obs_times: list[datetime.datetime] = [
            true_cme_t_init + datetime.timedelta(hours=10 + (obs_cadence_hr * i))
            for i in range(1, 1 + n_obs)
        ]

        use_synthetic_obs: bool = self.get_use_synthetic_obs()
        obs_rng_seed: int = 4096

        craft: str
        img: str
        ssw_event: str
        ssw_event, craft, img = self.get_ssw_event_craft()
        if use_synthetic_obs:
            # Synthetic (twin/OSSE) observations are taken from the observer defined above (obs_radius,
            #  obs_lon, obs_lat). Without this, craft is still the real spacecraft of ssw_event (e.g.
            #  "sta"), which ObservationOperator uses to overwrite obs_lon with that craft's real position
            #  for both the truth run and every ensemble member, so obs_lon = 300 deg was silently ignored.
            craft = "twin"
            obs_filenames: list[str] = ["Synth_obs"]
        else:
            obs_file_name: str = get_event(ssw_event)["obs_file"]

            # Initialise a dummy SURF model to get
            obs_filenames: list[str] = [
                os.path.join(str(os.getenv("OBS_DIR")), obs_file_name)
            ]
        use_parallel: bool = self.get_use_parallel()

        # Set limits on minimum and maximum elongation to use for simulating HI
        elong_min: float = 4.0
        elong_max: float = 30.0
        # Set limits on minimum and maximum elongation observations to use
        elong_min_obs: float = 4.0
        elong_max_obs: float = 30.0

        obs_par_dict: dict[str, Any] = {
            "n_obs": n_obs,
            "obs_radius": obs_radius,
            "obs_lon": obs_lon,
            "obs_lat": obs_lat,
            "obs_cov": obs_cov,
            "elong_min": elong_min,
            "elong_max": elong_max,
            "elong_min_obs": elong_min_obs,
            "elong_max_obs": elong_max_obs,
            "obs_times": obs_times,
            "use_synthetic_obs": use_synthetic_obs,
            "obs_filenames": obs_filenames,
            "obs_rng_seed": obs_rng_seed,
            "ssw_event": ssw_event,
            "craft": craft,
            "img": img,
            "use_parallel": use_parallel,
        }

        return obs_par_dict

    def initialise_da_parameters(self) -> dict[str, Any]:
        n_members: int = 50

        run_start: int = 0
        n_runs: int = 20

        delta_aux_pf: float = 0.98
        if self.get_use_synthetic_obs():
            pars_in_state_vector: list[str] = ["v", "width", "lon"]
        else:
            pars_in_state_vector: list[str] = ["t_init", "v", "width", "lon", "lat"]
        time_tolerance: Quantity[u.day] = 0.5 * u.day

        # Estimate the observation-bias coefficient with the CME parameters (see get_bias_settings)
        bias_settings: dict[str, Any] = self.get_bias_settings()
        if (
            bias_settings.get("mode", "off") == "estimate"
            and "bias" not in pars_in_state_vector
        ):
            pars_in_state_vector = pars_in_state_vector + ["bias"]

        da_par_dict: dict[str, Any] = {
            "n_members": n_members,
            "run_start": run_start,
            "n_runs": n_runs,
            "delta_aux_pf": delta_aux_pf,
            "pars_in_state": pars_in_state_vector,
            "time_tolerance": time_tolerance,
            "bias_settings": bias_settings,
            "correlation_settings": self.get_correlation_settings(),
            "pf_settings": self.get_pf_settings(),
            "obs_op_settings": self.get_obs_op_settings(),
        }

        return da_par_dict
