"""
da_surf_setup.py

Configuration for a data-assimilation experiment. ``Setup()`` collects every setting into six
dictionaries that ``RunDataAssimilationRoutine`` (data_assimilation_routine.py) reads:

    surf_par_dict      : forward model, start time, ambient solar wind (WSA/MAS/constant), domain
    obs_par_dict       : observation times / files, observer position, error variance, elongation range
    true_cme_par_dict  : "truth" CME used to make synthetic observations (OSSEs)
    fg_cme_par_dict    : first-guess CME (prior mean) and its spread
    prior_cme_cov      : how the prior ensemble is drawn (cme_cov_type, standard deviations, log-speed)
    da_par_dict        : ensemble size, number of runs, Aux-PF settings, parameters to estimate

To change the experiment, edit the ``get_*`` switches (model, event, synthetic vs real obs, ...) and the
values in the ``initialise_*`` methods. Per-event tables (start times, first guesses, observation files)
are in event_catalogue.py.

NB: the .env file in the working directory is loaded on import (WSA_DIR, OBS_DIR, OUT_BASE_DIR, ...).
"""
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

import surf.surf as S
import surf.surf_inputs as Sin

import huxt.huxt_inputs as Hin

import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from dotenv import load_dotenv

from cme_par_ens import CmeParEns
from event_catalogue import (
    FIRST_GUESS_DONKI, FIRST_GUESS_LUKE_PAPER, OBS_FILE_NAMES, SURF_INIT_TIMES, FirstGuessCME,
)
from init_sir import initialise_cme_parameter_ensemble_dict

env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path, override=True)

UNKNOWN_EVENT_MSG = "Unknown ssw_event name, expected ssw_event = 'ssw_007', 'ssw_008', 'ssw_009' or 'ssw_012'"


class Setup:
    """
    All settings for one data-assimilation experiment (see module docstring).

    Example::

        setup = Setup()
        setup.surf_par_dict["use_model"], setup.da_par_dict["n_members"]
    """
    def __init__(self) -> None:
        # NB: the order matters: the first-guess parameters reuse self.surf_par_dict
        self.surf_par_dict = self.initialise_surf_parameters()
        self.obs_par_dict = self.initialise_observation_parameters()
        self.true_cme_par_dict = self.initialise_true_parameters()
        self.fg_cme_par_dict = self.initialise_fg_cme_parameters()
        self.prior_cme_cov = self.initialise_prior_cme_cov()
        self.da_par_dict = self.initialise_da_parameters()


    def allowed_cov_types(self) -> set[str]:
        """Allowed values of cme_cov_type (see initialise_prior_cme_cov)."""
        return {"uncorr", "donki", "mo_cone"}


    # ------------------------------------------------------------------------------------------
    # Experiment switches
    # ------------------------------------------------------------------------------------------
    def get_use_parallel(self) -> bool:
        """Run the ensemble members in parallel (multiprocessing)."""
        use_parallel = True

        return use_parallel


    def get_mas_wsa_const(self) -> str:
        """Ambient solar wind source: "wsa", "mas" or "const" (uniform 400 km/s)."""
        mas_wsa_const = "wsa"

        return mas_wsa_const


    def get_use_synthetic_obs(self) -> bool:
        """True for an OSSE (synthetic observations of a truth CME), False for real observations."""
        use_synthetic_obs = False

        return use_synthetic_obs


    def get_ssw_event_craft(self) -> tuple[str, str, str]:
        """
        Event to assimilate and the spacecraft / image type it was observed with. For the "NNA"/"NNB"
        events the craft follows from the last letter (A -> STEREO-A, B -> STEREO-B).
        :return: ssw_event, craft ("sta"/"stb"), img ("diff" = differenced images)
        """
        ssw_event = "09A"

        if ssw_event[0:3] != "ssw":
            if ssw_event[2] == "A":
                craft = "sta"
            elif ssw_event[2] == "B":
                craft = "stb"
        else:
            craft="sta"

        img = "diff"

        return ssw_event, craft, img


    def get_surf_init_time(self) -> datetime.datetime:
        """
        Start time of the model run: 2012-01-01 for synthetic observations, otherwise looked up for
        the current event in event_catalogue.SURF_INIT_TIMES.
        :return: surf_init_time
        """
        ssw_event, _, _ = self.get_ssw_event_craft()
        if self.get_use_synthetic_obs():
            surf_init_time: datetime.datetime = datetime.datetime(2012, 1, 1, 0, 0)
        else:
            if ssw_event not in SURF_INIT_TIMES:
                sys.exit(UNKNOWN_EVENT_MSG)
            surf_init_time: datetime.datetime = SURF_INIT_TIMES[ssw_event]

        return surf_init_time


    def get_use_model(self) -> str:
        """
        Forward model to use
        :return: use_model: "surf", "compress_surf", "compress_surf3d" or "huxt"
        """

        poss_models = ["surf", "compress_surf", "compress_surf3d", "huxt"]
        use_model = "compress_surf3d"

        assert use_model.lower() in poss_models

        return use_model.lower()


    # ------------------------------------------------------------------------------------------
    # Parameter dictionaries
    # ------------------------------------------------------------------------------------------
    def initialise_surf_parameters(self) -> dict:
        """
        Model settings and the ambient solar wind at the inner boundary.
        WSA maps (calibrated at 1 AU / 215 rS) are mapped inwards to r_min; for the 3D model the full
        lat-lon map (and B map, if present) is used, otherwise a longitude profile at Earth's latitude.
        :return: surf_par_dict
        """
        mas_or_wsa = self.get_mas_wsa_const()

        surf_init_time = self.get_surf_init_time()
        use_model = self.get_use_model().lower()

        vr_map = np.nan * (u.km / u.s)
        vr_longs = np.nan * u.rad
        vr_lats = np.nan * u.rad
        br_in = np.nan * u.nanotesla
        br_longs = np.nan * u.rad
        br_lats = np.nan * u.rad
        cr_num = np.nan

        surf_init_time_astro = Time(surf_init_time, format="datetime")
        ert = S.Observer("EARTH", surf_init_time_astro)
        earth_lat = ert.lat.to(u.deg)
        earth_lon = ert.lon.to(u.deg)
        print(use_model)

        # Model domain and CME settings
        lon_start: Quantity[u.deg] = -70 * u.deg
        lon_stop: Quantity[u.deg] = 70 * u.deg
        latitude_min = -30 * u.deg
        latitude_max = 30 * u.deg
        n_lat_surf = 45
        sim_time: Quantity[u.day] = 2 * u.day
        dt_scale: int = 1
        cme_init_rad: Quantity[u.solRad] = 12 * u.solRad
        r_min: Quantity[u.solRad] = 21.5 * u.solRad
        cme_fixed_duration: bool = True
        fixed_duration: Quantity[u.s] = 12 * 3600 * u.s

        # Ambient solar wind at the inner boundary
        if mas_or_wsa.upper() == "MAS":
            cr_num: int = int(np.trunc(sn.carrington_rotation_number(surf_init_time)))

            if use_model in ["surf", "compress_surf"]:
                vr_in = Sin.get_MAS_long_profile(cr_num, lat=0.0 * u.deg)
            elif use_model in ["huxt"]:
                vr_in = Hin.get_MAS_long_profile(cr_num, lat=0.0 * u.deg)
            else:
                sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

        elif mas_or_wsa.upper() == "WSA":
            wsa_dir = os.getenv('WSA_DIR')
            output_path = Sin.get_WSA_from_ISWA(
                timestamp=surf_init_time,
                datadir=wsa_dir,
            )
            # (alternative: wsa_reader.ReadWSAFiles(surf_init_time, output_dir=wsa_dir).download_file())

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
                n_lon = len(vr_in)
                v_lon_grid_step = 2 * np.pi / n_lon
                v_longs = np.array([
                    i * v_lon_grid_step for i in range(n_lon)
                ]) * u.rad

                vr_in, lon_temp = Sin.map_v_inwards(vr_in, 215 * u.solRad, v_longs, r_min)
            elif use_model in ["surf3d", "compress_surf3d"]:
                print("Hello")
                vr_in, vr_longs, vr_lats, br_in, br_longs, br_lats, cr_num = Sin.get_WSA_maps(output_path)
                print(f"np.shape(vr_lats) = {np.shape(vr_lats)}, np.shape(vr_longs) = {np.shape(vr_longs)}")
                if np.isfinite(br_in).any():
                    vr_in, br_in = Sin.map_vmap_inwards(
                        v_map=vr_in,
                        v_map_lat=vr_lats,
                        v_map_long=vr_longs,
                        b_map=br_in,
                        r_orig=215 * u.solRad,
                        r_new=r_min,
                        gamma=1.5
                    )
                else:
                    vr_in = Sin.map_vmap_inwards(
                        v_map=vr_in,
                        v_map_lat=vr_lats,
                        v_map_long=vr_longs,
                        r_orig=215 * u.solRad,
                        r_new=r_min,
                        gamma=1.5
                    )
                print(np.shape(vr_in))

            elif use_model in ["huxt"]:
                vr_in = Hin.get_WSA_long_profile(output_path, lat=earth_lat)
                n_lon = len(vr_in)
                v_lon_grid_step = 2 * np.pi / n_lon
                v_longs = np.array([
                    i * v_lon_grid_step for i in range(n_lon)
                ]) * u.rad

                vr_in, lon_temp = Hin.map_v_inwards(vr_in, 215 * u.solRad, v_longs, r_min)
            else:
                sys.exit("Unknown model name, expected either 'surf', 'compress_surf' or 'huxt'")


        elif mas_or_wsa.upper() == "CONST":
            vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.zeros(128) + 400 * u.km / u.s
        else:
            print("Defaulting to constant ambient solar wind of 400 km/s")
            vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.zeros(128) + 400 * u.km / u.s

        out_surf_par = {
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
        """Truth CME for synthetic observations: launched (at cme_init_rad) after the model start."""
        surf_init_time = self.get_surf_init_time()
        true_cme_par_dict = initialise_cme_parameter_ensemble_dict(
            1, surf_init_time
        )

        # Initialise true CME parameters
        true_cme_t_init: datetime.datetime = surf_init_time + datetime.timedelta(hours=1)
        true_cme_speed: float = 495
        true_cme_width: float = 37.4
        true_cme_lon: float = 0
        true_cme_lat: float = 0
        true_cme_thick: float = 0

        true_cme_par_dict["t_init"]: datetime.datetime = true_cme_t_init
        true_t_init_str: str = true_cme_par_dict["t_init"].strftime("%Y%m%d-%H%M")

        true_cme_par_dict["v"]: Quantity[u.km / u.s] = true_cme_speed * u.km / u.s
        true_cme_par_dict["width"]: Quantity[u.deg] = true_cme_width * u.deg
        true_cme_par_dict["lon"]: Quantity[u.deg] = true_cme_lon * u.deg
        true_cme_par_dict["lat"]: Quantity[u.deg] = true_cme_lat * u.deg
        true_cme_par_dict["thick"]: Quantity[u.solRad] = true_cme_thick * u.solRad

        return true_cme_par_dict


    def initialise_fg_cme_parameters(self) -> dict:
        """
        First-guess CME parameters (the mean the prior ensemble is drawn around) and their spread.
          * Synthetic observations: the truth CME's speed/width, at Earth's longitude/latitude, launched
            after the model start.
          * Real observations: the event's entry in event_catalogue (DONKI by default), whose time is
            the arrival at 21.5 rS. That time is converted to the time at 12 rS assuming constant speed.
        Finally the launch time is moved from 12 rS to cme_init_rad -> r_min of the model in the same way.
        :return: fg_cme_par_dict with keys fg_cme_* (parameters) and sd_cme_* (spreads)
        """
        ssw_event, _, _ = self.get_ssw_event_craft()
        # Choose the first-guess catalogue: DONKI (True) or the "Luke paper" values (False)
        donki_cme = True

        if self.get_use_synthetic_obs():
            surf_init_time = self.get_surf_init_time()
            cme_at_12rs: datetime.datetime = (
                    surf_init_time + datetime.timedelta(hours=1)
            )
            surf_init_time_astro = Time(surf_init_time, format="datetime")
            ert = S.Observer("EARTH", surf_init_time_astro)
            earth_lat = ert.lat.to(u.deg)
            earth_lon = ert.lon.to(u.deg)

            fg_cme_speed: float = 495
            fg_cme_width: float = 37.4
            fg_cme_lon: float = earth_lon.value
            fg_cme_lat: float = earth_lat.value
            fg_cme_thick: float = 0
        else:
            fg_table: dict[str, FirstGuessCME] = FIRST_GUESS_DONKI if donki_cme else FIRST_GUESS_LUKE_PAPER
            if ssw_event not in fg_table:
                sys.exit(UNKNOWN_EVENT_MSG)
            fg_cme: FirstGuessCME = fg_table[ssw_event]

            cme_at_21rs: datetime.datetime = fg_cme.t_21rs
            fg_cme_speed: float = fg_cme.speed

            # Time at 12 rS = time at 21.5 rS - (9.5 rS / speed)
            dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
            time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
            cme_at_12rs: datetime.datetime = (
                    cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
            )

            fg_cme_width: float = fg_cme.width
            fg_cme_lon: float = fg_cme.lon
            fg_cme_lat: float = fg_cme.lat
            fg_cme_thick: float = fg_cme.thick

        # Time taken to get from cme_init_rad to r_min
        #  (reuse self.surf_par_dict if already built: initialise_surf_parameters re-reads the WSA map)
        if hasattr(self, "surf_par_dict"):
            surf_dict = self.surf_par_dict
        else:
            surf_dict = self.initialise_surf_parameters()
        surf_cme_ir: Quantity[u.solRad] = surf_dict["cme_init_rad"]
        surf_r_min: Quantity[u.solRad] = surf_dict["r_min"]

        dist_in_km: Quantity[u.km] = (surf_r_min - surf_cme_ir).to(u.km)
        seconds_to_r_min: float = dist_in_km.to(u.km).value / fg_cme_speed

        fg_cme_t_init: datetime.datetime = (
                cme_at_12rs + datetime.timedelta(seconds=seconds_to_r_min)
        )

        # Spread of the first guess (used to perturb the prior mean between runs)
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

        fg_cme_par_dict = {
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
            "sd_cme_thick": sd_cme_thick
        }

        return fg_cme_par_dict


    def initialise_prior_cme_cov(self) -> dict:
        """
        How the prior ensemble is drawn around the (perturbed) first guess:
          cme_cov_type "uncorr" (uniform, OSSEs) or "mo_cone" (Met Office cone-CME covariance, real obs);
          scale_corr: rescale the catalogue correlation matrix by the sd_* values;
          use_log_v: sample log(speed); sd_speed is then a standard deviation of log(v).
        :return: prior_cme_cov_dict
        """
        if self.get_use_synthetic_obs():
            cme_cov_type = "uncorr"
        else:
            cme_cov_type = "mo_cone"

        # Scale correlation matrix to make covariance matrix
        scale_corr = True
        use_log_v = True

        if self.get_use_synthetic_obs():
            # Initialise CME parameters for uniform distribution
            sd_t_init: float = 0  # In seconds

            if use_log_v:
                sd_speed: float = 0.1  # Multiplicative variance
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
                sd_speed: float = 0.1  # Multiplicative variance
            else:
                sd_speed: float = 50  # In km/s
            sd_width: float = 5  # In deg
            sd_lon: float = 5  # In deg
            sd_lat: float = 5  # In deg
            sd_thick: float = 0  # In solRad

        prior_cme_cov_dict = {
            "cme_cov_type": cme_cov_type,
            "scale_corr": scale_corr,
            "use_log_v": use_log_v,
            "sd_t_init": sd_t_init,
            "sd_speed": sd_speed,
            "sd_width": sd_width,
            "sd_lon": sd_lon,
            "sd_lat": sd_lat,
            "sd_thick": sd_thick
        }

        return prior_cme_cov_dict


    def initialise_observation_parameters(self) -> dict:
        """
        Observation settings. For synthetic observations, n_obs observations are made every
        obs_cadence_hr hours, starting 13 hours after the truth CME launch, by an observer at obs_lon.
        For real observations the times/positions come from the observation file of the event
        (obs_times, n_obs and obs_lon are then replaced by the values read from file).
        :return: obs_par_dict
        """
        n_obs: int = 8
        obs_cadence_hr = 3  # Observations cadence in hours
        true_cme_dict = self.initialise_true_parameters()
        true_cme_t_init: datetime.datetime = true_cme_dict["t_init"]

        surf_init_time = self.get_surf_init_time()
        obs_radius: Quantity[u.AU] = 1.0 * u.AU
        obs_lon: Quantity[u.deg] = 300 * u.deg
        obs_lat: Quantity[u.deg] = 0 * u.deg

        # Observation error *variance* (deg^2)
        obs_cov: list[float] = [0.5]

        obs_times: list[datetime.datetime] = [
            true_cme_t_init + datetime.timedelta(hours=10 + (obs_cadence_hr * i))
            for i in range(1, 1 + n_obs)
        ]

        use_synthetic_obs: bool = self.get_use_synthetic_obs()
        obs_rng_seed: int = 4096


        ssw_event, craft, img = self.get_ssw_event_craft()
        if use_synthetic_obs:
            bias_term_bool = False  # True
            obs_filenames: list[str] = [
                "Synth_obs"
            ]
        else:
            bias_term_bool = False

            # Observation file for this event (see event_catalogue.OBS_FILE_NAMES)
            if ssw_event not in OBS_FILE_NAMES:
                sys.exit(UNKNOWN_EVENT_MSG)
            obs_file_name = OBS_FILE_NAMES[ssw_event]

            obs_filenames: list[str] = [
                os.path.join(
                    str(os.getenv("OBS_DIR")), obs_file_name
                )
            ]
        # Linear elongation bias correction (only applied if bias_term_bool)
        bias_term_5 = 2.5
        bias_term_21 = 2.5

        use_parallel: bool = self.get_use_parallel()

        # Set limits on minimum and maximum elongation observations to use
        elong_min = 6.0
        elong_max = 10.0

        obs_par_dict = {
            "n_obs": n_obs,
            "obs_radius": obs_radius,
            "obs_lon": obs_lon,
            "obs_lat": obs_lat,
            "obs_cov": obs_cov,
            "elong_min": elong_min,
            "elong_max": elong_max,
            "obs_times": obs_times,
            "use_synthetic_obs": use_synthetic_obs,
            "obs_filenames": obs_filenames,
            "obs_rng_seed": obs_rng_seed,
            "ssw_event": ssw_event,
            "craft": craft,
            "img": img,
            "bias_term_bool": bias_term_bool,
            "bias_term_5rs": bias_term_5,
            "bias_term_21rs": bias_term_21,
            "use_parallel": use_parallel,
        }

        return obs_par_dict


    def initialise_da_parameters(self) -> dict:
        """
        Data-assimilation settings. run_start < 0 runs (n_runs) unperturbed-first-guess runs counting down
        from -1 (used for real observations); otherwise runs run_start .. run_start + n_runs - 1.
        :return: da_par_dict
        """
        n_members: int = 25

        run_start = -1
        n_runs: int = 1

        delta_aux_pf: float = 0.98
        if self.get_use_synthetic_obs():
            pars_in_state_vector: list[str] = ["v", "width", "lon"]
        else:
            pars_in_state_vector: list[str] = ["t_init", "v", "width", "lon", "lat"]
        time_tolerance: Quantity[u.day] = 0.5 * u.day

        da_par_dict = {
            "n_members": n_members,
            "run_start": run_start,
            "n_runs": n_runs,
            "delta_aux_pf": delta_aux_pf,
            "pars_in_state": pars_in_state_vector,
            "time_tolerance": time_tolerance,
            # Speed option: False runs the compressible observation-operator models only up to each observation
            #  time + time_tolerance instead of a fixed 3 days (faster, may change results slightly)
            "fixed_compressible_sim_time": True,
            # Diagnostic outputs in each run directory (see da_output.py):
            #  elongations.nc: observed and modelled elongations at every observation step (cheap)
            "save_elongation_output": True,
            #  cme_arrival_times.nc: every member's arrival at arrival_body after every step. This needs one
            #  extra model run of arrival_sim_time per member per assimilated observation (and for the prior);
            #  set False to skip it. Members that have not arrived by the end of the run count as misses.
            "compute_arrival_times": True,
            "arrival_sim_time": 5 * u.day,
            "arrival_body": "EARTH",
            #  output cadence of the arrival runs (None = same as the DA's dt_scale; larger = less memory)
            "arrival_dt_scale": None,
        }

        return da_par_dict
