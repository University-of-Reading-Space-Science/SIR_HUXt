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

from dotenv import load_dotenv
from pathlib import Path

from wsa_reader import ReadWSAFiles

from init_sir import initialise_cme_parameter_ensemble_dict

env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path, override=True)


class Setup:
    def __init__(self):
        self.surf_par_dict = self.initialise_surf_parameters()
        self.obs_par_dict = self.initialise_observation_parameters()
        self.true_cme_par_dict = self.initialise_true_parameters()
        self.fg_cme_par_dict = self.initialise_fg_cme_parameters()
        self.prior_cme_cov = self.initialise_prior_cme_cov()
        self.da_par_dict = self.initialise_da_parameters()


    def allowed_cov_types(self):
        return {"uncorr", "donki", "mo_cone"}


    def get_use_parallel(self):
        use_parallel = True

        return use_parallel


    def get_mas_wsa_const(self):
        mas_wsa_const = "const"

        return mas_wsa_const


    def get_use_synthetic_obs(self):
        use_synthetic_obs = True

        return use_synthetic_obs


    def get_ssw_event_craft(self):
        ssw_event = "ssw_012"
        craft = "stb"
        img = "diff"

        return ssw_event, craft, img


    def get_surf_init_time(self):
        ssw_event, _, _ = self.get_ssw_event_craft()
        if self.get_use_synthetic_obs():
            surf_init_time: datetime.datetime = datetime.datetime(2012, 1, 1, 0, 0)
        else:
            if ssw_event == "ssw_007":
                surf_init_time: datetime.datetime = datetime.datetime(2012, 8, 31, 0, 0, 0)

            elif ssw_event == "ssw_008":
                surf_init_time: datetime.datetime = datetime.datetime(2012, 9, 27, 15, 0, 0)

            elif ssw_event == "ssw_009":
                surf_init_time: datetime.datetime = datetime.datetime(2012, 10, 4, 20, 0, 0)

            elif ssw_event == "ssw_012":
                surf_init_time: datetime.datetime = datetime.datetime(2012, 11, 20, 0, 0, 0)

            else:
                sys.exit("Unknown ssw_event name, expected ssw_event = 'ssw_007', 'ssw_008', 'ssw_009' or 'ssw_012'")

        return surf_init_time


    def get_use_model(self):
        """
        Function to get model we require to use
        :return: use_model: String containing either "surf", "surf_compress", "huxt"
        """

        poss_models = ["surf", "compress_surf", "huxt"]
        use_model = "compress_surf"

        assert use_model.lower() in poss_models

        return use_model.lower()


    def initialise_surf_parameters(self):
        """
        Function to initialise the SURF parameters for simulation
        :return:
        """
        mas_or_wsa = self.get_mas_wsa_const()

        surf_init_time = self.get_surf_init_time()
        use_model = self.get_use_model().lower()

        surf_init_time_astro = Time(surf_init_time, format="datetime")
        ert = S.Observer("EARTH", surf_init_time_astro)
        earth_lat = ert.lat.to(u.deg)
        earth_lon = ert.lat.to(u.deg)

        lon_start: Quantity[u.deg] = -70 * u.deg
        lon_stop: Quantity[u.deg] = 70 * u.deg
        sim_time: Quantity[u.day] = 3 * u.day
        dt_scale: int = 1
        cme_init_rad: Quantity[u.solRad] = 12 * u.solRad
        r_min: Quantity[u.solRad] = 21.5 * u.solRad
        cme_fixed_duration: bool = True
        fixed_duration: Quantity[u.s] = 12 * 3600 * u.s

        if mas_or_wsa.upper() == "MAS":
            cr_num: int = np.trunc(sn.carrington_rotation_number(surf_init_time))

            if use_model in ["surf", "compress_surf"]:
                vr_in = Sin.get_MAS_long_profile(cr_num, lat=0.0 * u.deg)
            elif use_model in ["huxt"]:
                vr_in = Hin.get_MAS_long_profile(cr_num, lat=0.0 * u.deg)
            else:
                sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

        elif mas_or_wsa.upper() == "WSA":
            wsa_dir = os.getenv('WSA_DIR')
            readWSAClass = ReadWSAFiles(surf_init_time, output_dir=wsa_dir)
            output_path = readWSAClass.download_file()

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
                v_lons = np.array([
                    i * v_lon_grid_step for i in range(n_lon)
                ]) * u.rad

                vr_in, lon_temp = Sin.map_v_inwards(vr_in, 215 * u.solRad, v_lons, r_min)
            elif use_model in ["huxt"]:
                vr_in = Hin.get_WSA_long_profile(output_path, lat=earth_lat)
                n_lon = len(vr_in)
                v_lon_grid_step = 2 * np.pi / n_lon
                v_lons = np.array([
                    i * v_lon_grid_step for i in range(n_lon)
                ]) * u.rad

                vr_in, lon_temp = Hin.map_v_inwards(vr_in, 215 * u.solRad, v_lons, r_min)
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
            "lon_start": lon_start,
            "lon_stop": lon_stop,
            "sim_time": sim_time,
            "dt_scale": dt_scale,
            "r_min": r_min,
            "cme_init_rad": cme_init_rad,
            "cme_fixed_duration": cme_fixed_duration,
            "fixed_duration": fixed_duration,
        }

        return out_surf_par


    def initialise_true_parameters(self):
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


    def initialise_fg_cme_parameters(self):
        # Initialise CME parameters
        ssw_event, _, _ = self.get_ssw_event_craft()
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
            if donki_cme:
                if ssw_event == "ssw_007":
                    cme_at_21rs: datetime.datetime = datetime.datetime(2012, 8, 31, 22, 43, 0)
                    print(f"CR = {sn.carrington_rotation_number(cme_at_21rs)}")
                    fg_cme_speed: float = 1498

                    dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                    time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                    cme_at_12rs: datetime.datetime = (
                            cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                    )

                    fg_cme_width: float = 150
                    fg_cme_lon: float = -63
                    fg_cme_lat: float = -15
                    fg_cme_thick: float = 0

                elif ssw_event == "ssw_008":
                    cme_at_21rs: datetime.datetime = datetime.datetime(2012, 9, 28, 1, 56, 0)

                    fg_cme_speed: float = 1160

                    dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                    time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                    cme_at_12rs: datetime.datetime = (
                            cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                    )

                    fg_cme_width: float = 170
                    fg_cme_lon: float = 30
                    fg_cme_lat: float = 5
                    fg_cme_thick: float = 0

                elif ssw_event == "ssw_009":
                    cme_at_21rs: datetime.datetime = datetime.datetime(2012, 10, 5, 9, 15, 0)

                    fg_cme_speed: float = 650

                    dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                    time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                    cme_at_12rs: datetime.datetime = (
                            cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                    )

                    fg_cme_width: float = 94
                    fg_cme_lon: float = 10
                    fg_cme_lat: float = -28
                    fg_cme_thick: float = 0

                elif ssw_event == "ssw_012":
                    cme_at_21rs: datetime.datetime = datetime.datetime(2012, 11, 20, 17, 32, 0)

                    fg_cme_speed: float = 725

                    dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                    time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                    cme_at_12rs: datetime.datetime = (
                            cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                    )

                    fg_cme_width: float = 60
                    fg_cme_lon: float = 90
                    fg_cme_lat: float = 5
                    fg_cme_thick: float = 0

                else:
                    sys.exit("Unknown ssw_event name, expected ssw_event = 'ssw_007', 'ssw_008', 'ssw_009' or 'ssw_012'")
            else:
                if ssw_event == "ssw_007":
                    cme_at_21rs: datetime.datetime = datetime.datetime(2012, 8, 31, 22, 46, 0)
                    print(f"CR = {sn.carrington_rotation_number(cme_at_21rs)}")
                    fg_cme_speed: float = 1010

                    dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                    time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                    cme_at_12rs: datetime.datetime = (
                            cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                    )

                    fg_cme_width: float = 66
                    fg_cme_lon: float = -30
                    fg_cme_lat: float = 0
                    fg_cme_thick: float = 0

                elif ssw_event == "ssw_008":
                    cme_at_21rs: datetime.datetime = datetime.datetime(2012, 9, 28, 3, 49, 0)

                    fg_cme_speed: float = 872

                    dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                    time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                    cme_at_12rs: datetime.datetime = (
                            cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                    )

                    fg_cme_width: float = 110
                    fg_cme_lon: float = 20
                    fg_cme_lat: float = 4
                    fg_cme_thick: float = 0

                elif ssw_event == "ssw_009":
                    cme_at_21rs: datetime.datetime = datetime.datetime(2012, 10, 5, 8, 47, 0)

                    fg_cme_speed: float = 698

                    dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                    time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                    cme_at_12rs: datetime.datetime = (
                            cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                    )

                    fg_cme_width: float = 84
                    fg_cme_lon: float = 9
                    fg_cme_lat: float = -24
                    fg_cme_thick: float = 0

                elif ssw_event == "ssw_012":
                    cme_at_21rs: datetime.datetime = datetime.datetime(2012, 11, 20, 17, 40, 0)

                    fg_cme_speed: float = 664

                    dist_from_12_to_21rs_km = (9.5 * u.solRad).to(u.km).value
                    time_from_12_to_21rs = dist_from_12_to_21rs_km / fg_cme_speed
                    cme_at_12rs: datetime.datetime = (
                            cme_at_21rs - datetime.timedelta(seconds=time_from_12_to_21rs)
                    )

                    fg_cme_width: float = 94
                    fg_cme_lon: float = 22
                    fg_cme_lat: float = 20
                    fg_cme_thick: float = 0

                else:
                    sys.exit("Unknown ssw_event name, expected ssw_event = 'ssw_007', 'ssw_008', 'ssw_009' or 'ssw_012'")

        # Time taken to get from cme_init_rad to r_min
        surf_dict = self.initialise_surf_parameters()
        surf_cme_ir: Quantity[u.solRad] = surf_dict["cme_init_rad"]
        surf_r_min: Quantity[u.solRad] = surf_dict["r_min"]

        dist_in_km: Quantity[u.km] = (surf_r_min - surf_cme_ir).to(u.km)
        seconds_to_r_min: float = dist_in_km.to(u.km).value / fg_cme_speed

        fg_cme_t_init: datetime.datetime = (
                cme_at_12rs + datetime.timedelta(seconds=seconds_to_r_min)
        )

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


    def initialise_prior_cme_cov(self):
        if self.get_use_synthetic_obs():
            cme_cov_type = "uncorr"
        else:
            cme_cov_type = "mo_cone"

        """# Initialise how to build the prior CME covariance matrix
        if cme_cov_type is None:
            cme_cov_type = "uncorr"
    
        try:
            assert cme_cov_type.lower() in allowed_cov_types()
        except AssertionError:
            print("Defaulting to uncorrelated covariance matrix")
            cme_cov_type = "uncorr"
        else:
            cme_cov_type = cme_cov_type.lower()"""

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


    def initialise_observation_parameters(self):
        n_obs: int = 8
        obs_cadence_hr = 3  # Observations cadence in hours
        true_cme_dict = self.initialise_true_parameters()
        true_cme_t_init: datetime.datetime = true_cme_dict["t_init"]

        surf_init_time = self.get_surf_init_time()
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
        obs_filenames: list[str] = [
            os.path.join(
                str(os.getenv("OBS_DIR")), "SSW_cme_classifications.hdf5"
            )
        ]

        ssw_event, craft, img = self.get_ssw_event_craft()
        if use_synthetic_obs:
            bias_term_bool = False  # True
        else:
            bias_term_bool = True

        bias_term_5 = 2.5
        bias_term_21 = 2.5

        bias_term_5 = 2.5
        bias_term_21 = 2.5

        use_parallel: bool = self.get_use_parallel()

        obs_par_dict = {
            "n_obs": n_obs,
            "obs_radius": obs_radius,
            "obs_lon": obs_lon,
            "obs_lat": obs_lat,
            "obs_cov": obs_cov,
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


    def initialise_da_parameters(self):
        n_members: int = 50
        n_runs: int = 25

        delta_aux_pf: float = 0.98
        if self.get_use_synthetic_obs():
            pars_in_state_vector: list[str] = ["v", "width", "lon"]
        else:
            pars_in_state_vector: list[str] = ["t_init", "v", "width", "lon", "lat"]
        time_tolerance: Quantity[u.day] = 0.5 * u.day

        da_par_dict = {
            "n_members": n_members,
            "n_runs": n_runs,
            "delta_aux_pf": delta_aux_pf,
            "pars_in_state": pars_in_state_vector,
            "time_tolerance": time_tolerance
        }

        return da_par_dict