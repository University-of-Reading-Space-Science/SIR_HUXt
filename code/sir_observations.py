import numpy as np
import numpy.typing as npt
import datetime

import astropy.units as u
from astropy.units import Quantity

from cme_par_ens import CmeParEns
from sir_observation_operator import ObservationOperator
from cme_par_dict_structure import required_dict_keys


class Observations:
    def __init__(
            self,
            obs_lon: Quantity[u.deg],
            obs_times_in_datetime: datetime.datetime | list[datetime.datetime],
            obs_filenames: list[str]=None,
            use_synthetic_obs: bool=False,
            true_cme_par_dict: CmeParEns=None,
            obs_cov: npt.NDArray[float] | float=None,
            huxt_init_time: datetime.datetime=None,
            vr_in: npt.NDArray[Quantity[u.km / u.s]]=None,
            lon_start: Quantity[u.deg]=None,
            lon_stop: Quantity[u.deg]=None,
            sim_time: Quantity[u.day]=None,
            dt_scale: int|float=None,
            r_min: Quantity[u.solRad]=None,
            cme_init_rad: Quantity[u.solRad]=None,
            cme_fixed_duration:bool = None,
            fixed_duration: Quantity[u.s] = None,
            plot_huxt_output: bool=False,
            rng: int=None
    ) -> None:
        """
        Class to get observations for DA from
        :param obs_lon: Longitude of observation source
        :param obs_times_in_datetime: Times observations are taken
        :param obs_filenames: A list of observation filenames that observations are to be read from
        :param use_synthetic_obs: A boolean indicating whether to use synthetic observations, this overrides
                                  obs_filenames and if set to False, will use synthetic observations
                                  even if obs_filenames are specified
        :param true_cme_par_dict: To be used to create synthetic observations, a dictionary containing true parameters
            to create a truth CME simulation
        :param obs_cov: To be used with synthetic observations, an observation covariance matrix to be used to
            add random noise to  truth CME elongations
        :param huxt_init_time: Initial datetime for HUXt model
        :param vr_in: Velocities at the inner boundary of the HUXt model domain
        :param lon_start: Longitude where we start the HUXt simulation
        :param lon_stop: Longitude where we stop the HUXt simulation
        :param sim_time: How long the HUXt simulation will run for
        :param dt_scale: Frequency with which the model will output
        :param r_min: Minimum radius of HUXt simulation
        :param cme_init_rad: Initial radius the CME is observed at
        :param cme_fixed_duration: Boolean to determine whether CMEs will be input with fixed duration
        :param fixed_duration: If CME is fixed duration, this variable defines that fixed duration in seconds
        :param plot_huxt_output: Boolean to determine whether to plot HUXt output at observation time
        :param rng: Seed to use for random number generator
        :TODO: CHANGED SUCH THAT MULTIPLE LONGITUDES AND RADII CAN BE
            INPUT AND TIMES OF OBSERVATIONS ARE TAKEN FROM INPUT FILE IN CASE OF REAL OBS
        :TODO: Change such that observations are downloaded if need be
        """

        # Check whether we're reading observations from a file or using synthetic observations
        self.use_synth_obs = use_synthetic_obs

        # If no observation filenames are specified, make synthetic observations
        if obs_filenames is None:
            self.use_synth_obs = True
        else:
            self.obs_filenames = obs_filenames

        if obs_lon is None:
            self.obs_lon = 0 * u.deg
        else:
            self.obs_lon = obs_lon

        assert obs_times_in_datetime is not None
        self.obs_times_in_datetime = obs_times_in_datetime

        # If we are using synthetic observations, ensure all variables required to initialise them are provided
        if self.use_synth_obs:
            assert(
                all([true_cme_par_dict, obs_cov]) is not None
            )
            assert all(p in true_cme_par_dict.keys() for p in required_dict_keys())

            self.true_cme_par_dict = true_cme_par_dict
            self.obs_cov = obs_cov

            # Initialise default huxt setup
            if huxt_init_time is None:
                self.huxt_init_time = datetime.datetime(2008, 1, 1, 0, 0, 0)
            else:
                self.huxt_init_time = huxt_init_time

            if vr_in is None:
                self.vr_in = np.ones(128) * 400 * u.km / u.s
            else:
                self.vr_in = vr_in

            if lon_start is None:
                self.lon_start = 290 * u.deg
            else:
                self.lon_start = lon_start

            if lon_stop is None:
                self.lon_stop = 380 * u.deg
            else:
                self.lon_stop = lon_stop

            if sim_time is None:
                self.sim_time = 5 * u.day
            else:
                self.sim_time = sim_time

            if dt_scale is None:
                self.dt_scale = 20
            else:
                self.dt_scale = dt_scale

            if r_min is None:
                self.r_min = 30 * u.solRad
            else:
                self.r_min = r_min

            if cme_init_rad is None:
                self.cme_init_rad = 12 * u.solRad
            else:
                self.cme_init_rad = cme_init_rad

            if cme_fixed_duration is None:
                self.cme_fixed_duration = False
            else:
                self.cme_fixed_duration = cme_fixed_duration

            if fixed_duration is None:
                self.fixed_duration = 12 * 60 * 60 * u.s
            else:
                self.fixed_duration = fixed_duration

            self.plot_huxt_output = plot_huxt_output
            self.rng = rng

            self.observations = self.make_synthetic_obs()

        else:
            self.observations = self.read_obs_from_file()


    def make_synthetic_obs(self) -> list[float]:
        """
        Function to create synthetic observations
        :param rng: Current seed of the random number generator
        :param plot_huxt: Boolean to determine whether to plot the HUXt output at observation times
        :return: synth_obs: List of synthetic observations
        """

        if self.rng is None:
            rng = np.random.default_rng()
        else:
            rng = self.rng

        # Initialise an observation operator instance
        obs_op_obj = ObservationOperator(
            cme_par_dict = self.true_cme_par_dict,
            obs_lon = self.obs_lon,
            obs_time_in_datetime=self.obs_times_in_datetime,
            obs_cov = self.obs_cov,
            huxt_init_time = self.huxt_init_time,
            vr_in = self.vr_in,
            lon_start = self.lon_start,
            lon_stop = self.lon_stop,
            sim_time = self.sim_time,
            dt_scale = self.dt_scale,
            r_min = self.r_min,
            cme_init_rad = self.cme_init_rad,
            cme_fixed_duration = self.cme_fixed_duration,
            fixed_duration = self.fixed_duration,
            plot_huxt_output = self.plot_huxt_output
        )

        unpert_obs = obs_op_obj.make_obs_op()

        synth_obs = [
            uo + rng.normal(loc=0, scale=self.obs_cov) for uo in unpert_obs
        ]

        return synth_obs

    def read_obs_from_file(self) -> list[float]:
        pass