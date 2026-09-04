import numpy as np
import numpy.typing as npt
import datetime

import huxt.huxt as H
import surf.surf as S

from huxt.huxt import Observer as Hobs
from surf.surf import Observer as Sobs

import pandas as pd
import os
from stereo_spice.coordinates import StereoSpice

from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity
import tables

from cme_par_ens import CmeParEns
from sir_observation_operator import ObservationOperator
from cme_par_dict_structure import required_dict_keys


def get_ssw_profile(use_model, ssw_event, craft, img, pa_center, pa_wid=1.0):
    """
    Compute the Solar Stormwatch profile of the CME along a fixed position angle window, for either STA or STB.
    Parameters
    ----------
    use_model: Variable to determine which model to use, accepted inputs are ['huxt', 'surf', 'compress_surf']
    ssw_event: String identifier of which SWPC ssw_event to analyse. Should be ssw_007, ssw_008, ssw_009, or ssw_012.
    craft: String identifier of which STEREO craft to analyse, should be 'STA' or 'STB'.
    img: String identifier of whether to retrieve the ssw profiles of normal, or differenced images. Should be
        'norm', or 'diff'
    pa_center: Float value of the central position angle to track the CME front along. In degrees.
    pa_wid: Float value of the width of the position angle slice to track the CME front along. Defaults to 1 degree

    Returns
    -------
    profile: A pandas dataframe giving the consensus time-elongation profile of the observed CME front derived from the
            full distribution of solar stormwatch classifications.

    """
    # Open up the SSW data
    if use_model in ['huxt']:
        project_dirs = H._setup_dirs_()
    elif use_model in ['surf', 'compress_surf']:
        project_dirs = S._setup_dirs_()
    else:
        project_dirs = {}
        sys.exit(
            f"Unrecognised use_model variable, {use_model}, "
            f"accepted use_model variables = ['huxt', 'surf', 'compress_surf']"
        )
    ssw_out = tables.open_file(project_dirs['SSW_data'], mode="r")

    # Pull out ssw_event
    ssw_path = "/".join(['', ssw_event, craft, img])
    ssw_event = ssw_out.get_node(ssw_path)

    # Now for each time, look up elongation at position angle nearest to requested pa.
    times = []
    el_best = []
    el_lo = []
    el_hi = []
    for cme_slice in ssw_event:
        frame_time = Time(cme_slice._v_title, format='isot', scale='utc')
        # Stash time
        times.append(frame_time.datetime)
        # Get the CME front data
        cme = pd.DataFrame.from_records(cme_slice.cme_coords.read())
        cme.replace(to_replace=[99999], value=np.NaN, inplace=True)
        # Look up the indices of this position angle slice, and average the elongation coords in this
        # window.
        id_pa = (cme['pa'] >= (pa_center - pa_wid)) & (cme['pa'] <= (pa_center + pa_wid))
        el_best.append(np.nanmean(cme['el'][id_pa]))
        el_lo.append(np.nanmean(cme['el_lo'][id_pa]))
        el_hi.append(np.nanmean(cme['el_hi'][id_pa]))

    # Make dataframe of the raw ssw profile
    # Get array of PA's to pass out too
    pa = np.zeros(len(times)) + pa_center

    # Convert profile into a dataframe and return.
    profile = pd.DataFrame({'time': times, 'el': el_best, 'el_lo': el_lo, 'el_hi': el_hi, 'pa': pa})

    profile['el_dlo'] = profile['el'] - profile['el_lo']
    profile['el_dhi'] = profile['el_hi'] - profile['el']
    # Set the time error to zero, as well defined for SSW
    profile['time_err'] = profile['time'] - profile['time']
    # Add in julian dates
    profile['time'] = pd.DatetimeIndex(profile['time']).to_julian_date()

    ssw_out.close()
    return profile


class Observations:
    def __init__(
            self,
            use_model: str,
            obs_radius: Quantity[u.AU] | list[Quantity[u.AU]]=None,
            obs_lon: Quantity[u.deg] | list[Quantity[u.deg]]=None,
            obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = None,
            obs_times_in_datetime: datetime.datetime | list[datetime.datetime]=None,
            obs_filenames: list[str] | str =None,
            use_synthetic_obs: bool=False,
            true_cme_par_dict: CmeParEns=None,
            obs_cov: npt.NDArray[float] | float=None,
            surf_init_time: datetime.datetime=None,
            vr_in: npt.NDArray[Quantity[u.km / u.s]]=None,
            lon_start: Quantity[u.deg]=None,
            lon_stop: Quantity[u.deg]=None,
            sim_time: Quantity[u.day]=None,
            dt_scale: int|float=None,
            r_min: Quantity[u.solRad]=None,
            cme_init_rad: Quantity[u.solRad]=None,
            cme_fixed_duration:bool = None,
            fixed_duration: Quantity[u.s] = None,
            plot_surf_output: bool=False,
            obs_rng_seed: int=None,
            ssw_event: str=None,
            craft:str=None,
            img:str=None,
            use_parallel: bool=False,
    ) -> None:
        """
        Class to get observations for DA from
        :param use_model: String to determine which model to use, possible models are ['surf', 'compress_surf', 'huxt']
        :param use_model: String to determine which model to use, possible models are ['surf', 'compress_surf', 'huxt']
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
        :param surf_init_time: Initial datetime for SURF model
        :param vr_in: Velocities at the inner boundary of the SURF model domain
        :param lon_start: Longitude where we start the SURF simulation
        :param lon_stop: Longitude where we stop the SURF simulation
        :param sim_time: How long the SURF simulation will run for
        :param dt_scale: Frequency with which the model will output
        :param r_min: Minimum radius of SURF simulation
        :param cme_init_rad: Initial radius the CME is observed at
        :param cme_fixed_duration: Boolean to determine whether CMEs will be input with fixed duration
        :param fixed_duration: If CME is fixed duration, this variable defines that fixed duration in seconds
        :param plot_surf_output: Boolean to determine whether to plot SURF output at observation time
        :param rng_seed: Seed to use for random number generator
        :param use_parallel: Boolean to determine whether to use parallel execution if using synthetic observations
        :TODO: CHANGED SUCH THAT MULTIPLE LONGITUDES AND RADII CAN BE
            INPUT AND TIMES OF OBSERVATIONS ARE TAKEN FROM INPUT FILE IN CASE OF REAL OBS
        :TODO: Change such that observations are downloaded if need be
        """

        # Read in string to determine which model to use
        self.use_model: str = use_model.lower()
        assert self.use_model in ["surf", "compress_surf", "huxt"]

        # Check whether we're reading observations from a file or using synthetic observations
        self.use_synth_obs: bool = use_synthetic_obs

        # If no observation filenames are specified, make synthetic observations
        if obs_filenames is None:
            self.use_synth_obs: bool = True
        else:
            self.obs_filenames: str | list[str] = obs_filenames
            if isinstance(self.obs_filenames, str):
                self.obs_filenames = [self.obs_filenames]
            self.ssw_event: str = ssw_event
            self.craft: str = craft
            self.img: str = img

        if obs_radius is None:
            self.obs_radius: Quantity[u.AU] | list[Quantity[u.AU]] = 1.0 * u.AU
        else:
            self.obs_radius: Quantity[u.AU] | list[Quantity[u.AU]] = obs_radius

        if obs_lon is None:
            self.obs_lon: Quantity[u.deg] | list[Quantity[u.deg]] = 0 * u.deg
        else:
            self.obs_lon: Quantity[u.deg] | list[Quantity[u.deg]] = obs_lon

        if obs_lat is None:
            self.obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = 0 * u.deg
        else:
            self.obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = obs_lat

        # If we are using synthetic observations, ensure all variables required to initialise them are provided
        if self.use_synth_obs:
            assert obs_times_in_datetime is not None
            self.obs_times_in_datetime: datetime.datetime | list[datetime.datetime] = obs_times_in_datetime

            assert(
                all([true_cme_par_dict, obs_cov]) is not None
            )
            assert all(p in true_cme_par_dict.keys() for p in required_dict_keys())

            self.true_cme_par_dict: CmeParEns = true_cme_par_dict
            self.obs_cov: float | list[float] = obs_cov

            # Initialise default surf setup
            if surf_init_time is None:
                self.surf_init_time: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0)
            else:
                self.surf_init_time: datetime.datetime = surf_init_time

            if vr_in is None:
                self.vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones(128) * 400 * u.km / u.s
            else:
                self.vr_in: npt.NDArray[Quantity[u.km / u.s]] = vr_in

            if lon_start is None:
                self.lon_start: Quantity[u.deg] = 290 * u.deg
            else:
                self.lon_start: Quantity[u.deg] = lon_start

            if lon_stop is None:
                self.lon_stop: Quantity[u.deg] = 380 * u.deg
            else:
                self.lon_stop: Quantity[u.deg] = lon_stop

            if sim_time is None:
                self.sim_time: Quantity[u.day] = 5 * u.day
            else:
                self.sim_time: Quantity[u.day] = sim_time

            if dt_scale is None:
                self.dt_scale: float = 20
            else:
                self.dt_scale: float = dt_scale

            if r_min is None:
                self.r_min: Quantity[u.solRad] = 30 * u.solRad
            else:
                self.r_min: Quantity[u.solRad] = r_min

            if cme_init_rad is None:
                self.cme_init_rad: Quantity[u.solRad] = 12 * u.solRad
            else:
                self.cme_init_rad: Quantity[u.solRad] = cme_init_rad

            if cme_fixed_duration is None:
                self.cme_fixed_duration: bool = False
            else:
                self.cme_fixed_duration: bool = cme_fixed_duration

            if fixed_duration is None:
                self.fixed_duration: Quantity[u.s] = 12 * 60 * 60 * u.s
            else:
                self.fixed_duration: Quantity[u.s] = fixed_duration

            if obs_rng_seed is None:
                self.rng: Generator = np.random.default_rng()
            else:
                self.rng: Generator = np.random.default_rng(obs_rng_seed)

            self.use_parallel: bool = use_parallel
            self.plot_surf_output: bool = plot_surf_output

            self.observations: list[float] = self.make_synthetic_obs()

        else:
            obs_tuple: tuple[list[datetime.datetime], list[Quantity[u.deg]], list[float]] = self.read_obs_from_file()

            self.obs_times_in_datetime: list[datetime.datetime] = obs_tuple[0]
            self.obs_lon: list[Quantity[u.deg]] = obs_tuple[1]
            self.observations: list[float] = obs_tuple[2]


    def make_synthetic_obs(self) -> list[float]:
        """
        Function to create synthetic observations
        :param rng: Current seed of the random number generator
        :param plot_surf: Boolean to determine whether to plot the SURF output at observation times
        :return: synth_obs: List of synthetic observations
        """

        # Initialise an observation operator instance
        obs_op_obj = ObservationOperator(
            use_model=self.use_model,
            cme_par_dict = self.true_cme_par_dict,
            obs_radius=self.obs_radius,
            obs_lon = self.obs_lon,
            obs_lat = self.obs_lat,
            obs_time_in_datetime=self.obs_times_in_datetime,
            obs_cov = self.obs_cov,
            surf_init_time = self.surf_init_time,
            vr_in = self.vr_in,
            lon_start = self.lon_start,
            lon_stop = self.lon_stop,
            sim_time = self.sim_time,
            dt_scale = self.dt_scale,
            r_min = self.r_min,
            cme_init_rad = self.cme_init_rad,
            cme_fixed_duration = self.cme_fixed_duration,
            fixed_duration = self.fixed_duration,
            plot_surf_output = self.plot_surf_output,
            use_parallel = self.use_parallel,
        )

        unpert_obs = obs_op_obj.make_obs_op()

        synth_obs = [
            uo + self.rng.normal(loc=0, scale=self.obs_cov)
            for uo in unpert_obs[0, :]
        ]

        return synth_obs


    def read_obs_from_file(self) -> tuple[list[datetime.datetime], list[Quantity[u.deg]], list[float]]:

        times_datetime = []
        el_best = []
        el_lo = []
        el_hi = []
        obs_lon = []
        for obs_file in self.obs_filenames:
            with tables.open_file(obs_file, mode="r") as f:

                # Pull out ssw_event
                # ssw_event = "ssw_012"
                # craft = "stb"
                # img = "diff"
                pa_wid=1.0
                ssw_path = "/".join(['', self.ssw_event, self.craft, self.img])

                print(f"Reading observation from: {ssw_path}")
                ssw_event = f.get_node(ssw_path)
                spice = StereoSpice()

                for cme_slice in ssw_event:
                    frame_time = Time(cme_slice._v_title, format='isot', scale='utc')

                    ert_hpc = spice.get_lonlat(frame_time, 'earth', system='hpc', observatory=self.craft)
                    ert_hpr = spice.convert_hpc_to_hpr(ert_hpc[1], ert_hpc[2])
                    ert_pa_avg = np.mean(ert_hpr[1])
                    pa_center = ert_pa_avg

                    cme_df = pd.DataFrame.from_records(cme_slice.cme_coords.read())
                    cme_df.replace(to_replace=[99999], value=np.nan, inplace=True)

                    # Look up the indices of this position angle slice, and average
                    #   the elongation coords in this window.
                    id_pa = (cme_df['pa'] >= (pa_center - pa_wid)) & (cme_df['pa'] <= (pa_center + pa_wid))

                    # Get observations longitude
                    if self.use_model in ["surf", "compress_surf"]:
                        h_obs_class = Sobs(self.craft.upper(), frame_time)
                    elif use_model in ["huxt"]:
                        h_obs_class = Hobs(self.craft.upper(), frame_time)
                    else:
                        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

                    obs_lon.append(h_obs_class.lon.to(u.deg))

                    times_datetime.append(frame_time.to_datetime())
                    el_best.append(np.nanmean(cme_df['el'][id_pa]))
                    el_lo.append(np.nanmean(cme_df['el_lo'][id_pa]))
                    el_hi.append(np.nanmean(cme_df['el_hi'][id_pa]))

        obs_times_out = [
            x for i, x in enumerate(times_datetime) if ~np.isnan(el_best[i])
        ]
        obs_lon_out = [
            x for i, x in enumerate(obs_lon) if ~np.isnan(el_best[i])
        ]
        obs_out = [
            x for i, x in enumerate(el_best) if ~np.isnan(el_best[i])
        ]

        return obs_times_out, obs_lon_out, obs_out



def main():
    h5_file_path = os.path.join(
        "C:\\", "Users", "ss905122", "PycharmProjects",
        "SIR_SUXt", "SSW_cme_classifications.hdf5"
    )
    use_model = "surf"
    use_model = "surf"
    ssw_event: str = "ssw_012"
    craft: str = "stb"
    img: str = "diff"
    ObsClass = Observations(
        use_model=use_model,
        obs_filenames=h5_file_path,
        ssw_event=ssw_event,
        craft=craft,
        img=img
    )
    obs_times_out, obs_lon, obs_out = ObsClass.read_obs_from_file()
    print(f"ssw_event: {ssw_event}, craft: {craft}, img: {img}")
    print(f"obs_times_out = {obs_times_out}")
    print(f"obs_out = {obs_out}")
    print(f"obs_lon_out = {obs_lon}")
    return


if __name__ == "__main__":
    main()