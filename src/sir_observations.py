import numpy as np
import numpy.typing as npt
import datetime
import sys

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
from sir_observation_operator import ObservationOperator, ObsOpSettings, WorkerPool
from cme_par_dict_structure import required_dict_keys


class Observations:
    def __init__(
        self,
        use_model: str,
        obs_radius: Quantity[u.AU] | list[Quantity[u.AU]] = None,
        obs_lon: Quantity[u.deg] | list[Quantity[u.deg]] = None,
        obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = None,
        obs_times_in_datetime: datetime.datetime | list[datetime.datetime] = None,
        obs_filenames: list[str] | str = None,
        use_synthetic_obs: bool = False,
        true_cme_par_dict: CmeParEns = None,
        obs_cov: npt.NDArray[np.float64] | float = None,
        surf_init_time: datetime.datetime = None,
        vr_in: Quantity[u.km / u.s] = None,
        vr_longs: Quantity[u.deg] = np.nan * u.deg,
        vr_lats: Quantity[u.deg] = np.nan * u.deg,
        n_lat_surf: int = 45,
        lon_start: Quantity[u.deg] = None,
        lon_stop: Quantity[u.deg] = None,
        sim_time: Quantity[u.day] = None,
        dt_scale: int | float = None,
        r_min: Quantity[u.solRad] = None,
        cme_init_rad: Quantity[u.solRad] = None,
        cme_fixed_duration: bool = None,
        fixed_duration: Quantity[u.s] = None,
        plot_surf_output: bool = False,
        obs_rng_seed: int = None,
        ssw_event: str = None,
        craft: str = None,
        img: str = None,
        use_parallel: bool = False,
        obs_op_settings: ObsOpSettings | None = None,
        worker_pool: WorkerPool | None = None,
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
        :param obs_op_settings: ObsOpSettings for the observation operator of synthetic observations
            (None = the defaults)
        :param worker_pool: WorkerPool shared by the run (None = a pool just for these observations)
        :TODO: CHANGED SUCH THAT MULTIPLE LONGITUDES AND RADII CAN BE
            INPUT AND TIMES OF OBSERVATIONS ARE TAKEN FROM INPUT FILE IN CASE OF REAL OBS
        :TODO: Change such that observations are downloaded if need be
        """

        # Read in string to determine which model to use
        self.use_model: str = use_model.lower()
        assert self.use_model in ["surf", "compress_surf", "compress_surf3d", "huxt"]

        # Check whether we're reading observations from a file or using synthetic observations
        self.use_synth_obs: bool = use_synthetic_obs

        # These are needed for both synthetic and real observations
        #  (craft is passed through to the observation operator for synthetic obs)
        self.ssw_event: str = ssw_event
        self.craft: str = craft
        self.img: str = img
        self.vr_longs: Quantity[u.deg] = vr_longs
        self.vr_lats: Quantity[u.deg] = vr_lats
        self.n_lat_surf: int = n_lat_surf
        self.obs_op_settings: ObsOpSettings | None = obs_op_settings
        self.worker_pool: WorkerPool | None = worker_pool

        # If no observation filenames are specified, make synthetic observations
        if obs_filenames is None:
            self.use_synth_obs: bool = True
        else:
            self.obs_filenames: str | list[str] = obs_filenames
            if isinstance(self.obs_filenames, str):
                self.obs_filenames = [self.obs_filenames]

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
            self.obs_times_in_datetime: datetime.datetime | list[datetime.datetime] = (
                obs_times_in_datetime
            )

            assert (
                true_cme_par_dict is not None
            ), "true_cme_par_dict required for synthetic obs"
            assert obs_cov is not None, "obs_cov required for synthetic obs"
            assert all(p in true_cme_par_dict.keys() for p in required_dict_keys())

            self.true_cme_par_dict: CmeParEns = true_cme_par_dict
            self.obs_cov: float | list[float] = obs_cov

            # Initialise default surf setup
            if surf_init_time is None:
                self.surf_init_time: datetime.datetime = datetime.datetime(
                    2008, 1, 1, 0, 0, 0
                )
            else:
                self.surf_init_time: datetime.datetime = surf_init_time

            if vr_in is None:
                self.vr_in: Quantity[u.km / u.s] = np.ones(128) * 400 * u.km / u.s
            else:
                self.vr_in: Quantity[u.km / u.s] = vr_in

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
                self.rng: np.random.Generator = np.random.default_rng()
            else:
                self.rng: np.random.Generator = np.random.default_rng(obs_rng_seed)

            self.use_parallel: bool = use_parallel
            self.plot_surf_output: bool = plot_surf_output

            self.observations: list[float] = self.make_synthetic_obs()

        else:
            if self.ssw_event in ["ssw_007", "ssw_008", "ssw_009", "ssw_012"]:
                obs_tuple: tuple[
                    list[datetime.datetime], list[Quantity[u.deg]], list[float]
                ] = self.read_obs_from_file()

                self.obs_times_in_datetime: list[datetime.datetime] = obs_tuple[0]
                self.obs_lon: list[Quantity[u.deg]] = obs_tuple[1]
                self.observations: list[float] = obs_tuple[2]
            else:
                obs_tuple: tuple[
                    list[datetime.datetime],
                    list[Quantity[u.deg]],
                    list[Quantity[u.deg]],
                    list[float],
                ] = self.l5_read_obs_from_file()

                self.obs_times_in_datetime: list[datetime.datetime] = obs_tuple[0]
                self.obs_lon: list[Quantity[u.deg]] = obs_tuple[1]
                self.obs_lat: list[Quantity[u.deg]] = obs_tuple[2]
                self.observations: list[float] = obs_tuple[3]

    def make_synthetic_obs(self) -> list[float]:
        """
        Function to create synthetic observations
        :param rng: Current seed of the random number generator
        :param plot_surf: Boolean to determine whether to plot the SURF output at observation times
        :return: synth_obs: List of synthetic observations
        """

        # Initialise an observation operator instance
        obs_op_obj: ObservationOperator = ObservationOperator(
            use_model=self.use_model,
            cme_par_dict=self.true_cme_par_dict,
            obs_radius=self.obs_radius,
            obs_lon=self.obs_lon,
            obs_lat=self.obs_lat,
            obs_time_in_datetime=self.obs_times_in_datetime,
            obs_cov=self.obs_cov,
            craft=self.craft,
            surf_init_time=self.surf_init_time,
            vr_in=self.vr_in,
            vr_longs=self.vr_longs,
            vr_lats=self.vr_lats,
            n_lat_surf=self.n_lat_surf,
            lon_start=self.lon_start,
            lon_stop=self.lon_stop,
            sim_time=self.sim_time,
            dt_scale=self.dt_scale,
            r_min=self.r_min,
            cme_init_rad=self.cme_init_rad,
            cme_fixed_duration=self.cme_fixed_duration,
            fixed_duration=self.fixed_duration,
            plot_surf_output=self.plot_surf_output,
            use_parallel=self.use_parallel,
            settings=self.obs_op_settings,
            worker_pool=self.worker_pool,
        )

        unpert_obs: npt.NDArray[np.float64] = obs_op_obj.make_obs_op()

        # obs_cov is an error (co)variance, so the noise standard deviation is sqrt(obs_cov)
        obs_sd: float = np.sqrt(np.ravel(np.asarray(self.obs_cov, dtype=float))[0])
        synth_obs: list[float] = [
            uo + self.rng.normal(loc=0, scale=obs_sd) for uo in unpert_obs[0, :]
        ]

        return synth_obs

    def read_obs_from_file(
        self,
    ) -> tuple[list[datetime.datetime], list[Quantity[u.deg]], list[float]]:

        times_datetime: list[datetime.datetime] = []
        el_best: list[float] = []
        el_lo: list[float] = []
        el_hi: list[float] = []
        obs_lon: list[Quantity] = []
        for obs_file in self.obs_filenames:
            with tables.open_file(obs_file, mode="r") as f:

                # Pull out ssw_event
                pa_wid: float = 1.0
                ssw_path: str = "/".join(["", self.ssw_event, self.craft, self.img])

                print(f"Reading observation from: {ssw_path}")
                ssw_event: tables.Node = f.get_node(ssw_path)
                spice: StereoSpice = StereoSpice()

                for cme_slice in ssw_event:
                    frame_time: Time = Time(
                        cme_slice._v_title, format="isot", scale="utc"
                    )

                    ert_hpc: npt.NDArray[np.float64] = spice.get_lonlat(
                        frame_time, "earth", system="hpc", observatory=self.craft
                    )
                    ert_hpr: tuple[
                        float | npt.NDArray[np.float64], float | npt.NDArray[np.float64]
                    ] = spice.convert_hpc_to_hpr(ert_hpc[1], ert_hpc[2])
                    ert_pa_avg: float = np.mean(ert_hpr[1])
                    pa_center: float = ert_pa_avg

                    cme_df: pd.DataFrame = pd.DataFrame.from_records(
                        cme_slice.cme_coords.read()
                    )
                    cme_df.replace(to_replace=[99999], value=np.nan, inplace=True)

                    # Look up the indices of this position angle slice, and average
                    #   the elongation coords in this window.
                    id_pa: pd.Series = (cme_df["pa"] >= (pa_center - pa_wid)) & (
                        cme_df["pa"] <= (pa_center + pa_wid)
                    )

                    # Get observations longitude
                    if self.use_model in ["surf", "compress_surf", "compress_surf3d"]:
                        h_obs_class: S.Observer | H.Observer = Sobs(
                            self.craft.upper(), frame_time
                        )
                    elif self.use_model in ["huxt"]:
                        h_obs_class = Hobs(self.craft.upper(), frame_time)
                    else:
                        sys.exit(
                            "Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'"
                        )

                    obs_lon.append(h_obs_class.lon.to(u.deg))

                    times_datetime.append(frame_time.to_datetime())
                    el_best.append(np.nanmean(cme_df["el"][id_pa]))
                    el_lo.append(np.nanmean(cme_df["el_lo"][id_pa]))
                    el_hi.append(np.nanmean(cme_df["el_hi"][id_pa]))

        obs_times_out: list[datetime.datetime] = [
            x for i, x in enumerate(times_datetime) if ~np.isnan(el_best[i])
        ]
        obs_lon_out: list[Quantity] = [
            x for i, x in enumerate(obs_lon) if ~np.isnan(el_best[i])
        ]
        obs_out: list[float] = [
            x for i, x in enumerate(el_best) if ~np.isnan(el_best[i])
        ]

        return obs_times_out, obs_lon_out, obs_out

    def l5_read_obs_from_file(self, obs_lat: float = 0.0) -> tuple[
        list[datetime.datetime],
        list[Quantity[u.deg]],
        list[Quantity[u.deg]],
        list[float],
    ]:
        times_datetime: list[datetime.datetime] = []
        elong: list[float] = []
        obs_lon: list[Quantity] = []

        for obs_file in self.obs_filenames:
            with open(obs_file, "r") as f:
                for line in f:
                    line_split: list[str] = line.split()

                    datetime_str: str = line_split[1].split(".")[0].strip()
                    datetime_obs: datetime.datetime = datetime.datetime.strptime(
                        datetime_str, "%Y-%m-%dT%H:%M:%S"
                    )
                    times_datetime.append(datetime_obs)

                    elong.append(float(line_split[2].strip()))
                    # Attach degree units, consistent with read_obs_from_file (the observation operator
                    #  calls obs_lon.to(u.deg) and ObserverTracer requires an angle Quantity; a bare
                    #  float raised there)
                    obs_lon.append(float(line_split[3].strip()) * u.deg)

        obs_times_out: list[datetime.datetime] = [
            x for i, x in enumerate(times_datetime) if ~np.isnan(elong[i])
        ]
        obs_lon_out: list[Quantity] = [
            x for i, x in enumerate(obs_lon) if ~np.isnan(elong[i])
        ]
        obs_lat_out: list[Quantity] = [
            obs_lat * u.deg for i, _ in enumerate(elong) if ~np.isnan(elong[i])
        ]
        obs_out: list[float] = [x for i, x in enumerate(elong) if ~np.isnan(elong[i])]

        return obs_times_out, obs_lon_out, obs_lat_out, obs_out


def main() -> None:
    h5_file_path: str = os.path.join(
        "C:\\",
        "Users",
        "ss905122",
        "PycharmProjects",
        "SIR_HUXt",
        "SSW_cme_classifications.hdf5",
    )
    use_model: str = "surf"
    use_model = "surf"
    ssw_event: str = "ssw_012"
    craft: str = "stb"
    img: str = "diff"
    ObsClass: Observations = Observations(
        use_model=use_model,
        obs_filenames=h5_file_path,
        ssw_event=ssw_event,
        craft=craft,
        img=img,
    )
    obs_lon: list[Quantity]
    obs_out: list[float]
    obs_times_out: list[datetime.datetime]
    obs_times_out, obs_lon, obs_out = ObsClass.read_obs_from_file()
    print(f"ssw_event: {ssw_event}, craft: {craft}, img: {img}")
    print(f"obs_times_out = {obs_times_out}")
    print(f"obs_out = {obs_out}")
    print(f"obs_lon_out = {obs_lon}")
    return


if __name__ == "__main__":
    main()
