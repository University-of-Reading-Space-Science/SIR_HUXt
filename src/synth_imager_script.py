"""
synth_imager_script.py

Scratch script for testing the SURF3d synthetic heliospheric imager (surf_imaging) on a uniform
400 km/s wind with a single cone CME. Not part of the data-assimilation pipeline.
"""
import datetime
import astropy.units as u
import numpy as np

import surf.surf as S
import surf.surf_imaging as Sim
import surf.surf_inputs as Sin
from init_sir import setup_surf, setup_surf3d, initialise_cme_parameter_ensemble_dict
from sir_observation_operator import ObservationOperator

import matplotlib.pyplot as plt

class SynthObsEphem:
    """Minimal observer ephemeris for surf_imaging (cf. sir_observers.SynthObsEphem)."""
    def __init__(self, surf_init_time, obs_times, obs_rads, obs_lats, obs_lons):
        self.datetime = [
            surf_init_time + datetime.timedelta(seconds=i) for i in obs_times.value
        ]
        self.time = obs_times
        self.r = obs_rads
        self.lat = obs_lats
        self.lon = obs_lons


def main() -> None:
    """Build a SURF3d run, add a CME and compute / track the synthetic j-maps."""
    surf_init_time = datetime.datetime(2012, 1, 1, 0, 0, 0, 0)
    vr_in = np.ones(128) * 400 * u.km / u.s
    vr_in_3d = np.ones((91, 181)) * 400 * u.km / u.s
    v_lons = np.arange(0, 361, 2) * u.deg
    v_lats = np.arange(-90, 91, 2) * u.deg
    lat_min = -30 * u.deg
    lat_max = 30 * u.deg
    lon_start = 290 * u.deg
    lon_stop = 430 * u.deg
    sim_time = 3 * u.day
    dt_scale = 1
    r_min = 21.5 * u.solRad
    solver = "hydro"

    print(
        f"vr_in_3d = {np.shape(vr_in_3d)},\n\n"
        f" v_lons = {np.shape(v_lons)},\n\n"
        f" v_lats = {np.shape(v_lats)}"
    )
    '''
    cr = 2000
    vmap, vlongs, vlats = Sin.get_MAS_vr_map(cr)
    print(f"vmap = {vmap},\n\n vlongs = {vlongs},\n\n vlats = {vlats}")
    print(f"shape(v_map) = {np.shape(vmap)}")
    print(f"shape(vlongs) = {np.shape(vlongs)}")
    print(f"shape(vlats) = {np.shape(vlats)}")

    print(f"vlongs = {vlongs.to(u.deg)},\n\n vlats = {vlats.to(u.deg)}")
    #sys.exit()
    vr_in_3d = np.ones((19, 73)) * 400 * u.km / u.s
    v_lons = np.arange(0, 361, 5) * u.deg
    v_lats = np.arange(-90, 91, 10) * u.deg
    lat_min = -30 * u.deg,
    lat_max = 30 * u.deg,
    lon_start = 290 * u.deg,
    lon_stop = 380 * u.deg,
    sim_time: Quantity[u.day] = 2 * u.day,
    dt_scale: int = 20,
    r_min: Quantity[u.solRad] = 21.5 * u.solRad,
    solver: str = "huxt",
    
    model: SURF = setup_surf(
        start_datetime=surf_init_time,
        v_map = vr_in,
        lon_start=lon_start,
        lon_stop=lon_stop,
        sim_time=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
    )
    '''

    model = setup_surf3d(
        start_datetime=surf_init_time,
        vr_in=vr_in_3d,
        vr_lons=v_lons,
        vr_lats=v_lats,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_start=lon_start,
        lon_stop=lon_stop,
        sim_time=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
    )

    cme_launch_datetime = datetime.datetime(2012, 1, 1, 1, 0, 0, 0)
    cme_launch_time = (cme_launch_datetime - surf_init_time).total_seconds() * u.s
    # print(cme_launch_time)
    cme_speed = 600 * u.km / u.s
    cme_width = 40 * u.deg
    cme_lon = 0 * u.deg
    cme_lat = 0 * u.deg
    cme_thickness = 0.0 * u.solRad
    cme_fixed_duration = True
    fixed_duration = 12 * 60 * 60 * u.s

    cme_object = S.ConeCME(
        t_launch=cme_launch_time,
        v=cme_speed,
        width=cme_width,
        longitude=cme_lon,
        latitude=cme_lat,
        thickness=cme_thickness,
        cme_fixed_duration=cme_fixed_duration,
        fixed_duration=fixed_duration
    )

    # Initialise CME parameter dictionary
    cme_par_dict = initialise_cme_parameter_ensemble_dict(
        n_ensemble=1,
        surf_init_time=surf_init_time
    )
    cme_par_dict["t_init"] = [cme_launch_datetime]
    cme_par_dict["v"] = [cme_speed]
    cme_par_dict["width"] = [cme_width]
    cme_par_dict["lon"] = [cme_lon]
    cme_par_dict["lat"] = [cme_lat]
    cme_par_dict["thick"] = [cme_thickness]

    # obs_time = datetime.datetime(2012, 1, 1, 1, 0, 0)
    # obs_radius = 1.0 * u.AU
    # obs_time_datetime = np.array([obs_time])
    #
    # obs_from_start_sec = (obs_time - surf_init_time).total_seconds()
    obs_cadence_sec = 60 * 60  # 1 * 60 * 60
    n_obs = int((3 * 24 * 3600) / obs_cadence_sec) - 1
    first_obs = datetime.datetime(
        2012, 1, 1, 1, 0, 0
    )
    obs_from_start_sec = (first_obs - surf_init_time).total_seconds()
    obs_times = np.array([
        obs_from_start_sec + (obs_cadence_sec * i)
        for i in range(n_obs)
    ]) * u.s
    obs_times_datetime = [
        surf_init_time + datetime.timedelta(seconds=obs_times[i].value)
        for i in range(n_obs)
    ]
    obs_rads = np.array([
        1.0 for i in range(n_obs)
    ]) * u.AU
    obs_lats = np.array([
        0.0 for i in range(n_obs)
    ]) * u.deg
    obs_lons = [
        60#(360.0 - 120.0)# + (i * 10.0 / float(n_obs)))
        for i in range(n_obs)
    ] * u.deg
    print(f"obs_times_datetime: {obs_times_datetime}")
    #sys.exit()
    print(cme_object.t_launch)
    model.solve([cme_object])

    synth_obs_class = SynthObsEphem(
        surf_init_time=surf_init_time,
        obs_times=obs_times,
        obs_rads=obs_rads,
        obs_lats=obs_lats,
        obs_lons=obs_lons
    )

    synth_imager_class = Sim.SyntheticImager3D(synth_obs_class)

    jmap, djmap = synth_imager_class.compute_jmap(model)
    cme_profile = synth_imager_class.track_cmes(model, djmap)[0]

    synth_imager_time = cme_profile["feature_00"]["t"][:]
    synth_imager_elon = cme_profile["feature_00"]["e"][:]
    print(f"synth_imager_time: {synth_imager_time}")
    print(f"synth_imager_elon: {synth_imager_elon}")

    #for i in range(n_obs):
    obs_op_class = ObservationOperator(
        use_model="surf",
        cme_par_dict=cme_par_dict,
        obs_radius=obs_rads,
        obs_lon=obs_lons,
        obs_lat=obs_lats,
        obs_time_in_datetime=obs_times_datetime,
        obs_cov=0,
        surf_init_time=surf_init_time,
        vr_in=vr_in,
        lon_start=lon_start,
        lon_stop=lon_stop,
        sim_time=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        cme_init_rad=r_min,
        cme_fixed_duration=cme_fixed_duration,
        fixed_duration=fixed_duration,
        plot_surf_output=False,
        bias_term_bool=False,
        bias_term_5rs=0,
        bias_term_21rs=0
    )
    cme_obs_op = obs_op_class.make_obs_op()[0, :]
    print(f"cme_obs_op: {np.shape(cme_obs_op)}")
    print(f"obs_times: {obs_times.to(u.day)}")
    fig, ax = synth_imager_class.plot_jmap_with_cme_profiles(model, jmap, djmap)
    ax[1].plot(obs_times.to(u.day), cme_obs_op, 'b.', label="SURF CME flanks")
    ax[1].legend()
    plt.show()

    #print(f"cme_obs_op: {obs_times.to(u.day)}")
    #print(f"cme_profiles: {synth_imager_time}")

    #synth_e_nearest = []
    cme_obs_op_nearest = []
    for i, synth_t in enumerate(synth_imager_time):
        # Calculate nearest time in cme_obs_op elongation profile
        ind_req = np.argmin(np.abs(obs_times.to(u.day).value - synth_t))

        cme_obs_op_nearest.append(cme_obs_op[ind_req])

    #print(f"diff cme_flanks_elon: {synth_imager_elon - cme_obs_op_nearest}")
    return None


if __name__ == "__main__":
    main()