"""
init_sir.py

Factory functions for the forward models used by the data assimilation, plus the initialiser for the
CME parameter ensemble dictionary.

    setup_huxt           -> HUXt (incompressible, 2D in longitude at one latitude)
    setup_surf           -> SURF (2D; solver "huxt" = incompressible, "hydro" = compressible)
    setup_surf_compress  -> SURF with the compressible "hydro" solver and a magnetic-field boundary
    setup_surf3d         -> SURF3d (stack of SURF latitudes, built from a 2D lat-lon speed map)

    initialise_cme_parameter_ensemble_dict -> empty ``CmeParEns`` dictionary for n ensemble members

All models are initialised for the Carrington rotation containing ``start_datetime``, with the
Carrington longitude of Earth at that time as the initial longitude.
"""
import datetime

import numpy as np
import numpy.typing as npt

import huxt.huxt as H
import surf.surf as S

import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from cme_par_ens import CmeParEns
from sir_observers import n_output_steps


def setup_huxt(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones(128) * 400 * u.km / u.s,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
) -> H.HUXt:
    """
    Initialise HUXt for the Carrington rotation containing start_datetime, at Earth's latitude.
    (By default a uniform 400 km/s wind, starting at 2008-01-01T00:00:00.)
    :param start_datetime: Initial datetime of HUXt simulation
    :param vr_in: Radial solar wind speed at the inner boundary, in km/s (one value per longitude)
    :param lon_start: Initial longitude of HUXt simulation in degrees
    :param lon_stop: Final longitude of HUXt simulation in degrees
    :param sim_time: HUXt simulation length
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii

    :return model: HUXt model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    # Carrington rotation and Earth's position at the start of the run
    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: H.Observer = H.Observer("EARTH", start_time)

    # Set up HUXt for a sim_time-day simulation, outputting every dt_scale
    # (NB: HUXt's accel_limit option is not used)
    model: H.HUXt = H.HUXt(
        v_boundary=vr_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=ert.lat.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
    )

    return model


def setup_surf(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones(128) * 400 * u.km / u.s,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
        solver: str = "huxt",
) -> S.SURF:
    """
    Initialise 2D SURF for the Carrington rotation containing start_datetime, in the equatorial plane
    (latitude 0 deg). (By default a uniform 400 km/s wind, starting at 2008-01-01T00:00:00.)
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_in: Radial solar wind speed at the inner boundary, in km/s (one value per longitude)
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param sim_time: SURF simulation length
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use, must be in ["huxt", "hydro"]
        ("huxt" = incompressible, "hydro" = compressible)

    :return model: SURF model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["huxt", "hydro"], "Solver must be in ['huxt', 'hydro']"

    # Carrington rotation and Earth's position at the start of the run
    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: S.Observer = S.Observer("EARTH", start_time)

    # Set up SURF for a sim_time-day simulation, outputting every dt_scale
    # (NB: latitude fixed at 0 deg rather than Earth's latitude, ert.lat)
    model: S.SURF = S.SURF(
        v_boundary=vr_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=0 * u.deg,
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
    )

    return model


def setup_surf_compress(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones(128) * 400 * u.km / u.s,
        br_in: npt.NDArray[Quantity[u.nT]] | Quantity[u.nT] = np.nan * u.nT,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
        solver: str = "hydro",
) -> S.SURF:
    """
    Initialise 2D compressible SURF (with a radial magnetic field boundary condition) for the Carrington
    rotation containing start_datetime, in the equatorial plane (latitude 0 deg).
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_in: Radial solar wind speed at the inner boundary, in km/s (one value per longitude)
    :param br_in: Radial magnetic field at the inner boundary, in nT
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param sim_time: SURF simulation length
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use; must be "hydro" (compressible)

    :return model: SURF model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["hydro"], "Solver must be in ['hydro']"

    # Carrington rotation and Earth's position at the start of the run
    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: S.Observer = S.Observer("EARTH", start_time)

    # Set up SURF for a sim_time-day simulation, outputting every dt_scale
    model: S.SURF = S.SURF(
        v_boundary=vr_in,
        b_boundary=br_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=0 * u.deg,
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        solver=solver,
    )

    return model


def setup_surf3d(
        start_datetime: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.ones((45, 128)) * 400 * u.km / u.s,
        n_in: npt.NDArray[Quantity[u.cm ** -3]]=np.nan * (u.cm ** -3),
        T_in: npt.NDArray[Quantity[u.K]]=np.nan * u.K,
        br_in: npt.NDArray[Quantity[u.nT]] | Quantity[u.nT]=np.nan * u.nT,
        br_lons: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg]=np.nan * u.deg,
        br_lats: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg]=np.nan * u.deg,
        vr_lons: npt.NDArray[Quantity[u.deg]] = np.arange(0, 361, 10) * u.deg,
        vr_lats: npt.NDArray[Quantity[u.deg]] = np.arange(-90, 90, 10) * u.deg,
        lat_min: Quantity[u.deg] = -30 * u.deg,
        lat_max: Quantity[u.deg] = 30 * u.deg,
        n_lat_surf: int=45,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        n_lon_surf: int=128,
        sim_time: Quantity[u.day] = 2 * u.day,
        dt_scale: int = 20,
        r_min: Quantity[u.solRad] = 21.5 * u.solRad,
        solver: str = "huxt",
        latitudes: Quantity[u.deg] | None = None,
        lazy: bool = False,
) -> S.SURF3d:
    """
    Initialise 3D SURF (SURF3d) from a 2D (latitude x longitude) map of inner-boundary wind speeds.
    NB: without ``latitudes``, SURF3d's latitudes are the centres of n_lat_surf equal bins in sine(latitude)
    over the WHOLE sphere, of which only those between lat_min and lat_max are kept (e.g. 45 bins over
    -30..30 deg gives 23 latitudes about 2.6 deg apart), not n_lat_surf latitudes between lat_min and lat_max.
    :param start_datetime: Initial datetime of SURF simulation
    :param vr_in: (n_vr_lats, n_vr_lons) map of radial solar wind speed at the inner boundary, km/s
    :param n_in, T_in: Inner-boundary density and temperature (currently unused)
    :param br_in, br_lons, br_lats: Inner-boundary magnetic field map and its coordinates (currently unused)
    :param vr_lons: Longitudes of the columns of vr_in, in degrees
    :param vr_lats: Latitudes of the rows of vr_in, in degrees
    :param lat_min, lat_max: Latitude range of the SURF3d grid (ignored if latitudes is given)
    :param n_lat_surf: Number of sine-latitude bins over the whole sphere (ignored if latitudes is given)
    :param latitudes: Explicit latitudes of the SURF runs (increasing), e.g. from fov_latitude_grid; one SURF
        plane is built at each of these and nowhere else
    :param lazy: (with latitudes) build the planes only when solved, one at a time, with solve_surf3d, which can
        also drop the arrays the caller does not need; much less memory (see solve_surf3d)
    :param lon_start: Initial longitude of SURF simulation in degrees
    :param lon_stop: Final longitude of SURF simulation in degrees
    :param n_lon_surf: Number of SURF longitudes
    :param sim_time: SURF simulation length
    :param dt_scale: Scalar specifying cadence of output timesteps
    :param r_min: Inner boundary radius in solar radii
    :param solver: Solver to use, must be in ["huxt", "hydro"]

    :return model3d: SURF3d model object with required ambient wind conditions at required
                    longitudes and latitude
    """

    assert type(start_datetime) == datetime.datetime

    assert solver in ["huxt", "hydro"], "Solver must be in ['huxt', 'hydro']"

    # Carrington rotation and Earth's position at the start of the run
    start_time: Time = Time(start_datetime, scale="utc")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert: S.Observer = S.Observer("EARTH", start_time)

    # Verify that the speed map (vr_in) has shape (n_vr_lats, n_vr_lons)
    if isinstance(vr_lats.value, float):
        len_lat = 1
    else:
        len_lat = len(vr_lats.value)

    if isinstance(vr_lons.value, float):
        len_lon = 1
    else:
        len_lon = len(vr_lons.value)

    shape_vr = np.shape(vr_in)
    assert len(shape_vr) == 2
    assert shape_vr == (len_lat, len_lon)

    if latitudes is not None:
        return surf3d_on_latitudes(
            latitudes=latitudes,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            v_map=vr_in,
            v_map_lat=vr_lats.to(u.rad),
            v_map_long=vr_lons.to(u.rad),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_stop.to(u.rad),
            nlon=n_lon_surf,
            dt_scale=dt_scale,
            simtime=sim_time,
            r_min=r_min,
            solver=solver,
            lazy=lazy,
        )

    # Set up SURF for a sim_time-day simulation, outputting every dt_scale
    model3d = S.SURF3d(
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        v_map=vr_in,
        v_map_lat=vr_lats.to(u.rad),
        v_map_long=vr_lons.to(u.rad),
        latitude_min=lat_min.to(u.deg),
        latitude_max=lat_max.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        nlon=n_lon_surf,
        nlat=n_lat_surf,
        dt_scale=dt_scale,
        simtime=sim_time,
        r_min=r_min,
        solver=solver,
    )

    return model3d


def fov_latitude_grid(
        fov_lat_min: Quantity[u.deg],
        fov_lat_max: Quantity[u.deg],
        buffer: Quantity[u.deg] = 2.0 * u.deg,
        resolution: Quantity[u.deg] = 1.0 * u.deg,
        include: list[Quantity[u.deg]] | None = None,
) -> Quantity[u.deg]:
    """
    Latitudes for a SURF3d that only covers an imager's field of view: every ``resolution`` (1 deg) from
    fov_lat_min - buffer to fov_lat_max + buffer, on whole multiples of the resolution (so the buffer is at least
    ``buffer`` on each side, and less than buffer + resolution), limited to -90..90 deg.
    :param fov_lat_min, fov_lat_max: Latitude range of the imager's lines of sight
    :param include: Extra latitudes that must also have a SURF plane (e.g. Earth's, for the arrival time): each
        one outside the grid adds the single grid latitude nearest to it
    :return: Increasing latitudes (deg)
    """
    res = resolution.to(u.deg).value
    assert res > 0, "resolution must be positive"
    lo = Quantity(fov_lat_min).to(u.deg).value - buffer.to(u.deg).value
    hi = Quantity(fov_lat_max).to(u.deg).value + buffer.to(u.deg).value
    assert hi >= lo, "fov_lat_max must not be below fov_lat_min"
    k_lo = int(np.floor(lo / res + 1e-9))
    k_hi = int(np.ceil(hi / res - 1e-9))
    k_lim = int(np.floor(90.0 / res + 1e-9))
    ks = set(range(max(k_lo, -k_lim), min(k_hi, k_lim) + 1))
    for lat in (include or []):
        k = int(np.clip(np.round(Quantity(lat).to(u.deg).value / res), -k_lim, k_lim))
        if k not in ks:
            ks.add(k)
    return np.array(sorted(ks), dtype=float) * res * u.deg


def surf3d_on_latitudes(
        latitudes: Quantity[u.deg],
        v_map: npt.NDArray[Quantity[u.km / u.s]],
        v_map_lat: Quantity[u.rad],
        v_map_long: Quantity[u.rad],
        cr_num: float,
        cr_lon_init: Quantity[u.deg],
        r_min: Quantity[u.solRad] = 30 * u.solRad,
        r_max: Quantity[u.solRad] = 240 * u.solRad,
        lon_out: Quantity[u.rad] = np.nan * u.rad,
        lon_start: Quantity[u.rad] = np.nan * u.rad,
        lon_stop: Quantity[u.rad] = np.nan * u.rad,
        simtime: Quantity[u.day] = 5.0 * u.day,
        dt_scale: float = 1.0,
        nlon: int = 128,
        dr: Quantity[u.solRad] = 1.5 * u.solRad,
        v_max: Quantity[u.km / u.s] = 3000 * (u.km / u.s),
        gamma: float = 1.5,
        solver: str = "huxt",
        track_cmes: bool = True,
        lazy: bool = False,
) -> S.SURF3d:
    """
    A SURF3d with one SURF plane at each of the given latitudes (SURF3d itself only offers its sine-latitude
    grid). It is a genuine S.SURF3d instance, built the way SURF3d.__init__ builds one (the boundary speed at
    each latitude is interpolated from v_map in latitude, then to the SURF longitude grid), so it solves,
    pickles and is imaged exactly like one; only ``lat`` / ``nlat`` / ``SURFlat`` differ.
    Arguments as for S.SURF3d (latitudes replaces latitude_min / latitude_max / nlat).
    :param lazy: Do not build the planes now: solve_surf3d builds, solves and (optionally) slims them one at a
        time, so only one plane's full arrays exist at once. model.solve() then raises (use solve_surf3d).
    """
    lat = np.sort(np.unique(Quantity(latitudes).to(u.rad).value)) * u.rad
    assert lat.size >= 1, "need at least one latitude"
    assert np.all(np.abs(lat.to(u.deg).value) <= 90.0)
    assert len(v_map_lat) == len(v_map[:, 1])
    assert len(v_map_long) == len(v_map[1, :])

    model3d = S.SURF3d.__new__(S.SURF3d)
    model3d.latitude_min = lat.min()
    model3d.latitude_max = lat.max()
    model3d.lat = lat
    model3d.nlat = int(lat.size)
    model3d.track_cmes = track_cmes

    # Boundary speeds at each latitude (as SURF3d.__init__)
    longs, _, _ = S.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad, nlon=nlon)
    model3d.v_in = []
    vlong = np.ones(len(v_map_long))
    for thislat in model3d.lat:
        for ilong in range(0, len(v_map_long)):
            vlong[ilong] = np.interp(thislat.value, v_map_lat.value, v_map[:, ilong].value)
        model3d.v_in.append(np.interp(longs.value, v_map_long.value, vlong) * (u.km / u.s))

    model3d._plane_kwargs = dict(cr_num=cr_num, cr_lon_init=cr_lon_init, r_min=r_min, r_max=r_max,
                                 lon_out=lon_out, lon_start=lon_start, lon_stop=lon_stop, simtime=simtime,
                                 dt_scale=dt_scale, nlon=nlon, dr=dr, v_max=v_max, gamma=gamma, solver=solver,
                                 track_cmes=track_cmes)
    if lazy:
        model3d._lazy_lat = model3d.lat.copy()
        model3d._lazy_v_in = list(model3d.v_in)
        model3d.SURFlat = []
        model3d.solve = _lazy_surf3d_solve_guard
        return model3d

    model3d.SURFlat = [_surf3d_plane(model3d, model3d.v_in[i], model3d.lat[i]) for i in range(model3d.nlat)]
    return model3d


def _surf3d_plane(model3d: S.SURF3d, v_in, lat) -> S.SURF:
    """One (unsolved) SURF plane of a SURF3d built by surf3d_on_latitudes."""
    return S.SURF(v_boundary=v_in, latitude=lat, **model3d._plane_kwargs)


def _lazy_surf3d_solve_guard(*args, **kwargs):
    raise RuntimeError("This SURF3d was built with lazy=True: solve it with init_sir.solve_surf3d(model, cmes)")


def _earth_plane_index(plane: S.SURF, lats: Quantity) -> int:
    """
    Index of the latitude nearest Earth's mean latitude over the run, exactly as solved_run in
    sir_observation_operator picks the arrival plane (S.SURF3d.get_observer("EARTH") = Earth at the first plane's
    time_init + time_out; every plane has the same times).
    """
    earth_lat = np.mean(S.Observer("EARTH", plane.time_init + plane.time_out).lat.to(u.rad).value)
    return int(np.argmin(np.abs(Quantity(lats).to(u.rad).value - earth_lat)))


def _slim_plane(plane: S.SURF, keep_output_steps: int | None, keep_full_run: bool,
                keep_temperature: bool = True) -> None:
    """
    Release the arrays of a solved plane that are not needed, keeping:
    * density and (keep_temperature) temperature over the first keep_output_steps output times (the imaging
      window; all if None);
    * if keep_full_run (the plane nearest Earth, for the arrival): the speed grid and the temperature over the
      whole run, so the plasma conditions at Earth are available after the imaging window;
    * r, lon, time_out, time_init and the CMEs (whose tracer coords, computed during solve, cover the whole run).
    Released: the magnetic field, the (NaN) CME tracer speeds, the speed grid of the other planes and the
    temperature if not keep_temperature.
    """
    for name in ("b_grid", "cme_particles_v"):
        if hasattr(plane, name):
            delattr(plane, name)
    if not keep_full_run and hasattr(plane, "v_grid"):
        delattr(plane, "v_grid")
    if not keep_temperature and hasattr(plane, "temp_grid"):
        delattr(plane, "temp_grid")

    def truncated(grid):
        if keep_output_steps is None or keep_output_steps >= grid.shape[0]:
            return grid
        return grid[:keep_output_steps].copy()          # a copy, so the full array is freed

    if hasattr(plane, "rho_grid"):
        plane.rho_grid = truncated(plane.rho_grid)
    if hasattr(plane, "temp_grid") and not keep_full_run:
        plane.temp_grid = truncated(plane.temp_grid)


def solve_surf3d(
        model3d: S.SURF3d,
        cme_list: list,
        keep_until: Quantity[u.s] | None = None,
        keep_wind_nearest_earth: bool = False,
        only_nearest_earth: bool = False,
        slim: bool = True,
        keep_temperature: bool = True,
) -> S.SURF3d:
    """
    Solve a SURF3d from surf3d_on_latitudes(..., lazy=True) one latitude plane at a time, keeping only what is
    needed (density and temperature over the imaging window; speed only for the plane nearest Earth), so a member
    needs a fraction of the memory of SURF3d.solve (which holds speed, density and temperature for every plane
    and the whole run). The planes are independent 2D SURF runs, so every kept value
    is identical to SURF3d.solve.

    :param keep_until: Keep density only up to this time after the model start (the synthetic imager's window,
        e.g. 3 days of a 5-day run; the same output steps as sir_observers.truncated_model_view keeps); None keeps
        all
    :param keep_wind_nearest_earth: Keep the speed grid (and temperature) of the plane nearest Earth over the whole
        run (the CME arrival speed, see sir_observation_operator.wind_speed_at_arrival)
    :param only_nearest_earth: Solve only the plane nearest Earth (arrival-time runs: the arrival is taken from that
        plane, so the result is the same as solving them all); model.lat / SURFlat then hold that plane only
    :param slim: Release unneeded arrays (_slim_plane); False keeps every array (still one plane built at a time)
    :param keep_temperature: Keep every plane's temperature (over the same window as density; default True)
    :return: model3d, solved (SURFlat and lat hold the solved planes)
    """
    if not hasattr(model3d, "_lazy_lat"):        # an ordinary SURF3d: nothing to save
        model3d.solve(cme_list)
        return model3d
    lats = model3d._lazy_lat
    v_in = model3d._lazy_v_in
    n = int(lats.size)

    if only_nearest_earth:
        plane = _surf3d_plane(model3d, v_in[n // 2], lats[n // 2])
        i_earth = _earth_plane_index(plane, lats)
        if i_earth != n // 2:
            plane = _surf3d_plane(model3d, v_in[i_earth], lats[i_earth])
        plane.solve(cme_list)
        if slim:
            _slim_plane(plane, keep_output_steps=None, keep_full_run=True, keep_temperature=keep_temperature)
        model3d.SURFlat, model3d.lat, model3d.nlat, model3d.v_in = [plane], lats[i_earth:i_earth + 1], 1, [v_in[i_earth]]
        return model3d

    planes = []
    i_earth = None
    for i in range(n):
        plane = _surf3d_plane(model3d, v_in[i], lats[i])
        if keep_wind_nearest_earth and i_earth is None:
            i_earth = _earth_plane_index(plane, lats)
        plane.solve(cme_list)
        if slim:
            keep_steps = None if keep_until is None else n_output_steps(plane, keep_until)
            _slim_plane(plane, keep_steps, keep_full_run=keep_wind_nearest_earth and i == i_earth,
                        keep_temperature=keep_temperature)
        planes.append(plane)
    model3d.SURFlat, model3d.lat, model3d.nlat, model3d.v_in = planes, lats, n, list(v_in)
    return model3d


def initialise_cme_parameter_ensemble_dict(
        n_ensemble: int,
        surf_init_time: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0),
) -> CmeParEns:
    """
    Initialise a CME parameter ensemble dictionary with placeholder values (zeros, uniform weights)
    for each ensemble member. The CME parameters are filled in afterwards (e.g. from prior samples).
    :param n_ensemble: The number of ensemble members in the SIR analysis
    :param surf_init_time: Initial datetime of SURF simulation
    :return parameter_dict: Dictionary conforming to ``CmeParEns``
    """

    parameter_dict: CmeParEns = {
        "surf_init_time": surf_init_time,
        "t_init": [
            datetime.datetime(2008, 1, 1, 0, 0, 0)
            for _ in range(n_ensemble)
        ],
        "v": [0 for _ in range(n_ensemble)] * u.km / u.s,
        "width": [0 for _ in range(n_ensemble)] * u.deg,
        "lon": [0 for _ in range(n_ensemble)] * u.deg,
        "lat": [0 for _ in range(n_ensemble)] * u.deg,
        "thick": [0 for _ in range(n_ensemble)] * u.solRad,
        "v_transit": [0 for _ in range(n_ensemble)],
        "v_hit": [0 for _ in range(n_ensemble)],
        "likelihood": [0 for _ in range(n_ensemble)],
        # Uniform prior weights: 1/N (linear) and -log(N) (log)
        "weight": [(1.0 / n_ensemble) for _ in range(n_ensemble)],
        "log_weight": [-np.log(n_ensemble) for _ in range(n_ensemble)],
        "n_members": n_ensemble
    }

    return parameter_dict
