"""
Tests for the configuration (da_surf_setup, init_sir), the observation-operator helpers, the arrival-time
helpers and the WSA reader. Need astropy plus the SURF/HUXt/sunpy packages (only imported, no model runs).
"""
import datetime
import types

import numpy as np
import pytest

u = pytest.importorskip("astropy.units")
pytest.importorskip("surf")
pytest.importorskip("huxt")
pytest.importorskip("sunpy")

from da_test_helpers import make_ensemble


# ---------------------------------------------------------------------------------------------
# init_sir
# ---------------------------------------------------------------------------------------------
def test_initialise_cme_parameter_ensemble_dict():
    from init_sir import initialise_cme_parameter_ensemble_dict
    t0 = datetime.datetime(2020, 5, 1)
    d = initialise_cme_parameter_ensemble_dict(n_ensemble=7, surf_init_time=t0)
    assert d["n_members"] == 7 and d["surf_init_time"] == t0
    for key in ["t_init", "v", "width", "lon", "lat", "thick", "weight", "log_weight"]:
        assert len(d[key]) == 7, key
    np.testing.assert_allclose(d["weight"], 1 / 7)
    np.testing.assert_allclose(d["log_weight"], -np.log(7))
    assert d["v"].unit == u.km / u.s and d["lon"].unit == u.deg and d["thick"].unit == u.solRad


@pytest.mark.parametrize("factory", ["setup_surf", "setup_surf3d", "setup_huxt"])
def test_model_factories_reject_non_datetime_start(factory):
    import init_sir
    with pytest.raises(AssertionError):
        getattr(init_sir, factory)(start_datetime="2012-01-01")


def test_setup_surf_rejects_unknown_solver():
    from init_sir import setup_surf
    with pytest.raises(AssertionError):
        setup_surf(solver="nonsense")


# ---------------------------------------------------------------------------------------------
# da_surf_setup.Setup (methods tested without running __init__, which downloads WSA maps)
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def setup_cls():
    pytest.importorskip("dotenv")
    import da_surf_setup
    return da_surf_setup


def _bare_setup(mod, event="09A", synthetic=False):
    s = object.__new__(mod.Setup)
    craft = "sta" if event.endswith("A") or event.startswith("ssw") else "stb"
    s.get_ssw_event_craft = lambda: (event, craft, "diff")
    s.get_use_synthetic_obs = lambda: synthetic
    s.surf_par_dict = {"cme_init_rad": 12 * u.solRad, "r_min": 21.5 * u.solRad}
    return s


def test_surf_init_time_comes_from_the_catalogue(setup_cls):
    import event_catalogue as ec
    for event, t in ec.SURF_INIT_TIMES.items():
        assert _bare_setup(setup_cls, event).get_surf_init_time() == t
    assert _bare_setup(setup_cls, synthetic=True).get_surf_init_time() == datetime.datetime(2012, 1, 1)


def test_unknown_event_exits(setup_cls):
    with pytest.raises(SystemExit):
        _bare_setup(setup_cls, "99Z").get_surf_init_time()


def test_first_guess_launch_time_round_trips_to_the_catalogue_time(setup_cls):
    """r_min = 21.5 rS: going 21.5 -> 12 rS and back 12 -> 21.5 rS must give the catalogue time."""
    import event_catalogue as ec
    for event, fg in ec.FIRST_GUESS_DONKI.items():
        out = _bare_setup(setup_cls, event).initialise_fg_cme_parameters()
        assert abs((out["fg_cme_t_init"] - fg.t_21rs).total_seconds()) < 1e-3, event
        assert out["fg_cme_speed"] == fg.speed and out["fg_cme_lon"] == fg.lon


def test_observation_file_comes_from_the_catalogue(setup_cls, monkeypatch):
    import event_catalogue as ec
    monkeypatch.setenv("OBS_DIR", "OBSDIR")
    for event, name in ec.OBS_FILE_NAMES.items():
        s = _bare_setup(setup_cls, event)
        s.initialise_true_parameters = lambda: {"t_init": datetime.datetime(2012, 1, 1, 1)}
        s.get_use_parallel = lambda: False
        out = s.initialise_observation_parameters()
        assert out["obs_filenames"][0].endswith(name), event
        assert out["elong_min"] < out["elong_max"]


def test_da_and_prior_settings_are_consistent(setup_cls):
    s = _bare_setup(setup_cls)
    da = s.initialise_da_parameters()
    from cme_par_dict_structure import required_dict_keys
    assert set(da["pars_in_state"]) <= required_dict_keys()
    assert 0 < da["delta_aux_pf"] <= 1 and da["n_members"] > 1
    prior = s.initialise_prior_cme_cov()
    assert prior["cme_cov_type"] in s.allowed_cov_types()
    for k in ["sd_t_init", "sd_speed", "sd_width", "sd_lon", "sd_lat", "sd_thick"]:
        assert prior[k] >= 0


# ---------------------------------------------------------------------------------------------
# Observation operator helpers
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def obs_op():
    """An ObservationOperator built with its real __init__ (no model is run)."""
    from sir_observation_operator import ObservationOperator
    d = make_ensemble(n=5, seed=11)
    return ObservationOperator(
        use_model="compress_surf", cme_par_dict=d, obs_time_in_datetime=datetime.datetime(2012, 1, 1, 12),
        obs_cov=0.5, craft="twin", obs_lon=300 * u.deg, surf_init_time=d["surf_init_time"],
        r_min=21.5 * u.solRad, cme_init_rad=12 * u.solRad, use_parallel=False,
    )


def test_launch_time_adds_the_travel_time_to_r_min(obs_op):
    t_launch, v = obs_op.extract_cme_launch_time_and_speed()
    d = obs_op.cme_par_dict
    for i in range(d["n_members"]):
        t_init_s = (d["t_init"][i] - d["surf_init_time"]).total_seconds()
        delay = (9.5 * u.solRad).to(u.km).value / d["v"][i].to(u.km / u.s).value
        assert t_launch[i].to(u.s).value == pytest.approx(t_init_s + delay)
    np.testing.assert_allclose(v.to(u.km / u.s).value, d["v"].to(u.km / u.s).value)


def test_cme_longitude_is_wrapped_for_the_model(obs_op):
    lon = obs_op.extract_cme_lon().to(u.deg).value
    assert np.all(lon <= 180) and np.all(lon > -180)
    np.testing.assert_allclose(np.mod(lon, 360), np.mod(obs_op.cme_par_dict["lon"].to(u.deg).value, 360))


def test_single_cme_dictionary_is_accepted(obs_op):
    d = make_ensemble(n=1, seed=12)
    single = dict(d, t_init=d["t_init"][0], v=d["v"][0], width=d["width"][0], lon=d["lon"][0],
                  lat=d["lat"][0], thick=d["thick"][0])
    obs_op.cme_par_dict, obs_op.n_members = single, 1
    assert len(obs_op.extract_cme_width()) == 1
    assert len(obs_op.extract_cme_launch_time_and_speed()[0]) == 1


@pytest.mark.parametrize("b5, b21", [(2.5, 2.5), (1.0, 3.0), (4.0, -1.0)])
def test_bias_correction_is_the_line_through_its_anchor_points(obs_op, b5, b21):
    obs_op.bias_term_5rs, obs_op.bias_term_21rs = b5, b21
    assert obs_op.obs_op_bias_correction(5.0) == pytest.approx(b5)
    assert obs_op.obs_op_bias_correction(21.0) == pytest.approx(b21)
    assert obs_op.obs_op_bias_correction(13.0) == pytest.approx((b5 + b21) / 2)


def test_cme_objects_one_per_member(obs_op):
    cmes = obs_op.make_cme_objects()
    assert len(cmes) == obs_op.n_members


# ---------------------------------------------------------------------------------------------
# Arrival-time helpers
# ---------------------------------------------------------------------------------------------
def test_launch_delay_to_r_min():
    xr = pytest.importorskip("xarray")
    from osse_arrival import launch_delay_to_r_min
    ds = xr.Dataset(data_vars=dict(r_min=21.5, cme_init_rad=12.0))
    v = [500.0, 1000.0] * u.km / u.s
    expected = (9.5 * u.solRad).to(u.km).value / np.array([500.0, 1000.0])
    np.testing.assert_allclose(launch_delay_to_r_min(ds, v).to(u.s).value, expected)


def test_solved_cmes_uses_the_slice_nearest_earth_for_surf3d():
    pytest.importorskip("xarray")
    from osse_arrival import solved_cmes
    flat = types.SimpleNamespace(cmes=["cme2d"])
    assert solved_cmes(flat) == ["cme2d"]
    slices = [types.SimpleNamespace(latitude=lat * u.deg, cmes=[f"cme@{lat}"]) for lat in (-10, 0, 10)]
    earth = types.SimpleNamespace(lat=[6.0, 7.0] * u.deg)
    model3d = types.SimpleNamespace(SURFlat=slices, get_observer=lambda body: earth)
    assert solved_cmes(model3d) == ["cme@10"]


# ---------------------------------------------------------------------------------------------
# WSA reader (no network access: only URL construction)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("date, month_url, prev_url", [
    (datetime.datetime(2012, 9, 11), "2012/09", "2012/08"),
    (datetime.datetime(2013, 1, 3), "2013/01", "2012/12"),
])
def test_wsa_urls(tmp_path, date, month_url, prev_url):
    pytest.importorskip("requests")
    pytest.importorskip("bs4")
    from wsa_reader import ReadWSAFiles
    r = ReadWSAFiles(date, output_dir=str(tmp_path / "wsa"))
    assert r.wsa_url.endswith(month_url) and r.wsa_url_prev_month.endswith(prev_url)
    assert (tmp_path / "wsa").is_dir()
