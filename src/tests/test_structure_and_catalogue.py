"""Tests for cme_par_dict_structure, event_catalogue and osse_stats (no astropy needed)."""
import datetime

import numpy as np
import pytest

from cme_par_dict_structure import cme_par_get_indices, cme_par_get_keys, cme_par_key_to_index, required_dict_keys
import event_catalogue as ec
from osse_stats import numpy_nan_mean, numpy_nan_median, numpy_nan_std, weighted_median


# ---------------------------------------------------------------------------------------------
# cme_par_dict_structure
# ---------------------------------------------------------------------------------------------
def test_key_index_mapping_is_a_bijection():
    mapping = cme_par_key_to_index()
    assert set(mapping) == required_dict_keys()
    assert sorted(mapping.values()) == list(range(6))
    for key, idx in mapping.items():
        assert cme_par_get_indices(key) == idx
        assert cme_par_get_keys(idx) == key


def test_column_order_is_the_documented_one():
    assert [cme_par_get_keys(i) for i in range(6)] == ["t_init", "v", "width", "lon", "lat", "thick"]


@pytest.mark.parametrize("bad", ["speed", "weight", ""])
def test_unknown_key_is_rejected(bad):
    with pytest.raises(AssertionError):
        cme_par_get_indices(bad)


@pytest.mark.parametrize("bad", [-1, 6, 2.5])
def test_unknown_index_is_rejected(bad):
    with pytest.raises(AssertionError):
        cme_par_get_keys(bad)


# ---------------------------------------------------------------------------------------------
# event_catalogue
# ---------------------------------------------------------------------------------------------
def test_every_event_is_in_every_table():
    events = set(ec.SURF_INIT_TIMES)
    assert len(events) == 19
    assert set(ec.FIRST_GUESS_DONKI) == events
    assert set(ec.FIRST_GUESS_LUKE_PAPER) == events
    assert set(ec.OBS_FILE_NAMES) == events
    assert set(ec.SSW_EVENTS) <= events


# Known data problem: for 02A the first-guess CME reaches 21.5 rS (2020-10-26 05:57) BEFORE the model start
#  (2020-10-26 12:00), so it is launched before the run begins. Remove the xfail once SURF_INIT_TIMES is fixed.
_KNOWN_BAD_START = {"02A": "first-guess CME reaches 21.5 rS before the model start time"}


def _events_with_xfail():
    return [
        pytest.param(e, marks=pytest.mark.xfail(reason=_KNOWN_BAD_START[e], strict=True))
        if e in _KNOWN_BAD_START else e
        for e in ec.SURF_INIT_TIMES
    ]


@pytest.mark.parametrize("event", _events_with_xfail())
@pytest.mark.parametrize("table_name", ["FIRST_GUESS_DONKI", "FIRST_GUESS_LUKE_PAPER"])
def test_first_guess_reaches_21rs_after_the_model_start(table_name, event):
    fg = getattr(ec, table_name)[event]
    dt = fg.t_21rs - ec.SURF_INIT_TIMES[event]
    # the CME reaches 21.5 rS after the model starts, and within the 3-day compressible model run
    assert datetime.timedelta(0) < dt < datetime.timedelta(days=3)


@pytest.mark.parametrize("table", [ec.FIRST_GUESS_DONKI, ec.FIRST_GUESS_LUKE_PAPER])
def test_first_guesses_are_physical(table):
    for event, fg in table.items():
        assert isinstance(fg.t_21rs, datetime.datetime)
        assert 50 < fg.speed < 3500, event
        assert 0 < fg.width <= 360, event
        assert -180 <= fg.lon <= 180, event
        assert -90 <= fg.lat <= 90, event
        assert fg.thick >= 0, event


def test_the_two_first_guess_tables_differ_only_for_the_ssw_events():
    for event in ec.SURF_INIT_TIMES:
        same = ec.FIRST_GUESS_DONKI[event] == ec.FIRST_GUESS_LUKE_PAPER[event]
        assert same == (event not in ec.SSW_EVENTS), event


def test_observation_files_match_their_event():
    for event, name in ec.OBS_FILE_NAMES.items():
        if event in ec.SSW_EVENTS:
            assert name == "SSW_cme_classifications.hdf5"
        else:
            assert name.startswith(f"CME_{event}__") and name.endswith(".dat"), (event, name)


# ---------------------------------------------------------------------------------------------
# osse_stats
# ---------------------------------------------------------------------------------------------
def test_weighted_median_equal_weights_is_lower_median():
    assert weighted_median([3.0, 1.0, 2.0], [1, 1, 1]) == 2.0
    assert weighted_median([4.0, 1.0, 3.0, 2.0], [1, 1, 1, 1]) == 2.0


def test_weighted_median_follows_the_weight():
    assert weighted_median([1.0, 2.0, 3.0], [0.1, 0.1, 10.0]) == 3.0
    assert weighted_median([1.0, 2.0, 3.0], [10.0, 0.1, 0.1]) == 1.0


def test_weighted_median_along_an_axis():
    v = np.array([[1.0, 5.0, 3.0], [9.0, 7.0, 8.0]])
    w = np.ones_like(v)
    np.testing.assert_array_equal(weighted_median(v, w, axis=1), [3.0, 8.0])


def test_weighted_median_rejects_bad_weights():
    with pytest.raises(ValueError):
        weighted_median([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        weighted_median([1.0, 2.0], [1.0, -1.0])


def test_nan_safe_statistics():
    x = [1.0, np.nan, 3.0]
    assert numpy_nan_mean(x) == 2.0
    assert numpy_nan_median(x) == 2.0
    assert numpy_nan_std(x) == 1.0
    for f in (numpy_nan_mean, numpy_nan_median, numpy_nan_std):
        assert np.isnan(f([np.nan, np.nan]))
