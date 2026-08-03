import numpy as np
import numpy.typing as npt
import datetime
import os
import sys
import pandas as pd
import xarray as xr
from typing import TypedDict
import json

import huxt.huxt as H
import surf.surf as S
import surf.surf_analysis as SA
from mypy.build import TypedDict
from scipy.special.cython_special import log_wright_bessel
import math
import sir_huxt_mono_obs as shmo
import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose

from astropy.time import Time
from sir_observation_operator import ObservationOperator
from tests.expected_obs_op import ExpectedObservationOperator, ExpectedCMEFlankSingle
import matplotlib.pyplot as plt
from cme_par_ens import CmeParEns
import seaborn as sns
import colorcet as cc
import pytest
from cme_par_dict_structure import required_dict_keys

@pytest.mark.usefixtures("init_obs_op", "par_dict")
class TestObservationOperator:
    @pytest.fixture
    def exp_obs_op(self):
        return ExpectedObservationOperator()

    @pytest.fixture
    def exp_cme_flank(self):
        return ExpectedCMEFlankSingle()

    """
    def test_extract_cme_launch_time_and_speed(self, init_obs_op, exp_obs_op) -> None:
        cme_launch_time, cme_speed = init_obs_op.extract_cme_launch_time_and_speed()

        expected_cme_launch_time: list[Quantity[u.s]] = exp_obs_op.expected_cme_launch_time()
        expected_cme_speed: list[Quantity[u.km / u.s]] = exp_obs_op.expected_cme_speed()

        assert (
            (cme_launch_time.value == pytest.approx(expected_cme_launch_time.value))
            and (cme_launch_time.unit == expected_cme_launch_time.unit)
        )
        assert (
                (cme_speed.value == pytest.approx(expected_cme_speed.value))
                and (cme_speed.unit == expected_cme_speed.unit)
        )

        return None

    def test_extract_cme_width(self, init_obs_op, exp_obs_op) -> None:
        cme_width = init_obs_op.extract_cme_width()

        expected_cme_width: list[Quantity[u.km]] = exp_obs_op.expected_cme_width()

        assert (
                (cme_width.value == pytest.approx(expected_cme_width.value))
                and (cme_width.unit == expected_cme_width.unit)
        )

        return None

    def test_extract_cme_lon(self, init_obs_op, exp_obs_op) -> None:
        cme_lon = init_obs_op.extract_cme_lon()
        expected_cme_lon: list[Quantity[u.deg]] = exp_obs_op.expected_cme_lon()

        assert (
            (cme_lon.value == pytest.approx(expected_cme_lon.value))
            and (cme_lon.unit == expected_cme_lon.unit)
        )
        return None

    def test_extract_cme_lat(self, init_obs_op, exp_obs_op) -> None:
        cme_lat = init_obs_op.extract_cme_lat()
        expected_cme_lat: list[Quantity[u.deg]] = exp_obs_op.expected_cme_lat()

        assert (
            (cme_lat.value == pytest.approx(expected_cme_lat.value))
            and (cme_lat.unit == expected_cme_lat.unit)
        )
        return None

    def test_extract_cme_thickness(self, init_obs_op, exp_obs_op) -> None:
        cme_thick = init_obs_op.extract_cme_thickness()
        expected_cme_thick: list[Quantity[u.km]] = exp_obs_op.expected_cme_thick()

        assert (
            (cme_thick.value == pytest.approx(expected_cme_thick.value))
            and (cme_thick.unit == expected_cme_thick.unit)
        )
        return None

    def test_extract_cme_parameters(self, init_obs_op, exp_obs_op) -> None:
        cme_parameters = init_obs_op.extract_cme_parameters()
        expected_cme_par = exp_obs_op.expected_cme_parameters()

        for i in range(6):
            assert (
                (cme_parameters[i].value == pytest.approx(expected_cme_par[i].value))
                and (cme_parameters[i].unit == expected_cme_par[i].unit)
            )
        return None

    def test_make_cme_objects(self, init_obs_op, exp_obs_op, mocker) -> None:
        mock_cme = mocker.patch("surf.surf.ConeCME")
        mock_cme.return_value = 500

        expected_cme_pars = exp_obs_op.expected_cme_parameters()
        expected_cme_launch_time = expected_cme_pars[0]
        expected_cme_speed = expected_cme_pars[1]
        expected_cme_width = expected_cme_pars[2]
        expected_cme_lon = expected_cme_pars[3]
        expected_cme_lat = expected_cme_pars[4]
        expected_cme_thickness = expected_cme_pars[5]

        expected_mock_calls = [
            mocker.call(
                t_launch=expected_cme_launch_time[i],
                v=expected_cme_speed[i],
                width=expected_cme_width[i],
                longitude=expected_cme_lon[i],
                latitude=expected_cme_lat[i],
                thickness=expected_cme_thickness[i],
                cme_fixed_duration=init_obs_op.cme_fixed_duration,
                fixed_duration=init_obs_op.fixed_duration,
            ) for i in range(exp_obs_op.n_members)
        ]

        assert init_obs_op.make_cme_objects() == [500 for _ in range(init_obs_op.n_members)]
        mock_cme_calls = mock_cme.call_args_list

        ##############################################################
        for i in range(exp_obs_op.n_members):
            mock_call_kwargs = mock_cme_calls[i].kwargs
            expected_kwargs = expected_mock_calls[i].kwargs

            assert (mock_call_kwargs.keys() == expected_kwargs.keys())
            print(expected_kwargs.keys())
            for key in mock_call_kwargs.keys():
                if key in ["cme_fixed_duration"]:
                    assert (mock_call_kwargs[key] == expected_kwargs[key])
                else:
                    assert (
                        (mock_call_kwargs[key].value == pytest.approx(expected_kwargs[key].value))
                        and (mock_call_kwargs[key].unit == expected_kwargs[key].unit)
                    )


        #for i, ic in enumerate(expected_mock_calls_values):


        return None

    @pytest.mark.parametrize(
        "init_test_cme_flank_single_ens, ens_no", [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)
        ], indirect=["init_test_cme_flank_single_ens"]
    )
    def test_get_cme_flank_single_ens_member(
            self, init_obs_op, init_test_cme_flank_single_ens, ens_no, exp_cme_flank, mocker
    ) -> None:

        # Initialise mockers for the S.SURF and S.ConeCME calls
        mock_surf = mocker.patch("surf.surf.SURF")
        mock_surf.return_value = S.SURF()

        mock_solve = mocker.patch("surf.surf.SURF.solve")
        mock_solve.return_value = 450

        mock_cme = mocker.patch("surf.surf.ConeCME")
        mock_cme.return_value = 500

        mock_flank = mocker.patch("SIR_SURF.code.sir_observation_operator.Observer.compute_flank_profile")
        mock_flank.return_value = init_test_cme_flank_single_ens

        expected_cme_flank_el = exp_cme_flank.expected_flank_el(ens_no)
        expected_cme_flank_r = exp_cme_flank.expected_flank_r(ens_no)
        expected_cme_flank_lon = exp_cme_flank.expected_flank_lon(ens_no)

        assert (init_obs_op.get_cme_flank_single_ens_member(mock_cme)['el'].values == expected_cme_flank_el).all()
        assert (init_obs_op.get_cme_flank_single_ens_member(mock_cme)['r'].values == expected_cme_flank_r).all()
        assert (init_obs_op.get_cme_flank_single_ens_member(mock_cme)['lon'].values == expected_cme_flank_lon).all()

        return None
    

    @pytest.mark.parametrize(
        "init_test_cme_flanks", [5], indirect=["init_test_cme_flanks"]
    )
    def test_get_cme_flanks(
            self, init_obs_op, init_test_cme_flanks, exp_cme_flank, mocker
    ) -> None:
        n_ensemble = init_obs_op.n_members

        # Initialise mockers for the S.SURF and S.ConeCME calls
        mock_surf = mocker.patch("surf.surf.SURF")
        mock_surf.return_value = S.SURF()

        mock_solve = mocker.patch("surf.surf.SURF.solve")
        mock_solve.return_value = 450

        mock_cme = mocker.patch("surf.surf.ConeCME")
        mock_cme_values = [i for i in range(n_ensemble)]
        def mock_cme_side_effect(*args, **kwargs):
            return mock_cme_values[np.mod(mock_cme.call_count - 1, len(mock_cme_values))]

        mock_cme.side_effect = mock_cme_side_effect

        mock_flank = mocker.patch("SIR_SURF.code.sir_observation_operator.Observer.compute_flank_profile")
        mock_flank_values = [
            init_test_cme_flanks[i] for i in range(n_ensemble)
        ]
        def mock_flank_side_effect(*args, **kwargs):
            return mock_flank_values[np.mod(mock_flank.call_count - 1, len(mock_flank_values))]
        mock_flank.side_effect = mock_flank_side_effect

        init_cme_flanks: list[pd.DataFrame] = init_obs_op.get_cme_flanks()
        expected_cme_flanks: list[pd.DataFrame] = exp_cme_flank.expected_flanks()

        assert (
            init_cme_flanks[i].keys() == expected_cme_flanks[i].keys() for i in range(n_ensemble)
        )

        for i in range(n_ensemble):
            init_cme_flank = init_cme_flanks[i]
            expected_cme_flank = expected_cme_flanks[i]

            for key in init_cme_flank.keys().values:
                icf = list(init_cme_flank.loc[:, key].values)
                ecf = list(expected_cme_flank.loc[:, key].values)

                assert (icf == pytest.approx(ecf))

        return None
        """

    @pytest.mark.parametrize(
        "init_test_cme_flanks, obs_op_tests", [
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7)
        ], indirect=["init_test_cme_flanks", "obs_op_tests"]
    )
    def test_make_obs_op(
            self, get_use_model, init_obs_op, obs_op_tests, init_test_cme_flanks, exp_cme_flank, mocker
    ) -> None:
        n_ensemble = init_obs_op.n_members

        # Update the init_obs_op_object with the observation datetimes from test test_no
        #init_obs_op.get_obs_times = exp_cme_flank.obs_op_tests(test_no)[0]
        #print(f"init_obs_op.get_obs_times: {init_obs_op.obs_time_in_datetime}")

        if get_use_model in ["surf", "compress_surf"]:
            # Initialise mockers for the S.SURF and S.ConeCME calls
            mock_surf = mocker.patch("surf.surf.SURF")
            mock_surf.return_value = S.SURF()

            mock_solve = mocker.patch("surf.surf.SURF.solve")
            mock_solve.return_value = 450

            mock_cme = mocker.patch("surf.surf.ConeCME")
            mock_cme_values = [i for i in range(n_ensemble)]
        elif get_use_model in ["huxt"]:
            # Initialise mockers for the S.SURF and S.ConeCME calls
            mock_surf = mocker.patch("huxt.huxt.HUXt")
            mock_surf.return_value = H.HUXt()

            mock_solve = mocker.patch("huxt.huxt.HUXt.solve")
            mock_solve.return_value = 450

            mock_cme = mocker.patch("huxt.huxt.ConeCME")
            mock_cme_values = [i for i in range(n_ensemble)]
        else:
            sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

        def mock_cme_side_effect(*args, **kwargs):
            return mock_cme_values[np.mod(mock_cme.call_count - 1, len(mock_cme_values))]
        mock_cme.side_effect = mock_cme_side_effect

        mock_flank = mocker.patch("sir_observation_operator.Observer.compute_flank_profile")
        mock_flank_values = [init_test_cme_flanks[i] for i in range(n_ensemble)]

        def mock_flank_side_effect(*args, **kwargs):
            return mock_flank_values[np.mod(mock_flank.call_count - 1, len(mock_flank_values))]
        mock_flank.side_effect = mock_flank_side_effect

        init_obs_operator: list[pd.DataFrame] = init_obs_op.make_obs_op()
        expected_hx: list[pd.DataFrame] = exp_cme_flank.expected_obs_op(obs_op_tests)
        print(f"np.shape(init_obs_operator): {np.shape(init_obs_operator)}")
        print(f"np.shape(expected_hx): {np.shape(expected_hx)}")

        print(f"init_obs_op={init_obs_operator}")
        print(f"expected_hx={expected_hx}")

        for m in range(n_ensemble):
            assert init_obs_operator[m] == pytest.approx(expected_hx[m])

        return None
