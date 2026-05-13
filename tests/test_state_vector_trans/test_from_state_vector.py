import pytest
import astropy.units as u
import datetime
import sys

from astropy.units import Quantity

#sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code')
from SIR_HUXt.code.cme_par_ens import CmeParEns
from SIR_HUXt.code.from_state_vector import FromStateVector

from SIR_HUXt.tests.test_state_vector_trans.expected_cme_par_arr import ExpCmeParArray, ExpCmeArrayFromStateVector
from SIR_HUXt.tests.test_state_vector_trans.expected_cme_par_dict import ExpCmeParDict

##################################################################
@pytest.mark.usefixtures("init_from_state_vector")
class TestFromStateVector:
    @pytest.fixture
    def exp_val_dict(self) -> ExpCmeParDict:
        return ExpCmeParDict()

    @pytest.fixture
    def exp_val_arr(self) -> ExpCmeParArray:
        return ExpCmeParArray()

    @pytest.fixture
    def exp_val_arr_from_state(self, request) -> ExpCmeArrayFromStateVector:

        if hasattr(request, "param"):
            pars_in_state: list[str] = request.param
        else:
            pars_in_state = None

        return ExpCmeArrayFromStateVector(pars_in_state_vector=pars_in_state)

    def test_convert_t_init_array_to_dict(self, init_from_state_vector, exp_val_dict) -> None:
        t_init_arr = init_from_state_vector.convert_t_init_array_to_dict()

        expected_value: list[datetime.datetime] = exp_val_dict.expected_t_init()

        assert t_init_arr == expected_value

        return None


    def test_convert_v_array_to_dict(self, init_from_state_vector, exp_val_dict) -> None:
        v_dict = init_from_state_vector.convert_v_array_to_dict()

        expected_value: list[Quantity[u.km / u.s]] = exp_val_dict.expected_v()

        assert (
            (v_dict.value == pytest.approx(expected_value.value))
            and (v_dict.unit == expected_value.unit)
        )

        return None


    def test_convert_width_array_to_dict(self, init_from_state_vector, exp_val_dict) -> None:
        width_dict = init_from_state_vector.convert_width_array_to_dict()

        expected_value: list[Quantity[u.deg]] = exp_val_dict.expected_width()

        assert (
            (width_dict.value == pytest.approx(expected_value.value))
            and (width_dict.unit == expected_value.unit)
        )

        return None


    def test_convert_lon_array_to_dict(self, init_from_state_vector, exp_val_dict) -> None:
        lon_dict = init_from_state_vector.convert_lon_array_to_dict()

        expected_value: list[Quantity[u.deg]] = exp_val_dict.expected_lon()

        assert (
            (lon_dict.value == pytest.approx(expected_value.value))
            and (lon_dict.unit == expected_value.unit)
        )

        return None


    def test_convert_lat_array_to_dict(self, init_from_state_vector, exp_val_dict) -> None:
        lat_dict = init_from_state_vector.convert_lat_array_to_dict()

        expected_value: list[Quantity[u.deg]] = exp_val_dict.expected_lat()

        assert (
            (lat_dict.value == pytest.approx(expected_value.value))
            and (lat_dict.unit == expected_value.unit)
        )

        return None


    def test_convert_thick_array_to_dict(self, init_from_state_vector, exp_val_dict) -> None:
        thick_dict = init_from_state_vector.convert_thick_array_to_dict()

        expected_value: list[Quantity[u.solRad]] = exp_val_dict.expected_thick()

        assert (
            (thick_dict.value == pytest.approx(expected_value.value))
            and (thick_dict.unit == expected_value.unit)
        )

        return None


    def test_cme_par_array_to_cme_par_dict(self, init_from_state_vector, exp_val_dict) -> None:
        cme_par_dict: CmeParEns = init_from_state_vector.cme_par_array_to_cme_par_dict()

        expected_dict: CmeParEns = exp_val_dict.expected_cme_par_dict()
        print(f"cme_par_dict: {cme_par_dict}")
        print(f"expected_dict: {expected_dict}")

        assert (cme_par_dict.keys() == expected_dict.keys())

        for key in cme_par_dict.keys():
            if key in ["huxt_init_time", "n_members"]:
                assert (cme_par_dict[key] == expected_dict[key])
            elif key == "t_init":
                assert (cme_par_dict[key] == expected_dict[key])
            elif key in ["v_transit", "v_hit", "likelihood", "log_weight", "weight"]:
                assert (cme_par_dict[key] == pytest.approx(expected_dict[key]))
            else:
                assert (
                        (cme_par_dict[key].value == pytest.approx(expected_dict[key].value))
                        and (cme_par_dict[key].unit == expected_dict[key].unit)
                )

        return None


    @pytest.mark.parametrize(
        "init_from_state_vector, exp_val_arr_from_state", [
        (["t_init"], ["t_init"]),
        (["v"], ["v"]),
        (["width"], ["width"]),
        (["lon"], ["lon"]),
        (["lat"], ["lat"]),
        (["thick"], ["thick"]),
        (["t_init", "width"], ["t_init", "width"]),
        (["width", "t_init", "v"], ["width", "t_init", "v"]),
        (["lat", "lon", "thick", "v"], ["lat", "lon", "thick", "v"]),
        (["thick", "t_init", "v"], ["thick", "t_init", "v"]),
        (["v", "width", "lon"], ["v", "width", "lon"])
    ], indirect=True)
    def test_cme_par_state_vector_to_cme_array(self, init_from_state_vector, exp_val_arr_from_state) -> None:
        cme_par: npt.NDArray[float] = init_from_state_vector.state_vector_to_cme_par_array()

        expected_value: npt.NDArray[float] = exp_val_arr_from_state.expected_state_to_array()

        assert (cme_par == pytest.approx(expected_value))
