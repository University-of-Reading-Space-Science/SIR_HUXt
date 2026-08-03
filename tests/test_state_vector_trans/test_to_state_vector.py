import pytest
import sys
import numpy.typing as npt

from SIR_HUXt.tests.test_state_vector_trans.expected_cme_par_arr import ExpCmeParArray, ExpStateVector

#sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code')
from SIR_HUXt.src.cme_par_ens import CmeParEns
from SIR_HUXt.src.to_state_vector import ToStateVector

@pytest.mark.usefixtures("init_to_state_vector")
class TestToStateVector:
    @pytest.fixture
    def exp_val_arr(self) -> ExpCmeParArray:
        return ExpCmeParArray()


    @pytest.fixture
    def exp_state_vector(self, request) -> ExpStateVector:
        if hasattr(request, "param"):
            pars_in_state: list[str] = request.param
        else:
            pars_in_state = None

        return ExpStateVector(pars_in_state_vector=pars_in_state)


    def test_convert_t_init_dict_to_array(self, init_to_state_vector, exp_val_arr) -> None:
        t_init_arr = init_to_state_vector.convert_t_init_dict_to_array()

        expected_value: npt.NDArray[float] = exp_val_arr.expected_t_init()

        assert (t_init_arr == expected_value).all()

        return None


    def test_convert_v_dict_to_array(self, init_to_state_vector, exp_val_arr) -> None:
        v_arr = init_to_state_vector.convert_v_dict_to_array()

        expected_value: npt.NDArray[float] = exp_val_arr.expected_v()

        assert (v_arr == expected_value).all()

        return None


    def test_convert_width_dict_to_array(self, init_to_state_vector, exp_val_arr) -> None:
        width_arr = init_to_state_vector.convert_width_dict_to_array()

        expected_value: npt.NDArray[float] = exp_val_arr.expected_width()

        assert (width_arr == pytest.approx(expected_value))

        return None


    def test_convert_lon_dict_to_array(self, init_to_state_vector, exp_val_arr) -> None:
        lon_arr = init_to_state_vector.convert_lon_dict_to_array()

        expected_value: npt.NDArray[float] = exp_val_arr.expected_lon()

        assert (lon_arr == pytest.approx(expected_value))

        return None


    def test_convert_lat_dict_to_array(self, init_to_state_vector, exp_val_arr) -> None:
        lat_arr = init_to_state_vector.convert_lat_dict_to_array()

        expected_value: npt.NDArray[float] = exp_val_arr.expected_lat()

        assert (lat_arr == pytest.approx(expected_value))

        return None


    def test_convert_thick_dict_to_array(self, init_to_state_vector, exp_val_arr) -> None:
        thick_arr = init_to_state_vector.convert_thick_dict_to_array()

        expected_value: npt.NDArray[float] = exp_val_arr.expected_thick()

        assert (thick_arr == pytest.approx(expected_value))

        return None


    def test_cme_par_dict_to_cme_par_array(self, init_to_state_vector, exp_val_arr) -> None:
        cme_par_arr = init_to_state_vector.cme_par_dict_to_cme_par_array()

        expected_value: npt.NDArray[float] = exp_val_arr.expected_cme_par_array()

        assert (cme_par_arr == pytest.approx(expected_value))

        return None

    @pytest.mark.parametrize("init_to_state_vector, exp_state_vector", [
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
    def test_cme_par_dict_to_state_vector(self, init_to_state_vector, exp_state_vector) -> None:
        state: npt.NDArray[float] = init_to_state_vector.cme_par_dict_to_state_vector()

        expected_value: npt.NDArray[float] = exp_state_vector.expected_state_vector()

        assert (state == pytest.approx(expected_value))

