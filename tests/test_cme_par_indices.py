import pytest
import sys

sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code')
from cme_par_ens import CmeParEns
from to_state_vector import ToStateVector
from from_state_vector import FromStateVector

import cme_par_dict_structure as cpds

class TestCmeParIndices:
    @pytest.mark.parametrize("key, expected", [
        ("t_init", 0),
        ("v", 1),
        ("width", 2),
        ("lon", 3),
        ("lat", 4),
        ("thick", 5)
    ])
    def test_cme_par_get_indices_exp_outputs(self, key, expected) -> None:
        assert cpds.cme_par_get_indices(key) == expected

        return None


    @pytest.mark.parametrize("ind, expected", [
        (0, "t_init"),
        (1, "v"),
        (2, "width"),
        (3, "lon"),
        (4, "lat"),
        (5, "thick")
    ])
    def test_cme_par_get_keys_exp_outputs(self, ind, expected) -> None:
        assert cpds.cme_par_get_keys(ind) == expected

        return None


    def test_cme_par_get_indices_err_handling(self) -> None:
        with pytest.raises(AssertionError):
            cpds.cme_par_get_indices("t_int")
            cpds.cme_par_get_indices("v_transit")
            cpds.cme_par_get_indices(5)

        return None


    def test_cme_par_get_keys_err_handling(self) -> None:
        with pytest.raises(AssertionError):
            cpds.cme_par_get_keys("t_int")
            cpds.cme_par_get_keys(6)
            cpds.cme_par_get_keys(-3)

        return None




