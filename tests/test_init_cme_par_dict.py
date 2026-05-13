import datetime
import sys

#sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code')
import SIR_HUXt.code.init_sir as init_sir

class TestInitCmeParDict:
    def test_initialise_cme_parameter_ensemble_dict(self) -> None:
        parameter_arrays = init_sir.initialise_cme_parameter_ensemble_dict(
            n_ensemble=17,
            huxt_init_time=datetime.datetime(2008, 1, 1, 0, 0, 0)
        )
        expected_par_dict_keys: list[str] = [
            'huxt_init_time', 't_init', 'v', 'width',
            'lon', 'lat', 'thick', 'v_transit', 'v_hit',
            'likelihood', 'weight', 'log_weight', 'n_members'
        ]

        assert (list(parameter_arrays.keys()) == expected_par_dict_keys)

        return None
