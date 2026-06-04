import pytest
import numpy as np
from sir_resampling_log_weights import ResampleParsLogWeights, jacobian_log


def list_log(list):
    assert (x >= 0 for x in list)

    list_log = [-np.inf if x == 0 else np.log(x) for x in list]
    return list_log


class TestResamplingLog:
    @pytest.fixture
    def init_resamp_par_log(self, request) -> ResampleParsLogWeights:
        if hasattr(request, 'param'):
            print(request.param)
            log_weights = request.param
        else:
            log_weights = None

        print(f"weights={log_weights}")

        return ResampleParsLogWeights(log_weights, rng=42)



    @pytest.fixture
    def expected_jacob_log(self, request):
        if hasattr(request, 'param'):
            exponents: list[float] = request.param
        else:
            exponents = None

        exp_sum = np.sum(np.exp(exponents))

        jacob_log = np.log(exp_sum)
        print(f"jacob_log={jacob_log}")

        return jacob_log


    @pytest.mark.parametrize("input_par, expected_jacob_log", [
        (list_log([0.1, 0.3, 0.5, 0.6, 0.5]), list_log([0.1, 0.3, 0.5, 0.6, 0.5])),
        (list_log([0.2, 0.2, 0.2, 0.2, 0.2]), list_log([0.2, 0.2, 0.2, 0.2, 0.2])),
        (list_log([0.1, 0.4, 0.05, 0.15, 0.3]), list_log([0.1, 0.4, 0.05, 0.15, 0.3])),
        ([4, 2, 22, 35, 17], [4, 2, 22, 35, 17]),
        (list_log([14, 0, 12, 2, 22]), list_log([14, 0, 12, 2, 22])),
        ([1, 5, 1.5, 0.5, 2], [1, 5, 1.5, 0.5, 2]),
        ([4, 6, 7, 2, 1, 5, 5, 0, 15, 5], [4, 6, 7, 2, 1, 5, 5, 0, 15, 5])
    ], indirect=["expected_jacob_log"])
    def test_jacob_log(self, input_par, expected_jacob_log):
        assert jacobian_log(input_par) == pytest.approx(expected_jacob_log)


    @pytest.mark.parametrize("init_resamp_par_log, expected", [
        (list_log([0.1, 0.3, 0.5, 0.6, 0.5]), list_log([0.05, 0.15, 0.25, 0.3, 0.25])),
        (list_log([0.2, 0.2, 0.2, 0.2, 0.2]), list_log([0.2, 0.2, 0.2, 0.2, 0.2])),
        (list_log([0.1, 0.4, 0.05, 0.15, 0.3]), list_log([0.1, 0.4, 0.05, 0.15, 0.3])),
        (list_log([4, 2, 22, 35, 17]), list_log([0.05, 0.025, 0.275, 0.4375, 0.2125])),
        (list_log([14, 0, 12, 2, 22]), list_log([0.28, 0, 0.24, 0.04, 0.44])),
        (list_log([1, 5, 1.5, 0.5, 2]), list_log([0.1, 0.5, 0.15, 0.05, 0.2])),
        (list_log([4, 6, 7, 2, 1, 5, 5, 0, 15, 5]), list_log([0.08, 0.12, 0.14, 0.04, 0.02, 0.1, 0.1, 0, 0.3, 0.1]))
    ], indirect=["init_resamp_par_log"])
    def test_norm_log_weights(self, init_resamp_par_log, expected):
        assert init_resamp_par_log.norm_log_weights() == pytest.approx(expected)


    @pytest.mark.parametrize("init_resamp_par_log, expected", [
        (list_log([0.1, 0.3, 0.5, 0.6, 0.5]), list_log([0.05, 0.2, 0.45, 0.75, 1])),
        (list_log([0.2, 0.2, 0.2, 0.2, 0.2]), list_log([0.2, 0.4, 0.6, 0.8, 1])),
        (list_log([0.1, 0.4, 0.05, 0.15, 0.3]), list_log([0.1, 0.5, 0.55, 0.7, 1])),
        (list_log([4, 2, 22, 35, 17]), list_log([0.05, 0.075, 0.35, 0.7875, 1])),
        (list_log([14, 0, 12, 2, 22]), list_log([0.28, 0.28, 0.52, 0.56, 1])),
        (list_log([1, 5, 1.5, 0.5, 2]), list_log([0.1, 0.6, 0.75, 0.8, 1])),
        (list_log([4, 6, 7, 2, 1, 5, 5, 0, 15, 5]), list_log([0.08, 0.2, 0.34, 0.38, 0.4, 0.5, 0.6, 0.6, 0.9, 1]))
    ], indirect=["init_resamp_par_log"])
    def test_make_cdf(self, init_resamp_par_log, expected):
        print(f"init_resamp_par={init_resamp_par_log.make_cdf_log_weights()}, expected={expected}")
        assert (init_resamp_par_log.make_cdf_log_weights() == pytest.approx(expected))


    @pytest.mark.parametrize("init_resamp_par_log, init_pos, expected", [
        (list_log([0.1, 0.3, 0.5, 0.6, 0.5]), 0.025, [0, 2, 2, 3, 4]),
        (list_log([0.1, 0.3, 0.5, 0.6, 0.5]), 0.1, [1, 2, 3, 3, 4]),
        (list_log([0.1, 0.3, 0.5, 0.6, 0.5]), 0.17, [1, 2, 3, 4, 4]),
        (list_log([0.1, 0.3, 0.5, 0.6, 0.5]), 0.2, [1, 2, 3, 4, 4]),
        (list_log([0.2, 0.2, 0.2, 0.2, 0.2]), 0.0, [0, 0, 1, 2, 3]),
        (list_log([0.2, 0.2, 0.2, 0.2, 0.2]), 0.03, [0, 1, 2, 3, 4]),
        (list_log([0.2, 0.2, 0.2, 0.2, 0.2]), 0.15, [0, 1, 2, 3, 4]),
        (list_log([0.2, 0.2, 0.2, 0.2, 0.2]), 0.2, [0, 1, 2, 3, 4]),
        (list_log([0.1, 0.4, 0.05, 0.15, 0.3]), 0.006, [0, 1, 1, 3, 4]),
        (list_log([0.1, 0.4, 0.05, 0.15, 0.3]), 0.1, [0, 1, 1, 3, 4]),
        (list_log([0.1, 0.4, 0.05, 0.15, 0.3]), 0.13, [1, 1, 2, 4, 4]),
        (list_log([0.1, 0.4, 0.05, 0.15, 0.3]), 0.16, [1, 1, 3, 4, 4]),
        (list_log([4, 2, 22, 35, 17]), 0.025, [0, 2, 3, 3, 4]),
        (list_log([4, 2, 22, 35, 17]), 0.05, [0, 2, 3, 3, 4]),
        (list_log([4, 2, 22, 35, 17]), 0.06875, [1, 2, 3, 3, 4]),
        (list_log([4, 2, 22, 35, 17]), 0.15, [2, 2, 3, 3, 4]),
        (list_log([14, 0, 12, 2, 22]), 0, [0, 0, 2, 4, 4]),
        (list_log([14, 0, 12, 2, 22]), 0.05, [0, 0, 2, 4, 4]),
        (list_log([14, 0, 12, 2, 22]), 0.1, [0, 2, 2, 4, 4]),
        (list_log([14, 0, 12, 2, 22]), 0.15, [0, 2, 3, 4, 4]),
        (list_log([1, 5, 1.5, 0.5, 2]), 0.05, [0, 1, 1, 2, 4]),
        (list_log([1, 5, 1.5, 0.5, 2]), 0.1, [0, 1, 1, 2, 4]),
        (list_log([1, 5, 1.5, 0.5, 2]), 0.15, [1, 1, 1, 2, 4]),
        (list_log([1, 5, 1.5, 0.5, 2]), 0.2, [1, 1, 1, 3, 4]),
        (list_log([4, 6, 7, 2, 1, 5, 5, 0, 15, 5]), 0.02, [0, 1, 2, 2, 5, 6, 8, 8, 8, 9]),
        (list_log([4, 6, 7, 2, 1, 5, 5, 0, 15, 5]), 0.05, [0, 1, 2, 3, 5, 6, 8, 8, 8, 9]),
        (list_log([4, 6, 7, 2, 1, 5, 5, 0, 15, 5]), 0.07, [0, 1, 2, 3, 5, 6, 8, 8, 8, 9]),
        (list_log([4, 6, 7, 2, 1, 5, 5, 0, 15, 5]), 0.09, [1, 1, 2, 4, 5, 6, 8, 8, 8, 9])
    ], indirect=["init_resamp_par_log"])
    def test_systematic_resampling_log_weights(self, init_resamp_par_log, init_pos, expected):
        print(
            f"init_resamp_par={init_resamp_par_log.systematic_resampling_log_weights(init_position=init_pos)}, "
            f"expected={expected}"
        )
        assert (init_resamp_par_log.systematic_resampling_log_weights(init_position=init_pos) == expected)