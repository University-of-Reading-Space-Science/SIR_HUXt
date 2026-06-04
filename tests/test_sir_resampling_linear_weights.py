import pytest

from sir_resampling_linear_weights import ResampleParsLinearWeights


class TestResamplingLinear:
    @pytest.fixture
    def init_resamp_par(self, request) -> ResampleParsLinearWeights:
        if hasattr(request, 'param'):
            print(request.param)
            weights = request.param
        else:
            weights = None

        print(f"weights={weights}")

        return ResampleParsLinearWeights(weights, rng=42)


    @pytest.mark.parametrize("init_resamp_par, expected", [
        ([0.1, 0.3, 0.5, 0.6, 0.5], [0.05, 0.15, 0.25, 0.3, 0.25]),
        ([0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]),
        ([0.1, 0.4, 0.05, 0.15, 0.3], [0.1, 0.4, 0.05, 0.15, 0.3]),
        ([4, 2, 22, 35, 17], [0.05, 0.025, 0.275, 0.4375, 0.2125]),
        ([14, 0, 12, 2, 22], [0.28, 0, 0.24, 0.04, 0.44]),
        ([1, 5, 1.5, 0.5, 2], [0.1, 0.5, 0.15, 0.05, 0.2]),
        ([4, 6, 7, 2, 1, 5, 5, 0, 15, 5], [0.08, 0.12, 0.14, 0.04, 0.02, 0.1, 0.1, 0, 0.3, 0.1])
    ], indirect=["init_resamp_par"])
    def test_norm_weights(self, init_resamp_par, expected):
        assert (init_resamp_par.norm_weights() == expected).all()

    #def test_norm_weights_invalid(self):

    @pytest.mark.parametrize("init_resamp_par, expected", [
        ([0.1, 0.3, 0.5, 0.6, 0.5], [0.05, 0.2, 0.45, 0.75, 1]),
        ([0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.4, 0.6, 0.8, 1]),
        ([0.1, 0.4, 0.05, 0.15, 0.3], [0.1, 0.5, 0.55, 0.7, 1]),
        ([4, 2, 22, 35, 17], [0.05, 0.075, 0.35, 0.7875, 1]),
        ([14, 0, 12, 2, 22], [0.28, 0.28, 0.52, 0.56, 1]),
        ([1, 5, 1.5, 0.5, 2], [0.1, 0.6, 0.75, 0.8, 1]),
        ([4, 6, 7, 2, 1, 5, 5, 0, 15, 5], [0.08, 0.2, 0.34, 0.38, 0.4, 0.5, 0.6, 0.6, 0.9, 1])
    ], indirect=["init_resamp_par"])
    def test_make_cdf(self, init_resamp_par, expected):
        print(f"init_resamp_par={init_resamp_par.make_cdf()}, expected={expected}")
        assert (init_resamp_par.make_cdf() == pytest.approx(expected))

    @pytest.mark.parametrize("init_resamp_par, init_pos, expected", [
        ([0.1, 0.3, 0.5, 0.6, 0.5], 0.025, [0, 2, 2, 3, 4]),
        ([0.1, 0.3, 0.5, 0.6, 0.5], 0.1, [1, 2, 3, 3, 4]),
        ([0.1, 0.3, 0.5, 0.6, 0.5], 0.17, [1, 2, 3, 4, 4]),
        ([0.1, 0.3, 0.5, 0.6, 0.5], 0.2, [1, 2, 3, 4, 4]),
        ([0.2, 0.2, 0.2, 0.2, 0.2], 0.0, [0, 0, 1, 2, 3]),
        ([0.2, 0.2, 0.2, 0.2, 0.2], 0.03, [0, 1, 2, 3, 4]),
        ([0.2, 0.2, 0.2, 0.2, 0.2], 0.15, [0, 1, 2, 3, 4]),
        ([0.2, 0.2, 0.2, 0.2, 0.2], 0.2, [0, 1, 2, 3, 4]),
        ([0.1, 0.4, 0.05, 0.15, 0.3], 0.006, [0, 1, 1, 3, 4]),
        ([0.1, 0.4, 0.05, 0.15, 0.3], 0.1, [0, 1, 1, 3, 4]),
        ([0.1, 0.4, 0.05, 0.15, 0.3], 0.13, [1, 1, 2, 4, 4]),
        ([0.1, 0.4, 0.05, 0.15, 0.3], 0.16, [1, 1, 3, 4, 4]),
        ([4, 2, 22, 35, 17], 0.025, [0, 2, 3, 3, 4]),
        ([4, 2, 22, 35, 17], 0.05, [0, 2, 3, 3, 4]),
        ([4, 2, 22, 35, 17], 0.06875, [1, 2, 3, 3, 4]),
        ([4, 2, 22, 35, 17], 0.15, [2, 2, 3, 3, 4]),
        ([14, 0, 12, 2, 22], 0, [0, 0, 2, 4, 4]),
        ([14, 0, 12, 2, 22], 0.05, [0, 0, 2, 4, 4]),
        ([14, 0, 12, 2, 22], 0.1, [0, 2, 2, 4, 4]),
        ([14, 0, 12, 2, 22], 0.15, [0, 2, 3, 4, 4]),
        ([1, 5, 1.5, 0.5, 2], 0.05, [0, 1, 1, 2, 4]),
        ([1, 5, 1.5, 0.5, 2], 0.1, [0, 1, 1, 2, 4]),
        ([1, 5, 1.5, 0.5, 2], 0.15, [1, 1, 1, 2, 4]),
        ([1, 5, 1.5, 0.5, 2], 0.2, [1, 1, 1, 3, 4]),
        ([4, 6, 7, 2, 1, 5, 5, 0, 15, 5], 0.02, [0, 1, 2, 2, 5, 6, 8, 8, 8, 9]),
        ([4, 6, 7, 2, 1, 5, 5, 0, 15, 5], 0.05, [0, 1, 2, 3, 5, 6, 8, 8, 8, 9]),
        ([4, 6, 7, 2, 1, 5, 5, 0, 15, 5], 0.07, [0, 1, 2, 3, 5, 6, 8, 8, 8, 9]),
        ([4, 6, 7, 2, 1, 5, 5, 0, 15, 5], 0.09, [1, 1, 2, 4, 5, 6, 8, 8, 8, 9])
    ], indirect=["init_resamp_par"])
    def test_systematic_resampling(self, init_resamp_par, init_pos, expected):
        print(f"init_resamp_par={init_resamp_par.systematic_resampling(init_position=init_pos)}, expected={expected}")
        assert (init_resamp_par.systematic_resampling(init_position=init_pos) == expected)