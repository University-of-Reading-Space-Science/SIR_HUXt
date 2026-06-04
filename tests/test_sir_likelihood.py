import pytest
import numpy as np

from sir_likelihood import LikelihoodFunction

class TestSirLikelihood:
    @pytest.fixture
    def init_test(self, request) -> LikelihoodFunction:
        if hasattr(request, 'param'):
            print(request.param)
            test_no = request.param
        else:
            test_no = -1

        print(f"test_no={test_no}")
        if test_no == 0:
            obs = 1
            obs_cov = 1
            hx = 2
        elif test_no == 1:
            obs = 34
            obs_cov = 12
            hx = 20
        elif test_no == 2:
            obs = [10, 90]
            obs_cov = [[1, 0], [0, 1]]
            hx = [14, 84]
        elif test_no == 3:
            obs = np.array([10, 90])
            obs_cov = np.array([[10, -5], [-5, 10]])
            hx = [14, 84]
        elif test_no == 4:
            obs = np.array([10, 20, 20])
            obs_cov = np.array([[10, -5, 12], [-5, 10, 4], [12, 4, 10]])
            hx = np.array([9, 14, 18])
        elif test_no == 5:
            obs = np.array([10, 20, 20])
            obs_cov = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
            hx = np.array([9, 14, 18])
        else:
            obs = np.array([2])
            obs_cov = np.array([1])
            hx = np.array([0])

        # Return likelihood function class
        return LikelihoodFunction(obs, obs_cov, hx)

    @pytest.fixture
    def expected_test_log_lik_gauss(self, request) -> float:
        if hasattr(request, 'param'):
            print(request.param)
            test_no: int = request.param
        else:
            test_no: int = -1

        if test_no == 0:
            exp_log_likelihood: float = -1.0
        elif test_no == 1:
            exp_log_likelihood: float = -49.0 / 3.0
        elif test_no == 2:
            exp_log_likelihood: float = -52.0
        elif test_no == 3:
            exp_log_likelihood: float = -56.0 / 15.0
        elif test_no == 4:
            exp_log_likelihood: float = -1492.0 / 665.0
        elif test_no == 5:
            exp_log_likelihood: float = -41.0 / 10.0
        else:
            exp_log_likelihood: float = -4.0

        return exp_log_likelihood

    @pytest.fixture
    def expected_test_lik_gauss(self, request) -> float:
        if hasattr(request, 'param'):
            print(request.param)
            test_no = request.param
        else:
            test_no = -1

        if test_no == 0:
            exp_likelihood: float = np.exp(-1.0)
        elif test_no == 1:
            exp_likelihood: float = np.exp(-49.0 / 3.0)
        elif test_no == 2:
            exp_likelihood: float = np.exp(-52.0)
        elif test_no == 3:
            exp_likelihood: float = np.exp(-56.0 / 15.0)
        elif test_no == 4:
            exp_likelihood: float = np.exp(-1492.0 / 665.0)
        elif test_no == 5:
            exp_likelihood: float = np.exp(-41.0 / 10.0)
        else:
            exp_likelihood: float = np.exp(-4.0)

        return exp_likelihood


    @pytest.mark.parametrize("init_test, expected_test_log_lik_gauss", [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5)
    ], indirect=True)
    def test_sir_log_likelihood_gaussian(self, init_test, expected_test_log_lik_gauss):
        assert init_test.log_likelihood_gaussian() == pytest.approx(expected_test_log_lik_gauss)

    @pytest.mark.parametrize("init_test, expected_test_lik_gauss", [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5)
    ], indirect=True)
    def test_sir_likelihood_gaussian(self, init_test, expected_test_lik_gauss):
        assert init_test.likelihood_gaussian() == pytest.approx(expected_test_lik_gauss)