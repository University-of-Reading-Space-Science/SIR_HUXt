import pytest
import numpy as np
import sys

sys.path.append('C:\\Users\\ss905122\\PycharmProjects\\SIR_HUXt\\code')
import sir

class TestZscores:
    def test_zscore(self) -> None:

        x1 = np.array([1, 2, 3, 4, 4, 5, 6, 7])
        x2 = np.array([3, 3, 3, 4, 4, 5, 5, 7, 11, 15])

        expected_z1 = 0.5 * np.array([-3, -2, -1, 0, 0, 1, 2, 3])
        expected_x1_avg = 4
        expected_x1_std = 2

        expected_z2 = 0.25 * np.array([-3, -3, -3, -2, -2, -1, -1, 1, 5, 9])
        expected_x2_avg = 6
        expected_x2_std = 4

        z1, x1_avg, x1_std = sir.zscore(x1)
        z2, x2_avg, x2_std = sir.zscore(x2)

        assert (z1 == pytest.approx(expected_z1))
        assert (x1_avg == pytest.approx(expected_x1_avg))
        assert (x1_std == pytest.approx(expected_x1_std))

        assert (z2 == pytest.approx(expected_z2))
        assert (x2_avg == pytest.approx(expected_x2_avg))
        assert (x2_std == pytest.approx(expected_x2_std))

        return None


    def test__inv_zscore(self) -> None:

        z1 = 0.5 * np.array([-3, -2, -1, 0, 0, 1, 2, 3])
        x1_avg = 4
        x1_std = 2

        z2 = 0.25 * np.array([-3, -3, -3, -2, -2, -1, -1, 1, 5, 9])
        x2_avg = 6
        x2_std = 4

        expected_x1 = np.array([1, 2, 3, 4, 4, 5, 6, 7])
        expected_x2 = np.array([3, 3, 3, 4, 4, 5, 5, 7, 11, 15])

        x1 = sir.inv_zscore(z1, x1_avg, x1_std)
        x2 = sir.inv_zscore(z2, x2_avg, x2_std)

        assert (x1 == pytest.approx(expected_x1))
        assert (x2 == pytest.approx(expected_x2))

        return None