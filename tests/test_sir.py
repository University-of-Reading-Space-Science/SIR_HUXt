from SIR_HUXt.tests.test_cme_par_indices import TestCmeParIndices
from SIR_HUXt.tests.test_zscores import TestZscores

from SIR_HUXt.tests.test_sir_resampling_linear_weights import TestResamplingLinear
from SIR_HUXt.tests.test_sir_resampling_log_weights import TestResamplingLog

##########################################
from SIR_HUXt.tests.test_init_cme_par_dict import TestInitCmeParDict

from SIR_HUXt.tests.test_state_vector_trans.test_to_state_vector import TestToStateVector

from SIR_HUXt.tests.test_state_vector_trans.test_from_state_vector import TestFromStateVector

from SIR_HUXt.tests.test_sir_likelihood import TestSirLikelihood

from tests.test_sir_observation_operator import TestObservationOperator



def test_sir() -> None:
    TestCmeParIndices()
    TestZscores()
    TestResamplingLinear()
    TestResamplingLog()

    TestInitCmeParDict()

    TestToStateVector()
    TestFromStateVector()

    TestSirLikelihood()
    TestObservationOperator()

    return None

def main():
    test_sir()

    return None

if __name__ == "__main__":
    main()