import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from typing import Callable

def diffusion_band(kdst         :   pd.DataFrame,
                   lower_limit  :   Callable,
                   upper_limit  :   Callable)->pd.DataFrame:

    mask        = ((kdst.Zrms**2 < upper_limit(kdst.DT)) &
                   (kdst.Zrms**2 > lower_limit(kdst.DT)))

    kdst_inband = kdst[mask]

    return kdst_inband



def test_diffusion_band():
    Zrms = np.arange(0,100, 0.01)**0.5
    DT = np.arange(0, 2000, 0.2)
    d = {'Zrms': Zrms, 'DT': DT}
    df_test = pd.DataFrame(data = d)
    lower_limit = lambda x: 0.05*x - 3
    upper_limit = lambda x: 0.05*x + 3
    fun_test = diffusion_band(df_test, lower_limit, upper_limit)

    assert fun_test.shape == df_test.shape

    assert np.all(fun_test.values == df_test.values)




test_diffusion_band()
