import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

def diffusion_band(kdst         :   pd.DataFrame,
                   lower_limit  :   function,
                   upper_limit  :   function)->pd.DataFrame:


    pass



def test_diffusion_band():
    Zrms = np.arange(0,10, 0.001)
    DT = np.arange(0, 2000, 0.001)
    d = {'Zrms': Zrms, 'DT': DT}
    df_test = pd.DataFrame(data = d)
    lower_limit = lambda x: 0.05*x - 3
    upper_limit = lambda x: 0.05*x + 3
    fun_test = diffusion_band(df_test, lower_limit, upper_limit)

    assert fun_test.shape == df_test.shape

    assert np.all(fun_test.values == df_test.values)
