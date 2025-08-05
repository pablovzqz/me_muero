import pandas as pd
import numpy as np
from invisible_cities.core.fit_functions import fit, expo

def f(df: pd.DataFrame,
      bins: np.ndarray,
      nsigma: float,
      ) -> pd.DataFrame:
    meansy , bin_edges, _ = stats.binned_statistic(df.DT, (df.Zrms)**2, bins = bins, statistic='mean')
    stds, _, _ = stats.binned_statistic(df.DT, (df.Zrms)**2, bins = bins, statistic='std')
    values_up = meansy + nsigma*stds
    values_down = meansy - nsigma*stds
    bin_centers = (bin_edges[:-1]+bin_edges[1:])*0.5
    f_up = fit(expo, bin_centers, values_up, seed = (meansy.mean(), -1))
    f_down = fit(expo, bin_centers, values_down, seed = (meansy.mean(), -1))



def test_f():

    DT          = np.linspace(0, 1400, 1000)
    S2e         = 8200 * np.exp(-DT/30000)

    test_matrix = np.column_stack((DT, S2e))

    df          = pd.DataFrame(test_matrix, columns = ['DT', 'S2e'])

    assert f(df, 1, 10e10).shape == df.shape

    assert f(df, 1, 0).shape == (0, 2)
