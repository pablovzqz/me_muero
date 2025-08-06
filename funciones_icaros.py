import pandas as pd
import numpy as np
from invisible_cities.core.fit_functions import fit, polynom, gauss
from scipy import stats

def f(df: pd.DataFrame,
      bins: np.ndarray,
      nsigma: float,
      ) -> pd.DataFrame:

    Zrms2 = df.Zrms**2
    meansy , bin_edges, _ = stats.binned_statistic(df.DT, Zrms2, bins = bins, statistic='mean')
    stds, _, _ = stats.binned_statistic(df.DT, Zrms2, bins = bins, statistic='std')
    values_up = meansy + nsigma*stds
    values_down = meansy - nsigma*stds
    bin_centers = (bin_edges[:-1]+bin_edges[1:])*0.5
    f_up = fit(polynom, bin_centers, values_up, seed = (0, 1))
    f_down = fit(polynom, bin_centers, values_down, seed = (0, 1))


    mask = (f_up.fn(df.DT) > Zrms2) & (f_down.fn(df.DT) < Zrms2)

    return df[mask]


def test_f():

    DT          = np.linspace(0, 1400, 1000)
    Zrms         = 8200 * np.exp(-DT/30000) + [1, -1]*(len(DT)//2)

    test_matrix = np.column_stack((DT, Zrms))

    df          = pd.DataFrame(test_matrix, columns = ['DT', 'Zrms'])

    var = f(df, 10, 100)

    assert f(df, 10, 100).shape == df.shape

    assert f(df, 10, 0).shape == (0, 2)


test_f()
