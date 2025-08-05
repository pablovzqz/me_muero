import pandas as pd
import numpy as np

def f(df: pd.DataFrame,
      bins: np.ndarray,
      nsigma: float,
      ) -> pd.DataFrame:
    pass

def test_f():

    DT          = np.linspace(0, 1400, 1000)
    S2e         = 8200 * np.exp(-DT/30000)

    test_matrix = np.column_stack((DT, S2e))

    df          = pd.DataFrame(test_matrix, columns = ['DT', 'S2e'])

    assert f(df, 1, 10e10).shape == df.shape

    assert f(df, 1, 0).shape == (0, 2)
