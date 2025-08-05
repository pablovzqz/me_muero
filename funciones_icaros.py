import pandas as pd
import numpy as np

def f(df: pd.DataFrame,
      bins: np.ndarray,
      nsigma: float,
      ) -> pd.DataFrame:
    pass

def test_f():

    test_matrix = np.array([[1200, 7000],
                           [200, 8200],
                           [300, 7200],
                           [450, 7800]])

    df          = pd.DataFrame(test_matrix, columns = ['DT', 'S2e'])

    assert f(df, 1, 10e10).shape == df.shape

    assert f(df, 1, 0).shape == (0, 2)
