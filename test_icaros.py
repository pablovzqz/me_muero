import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from invisible_cities.core.fit_functions import fit, polynom, gauss
from scipy.stats import binned_statistic
from typing import Callable

def diffusion_band(kdst         :   pd.DataFrame,
                   lower_limit  :   Callable,
                   upper_limit  :   Callable)->pd.DataFrame:

    mask        = ((kdst.Zrms**2 < upper_limit(kdst.DT)) &
                   (kdst.Zrms**2 > lower_limit(kdst.DT)))

    kdst_inband = kdst[mask]

    return kdst_inband



def diffusion_band2(kdst    : pd.DataFrame,
                     bins   : np.ndarray,
                     sigmas : float) -> pd.DataFrame:

    means , bin_edges, _ = binned_statistic(kdst.DT, kdst.Zrms**2, bins = bins, statistic = 'mean')
    std   , _        , _ = binned_statistic(kdst.DT, kdst.Zrms**2, bins = bins, statistic = 'std')

    top    = means + sigmas * std
    bottom = means - sigmas * std

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    top_line  = fit(polynom, bin_centers, top    , seed = (0, 1))
    bot_line  = fit(polynom, bin_centers, bottom , seed = (0, 1))


    mask = (top_line.fn(kdst.DT) > kdst.Zrms**2) & (bot_line.fn(kdst.DT) < kdst.Zrms**2)

    return kdst[mask]


def dif_band(kdst   : pd.DataFrame,
             method : str = 'functions',
             **kwargs) -> pd.DataFrame:

    if method == 'statistical':
        kdst = diffusion_band (kdst, **kwargs)

    else:
        kdst = diffusion_band2(kdst, **kwargs)

    return kdst


def test_dif_band():

    Zrms    = np.arange(0, 100,  0.01)**0.5
    DT      = np.arange(0, 2000, 0.2)

    d        = {'Zrms': Zrms, 'DT': DT}
    df_test  = pd.DataFrame(data = d)

    kwargs_f = {'lower_limit': lambda x: 0.05*x - 3,
                'upper_limit': lambda x: 0.05*x + 3}

    kwargs_s = {'bins': 50, 'sigmas': 3}

    methods = ('functions', 'statistical')
    kwarg  = (kwargs_f, kwargs_s)

    for method, args in zip(methods, kwarg):
        assert not dif_band(df_test, method, **args).empty




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


# test_diffusion_band()

def unique_S1_S2(kdst   : pd.DataFrame) -> pd.DataFrame:

    mask = (kdst.nS1 == 1) & (kdst.nS2 == 1)

    if kdst.empty:
        raise ValueError ('No objets to filter')

    return kdst[mask]

def test_unique_S1_S2():

    test_matrix = np.array([[1,1], [12,1], [1,2], [2,2], [1,1]])

    df          = pd.DataFrame((test_matrix),  columns = ['nS1', 'nS2'])

    assert np.shape(unique_S1_S2(df)) == (2, 2)


def test_unique_S1_S2_2():

    empty_df    = pd.DataFrame(columns = ['nS1', 'nS2'])

    try:
        unique_S1_S2(empty_df)

    except ValueError:
        print('Test passed!')

    except Exception as e:
        raise


 test_unique_S1_S2()
 test_unique_S1_S2_2()

 def lifetime_fit(kdst: pd.DataFrame) -> pd.DataFrame:

    def exponential(time: np.array, e0: float, lifetime: float) -> np.array:
        return e0 * np.exp(-time / lifetime)

    try:
        popt, pcov = curve_fit(exponential, kdst.DT, kdst.S2e, p0=[8200, 30000])
        E0, lifetime, uE0, ulifetime = (*popt, *np.sqrt(np.diag(pcov)))

    except Exception as e:
        print(f"Error fitting data: {e}")
        E0, lifetime, uE0, ulifetime = np.nan, np.nan, np.nan, np.nan

    columns = ['E0', 'lifetime', 'uE0', 'ulifetime']
    return pd.DataFrame([E0, lifetime, uE0, ulifetime], index=columns).T


def create_maps(kdst                : pd.DataFrame
                ) -> pd.DataFrame:

    x_bins = np.linspace(-500, 500, 101)
    y_bins = np.linspace(-500, 500, 101)

    kdst['X_bin'] = np.digitize(kdst['X'], x_bins)
    kdst['Y_bin'] = np.digitize(kdst['Y'], y_bins)


    grouped = kdst.groupby(['X_bin', 'Y_bin']).apply(lifetime_fit).reset_index(level=2, drop=True)
    # grouped = grouped[(grouped['E0'] > 7500) & (grouped['lifetime'] < 50000)]

    return grouped

def apply_corrections(kdst                     : pd.DataFrame,
                      map                      : pd.DataFrame,
                      NormStrategy             : str  = None,
                      lifetime_correction      : bool = True) -> pd.DataFrame:

    x_bins = np.linspace(-500, 500, 101)
    y_bins = np.linspace(-500, 500, 101)

    if NormStrategy is not None:

        kdst_group_center = map.reset_index()

        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2
        kdst_group_center['X_mm'] = kdst_group_center['X_bin'].astype(int).apply(lambda i: x_centers[i - 1])
        #kdst_group_center['X_mm'] = x_centers[kdst_group_center['X_bin'].astype(int) - 1]
        kdst_group_center['Y_mm'] = kdst_group_center['Y_bin'].astype(int).apply(lambda i: y_centers[i - 1])

        kdst_group_center = kdst_group_center[kdst_group_center.X_mm**2 + kdst_group_center.Y_mm**2 < 300**2]


        if NormStrategy == 'Mean':
            norm = kdst_group_center['E0'].mean()

        elif NormStrategy == 'Maximum':
            norm = kdst_group_center['E0'].max()

        else:
            raise ValueError("NormStrategy must be 'Mean' or 'Maximum'")

        map = map.reset_index()

        kdst = kdst.merge(map[['X_bin', 'Y_bin', 'E0', 'lifetime']], on=['X_bin', 'Y_bin'], how='left')

        kdst['corr_geom'] = norm / kdst['E0']

        if lifetime_correction == False:

            return kdst

        else:

            kdst['corr_lifetime'] = np.exp(kdst['DT'] / kdst['lifetime'])
            kdst['correction_total'] = kdst['corr_geom'] * kdst['corr_lifetime']

            kdst['S2e_corr'] = kdst['S2e'] * kdst['correction_total']

    return kdst.drop(columns=['X_bin', 'Y_bin'])
