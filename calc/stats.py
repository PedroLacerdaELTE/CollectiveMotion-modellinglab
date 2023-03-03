import numbers
import os
import pickle
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import scipy.signal
import yaml
from matplotlib import pyplot as plt
from numba import njit, prange
from calc.auxiliar import get_numba_meshgrid_3D
from scipy.optimize import curve_fit


def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c


def quadratic_function2D(X, a, b, c, d, e, f):
    x, y = X
    return a * x ** 2 \
           + b * y ** 2 \
           + c * x  \
           + d * y \
           + e * x * y \
           + f


def jacobian_quadratic_function2D(X, a, b, c, d, e, f):
    x, y = X
    return np.array([x ** 2, y ** 2, x, y, x * y, np.ones(shape=x.shape)]).T

def assign_bins(df, df_bins, is_adaptive=True, grid_cols_to_bin=None, adaptive_col_to_bin=None, grid_index_cols=None,
                adaptive_index_col=None):
    df_calc = df.copy()
    if grid_cols_to_bin is None:
        cols_to_bin = ['Z', 'phi']
    if grid_index_cols is None:
        grid_index_cols = [f'bin_index_{col}' for col in grid_cols_to_bin]
    assigned_bins = []
    grid_bin_edges = {}
    for col in grid_cols_to_bin:
        grid_bin_edges[col] = np.sort(df_bins[col + '_min'].unique())
        grid_bin_edges[col] = np.append(grid_bin_edges[col], [df_bins[col + '_max'].max()])

    # -1 to make it start at zero
    grid_index_co = {col: np.digitize(df.loc[:, col].values, grid_bin_edges[col]) - 1
                     for col in grid_cols_to_bin}

    df_assigned_bins = pd.DataFrame().from_dict(grid_index_co)
    df_assigned_bins.columns = grid_index_cols
    df_assigned_bins.index = df.index
    df_calc = pd.merge(df_calc, df_assigned_bins, left_index=True, right_index=True)

    from itertools import product
    df_adaptive_bins = pd.DataFrame(columns=[adaptive_index_col], index=df.index)
    # For each grid bin, assign adaptive bin
    for ij_tuple in product(*[np.unique(grid_index_co[col]) for col in grid_cols_to_bin]):

        # Get Data from current grid bin
        data_on_current_grid_bin = df_calc.copy()
        current_grid_bin = df_bins.copy()
        for i, bin_index in enumerate(ij_tuple):
            data_on_current_grid_bin = data_on_current_grid_bin[
                data_on_current_grid_bin[grid_index_cols[i]] == bin_index]
            current_grid_bin = current_grid_bin[current_grid_bin[grid_index_cols[i]] == bin_index]

        # Get bins of current grid bin
        adaptive_bin_edges = current_grid_bin[adaptive_col_to_bin + '_min'].sort_values().values
        adaptive_bin_edges = np.append(adaptive_bin_edges, [current_grid_bin[adaptive_col_to_bin + '_max'].max()])

        # Assign adaptive bin to each data point. -1 to be zero-indexed
        adaptive_bin_indices = np.digitize(data_on_current_grid_bin.loc[:, adaptive_col_to_bin].values,
                                           adaptive_bin_edges) - 1

        df_adaptive_bins.loc[data_on_current_grid_bin.index, adaptive_index_col] = adaptive_bin_indices

    df_assigned_bins = pd.merge(df_calc[grid_index_cols], df_adaptive_bins, left_index=True, right_index=True)

    return df_assigned_bins


def get_bin_edges_adaptive(df, grid_cols=None, adaptive_col='Z', n_grid_bins=(10, 10),
                           adaptive_bin_min_size=0,
                           adaptive_bin_max_size=25,
                           max_bin_count=5):
    if grid_cols is None:
        grid_cols = ['X', 'Y']
    from itertools import product

    assert np.all(np.isin(grid_cols, df.columns)), 'all values in grid_keys must be columns of df'
    assert adaptive_col in df.columns, 'adaptive_key must a column of df'
    assert len(grid_cols) == len(n_grid_bins), 'length of n_grid_bins must match length of grid_cols'

    if isinstance(grid_cols, np.ndarray):
        grid_cols = grid_cols.tolist()
    all_cols = grid_cols + [adaptive_col]

    # Calculate grid binning, using np.histogramdd

    hist, grid_bin_edges_arr = np.histogramdd(df.loc[:, grid_cols].values, bins=n_grid_bins, normed=False)
    grid_bin_edges = {grid_cols[i]: val for i, val in enumerate(grid_bin_edges_arr)}

    # This is here to prevent the maximum value be excluded
    for col in grid_cols:
        grid_bin_edges[col][-1] += 0.001

    # -1 to make it start at zero
    grid_bins_index = {col: np.digitize(df.loc[:, col].values, grid_bin_edges[col]) - 1
                       for col in grid_cols}

    df_aug = df.copy()[all_cols]
    for k, v in grid_bins_index.items():
        df_aug[f'bin_index_{k}'] = v

    adaptive_bin_edges = {}
    adaptive_bin_min, adaptive_bin_max = df[adaptive_col].min() - 1, df[adaptive_col].max() + 1

    # for each grid bin, define adaptive bin edges
    for grid_indices in product(*[np.arange(n) for n in n_grid_bins]):
        current_bin = df_aug.copy()
        for col_idx, i in enumerate(grid_indices):
            current_bin = current_bin[current_bin[f'bin_index_{grid_cols[col_idx]}'] == i]

        # if current bin does not contain any datapoint fill with bins of maximum size
        if len(current_bin) == 0:
            adaptive_bin_edges[grid_indices] = np.arange(adaptive_bin_min, adaptive_bin_max, adaptive_bin_max_size)
            if adaptive_bin_edges[grid_indices][-1] != adaptive_bin_max:
                adaptive_bin_edges[grid_indices] = np.r_[adaptive_bin_edges[grid_indices], adaptive_bin_max]
            continue

        current_bin = current_bin.sort_values(adaptive_col).reset_index()

        # Left Padding
        # Fill with bins of maximum size until there is data on current_bin

        current_bin_edges = np.arange(adaptive_bin_min, current_bin.loc[0, adaptive_col],
                                      adaptive_bin_max_size).tolist()
        if len(current_bin_edges) == 0:
            current_bin_edges = [adaptive_bin_min]
        current_bin_count = 0

        for idx, row in current_bin.iterrows():
            while True:  # This probably never happens because of the left padding
                if row[adaptive_col] > current_bin_edges[-1] + adaptive_bin_max_size:
                    current_bin_edges.append(current_bin_edges[-1] + adaptive_bin_max_size)
                    current_bin_count = 0
                else:
                    break

            if current_bin_count + 1 == max_bin_count:
                current_bin_edges.append(row[adaptive_col] + 0.001)
                current_bin_count = 0
            else:
                current_bin_count += 1

        # Right Padding
        # Fill with bins of maximum size until there is z_max
        current_bin_edges += np.arange(current_bin[adaptive_col].values[-1], adaptive_bin_max,
                                       adaptive_bin_max_size).tolist()

        if current_bin_edges[-1] != adaptive_bin_max:
            current_bin_edges.append(adaptive_bin_max)

        adaptive_bin_edges[grid_indices] = np.unique(current_bin_edges)

    all_bin_edges = []

    for grid_indices, adaptive_col_edges_arr in adaptive_bin_edges.items():

        for adaptive_idx in range(len(adaptive_col_edges_arr) - 1):

            all_indices = grid_indices + (adaptive_idx,)
            current_bin = {f'bin_index_{col}': grid_indices[grid_col_idx] for grid_col_idx, col in enumerate(grid_cols)}
            current_bin[f'bin_index_{adaptive_col}'] = adaptive_idx

            for grid_col_idx, col in enumerate(grid_cols):
                current_bin[f'{col}_min'] = grid_bin_edges[col][grid_indices[grid_col_idx]]
                current_bin[f'{col}_max'] = grid_bin_edges[col][grid_indices[grid_col_idx] + 1]
                current_bin[f'{col}_avg'] = (grid_bin_edges[col][grid_indices[grid_col_idx]]
                                             + grid_bin_edges[col][grid_indices[grid_col_idx] + 1]) / 2

            current_bin.update({f'{adaptive_col}_min': adaptive_col_edges_arr[adaptive_idx],
                                f'{adaptive_col}_max': adaptive_col_edges_arr[adaptive_idx + 1],
                                f'{adaptive_col}_avg': (adaptive_col_edges_arr[adaptive_idx]
                                                        + adaptive_col_edges_arr[adaptive_idx + 1]) / 2}
                               )

            all_bin_edges.append(current_bin)

    index_col = [f'bin_index_{col}' for col in grid_cols + [adaptive_col]]
    df_bins = pd.DataFrame(all_bin_edges)

    return df_bins, index_col


def get_bin_edges_grid(df, cols_to_bins, n_bins):
    X = df[cols_to_bins].values
    n_dims = X.shape[-1]

    hist, bin_edges = np.histogramdd(X, bins=n_bins, normed=False)  # weights= df['dZdT'].values,
    bin_edges = list(bin_edges)

    n_bins = [len(bin_edges[i]) - 1 for i in range(n_dims)]

    all_bin_edges = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            for k in range(n_bins[2]):
                all_bin_edges[(i, j, k)] = {f'{cols_to_bins[0]}_min': bin_edges[0][:-1][i],
                                            f'{cols_to_bins[0]}_max': bin_edges[0][1:][i],
                                            f'{cols_to_bins[1]}_min': bin_edges[1][:-1][j],
                                            f'{cols_to_bins[1]}_max': bin_edges[1][1:][j],
                                            f'{cols_to_bins[2]}_min': bin_edges[2][:-1][k],
                                            f'{cols_to_bins[2]}_max': bin_edges[2][1:][k],
                                            f'{cols_to_bins[0]}_avg': (bin_edges[0][:-1][i] + bin_edges[0][1:][i]) / 2,
                                            f'{cols_to_bins[1]}_avg': (bin_edges[1][:-1][j] + bin_edges[1][1:][j]) / 2,
                                            f'{cols_to_bins[2]}_avg': (bin_edges[2][:-1][k] + bin_edges[2][1:][k]) / 2
                                            }

    df_bins = pd.DataFrame().from_dict(all_bin_edges, orient='index')

    index_cols = [f'bin_index_{col}' for col in cols_to_bins]
    df_bins = df_bins.reset_index().rename(columns={f'level_{i}': col for i, col in enumerate(index_cols)})

    return df_bins, index_cols


def get_bins(df, cols_to_bin, n_bins=(10, 10, 40), method=None, adaptive_kwargs=None):
    # assert len(n_bins) == X.shape[-1], 'n_bins must have the same dimension as X'

    if adaptive_kwargs is None:
        adaptive_kwargs = {}
    if method == 'grid':
        df_bins, index_cols = get_bin_edges_grid(df, cols_to_bin, n_bins)
    else:
        df_bins, index_cols = get_bin_edges_adaptive(df, grid_cols=cols_to_bin[:-1], adaptive_col=cols_to_bin[-1],
                                                     n_grid_bins=n_bins, **adaptive_kwargs)

    return df_bins, index_cols


def get_histogram_from_bins(df, bin_index_cols):
    df_calc = df[bin_index_cols].copy()
    df_calc['counts'] = 1
    # df_calc = pd.merge(df_bins[bin_index_cols], df_calc, left_on=bin_index_cols, right_on=bin_index_cols, how='left')

    df_group = df_calc.groupby(bin_index_cols, as_index=False)
    df_count = df_group.sum()

    # df_count = pd.merge(df_bins, df_count, left_on=bin_index_cols, right_on=bin_index_cols, how='left')

    return df_count


def get_bin_mean(df, i, j, column, replace_value=np.nan):
    current_bin = df[(df['bin_index_x'] == i) & (df['bin_index_y'] == j)]

    if current_bin.empty:
        return replace_value
    else:
        return current_bin[column].mean()


def weighted_average_groupby(df_group, value_col, weights_col):
    values = df_group[value_col].values
    weights = df_group[weights_col].values

    if len(values) == 1:
        return values[0]

    if np.nansum(weights) == 0:
        return 0

    weights = weights_preprocessing(weights)

    weighted_average = np.average(values, weights=weights)
    return weighted_average


def weighted_std_groupby(df_group, value_col, weights_col):
    values = df_group[value_col].values
    weights = df_group[weights_col].values

    if len(values) == 1:
        return 0

    if np.nansum(weights) == 0:
        return 0

    weighted_average = weighted_average_groupby(df_group, value_col, weights_col)

    weights = weights_preprocessing(weights)

    # weighted 2nd Moment
    weighted_values = weights * (values - weighted_average) ** 2
    # Normalize
    weighted_values = weighted_values / weights.sum()

    weighted_std = np.sqrt(weighted_values.sum())

    return weighted_std


def weighted_moving_average(df, values_col, weights_col, ma_args=None):
    from types import LambdaType
    df_copy = df.copy()
    df_return = pd.DataFrame()

    if ma_args is None:
        ma_args = {'window': 12, 'min_periods': 1, 'center': True}
    else:
        ma_args = ma_args.copy()

    if 'window_args' in ma_args.keys():
        window_args = ma_args.pop('window_args')
    else:
        window_args = {}

    if isinstance(weights_col, LambdaType):
        df_copy['weights'] = df_copy.apply(weights_col, axis=1)
    else:
        df_copy['weights'] = df_copy[weights_col]

    # Deal with NA and negative values on the weight values
    df_copy['weights'] = weights_preprocessing(df_copy['weights'].values)

    # Weighted Values
    df_copy['weighted_values'] = df_copy[values_col] * df_copy['weights']

    # Moving Sums
    df_roll = pd.DataFrame()
    df_roll['weighted_sum'] = df_copy['weighted_values'].rolling(**ma_args).sum(**window_args)
    df_roll['sum_of_weights'] = df_copy['weights'].rolling(**ma_args).sum(**window_args)

    # Moving Average
    df_roll['weighted_moving_average'] = df_roll['weighted_sum'] / df_roll['sum_of_weights']

    return df_roll['weighted_moving_average'].values


def weights_preprocessing(weights_array, impute_value=0):
    weights_array.copy()
    if isinstance(weights_array, list):
        weights_array = np.array(weights_array)
    if weights_array.min() < 0:
        weights_array = weights_array - weights_array.min()

    mask_na = np.where(np.isnan(weights_array))
    weights_array[mask_na] = impute_value

    return weights_array


def weighted_moving_std(df, values_col, weights_col, weighted_average_col, ma_args=None):
    from types import LambdaType
    ma_args = ma_args.copy()
    df_copy = df.copy()

    if ma_args is None:
        ma_args = {'window': 12, 'min_periods': 1, 'center': True}

    if 'window_args' in ma_args.keys():
        window_args = ma_args.pop('window_args')
    else:
        window_args = {}

    if isinstance(weights_col, LambdaType):
        df_copy['weights'] = df_copy.apply(weights_col, axis=1)
    else:
        df_copy['weights'] = df_copy[weights_col]

    # Deal with NA and negative values on the weight values

    df_copy['weights'] = weights_preprocessing(df_copy['weights'].values)

    # Weighted Values

    df_copy['weighted_values'] = df_copy[values_col] * df_copy['weights']
    df_copy['deviation'] = (df_copy['weighted_values'] - df_copy[weighted_average_col]) ** 2

    # Moving Standard Deviation
    df_roll = pd.DataFrame()
    df_roll['moving_variance'] = df_copy['deviation'].rolling(**ma_args).mean(**window_args)
    try:
        df_roll['moving_standard_deviation'] = np.sqrt(df_roll['moving_variance'])
    except RuntimeWarning as e:
        print(e)
    else:
        pass
    return df_roll['moving_standard_deviation'].values


def get_weighted_average(x_array, weights):
    x_array = np.array(x_array)
    weights = np.array(weights)

    weighted_average, sum_of_weights = np.average(x_array, weights=weights, returned=True)

    n = len(weights[weights != 0])
    # http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)

    numerator = n_eff * np.sum(weights * (x_array - weighted_average) ** 2.0)
    denominator = (n_eff - 1) * sum_of_weights

    weighted_std = np.sqrt(numerator / denominator)
    weighted_sem = weighted_std / np.sqrt(n_eff)
    return weighted_average, weighted_std, weighted_sem, n


def rolling_apply_multiple_columns(func, window_size: int, min_periods: int, *arrays: np.ndarray, n_jobs: int = 1,
                                   drop_na=True, **kwargs):
    import numpy_ext as npext
    with warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', category=DeprecationWarning)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        # From 0 to window_size - 1
        return_arr_before = npext.expanding_apply(func,
                                                  min_periods,
                                                  *[arr[:window_size - 1] for arr in arrays],
                                                  n_jobs=n_jobs,
                                                  **kwargs
                                                  )
        # Remove the NaN from the beginning
        if drop_na:
            return_arr_before = return_arr_before[min_periods-1:]

        # From window_size to N-window_size
        return_arr = npext.rolling_apply(func,
                                         window_size,
                                         *arrays, # window_size - 1 [:-window_size + 1]
                                         n_jobs=n_jobs,
                                         **kwargs)

        # Remove the NaN from the beginning
        return_arr = return_arr[window_size-1:]

        # From N-window_size to N
        return_arr_after = npext.expanding_apply(func,
                                                 min_periods,
                                                 *[arr[-window_size + 1:][::-1] for arr in arrays],
                                                 n_jobs=n_jobs,
                                                 **kwargs
                                                 )
        # Remove the NaN from the beginning
        if drop_na:
            return_arr_after = return_arr_after[min_periods-1:]

        return_arr = np.vstack([return_arr_before, return_arr, return_arr_after[::-1]])
    return return_arr


def get_maximum_by_quadratic_fit(x_values, y_values, bounds=None):
    from scipy.optimize.optimize import OptimizeWarning

    try:
        popt, pcov = curve_fit(quadratic_function, x_values, y_values,
                               bounds=bounds
                               )
    except OptimizeWarning as e:
        print(e)
        return np.nan, np.nan, np.nan, np.nan
    a, b, c = popt
    x_max = -b / (2 * a)
    y_max = c - b ** 2 / (4 * a)
    is_maximum = a < 0
    return x_max, y_max, popt, pcov, is_maximum


def get_maximum_by_quadratic_fit2D(x_values, y_values, f_values, method, bounds=(-np.inf, np.inf),
                                   ftol=0.02, p0=None):
    from scipy.optimize.optimize import OptimizeWarning

    if method == 'fit':
        try:
            popt, pcov = curve_fit(quadratic_function2D,
                                   np.vstack([x_values, y_values]),
                                   f_values,
                                   p0=p0,
                                   bounds=bounds,
                                   ftol=ftol,
                                   jac=jacobian_quadratic_function2D
                                   )
            a, b, c, d, e, f = popt
        except OptimizeWarning as e:
            print(e)
            return np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        X = np.vstack([x_values ** 2,
                       y_values ** 2,
                       x_values,
                       y_values,
                       x_values * y_values,
                       np.ones(shape=x_values.shape)]).T
        Y = f_values
        if np.array(bounds).ndim == 1:
            epsilon = 0
        elif np.array(bounds).ndim == 2:
            epsilon = 1 / bounds[1][2]

        l_regularization = np.diag([0, 0, epsilon, epsilon, 0, 0])
        popt = np.linalg.inv(X.T @ X + l_regularization.T @ l_regularization) @ X.T @ Y
        pcov = None

        a, b, c, d, e, f = popt

    x_max = (2 * b * c - d * e) / (e ** 2 - 4 * a * b)
    y_max = (2 * a * d - c * e) / (e ** 2 - 4 * a * b)
    f_max = quadratic_function2D([x_max, y_max],
                                 a, b, c, d, e, f)

    hess_matrix = np.array([[2 * a, e], [e, 2 * b]])
    is_maximum = np.all(np.linalg.eigvals(hess_matrix) < 0)
    # If the hessian matrix is positive-definite: the function has a minimum
    # If the hessian matrix is negative-definite: the function has a maximum
    # Otherwise, the test is inclusive or the extreme is a saddle point
    return x_max, y_max, f_max, popt, pcov, is_maximum


def get_maximum_vertical_velocity2d(X_array, Y_array, Vz_array, Z_array, n_sigma=0, n_min=50, derivative_epsilon=None,
                                    method='analytic', debug=False):
    if derivative_epsilon is None:
        derivative_epsilon = np.inf
    #else:
    #             a       b                c                  d               e         f
    bounds = ([-np.inf, -np.inf, -derivative_epsilon, -derivative_epsilon, -np.inf, -np.inf],
              [0, 0, derivative_epsilon, derivative_epsilon, np.inf, np.inf]
              )
    rho_array = np.sqrt(X_array ** 2 + Y_array ** 2)
    X_array_mask = np.logical_not(np.isnan(X_array))
    Y_array_mask = np.logical_not(np.isnan(Y_array))
    Vz_array_mask = np.logical_not(np.isnan(Vz_array))
    Z_array_mask = np.logical_not(np.isnan(Z_array))

    na_mask = X_array_mask.copy()
    for arr in [Y_array_mask, Vz_array_mask, Z_array_mask]:
        na_mask = np.logical_and(na_mask, arr)

    if n_sigma is not None:
        rho_median = np.median(rho_array)
        rho_std = np.std(rho_array)
        rho_maximum_allowed = rho_median + n_sigma * rho_std
        mask = rho_array < rho_maximum_allowed
        mask = np.logical_and(mask, na_mask)
    else:
        mask = na_mask
        rho_maximum_allowed = np.inf
    Z_avg = np.mean(Z_array)

    N = mask.sum()
    if N < n_min:
        mask_idx = np.argpartition(rho_array, n_min)
        mask = np.full(shape=len(rho_array), fill_value=False)
        mask[mask_idx[:n_min]] = True
        mask[~na_mask] = False
        N = mask.sum()
    if N < n_min:   # It is possible that there are NaNs in the first n_min elements. If this happens,
                    # nans are carried to the fitting part. This lines makes sure this doesn't happen.
        return np.nan, np.nan, np.nan, Z_avg, np.nan, np.nan, N

    X_max, Y_max, dZdT_air_max, popt, pcov, is_maximum = get_maximum_by_quadratic_fit2D(X_array[mask],
                                                                                        Y_array[mask],
                                                                                        Vz_array[mask],
                                                                                        bounds=bounds,
                                                                                        method=method
                                                                                        )
    if debug and Z_avg > 300:
        from plotting.decomposition.debug import debug_plot_decomposition_get_maximum_vertical_velocity2d
        debug_plot_decomposition_get_maximum_vertical_velocity2d(X_array, Y_array, Vz_array, mask, popt,
                                                                 X_max, Y_max, dZdT_air_max)
    if not is_maximum:
        return np.nan, np.nan, np.nan, Z_avg, popt, pcov, N

    return X_max, Y_max, dZdT_air_max, Z_avg, popt, pcov, N


def update_aero_params(df_iteration, df_real, params):
    delta = {}
    loss_sum = ((df_real['dZdT_bird_real'] - df_iteration['dZdT_bird_i']) * df_iteration['dZdT_bird_i']).mean()
    delta['CL'] = loss_sum / params['CL'] * (-3 / 2)
    delta['CD'] = loss_sum / params['CD']
    delta['mass'] = loss_sum / params['mass'] * (1 / 2)
    delta['wing_area'] = loss_sum / params['wing_area'] * (-1 / 2)

    return delta


def average_over_vars(sweep_array, integrate_array, independent_var):
    # independent_var is a function or lambda where the first argument must be the sweep var,
    # followed by the integrated values
    output_avg = []
    output_std = []
    for sweep in sweep_array:
        array_with_fixed_sweep_var = []
        for integrate_x, integrate_y in product(*integrate_array):
            fixed_sweep_value = independent_var(sweep, integrate_x, integrate_y)

            array_with_fixed_sweep_var.append(fixed_sweep_value)
        output_avg.append(np.average(array_with_fixed_sweep_var))
        output_std.append(np.std(array_with_fixed_sweep_var, ddof=1))

    output_avg = np.array(output_avg)
    output_std = np.array(output_std)

    return output_avg, output_std


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if isinstance(points, pd.Series):
        points = points.values
    median = np.median(points)
    var = np.sum((points - median) ** 2)
    std = np.sqrt(var)

    mask = np.abs(points - median) < thresh * std

    return mask


def get_correlation(field, field_dimensions=1, N_periods=3, scale=(1, 1), debug=False):
    # field 1 x Lx x Ly,
    # N number of periods on extended_field
    coord_list = ['x', 'y', 'z']
    n_dimensions = len(scale)
    N = {coord_list[i]: field.shape[i] for i in np.arange(n_dimensions)}
    L = {coord: n * scale[i] for i, (coord, n) in enumerate(N.items())}
    if N_periods % 2 == 0:
        N_periods = N_periods + 1
    padding_size = int((N_periods - 1) / 2)  # padding per side: left, right, up and down

    extended_field = np.zeros(shape=list(N_periods * n for n in N.values()))

    for indices in product(*([np.arange(N_periods)] * n_dimensions)):
        current_slice = tuple((slice(indices[i] * n, (indices[i] + 1) * n )
                               for i, n in enumerate(N.values()))
                              )

        # [n * Nx: (n + 1) * Nx, m * Ny: (m + 1) * Ny]
        extended_field[current_slice] = field

    if debug:
        mg_array = np.meshgrid(*[np.arange(0, n) * scale[i] for i, n in enumerate(N.values())] )

        extended_mg_array = np.meshgrid(*[np.arange(-padding_size * n,
                                                              (N_periods - padding_size) * n)
                                                    * scale[i]
                                                    for i, n in enumerate(N.values())]
                                                   )

        fig, ax = plt.subplots(1, 2)
        if n_dimensions != 3:
            z_mask = np.full(shape=mg_array[0].shape, fill_value=True)
            extended_z_mask = np.ones(shape=extended_mg_array[0].shape, dtype=bool)
        else:
            z_mask = mg_array[-1] == 0
            extended_z_mask = extended_mg_array[-1] == 0
        m = ax[0].contourf(extended_mg_array[0][extended_z_mask].reshape(extended_mg_array[0].shape[:-1]),
                           extended_mg_array[1][extended_z_mask].reshape(extended_mg_array[1].shape[:-1]),
                           extended_field[extended_z_mask].reshape(extended_mg_array[0].shape[:-1]), levels=100)
        plt.colorbar(m, ax=ax[0], label='Vx (m/s)')

        m = ax[1].contourf(mg_array[0][z_mask].reshape(mg_array[0].shape[:-1]),
                           mg_array[1][z_mask].reshape(mg_array[1].shape[:-1]),
                           field[z_mask].reshape(field.shape[1:]),
                           levels=100)
        plt.colorbar(m, ax=ax[1], label='Vx (m/s)')
        ax[0].vlines([n * L['x'] for n in range(-(padding_size - 1), N_periods - padding_size)],
                     ymin=-padding_size * L['y'],
                     ymax=(N_periods - padding_size) * L['y'],
                     colors='r', linestyles='dashed')
        ax[0].hlines([n * L['y'] for n in range(-(padding_size - 1), N_periods - padding_size)],
                     xmin=-padding_size * L['x'],
                     xmax=(N_periods - padding_size) * L['x'],
                     colors='r', linestyles='dashed')

        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[0].set_title('extended field')
        ax[1].set_title('field')
        ax[0].set_xlabel('X (m)')
        ax[0].set_ylabel('Y (m)')
        ax[1].set_xlabel('X (m)')
        ax[1].set_ylabel('Y (m)')
        fig.tight_layout()
        plt.show(block=True)

    #output_shape = tuple(((N_periods - 1) * n + 1 for n in N.values()))

    #corr = np.zeros(shape=output_shape)
    #for delta_indices in product(*[range(output_shape[i_]) for i_ in range(len(output_shape))]):
    #    if debug:
    #        print(delta_indices)
    #    current_slice = tuple((slice(delta_indices[i], delta_indices[i] + n) for i, n in enumerate(N.values()) ))
    #    #corr[delta_indices] = np.mean(extended_field[current_slice] * field)
    corr = scipy.signal.correlate(extended_field, field, mode='valid') / field.size
    deltas_mg = np.meshgrid(*[np.arange(-n, n + 1, dtype=float) * scale[i]
                              for i, n in enumerate(N.values())]
                            )

    return corr, deltas_mg


def get_correlation_differences_with_extension_3d(field, N_periods=3, scale=(1, 1, 1), debug=False, plot_debug=False):
    # field 1 x Lx x Ly,
    # N number of periods on extended_field
    coord_list = ['x', 'y', 'z']
    n_dimensions = len(scale)
    N = {coord_list[i]: field.shape[i] for i in np.arange(n_dimensions)}
    L = {coord: n * scale[i] for i, (coord, n) in enumerate(N.items())}
    if N_periods % 2 == 0:
        N_periods = N_periods + 1
    extension_fraction = 1/2  # padding per side: left, right, up and down
    extension_N = {k: int(v * extension_fraction) for k, v in N.items()}
    extension_L = {coord: n * scale[i] for i, (coord, n) in enumerate(extension_N.items())}

    extended_field = np.zeros(shape=[extension_N[coord] + n + extension_N[coord] for coord, n in N.items()])

    if n_dimensions == 3:
        #center
        extended_field[extension_N['x']: extension_N['x'] + N['x'],
                       extension_N['y']: extension_N['y'] + N['y'],
                       extension_N['z']: extension_N['z'] + N['z']] = field

        #sides
        # YZ Plane
        extended_field[0: extension_N['x'], extension_N['y']: extension_N['y'] + N['y'], extension_N['z']: extension_N['z'] + N['z']] = field[-extension_N['x']:]
        extended_field[-extension_N['x']:, extension_N['y']: extension_N['y'] + N['y'], extension_N['z']: extension_N['z'] + N['z']] = field[0:extension_N['x']]

        # XZ Plane
        extended_field[extension_N['x']: extension_N['x'] + N['x'], 0: extension_N['y'], extension_N['z']: extension_N['z'] + N['z']] = field[:, -extension_N['y']:, :]
        extended_field[extension_N['x']: extension_N['x'] + N['x'], -extension_N['y']:, extension_N['z']: extension_N['z'] + N['z']] = field[:, 0:extension_N['y']:, :]

        # XY Plane
        extended_field[extension_N['x']: extension_N['x'] + N['x'], extension_N['y']: extension_N['y'] + N['y'], 0: extension_N['z']] = field[:, :, -extension_N['z']:]
        extended_field[extension_N['x']: extension_N['x'] + N['x'], extension_N['y']: extension_N['y'] + N['y'], -extension_N['z']:] = field[:, :, 0:extension_N['z']:]

        # edges
        # along z
        extended_field[0: extension_N['x'], 0: extension_N['y'], extension_N['z']:extension_N['z'] + N['z']] = field[-extension_N['x']:, -extension_N['y']:, :]
        extended_field[0: extension_N['x'], -extension_N['y']:, extension_N['z']:extension_N['z'] + N['z']] = field[-extension_N['x']:, 0: extension_N['y'], :]
        extended_field[-extension_N['x']:, 0: extension_N['y'], extension_N['z']:extension_N['z'] + N['z']] = field[0:extension_N['x'], -extension_N['y']:, :]
        extended_field[-extension_N['x']:, -extension_N['y']:, extension_N['z']:extension_N['z'] + N['z']] = field[0:extension_N['x'], 0:extension_N['y'], :]

        # Along x
        extended_field[extension_N['x']:extension_N['x'] + N['x'], 0: extension_N['y'], -extension_N['z']:] = field[:, -extension_N['y']:, 0:extension_N['z']]
        extended_field[extension_N['x']:extension_N['x'] + N['x'], 0: extension_N['y'], 0:extension_N['z']] = field[:, -extension_N['y']:, -extension_N['z']:]
        extended_field[extension_N['x']:extension_N['x'] + N['x'], -extension_N['y']:, -extension_N['z']:] = field[:, 0:extension_N['y'], 0:extension_N['z']]
        extended_field[extension_N['x']:extension_N['x'] + N['x'], -extension_N['y']:, 0:extension_N['z']] = field[:, 0:extension_N['y'], -extension_N['z']:]

        # Along y
        extended_field[0: extension_N['x'], extension_N['y']: extension_N['y'] + N['y'], -extension_N['z']:] = field[-extension_N['x']:, :, 0:extension_N['z']]
        extended_field[0: extension_N['x'], extension_N['y']: extension_N['y'] + N['y'], 0:extension_N['z']] = field[-extension_N['x']:, :, -extension_N['z']:]
        extended_field[-extension_N['x']:, extension_N['y']: extension_N['y'] + N['y'], 0:extension_N['z']:] = field[0:extension_N['x'], :, -extension_N['z']:]
        extended_field[-extension_N['x']:, extension_N['y']: extension_N['y'] + N['y'], -extension_N['z']:] = field[0:extension_N['x'], :, 0:extension_N['z']]

        # Vertices
        extended_field[0: extension_N['x'], 0:extension_N['y'], 0:extension_N['z']] = field[-extension_N['x']:, -extension_N['y']:, -extension_N['z']:]
        extended_field[0: extension_N['x'], 0:extension_N['y'], -extension_N['z']:] = field[-extension_N['x']:, -extension_N['y']:, 0:extension_N['z']]
        extended_field[0: extension_N['x'], -extension_N['y']:, 0:extension_N['z']] = field[-extension_N['x']:, 0:extension_N['y'], -extension_N['z']:]
        extended_field[0: extension_N['x'], -extension_N['y']:, -extension_N['z']:] = field[-extension_N['x']:, 0:extension_N['y']:, 0:extension_N['z']]
        extended_field[-extension_N['x']:, 0:extension_N['y'], 0:extension_N['z']] = field[0:extension_N['x'], -extension_N['y']:, -extension_N['z']:]
        extended_field[-extension_N['x']:, 0:extension_N['y'], -extension_N['z']:] = field[0:extension_N['x'], -extension_N['y']:, 0:extension_N['z']]
        extended_field[-extension_N['x']:, -extension_N['y']:, 0:extension_N['z']] = field[0:extension_N['x'], 0:extension_N['y'], -extension_N['z']:]
        extended_field[-extension_N['x']:, -extension_N['y']:, -extension_N['z']:] = field[0:extension_N['x'], 0:extension_N['y']:, 0:extension_N['z']]

    if plot_debug:
        mg_array = np.meshgrid(*[np.arange(0, n) * scale[i] for i, n in enumerate(N.values())] )

        extended_mg_array = np.meshgrid(*[np.arange(-extension_N[coord], n + extension_N[coord]
                                                    ) * scale[i]
                                          for i, (coord, n) in enumerate(N.items())]
                                        )


        fig, ax = plt.subplots(1, 2)
        if n_dimensions != 3:
            z_mask = np.full(shape=mg_array[0].shape, fill_value=True, dtype=bool)
            extended_z_mask = np.full(shape=extended_mg_array[0].shape, fill_value=True, dtype=bool)
            m_extended = ax[0].contourf(extended_mg_array[0][extended_z_mask].reshape(extended_mg_array[0].shape),
                               extended_mg_array[1][extended_z_mask].reshape(extended_mg_array[1].shape),
                               extended_field[extended_z_mask].reshape(extended_mg_array[0].shape), levels=100)
            m = ax[1].contourf(mg_array[0][z_mask].reshape(mg_array[0].shape),
                               mg_array[1][z_mask].reshape(mg_array[1].shape),
                               field[z_mask].reshape(field.shape),
                               levels=100)
        else:
            z_mask = mg_array[-1] == 0
            extended_z_mask = extended_mg_array[-1] == 0
            m_extended = ax[0].contourf(extended_mg_array[0][extended_z_mask].reshape(extended_mg_array[0].shape[:-1]),
                               extended_mg_array[1][extended_z_mask].reshape(extended_mg_array[1].shape[:-1]),
                               extended_field[extended_z_mask].reshape(extended_mg_array[0].shape[:-1]), levels=100)
            m = ax[1].contourf(mg_array[0][z_mask].reshape(mg_array[0].shape[:-1]),
                               mg_array[1][z_mask].reshape(mg_array[1].shape[:-1]),
                               field[z_mask].reshape(field.shape[1:]),
                               levels=100)
        plt.colorbar(m_extended, ax=ax[0], label='Vx (m/s)')

        plt.colorbar(m, ax=ax[1], label='Vx (m/s)')
        ax[0].vlines(0,
                     ymin=-extension_L['y'],
                     ymax=N['y'] + extension_L['y'],
                     colors='r', linestyles='dashed')
        ax[0].vlines(L['x'],
                     ymin=-extension_L['y'],
                     ymax=N['y'] + extension_L['y'],
                     colors='r', linestyles='dashed')
        ax[0].hlines(0,
                     xmin=-extension_L['x'],
                     xmax=N['x'] + extension_L['x'],
                     colors='r', linestyles='dashed')
        ax[0].hlines(L['y'],
                     xmin=-extension_L['x'],
                     xmax=N['x'] + extension_L['x'],
                     colors='r', linestyles='dashed')

        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[0].set_title('extended field')
        ax[1].set_title('field')
        ax[0].set_xlabel('X (m)')
        ax[0].set_ylabel('Y (m)')
        ax[1].set_xlabel('X (m)')
        ax[1].set_ylabel('Y (m)')
        fig.tight_layout()
        plt.show(block=True)
    output_shape = tuple(((N_periods - 1) * n + 1 for n in N.values()))
    start_time = time.time()
    corr = get_differences_convolution(field, extended_field, debug=debug)
    end_time = time.time()
    print(end_time - start_time)


    deltas_mg = np.meshgrid(*[np.arange(-extension_N[coord], extension_N[coord] + 1, dtype=float) * scale[i]
                              for i, (coord, n) in enumerate(N.items())]
                            )

    return corr, deltas_mg


def get_differences_convolution(field, extended_field, debug=False):
    N_array = np.array(field.shape)
    n_dim = field.ndim
    output_shape = np.array(extended_field.shape, dtype=np.int32) - np.array(field.shape, dtype=np.int32) + 1
    corr = np.zeros(shape=output_shape, dtype=field.dtype)
    array_of_delta_indices = np.indices(output_shape)
    flatten_indices = np.empty(shape=(array_of_delta_indices[0].size, array_of_delta_indices.shape[0]),
                               dtype=array_of_delta_indices.dtype)

    for i in range(n_dim):
        flatten_indices[:, i] = array_of_delta_indices[i].flatten()

    if n_dim == 2:
        corr = get_difference_sweep_2d(field, extended_field, flatten_indices, N_array, corr, debug)
    else:
        corr = get_difference_sweep_3d(field, extended_field, flatten_indices, N_array, corr, debug)
    return corr


def get_difference_sweep_2d(field, extended_field, N_array, corr, debug):
    output_shape = np.array(extended_field.shape, dtype=np.int32) - np.array(field.shape, dtype=np.int32) + 1

    for delta_x in prange(output_shape[0]):
        for delta_y in prange(output_shape[1]):
                delta_indices = np.array([delta_x, delta_y])
                if debug:
                    print(delta_indices)

                current_extended_field = extended_field[delta_indices[0]: delta_indices[0] + N_array[0], delta_indices[1]: delta_indices[1] + N_array[1]]
                current_curr = get_difference(current_extended_field, field)

                corr[delta_indices[0], delta_indices[1]] = current_curr
    return corr


def get_correlation_differences_from_sample(field, correlation_span, scale=(1, 1), n_dimensions=2,
                                            plot_debug=False):

    if isinstance(scale, numbers.Number):
        scale = [scale] * n_dimensions
    correlation_span_index = round(correlation_span / scale[0])

    if n_dimensions == 2:
        corr = get_correlation_differences_from_sample_new2d(field=field,
                                                             correlation_index_span=correlation_span_index,
                                                             plot_debug=plot_debug)
    else:
        corr = get_correlation_differences_from_sample_new3D(field=field,
                                                             correlation_index_span=correlation_span_index,
                                                             plot_debug=plot_debug)
    array_of_deltas = []
    for s in scale:
        array_of_deltas.append(np.arange(-correlation_span_index, correlation_span_index + 1) * s)

    deltas_mg = np.meshgrid(*array_of_deltas)
    return corr, deltas_mg


def get_correlation_differences_and_stats_from_sample(field, correlation_span, scale=(1, 1), n_dimensions=2,
                                                      fitting_limits=None, save_folder=None,
                                                      correlation_data_filename='all_correlation_data.pkl',
                                                      correlation_fit_filename='linear_fit_original.yaml',
                                                      plot_debug=False):

    corr, deltas_mg = get_correlation_differences_from_sample(field, correlation_span, scale=scale,
                                                              n_dimensions=n_dimensions,
                                                              plot_debug=plot_debug)

    (radial_corr, radial_corr_std, delta_R) = get_radial_correlation_from_grid_correlation(corr, deltas_mg)

    mask_to_fit = np.full(shape=delta_R.shape, fill_value=True)
    if fitting_limits is not None:
        if fitting_limits[0] is not None:
            mask_to_fit = mask_to_fit & (delta_R > fitting_limits[0])
        if fitting_limits[1] is not None:
            mask_to_fit = mask_to_fit & (delta_R < fitting_limits[1])

    errors_for_fitting = [1 / s if s != 0 else 0 for s in radial_corr_std[mask_to_fit]]

    fit_return = np.polyfit(np.log(delta_R[mask_to_fit]),
                            np.log(radial_corr[mask_to_fit]),
                            w=errors_for_fitting, deg=1,
                            cov=True)

    fit_parameters, fit_cov = fit_return
    fit_errorbars = np.sqrt(np.diag(fit_cov))

    return_dict = {'cartesian_correlation': {
                        'deltas': deltas_mg,
                        'values': corr,
                    },
                   'radial_correlation': {
                       'deltas': delta_R,
                       'values': radial_corr,
                       'std': radial_corr_std,
                   },
                   'linear_fit': {
                       'parameters': fit_parameters,
                       'parameters_cov': fit_cov,
                       'errorbars': fit_errorbars
                   }}

    return return_dict


def get_correlation_differences_from_sample_new2d(field, correlation_index_span, plot_debug=False):

    output_shape = (2 * correlation_index_span + 1, 2 * correlation_index_span + 1)
    sample_field = field[correlation_index_span:-correlation_index_span, correlation_index_span:-correlation_index_span]

    corr = np.empty(shape=output_shape)
    for delta_x in prange(output_shape[0]):
        for delta_y in prange(output_shape[1]):
            delta_indices = np.array([delta_x, delta_y])
            bottom_left_corner = [delta_indices[0],
                                  delta_indices[1]]
            top_right_corner = [bottom_left_corner[0] + sample_field.shape[0],
                                bottom_left_corner[1] + sample_field.shape[1]]
            current_slice = field[bottom_left_corner[0]: top_right_corner[0], bottom_left_corner[1]: top_right_corner[1]]
            current_curr = get_difference(current_slice, sample_field)
            corr[delta_x, delta_y] = current_curr

    if plot_debug:
        levels = 30
        fig, ax = plt.subplots(1, 3)
        delta_x, delta_y = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]))
        ax[0].contourf(delta_x, delta_y, field, levels=levels)
        delta_x, delta_y = np.meshgrid(np.arange(sample_field.shape[0]), np.arange(sample_field.shape[1]))
        ax[1].contourf(delta_x, delta_y, sample_field, levels=levels)
        delta_x, delta_y = np.meshgrid(np.arange(output_shape[0]), np.arange(output_shape[1]))
        ax[2].contourf(delta_x, delta_y, corr, levels=levels)
        ax[0].hlines(y=[correlation_index_span, correlation_index_span + sample_field.shape[1]],
                     xmin=[correlation_index_span] * 2, xmax=[correlation_index_span] * 2,
                     color='r', linestyles='dashed')
        ax[0].vlines(x=[correlation_index_span, correlation_index_span + sample_field.shape[0]],
                     ymin=[correlation_index_span] * 2, ymax=[correlation_index_span] * 2,
                     color='r', linestyles='dashed')
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[2].set_aspect('equal')

        plt.show(block=True)

    return corr


def get_correlation_differences_from_sample_2D(field, correlation_span, scale=(1, 1), plot_debug=False):

    if isinstance(scale, numbers.Number):
        scale = (scale, scale)
    correlation_span_index = round(correlation_span / scale[0])
    sample_field = field[correlation_span_index:-correlation_span_index, correlation_span_index:-correlation_span_index]
    if plot_debug:
        fig, ax = plt.subplots(1, 2, figsize=(19,16), tight_layout=True)
        xy_mg = np.meshgrid(np.arange(0, field.shape[0]) * scale[0],
                            np.arange(0, field.shape[1]) * scale[1])
        sample_xy_mg = np.meshgrid(np.arange(correlation_span_index, field.shape[0] - correlation_span_index) * scale[0],
                                   np.arange(correlation_span_index, field.shape[1] - correlation_span_index) * scale[1])
        m = ax[0].contourf(xy_mg[0], xy_mg[1], field, levels=100)
        m_sample = ax[1].contourf(sample_xy_mg[0], sample_xy_mg[1], sample_field, levels=50)
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')

        plt.colorbar(m, ax=ax[0], label='Vx (m/s)')
        plt.colorbar(m_sample, ax=ax[1], label='Vx (m/s)')
        ax[0].vlines(sample_xy_mg[0].min(),
                     ymin=sample_xy_mg[1].min(),
                     ymax=sample_xy_mg[1].max(),
                     colors='r', linestyles='dashed')
        ax[0].vlines(sample_xy_mg[0].max(),
                     ymin=sample_xy_mg[1].min(),
                     ymax=sample_xy_mg[1].max(),
                     colors='r', linestyles='dashed')

        ax[0].hlines(sample_xy_mg[1].min(),
                     xmin=sample_xy_mg[1].min(),
                     xmax=sample_xy_mg[0].max(),
                     colors='r', linestyles='dashed')
        ax[0].hlines(sample_xy_mg[1].max(),
                     xmin=sample_xy_mg[1].min(),
                     xmax=sample_xy_mg[0].max(),
                     colors='r', linestyles='dashed')

        plt.show(block=True)
    output_shape = (2 * correlation_span_index + 1, 2 * correlation_span_index + 1)
    corr = np.empty(shape=output_shape)
    for delta_x in prange(output_shape[0]):
        for delta_y in prange(output_shape[1]):
            delta_indices = np.array([delta_x, delta_y])
            bottom_left_corner = [delta_indices[0],
                                  delta_indices[1]]
            top_right_corner = [bottom_left_corner[0] + sample_field.shape[0],
                                bottom_left_corner[1] + sample_field.shape[1]]
            current_slice = field[bottom_left_corner[0]: top_right_corner[0], bottom_left_corner[1]: top_right_corner[1]]
            current_curr = get_difference(current_slice, sample_field)
            corr[delta_x, delta_y] = current_curr

    deltas_mg = np.meshgrid(np.arange(-correlation_span_index, correlation_span_index + 1) * scale[0],
                            np.arange(-correlation_span_index, correlation_span_index + 1) * scale[1])

    return corr, deltas_mg


@njit(parallel=True)
def get_correlation_differences_from_sample_new3D(field, correlation_index_span, debug=True, plot_debug=False):
    #scale = np.array([scale, scale, scale])
    sample_field = field[correlation_index_span:-correlation_index_span, correlation_index_span:-correlation_index_span, correlation_index_span:-correlation_index_span]
    output_shape = (2 * correlation_index_span + 1, 2 * correlation_index_span + 1, 2 * correlation_index_span + 1)
    corr = np.empty(shape=output_shape)
    for delta_x in prange(output_shape[0]):
        for delta_y in prange(output_shape[1]):
            for delta_z in prange(output_shape[2]):
                delta_indices = np.array([delta_x, delta_y, delta_z])
                if debug:
                    if delta_z % 10 == 0:
                        print(delta_indices)
                bottom_left_corner = [delta_indices[0],
                                      delta_indices[1],
                                      delta_indices[2]]
                top_right_corner = [bottom_left_corner[0] + sample_field.shape[0],
                                    bottom_left_corner[1] + sample_field.shape[1],
                                    bottom_left_corner[2] + sample_field.shape[2]]

                current_slice = field[bottom_left_corner[0]: top_right_corner[0], bottom_left_corner[1]: top_right_corner[1], bottom_left_corner[2]: top_right_corner[2]]
                current_curr = get_difference(current_slice, sample_field)
                corr[delta_x, delta_y, delta_z] = current_curr

    return corr

@njit(parallel=True)
def get_correlation_differences_from_sample_3D(field, correlation_span, scale=1.0, debug=True, plot_debug=False):
    #scale = np.array([scale, scale, scale])
    correlation_span_index = round(correlation_span / scale)
    sample_field = field[correlation_span_index:-correlation_span_index, correlation_span_index:-correlation_span_index, correlation_span_index:-correlation_span_index]
    output_shape = (2 * correlation_span_index + 1, 2 * correlation_span_index + 1, 2 * correlation_span_index + 1)
    corr = np.empty(shape=output_shape)
    for delta_x in prange(output_shape[0]):
        for delta_y in prange(output_shape[1]):
            for delta_z in prange(output_shape[2]):
                delta_indices = np.array([delta_x, delta_y, delta_z])
                if debug:
                    if delta_z % 10 == 0:
                        print(delta_indices)
                bottom_left_corner = [delta_indices[0],
                                      delta_indices[1],
                                      delta_indices[2]]
                top_right_corner = [bottom_left_corner[0] + sample_field.shape[0],
                                    bottom_left_corner[1] + sample_field.shape[1],
                                    bottom_left_corner[2] + sample_field.shape[2]]

                current_slice = field[bottom_left_corner[0]: top_right_corner[0], bottom_left_corner[1]: top_right_corner[1], bottom_left_corner[2]: top_right_corner[2]]
                current_curr = get_difference(current_slice, sample_field)
                corr[delta_x, delta_y, delta_z] = current_curr

    deltas_mg_tuple = get_numba_meshgrid_3D(np.arange(-correlation_span_index,
                                                correlation_span_index + 1).astype(np.float64) * scale,
                                      np.arange(-correlation_span_index,
                                                correlation_span_index + 1).astype(np.float64) * scale,
                                      np.arange(-correlation_span_index,
                                                correlation_span_index + 1).astype(np.float64) * scale)
    deltas_mg = np.empty(shape=(3, deltas_mg_tuple[0].shape[0], deltas_mg_tuple[0].shape[1], deltas_mg_tuple[0].shape[2]),
                         dtype=deltas_mg_tuple[0].dtype)
    for i in np.arange(len(deltas_mg_tuple)):
        deltas_mg[i] = (deltas_mg_tuple[i] - correlation_span_index) * scale
    return corr, deltas_mg


@njit(parallel=True)
def get_correlation_differences_from_sample_in_time(field, correlation_span, scale=1.0, debug=True, plot_debug=False):
    #scale = np.array([scale, scale, scale])
    correlation_span_index = round(correlation_span / scale)
    sample_field = field[correlation_span_index:-correlation_span_index, correlation_span_index:-correlation_span_index, correlation_span_index:-correlation_span_index]
    output_shape = (2 * correlation_span_index + 1, 2 * correlation_span_index + 1, 2 * correlation_span_index + 1)
    corr = np.empty(shape=output_shape)
    for delta_x in prange(output_shape[0]):
        for delta_y in prange(output_shape[1]):
            for delta_z in prange(output_shape[2]):
                delta_indices = np.array([delta_x, delta_y, delta_z])
                if debug:
                    if delta_z % 10 == 0:
                        print(delta_indices)
                bottom_left_corner = [delta_indices[0],
                                      delta_indices[1],
                                      delta_indices[2]]
                top_right_corner = [bottom_left_corner[0] + sample_field.shape[0],
                                    bottom_left_corner[1] + sample_field.shape[1],
                                    bottom_left_corner[2] + sample_field.shape[2]]

                current_slice = field[bottom_left_corner[0]: top_right_corner[0], bottom_left_corner[1]: top_right_corner[1], bottom_left_corner[2]: top_right_corner[2]]
                current_curr = get_difference(current_slice, sample_field)
                corr[delta_x, delta_y, delta_z] = current_curr

    deltas_mg_tuple = get_numba_meshgrid_3D(np.arange(-correlation_span_index,
                                                correlation_span_index + 1).astype(np.float64) * scale,
                                      np.arange(-correlation_span_index,
                                                correlation_span_index + 1).astype(np.float64) * scale,
                                      np.arange(-correlation_span_index,
                                                correlation_span_index + 1).astype(np.float64) * scale)
    deltas_mg = np.empty(shape=(3, deltas_mg_tuple[0].shape[0], deltas_mg_tuple[0].shape[1], deltas_mg_tuple[0].shape[2]),
                         dtype=deltas_mg_tuple[0].dtype)
    for i in np.arange(len(deltas_mg_tuple)):
        deltas_mg[i] = (deltas_mg_tuple[i] - correlation_span_index) * scale
    return corr, deltas_mg


@njit(parallel=True)
def get_difference_sweep_3d(field, extended_field, flatten_indices, N_array, corr, debug):
    output_shape = np.array(extended_field.shape, dtype=np.int32) - np.array(field.shape, dtype=np.int32) + 1

    for delta_x in prange(output_shape[0]):
        for delta_y in prange(output_shape[1]):
            for delta_z in prange(output_shape[2]):
                delta_indices = np.array([delta_x, delta_y, delta_z])
                if debug:
                    print(delta_indices)
                #current_slice = tuple((slice(delta_indices[i], delta_indices[i] + n ) for i, n in enumerate(N_array) ))
                current_extended_field = extended_field[delta_indices[0]: delta_indices[0] + N_array[0], delta_indices[1]: delta_indices[1] + N_array[1], delta_indices[2]: delta_indices[2] + N_array[2]]
                current_curr = get_difference(current_extended_field, field)

                corr[delta_indices[0], delta_indices[1], delta_indices[2]] = current_curr
    return corr

@njit
def get_difference(current_extended_field, field):
    return np.mean(np.abs(current_extended_field - field))

def get_radial_correlation_from_grid_correlation(correlation, deltas_array):

    delta_R_mg = np.sqrt(np.sum([delta_mg ** 2 for delta_mg in deltas_array],
                                axis=0))
    unique_delta_R = np.unique(delta_R_mg)

    unique_delta_R = np.sort(unique_delta_R)

    average_correlations = []
    std_average_correlations = []
    for displacement in unique_delta_R:
        mask_displacement = np.isclose(delta_R_mg, displacement)
        average_correlations.append((correlation[mask_displacement]).mean()
                                    )
        std_average_correlations.append(np.std(correlation[mask_displacement]) / np.sqrt(mask_displacement.sum()))

    average_correlations = np.array(average_correlations)
    std_average_correlations = np.array(std_average_correlations)

    return average_correlations, std_average_correlations, unique_delta_R


def get_radial_correlation(field, N_periods=3, scale=(1, 1)):
    corr, deltas_mg = get_correlation_differences(field, N_periods=N_periods, scale=scale)
    return get_radial_correlation_from_grid_correlation(corr, deltas_mg)
