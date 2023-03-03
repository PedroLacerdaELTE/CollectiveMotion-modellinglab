import warnings
from types import LambdaType

import numpy as np
from scipy.interpolate import UnivariateSpline

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import pandas as pd
from matplotlib import pyplot as plt

from calc.auxiliar import SplineWrapper, UnivariateSplineWrapper
from calc.stats import weighted_moving_average, weighted_moving_std, rolling_apply_multiple_columns, \
    weights_preprocessing, get_weighted_average, get_maximum_vertical_velocity2d, get_maximum_by_quadratic_fit


def get_maximum_vertical_velocity_by_moving_average(df_iteration, independent_var_col, vz_col, Z_col, window,
                                                    min_periods, n_sigma=0, debug=False):
    n_dim = len(independent_var_col)
    df_calc = df_iteration.sort_values(Z_col)
    if n_dim == 2:
        x_col, y_col = independent_var_col
        array_vz_max = rolling_apply_multiple_columns(get_maximum_vertical_velocity2d,
                                                      window, min_periods,
                                                      df_calc[x_col].values,
                                                      df_calc[y_col].values,
                                                      df_calc[vz_col].values,
                                                      df_calc[Z_col].values, n_sigma=n_sigma, debug=debug,
                                                      derivative_epsilon=None  # 1e-5
                                                      )

        df_vz_max = pd.DataFrame(array_vz_max, columns=['X_max',
                                                        'Y_max',
                                                        'dZdT_air_max',
                                                        'Z_avg',
                                                        'fit_coefficients',
                                                        'fit_pcov',
                                                        'N']
                                 )
        df_vz_max['X_max'] = df_vz_max['X_max'].astype(float)
        df_vz_max['Y_max'] = df_vz_max['Y_max'].astype(float)
        df_vz_max['dZdT_air_max'] = df_vz_max['dZdT_air_max'].astype(float)
        df_vz_max['Z_avg'] = df_vz_max['Z_avg'].astype(float)
        df_vz_max['N'] = df_vz_max['N'].astype(int)
    else:
        rho_col = independent_var_col
        array_vz_max = rolling_apply_multiple_columns(get_maximum_vertical_velocity,
                                                      window, min_periods,
                                                      df_calc[rho_col].values,
                                                      df_calc[vz_col].values,
                                                      df_calc[Z_col].values, n_sigma=n_sigma, debug=debug)

        df_vz_max = pd.DataFrame(array_vz_max,
                                         columns=['rho_max',
                                                  'dZdT_air_max',
                                                  'Z_avg',
                                                  'fit_coefficients',
                                                  'fit_pcov',
                                                  'N']
                                         )
        df_vz_max['rho_max'] = df_vz_max['x_max'].astype(float)
        df_vz_max['dZdT_air_max'] = df_vz_max['dZdT_air_max'].astype(float)
        df_vz_max['Z_avg'] = df_vz_max['Z_avg'].astype(float)
        df_vz_max['N'] = df_vz_max['N'].astype(int)

    # It's likely that there are several duplicates is the point that gets in and the point that gets out of the window
    # is filtered out and therefore has no consequence.
    df_vz_max.dropna(subset=['dZdT_air_max'], inplace=True)
    df_vz_max.drop_duplicates(subset=['dZdT_air_max', 'Z_avg'], inplace=True, ignore_index=True)

    return df_vz_max


def get_wind_from_thermal_core(df, vx_ground_col='dXdT_ground_avg',
                               vy_ground_col='dYdT_ground_avg',
                               vz_ground_col='dZdT_air_0_avg',
                               vz_air_max_col='dZdT_air_max',
                               x_col='X_avg', y_col='Y_avg', z_col='Z_avg',
                               sort_on='Z_avg',
                               method='velocities',
                               spline_parameters=None,
                               smoothing_ma_args=None,
                               debug=False):
    if smoothing_ma_args is None:
        smoothing_ma_args = {'window': 720,
                             'win_type': 'gaussian',
                             'window_args': {'std': 300},
                             'min_periods': 1,
                             'center': True}
    else:
        smoothing_ma_args = smoothing_ma_args.copy()

    if 'window_args' in smoothing_ma_args.keys():
        window_args = smoothing_ma_args.pop('window_args')
    else:
        window_args = {}

    df_calc = df.copy()
    df_calc = df_calc.drop_duplicates(subset=['Z_avg'], ignore_index=True)
    df_calc.sort_values('Z_avg', inplace=True)

    if method == 'velocities':
        df_calc['wind_X_raw'] = df_calc[vx_ground_col] / df_calc[vz_ground_col] * df_calc[vz_air_max_col]  # dXdt = dXdZ . dZdT
        df_calc['wind_Y_raw'] = df_calc[vy_ground_col] / df_calc[vz_ground_col] * df_calc[vz_air_max_col]
    else:
        N = df_calc['Z_avg'].size
        if spline_parameters is None:
            spline_parameters = {'degree': 1,
                                 'smoothing_factor': 0.1
                                 }
        for i_col, coord_col in enumerate([x_col, y_col]):

            weights = df_calc['X_sem'].values if i_col == 0 else df_calc['Y_sem'].values

            weights = 1 / weights

            merging_spline = UnivariateSplineWrapper(df_calc[z_col].values,
                                                     df_calc[coord_col].values,
                                                     w=weights,
                                                     degree=spline_parameters['degree'],
                                                     s=spline_parameters['smoothing_factor'] * N,
                                                     weight_normalization=spline_parameters['weight_normalization']
                                                     )

            df_calc[['dXdZ', 'dYdZ'][i_col]] = merging_spline(df_calc[z_col].values, nu=1,
                                                              extrapolate=spline_parameters['extrapolate'])
        for coord in ['X', 'Y']:
            df_calc[f'wind_{coord}_raw'] = df_calc[f'd{coord}dZ'] * df_calc[f'dZdT_air_max']

    df_calc['wind_X'] = df_calc['wind_X_raw'].rolling(**smoothing_ma_args).mean(**window_args)
    df_calc['wind_Y'] = df_calc['wind_Y_raw'].rolling(**smoothing_ma_args).mean(**window_args)

    if debug:
        from plotting.decomposition.debug import debug_plot_decomposition_wind
        debug_plot_decomposition_wind(df_calc)

    df_calc.dropna(inplace=True)
    return df_calc


def get_wind_from_thermal_core_regular(df, position_cols=None, vz_air_col='dZdT_air_i_max',
                                       smoothing_ma_args=None, vz_max=10, debug=False):
    if smoothing_ma_args is None:
        smoothing_ma_args = {'window': 720,
                             'win_type': 'gaussian',
                             'window_args': {'std': 300},
                             'min_periods': 1,
                             'center': True}
    else:
        smoothing_ma_args = smoothing_ma_args.copy()

    if 'window_args' in smoothing_ma_args.keys():
        window_args = smoothing_ma_args.pop('window_args')
    else:
        window_args = {}

    if position_cols is None:
        position_cols = ['X_thermal_avg',
                         'Y_thermal_avg',
                         'Z_thermal_avg']
    df_calc = df.copy()
    df_calc = df_calc.sort_values([position_cols[-1]])

    df_calc['dX'] = df_calc[position_cols[0]].diff()
    df_calc['dX_smoothed'] = df_calc['dX'].rolling(**smoothing_ma_args).mean(**window_args)
    df_calc['dY'] = df_calc[position_cols[1]].diff()
    df_calc['dY_smoothed'] = df_calc['dY'].rolling(**smoothing_ma_args).mean(**window_args)
    df_calc['dZ'] = df_calc[position_cols[2]].diff()
    df_calc['dZ_smoothed'] = df_calc['dZ'].rolling(**smoothing_ma_args).mean(**window_args)

    df_calc['dXdZ'] = df_calc['dX_smoothed'] / df_calc['dZ_smoothed']
    df_calc['dYdZ'] = df_calc['dY_smoothed'] / df_calc['dZ_smoothed']

    dxdy_dz_spline = SplineWrapper(df_calc[['dXdZ', 'dYdZY']].values,
                                   df_calc[position_cols[-1]]).fit()

    z_array = np.linspace(df_calc[position_cols[2]].min(), df_calc[position_cols[2]].max(),
                          df_calc[position_cols[2]].count())
    df_calc['dXdZ_regular'] = dxdy_dz_spline(z_array)
    df_calc['dYdZ_regular'] = dxdy_dz_spline(z_array)
    df_calc['wind_X'] = df_calc['dXdZ_regular'] * np.clip(df_calc[vz_air_col], -vz_max,
                                                          vz_max)  # deltaX / deltat = deltaX/deltaZ . max(dZ/dT_air)_1080
    df_calc['wind_Y'] = df_calc['dYdZ_regular'] * np.clip(df_calc[vz_air_col], -vz_max,
                                                          vz_max)  # dZ/dT_air = dZ/dT_ground - dZ/dT_bird

    if not debug:
        df_calc = df_calc[['wind_X', 'wind_Y']]
    df_calc = df_calc[['wind_X_avg', 'wind_Y_avg']]
    return df_calc


def get_thermal_core_by_weighted_average(df, columns_and_weights, max_col=None, sort_by='Z', filter_col=None,
                                         avg_suffix='_avg', std_suffix='_std', sem_suffix='_sem', count_suffix='_N',
                                         ma_args=None):
    df_calc = df.copy()
    if filter_col is not None:
        df_calc = df_calc[df_calc[filter_col]]
    df_return = pd.DataFrame()

    if ma_args is None:
        ma_args = {'window': 36,
                   'min_periods': 1,
                   'center': True}

    df_calc.sort_values(by=[sort_by], inplace=True)

    # Moving Weighted Average
    for value_col, weights in columns_and_weights.items():
        current_df = df_calc.copy()

        if isinstance(weights, LambdaType):
            current_df['weights'] = current_df.apply(weights, axis=1)
        elif isinstance(weights, str):
            current_df['weights'] = current_df[weights]
        else:
            current_df['weights'] = weights

        # Deal with NA and negative values on the weight values
        current_df['weights'] = weights_preprocessing(current_df['weights'].values)

        array_vz_max = rolling_apply_multiple_columns(get_weighted_average,
                                                      ma_args['window'],
                                                      ma_args['min_periods'],
                                                      current_df[value_col].values,
                                                      current_df['weights'].values,
                                                      drop_na=True)

        df_return[value_col + avg_suffix] = array_vz_max[:, 0]
        df_return[value_col + std_suffix] = array_vz_max[:, 1]
        df_return[value_col + sem_suffix] = array_vz_max[:, 2]
        df_return[value_col + count_suffix] = array_vz_max[:, 3]

    if max_col is not None:
        if 'win_type' in ma_args.keys():
            ma_args.pop('win_type')
        if 'window_args' in ma_args.keys():
            ma_args.pop('window_args')
        df_return[max_col + '_max'] = df_calc[max_col].rolling(**ma_args).max().values

    df_return.dropna(inplace=True)
    df_return.drop_duplicates(inplace=True)
    return df_return


def merge_dfs_with_spline(df, df_other, other_cols_to_merge, other_merge_on, merge_on,
                          spline_degree=1, smoothing_factor=0, extrapolate=0, weight_normalization=False, debug=False):
    if other_cols_to_merge is None:
        other_cols_to_merge = df_other.columns

    df_calc = df.copy()
    df_other_calc = df_other.copy()

    df_other_calc.drop_duplicates(subset=other_merge_on, inplace=True, ignore_index=True)
    df_other_calc.sort_values([other_merge_on], inplace=True)
    # The amount of smoothness is determined by satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s,
    # So if the weights w are normalized, that sum will be of the order of N, so the smoothing_factor should be of the
    # order of 1
    if smoothing_factor is None:
        smoothing = df_other_calc[other_merge_on].size
    else:
        smoothing = smoothing_factor * df_other_calc[other_merge_on].size

    # Weighted spline
    merging_spline = {}
    if isinstance(other_cols_to_merge, dict):
        for col, w in other_cols_to_merge.items():

            if isinstance(w, LambdaType):
                weights = df_other_calc.apply(w, axis=1).values
            elif isinstance(w, str):
                weights = df_other_calc[w].values
            else:
                raise TypeError
            weights = np.array(weights)

            merging_spline[col] = UnivariateSplineWrapper(df_other_calc[other_merge_on].values.T,
                                                          df_other_calc[col].values,
                                                          degree=spline_degree, s=smoothing, w=weights,
                                                          weight_normalization=weight_normalization
                                                          )
            df_calc[col] = merging_spline[col](df_calc[merge_on].values, extrapolate=extrapolate)

    else:  # Not Weighted Spline
        for col in other_cols_to_merge:
            merging_spline[col] = UnivariateSplineWrapper(df_other_calc[other_merge_on].values.T,
                                                          df_other_calc[col].values,
                                                          degree=spline_degree, s=smoothing,
                                                          )
            df_calc[col] = merging_spline[col](df_calc[merge_on].values,  extrapolate=extrapolate)
    if debug:
        for i_col, col in enumerate(other_cols_to_merge):
            import matplotlib
            matplotlib.use('QtAgg')
            import matplotlib.pyplot as plt
            x_values = df_calc.sort_values(merge_on)[merge_on]
            plt.plot(x_values,
                     merging_spline[col](x_values)
                     )
            plt.xlabel(merge_on)
            plt.ylabel(col)
            plt.show(block=True)
    spline_stats = {col: {'tck': spl.get_tck(),
                          'residuals': spl.get_residual()}
                    for col, spl in merging_spline.items()}
    return df_calc, spline_stats


def get_gaussian_thermal_from_position(x_relative, y_relative, z, A, radius):
    try:
        distance = np.sqrt(x_relative ** 2 + y_relative ** 2)
        Vz = A * np.exp(-np.power(distance, 2.) / (2 * np.power(radius, 2.)))

    except RuntimeWarning as e:
        print(e)
    else:
        return Vz


def get_paraboloid_thermal_from_position(x_relative, y_relative, z_pos, A, radius):
    a = - A / radius ** 2
    b = a
    c = 0
    d = A
    Vz = a * (x_relative) ** 2 + b * (y_relative) ** 2 + c * (x_relative) * (y_relative) + A

    if Vz < 0:
        Vz = 0
    return Vz


def get_predefined_thermal_model(model):
    if model == 'gaussian':
        return get_gaussian_thermal_from_position
    elif model == 'paraboloid':
        return get_paraboloid_thermal_from_position


def get_thermal_from_position(x_pos, y_pos, z_pos, model='guassian', **args):
    if model == 'gaussian':
        return get_gaussian_thermal_from_position(x_pos, y_pos, z_pos, **args)
    elif model == 'paraboloid':
        return get_paraboloid_thermal_from_position(x_pos, y_pos, z_pos, **args)


def get_thermal_average_speed(z_core, model='gaussian', **thermal_parameters):
    if model == 'gaussian':
        return thermal_parameters['A'] / 4
    elif model == 'paraboloid':
        return thermal_parameters['A'] / 8
    else:
        return thermal_parameters['A'] / 2


def get_thermal_core(wind, z_pos, model='gaussian', **thermal_parameters):
    dz = 0.1
    z_core = np.arange(0.1, z_pos, dz)
    thermal_parameters_updated = thermal_parameters.copy()
    thermal_parameters_updated.update({'x_core': 0, 'y_core': 0})
    vz = get_thermal_average_speed(z_core, model=model, **thermal_parameters_updated)

    wind_x = []
    wind_y = []
    for z in z_core:
        current_wind = wind(z)
        wind_x.append(current_wind[0])
        wind_y.append(current_wind[1])

    wind_x = np.array(wind_x)
    wind_y = np.array(wind_y)

    dx = wind_x / vz * dz
    x_core = np.cumsum(dx)

    dy = wind_y / vz * dz
    y_core = np.cumsum(dy)

    # from scipy.interpolate import splprep, splev
    # tck, u = splprep([x_core, y_core, z_core], s=0)
    # new_points = splev(u, tck)

    return x_core, y_core, z_core


def get_random_walk_wind_space_time(wind_avg, wind_std, z_max, x_max, time_max, n_steps=200):

    x_values = np.linspace(-x_max, x_max, n_steps)
    z_values = np.linspace(0, z_max, n_steps)
    t_values = np.linspace(0, time_max, n_steps)
    mgs = np.meshgrid(x_values, z_values,t_values)

    xyzt_array = np.stack([mg.flatten() for mg in mgs], axis=-1)

    for i in range(n_steps * n_steps):
        wind_x_values_avg = wind_avg[0]
        wind_x_values = wind_std[0] * np.random.standard_normal(n_steps).cumsum()
        wind_x_values = wind_x_values_avg + (wind_x_values - wind_x_values.mean())

        wind_y_values_avg = wind_avg[1]

        wind_y_values = wind_std[1] * np.random.standard_normal(n_steps).cumsum()
        wind_y_values = wind_y_values_avg + (wind_y_values - wind_y_values.mean())

        current_wind_values = np.vstack([wind_x_values, wind_y_values]).T
        if i == 0:
            wind_values = current_wind_values
        else:
            wind_values = np.vstack([wind_values, current_wind_values])


    return xyzt_array, wind_values


def get_ND_random_walk(mean, std, n_vars, n_steps=200, shape=2):

    RW_values = mean + (std * np.random.standard_normal((n_steps ** n_vars, shape))).cumsum(axis=0)
    RW_values = mean + (RW_values - RW_values.mean())

    return RW_values


def get_4D_random_walk_wind(mean, std, limits, n_steps=200):

    mgs = np.meshgrid(*[np.linspace(*limits, n_steps) for limits in limits.values()])
    used_vars = list(limits.keys())
    used_vars = ''.join(used_vars)

    xyzt_array = np.stack([mg.flatten() for mg in mgs], axis=-1)

    for i in range(len(used_vars)):
        wind_x_values_avg = mean[0]
        wind_x_values = std[0] * np.random.standard_normal(n_steps).cumsum()
        wind_x_values = wind_x_values_avg + (wind_x_values - wind_x_values.mean())

        wind_y_values_avg = mean[1]

        wind_y_values = std[1] * np.random.standard_normal(n_steps).cumsum()
        wind_y_values = wind_y_values_avg + (wind_y_values - wind_y_values.mean())

        current_wind_values = np.vstack([wind_x_values, wind_y_values]).T
        if i == 0:
            wind_values = current_wind_values
        else:
            wind_values = np.vstack([wind_values, current_wind_values])

    if len(used_vars) == 1:
        xyzt_array = xyzt_array.flatten()
    return xyzt_array, wind_values, used_vars


def get_random_walk_wind_with_time(wind_avg, wind_std, z_max, x_max, time_max, n_steps=200):

    x_values = np.linspace(-x_max, x_max, n_steps)
    z_values = np.linspace(0, z_max, n_steps)
    t_values = np.linspace(0, time_max, n_steps)
    mgs = np.meshgrid(x_values, z_values,t_values)

    xyzt_array = np.stack([mg.flatten() for mg in mgs], axis=-1)

    for i in range(n_steps * n_steps):
        wind_x_values_avg = wind_avg[0]
        wind_x_values = wind_std[0] * np.random.standard_normal(n_steps).cumsum()
        wind_x_values = wind_x_values_avg + (wind_x_values - wind_x_values.mean())

        wind_y_values_avg = wind_avg[1]

        wind_y_values = wind_std[1] * np.random.standard_normal(n_steps).cumsum()
        wind_y_values = wind_y_values_avg + (wind_y_values - wind_y_values.mean())

        current_wind_values = np.vstack([wind_x_values, wind_y_values]).T
        if i == 0:
            wind_values = current_wind_values
        else:
            wind_values = np.vstack([wind_values, current_wind_values])


    return xyzt_array, wind_values


def get_random_walk_wind(wind_avg, wind_std, z_max, n_steps=200):

    wind_x_values_avg = wind_avg[0]
    wind_x_values = wind_std[0] * np.random.standard_normal(n_steps).cumsum()
    wind_x_values = wind_x_values_avg + (wind_x_values - wind_x_values.mean())

    wind_y_values_avg = wind_avg[1]

    wind_y_values = wind_std[1] * np.random.standard_normal(n_steps).cumsum()
    wind_y_values = wind_y_values_avg + (wind_y_values - wind_y_values.mean())

    z_values = np.linspace(0, z_max, n_steps)

    return np.vstack([z_values, wind_x_values, wind_y_values])


def get_hierarchical_turbulence(X, t, scales_dict, list_of_interpolator, interpolator_parameters):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    n_dim = X.ndim
    if X.ndim == 1:
        X = X.reshape(1, X.shape[0])
    output_shape = np.hstack([X.shape[:-1], [len(list_of_interpolator), len(scales_dict)]])
    results = np.empty(shape=output_shape, dtype=float)

    for j in range(len(list_of_interpolator)):  # loop on the three components of velocity
        for i, (spatial_scale, velocity_scale) in enumerate(scales_dict.items()):

            X_alt = np.mod(X, spatial_scale) * interpolator_parameters['spatial_scale'] / spatial_scale
            Xt_shape = np.hstack([X.shape[:-1], 4])
            Xt = np.empty(shape=Xt_shape)
            Xt[..., 1:] = X_alt
            Xt[:, 0] = t
            results[..., j, i] = list_of_interpolator[j](Xt) * velocity_scale / interpolator_parameters['velocity_scale']
        #print(3)
    if len(list_of_interpolator) == 1:
        results = results[..., 0, :]
    if n_dim == 1:
        results = results[0]
    return results


def get_maximum_vertical_velocity(rho_array, Vz_array, Z_array, n_sigma=None, b_epsilon=None, debug=False):
    rho_array_mask = np.logical_not(np.isnan(rho_array))
    Vz_array_mask = np.logical_not(np.isnan(Vz_array))
    Z_array_mask = np.logical_not(np.isnan(Z_array))

    na_mask = np.logical_and(rho_array_mask, Vz_array_mask, Z_array_mask)
    if b_epsilon is None:
        b_epsilon = np.inf
    if n_sigma is not None:
        rho_median = np.median(rho_array)
        rho_std = np.std(rho_array)
        mask = rho_array < rho_median + n_sigma * rho_std
        mask = np.logical_and(mask, na_mask)
    else:
        mask = na_mask
    Z_avg = np.mean(Z_array)

    N = mask.sum()

    rho_max, dZdT_air_max, popt, pcov, is_maximum = get_maximum_by_quadratic_fit(rho_array[mask],
                                                                     Vz_array[mask],
                                                                     bounds=([-np.inf, -b_epsilon, -np.inf],
                                                                             [np.inf, b_epsilon, np.inf]))

    if debug and Z_avg > 300:
        from plotting.decomposition.debug import debug_plot_decomposition_get_maximum_vertical_velocity
        debug_plot_decomposition_get_maximum_vertical_velocity(rho_array, Vz_array, mask, rho_max, dZdT_air_max, popt)
    if not is_maximum:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return rho_max, dZdT_air_max, Z_avg, popt, pcov, N


def get_rolling_maximum_vertical_velocity(df_iteration, radius_col='rho_thermal', Vz_col='dZdT_air_0',
                                          sort_by_col='Z_thermal',
                                          ma_args=None, debug=False):
    ma_args = ma_args.copy()

    if ma_args is None:
        ma_args = {'window': 12, 'min_periods': 1, 'center': True}

    if 'window_args' in ma_args.keys():
        window_args = ma_args.pop('window_args')
    else:
        window_args = {}

    df_calc = df_iteration.copy()
    df_calc = df_calc.sort_values(sort_by_col)

    # Remove outliers that strongly bias the fit
    df_calc = df_calc.dropna()

    df_return = pd.DataFrame()

    # min_periods to window_size
    for i in range(ma_args['min_periods'], ma_args['window']):
        current_row = {}
        current_df = df_calc.copy().iloc[0: i]

        rho_max_return, dZdT_air_max_return, rho_max, dZdT_air_max, (a, b, c), status = get_maximum_vertical_velocity(
            current_df[radius_col], current_df[Vz_col])
        current_row['Z_avg'] = current_df['Z_thermal'].mean()
        current_row['rho_max'] = rho_max_return
        current_row['dZdT_air_max'] = dZdT_air_max_return

        if debug:
            current_row['real_rho_max'] = rho_max
            current_row['real_dZdT_air_max'] = dZdT_air_max
            current_row['quadratic_fit'] = (a, b, c)
            current_row['status'] = status

        df_return = df_return.append(current_row, ignore_index=True)

    for i in range(len(df_calc.index) - ma_args['window']):
        # print(i)
        current_row = {}
        current_df = df_calc.copy().iloc[i: i + ma_args['window']]

        rho_max_return, dZdT_air_max_return, rho_max, dZdT_air_max, (a, b, c), status = get_maximum_vertical_velocity(
            current_df[radius_col], current_df[Vz_col])
        current_row['Z_avg'] = current_df['Z_thermal'].mean()
        current_row['rho_max'] = rho_max_return
        current_row['dZdT_air_max'] = dZdT_air_max_return

        if debug:
            current_row['real_rho_max'] = rho_max
            current_row['real_dZdT_air_max'] = dZdT_air_max
            current_row['quadratic_fit'] = (a, b, c)
            current_row['status'] = status

        df_return = df_return.append(current_row, ignore_index=True)

    # From window_size to min_periods
    for i in range(ma_args['window']):
        current_row = {}
        current_df = df_calc.copy().iloc[-ma_args['window'] + i:]
        if len(current_df.index) < ma_args['min_periods']:
            break

        rho_max_return, dZdT_air_max_return, rho_max, dZdT_air_max, (a, b, c), status = get_maximum_vertical_velocity(
            current_df[radius_col], current_df[Vz_col])
        current_row['Z_avg'] = current_df['Z_thermal'].mean()
        current_row['rho_max'] = rho_max_return
        current_row['dZdT_air_max'] = dZdT_air_max_return

        if debug:
            current_row['real_rho_max'] = rho_max
            current_row['real_dZdT_air_max'] = dZdT_air_max
            current_row['quadratic_fit'] = (a, b, c)
            current_row['status'] = status

        df_return = df_return.append(current_row, ignore_index=True)

    return df_return
