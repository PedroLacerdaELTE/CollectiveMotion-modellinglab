import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev, splrep, UnivariateSpline, RegularGridInterpolator

from calc.flight import get_bank_angle_from_radius, \
    get_min_sink_rate_from_bank_angle, get_horizontal_velocity_from_bird_parameters
from calc.geometry import pf_get_curvature_from_trajectory, pf_get_radius_from_curvature


def get_regular_grid_from_irregular_data(x_array, y_array, *dependent_arrays, resolution):

    from matplotlib import tri
    xi = np.linspace(np.min(x_array) - 0.1,
                     np.max(x_array) + 0.1, resolution, endpoint=True)
    yi = np.linspace(np.min(y_array) - 0.1,
                     np.max(y_array) + 0.1, resolution, endpoint=True)
    Xi, Yi = np.meshgrid(xi, yi)

    triang = tri.Triangulation(x_array.flatten(),
                               y_array.flatten())
    interpolators = [tri.LinearTriInterpolator(triang, current_array.flatten())
                     for current_array in dependent_arrays if current_array is not None]

    v_i = [interpolator(Xi, Yi) for interpolator in interpolators]

    return Xi, Yi, v_i


def parse_projection_string(projection_str):
    # Parse cross-section type
    projection_str = projection_str.lower()
    first_var, second_var = projection_str

    first_index = 'xyz'.index(first_var)
    second_index = 'xyz'.index(second_var)

    section_index = int('012'.replace(str(first_index), '').replace(str(second_index), ''))
    section_var = 'xyz'[section_index]

    return first_var, second_var, first_index, second_index, section_index, section_var


class SplineWrapper(object):
    def __init__(self, X, y, degree=3, s=0, w=None):
        self.X = X
        self.y = y
        self.s = s
        self.w = w
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        self.degree = degree
        self.tck = None

        if isinstance(self.X, (pd.Series, pd.DataFrame)):
            self.X = self.X.values

        if isinstance(self.y, (pd.Series, pd.DataFrame)):
            self.y = self.y.values

    def fit(self):
        if self.X.ndim == 1:
            self.tck = splrep(y=self.X, x=self.y, w=self.w, s=self.s, k=self.degree)
        else:
            self.tck, _ = splprep(x=self.X, u=self.y, w=self.w, s=self.s, k=self.degree)

    def __call__(self, x, der=0, extrapolate=0):
        if self.tck is None:
            raise RuntimeError('must run fit method first')
        result = splev(x=x, tck=self.tck, der=der, ext=extrapolate)
        return np.array(result)


class UnivariateSplineWrapper(UnivariateSpline):
    def __init__(self, x, y, degree=3, s=0, w=None, weight_normalization=False):
        super().__init__(x, y, k=degree, w=w, s=s)
        self.x = x
        self.degree = degree
        self.y = y
        self.s = s
        self.w = w
        self.weight_normalization = weight_normalization
        if self.weight_normalization and (self.w is not None):
            self.w = self.w / np.sum(self.w)
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.degree = degree
        self.tck = None

    @classmethod
    def from_tck(cls, tck, ext=0):
        self = super()._from_tck(tck, ext=ext)
        self.x = None
        self.degree = tck[-1]
        self.y = None
        self.s = None
        self.w = None
        self.weight_normalization = None
        if self.weight_normalization and (self.w is not None):
            self.w = self.w / np.sum(self.w)
        self.x_min = np.min(self.get_knots())
        self.x_max = np.max(self.get_knots())

        return self

    def _linear_interpolation(self, x, x0, nu):

        f0 = super().__call__(x0, nu=nu, ext=0)
        slope = super().__call__(x0, nu=nu + 1, ext=0)
        result = f0 + slope * (x - x0)
        return result

    def get_tck(self):
        return self._eval_args

    def __call__(self, x, nu=0, extrapolate=0):
        if isinstance(x, (np.ndarray, list)):
            if extrapolate == 'linear':
                result = list(map(lambda elem: self(elem, nu=nu, extrapolate=extrapolate), x))
            else:
                result = super().__call__(x, nu=nu, ext=extrapolate)
        elif extrapolate == 'linear':
            if x < self.x_min:
                result = self._linear_interpolation(x, self.x_min, nu)
            elif x > self.x_max:
                result = self._linear_interpolation(x, self.x_max, nu)
            else:
                result = super().__call__(x, nu=nu, ext=0)
        else:
            result = super().__call__(x, nu=nu, ext=extrapolate)

        return np.array(result)


def get_geometric_characteristics(df_track, state_vector_columns=None):
    df_calc = df_track.copy()

    if state_vector_columns is None:
        state_vector_columns = {'X': 'X_bird',
                                'Y': 'Y_bird',
                                'Z': 'Z_bird',
                                'Vx': 'dXdT_bird',
                                'Vy': 'dYdT_bird',
                                'Ax': 'd2XdT2_bird',
                                'Ay': 'd2YdT2_bird'}
    df_calc['angle'] = (np.arctan2(df_calc[state_vector_columns['Y']],
                                   df_calc[state_vector_columns['X']]
                                   )
                        + np.pi) * 180 / np.pi

    df_calc['new_circle'] = (((df_calc['angle'] < df_calc.shift(1)['angle']) & (
            df_calc['angle'].shift(1) > df_calc.shift(2)['angle'])) |
                             ((df_calc['angle'] > df_calc.shift(1)['angle']) & (
                                     df_calc['angle'].shift(1) < df_calc.shift(2)['angle']))
                             )

    df_calc['curvature'] = df_calc.apply(pf_get_curvature_from_trajectory, args=(state_vector_columns['Vx'],
                                                                                 state_vector_columns['Vy'],
                                                                                 state_vector_columns['Ax'],
                                                                                 state_vector_columns['Ay']), axis=1)

    df_calc['radius_raw'] = df_calc.apply(pf_get_radius_from_curvature, args=('curvature',), axis=1)
    list_of_birds = df_calc['bird_name'].unique()
    df_temp = pd.Series(dtype=np.float64)
    for bird in list_of_birds:
        current_bird = df_calc[df_calc['bird_name'] == bird]
        df_temp = pd.concat([df_temp, current_bird['radius_raw'].rolling(3, center=True, min_periods=1,
                                                                         win_type='gaussian').mean(std=1)
                             ])
    df_calc['radius'] = df_temp

    df_track[['angle', 'new_circle', 'curvature', 'radius', 'radius_raw']] = df_calc[['angle', 'new_circle',
                                                                                      'curvature', 'radius_raw',
                                                                                      'radius']]

    return df_track


def get_flight_characteristics(df_track, bird_parameters, state_vector_columns=None, radius_col='radius'):
    if state_vector_columns is None:
        state_vector_columns = {'X': 'X_bird',
                                'Y': 'Y_bird',
                                'Z': 'Z_bird',
                                'Vx': 'dXdT_bird',
                                'Vy': 'dYdT_bird',
                                'Vz': 'dZdT_bird'}
    df_calc = pd.DataFrame()

    df_calc['bank_angle'] = df_track[radius_col].apply(get_bank_angle_from_radius, **bird_parameters)

    df_calc['Vh'] = df_calc['bank_angle'].apply(get_horizontal_velocity_from_bird_parameters,
                                                **{'mass': bird_parameters['mass'],
                                                   'wing_area': bird_parameters['wing_area'],
                                                   'CL': bird_parameters['CL']})

    df_calc['min_sink_rate'] = df_calc['bank_angle'].apply(get_min_sink_rate_from_bank_angle,
                                                           **bird_parameters)

    df_calc['Vx'] = df_track[state_vector_columns['Vx']] / np.sqrt(df_track[state_vector_columns['Vx']] ** 2
                                                                   + df_track[state_vector_columns['Vy']] ** 2)
    df_calc['Vx'] = df_calc['Vx'] * df_calc['Vh']

    df_calc['Vy'] = df_track[state_vector_columns['Vy']] / np.sqrt(df_track[state_vector_columns['Vx']] ** 2
                                                                   + df_track[state_vector_columns['Vy']] ** 2)
    df_calc['Vy'] = df_calc['Vy'] * df_calc['Vh']

    return df_calc


def get_3_point_stencil_differentiation(data_array, dt, n=1):
    assert n in [1, 2], 'the order of differentiation n must be 1, 2, 3 or 4'
    if n == 1:
        weights = np.array([1, 0, -1])
        weights = weights / (2 * dt)
    elif n == 2:
        weights = np.array([1, -2, 1])
        weights = weights / (dt ** 2)
    N = data_array.size
    M = weights.size
    result_size = max(M, N) - min(M, N) + 1
    # numba does not support numpy.convolve with the 'mode' argument.
    # Therefore we need to cut the unwanted data manually
    result = np.empty(shape=result_size, dtype=float)
    result = np.convolve(data_array, weights,
                         #mode='valid'
                         )[2:-2]
    return result


def get_5_point_stencil_differentiation(data_array, dt, n=1):
    assert n in [1, 2, 3, 4], 'the order of differentiation n must be 1, 2, 3 or 4'
    if n == 1:
        weights = np.array([-1, 8, 0, -8, 1])
        weights = weights / (12 * dt)
    elif n == 2:
        weights = np.array([-1, 16, -30, 16, -1])
        weights = weights / (12 * dt ** 2)
    elif n == 3:
        weights = np.array([1, -2, 0, 2, -1])
        weights = weights / (2 * dt ** 3)
    elif n == 4:
        weights = np.array([1, -4, 6, -4, 1])
        weights = weights / (1 * dt ** 4)
    N = data_array.size
    M = weights.size
    result_size = max(M, N) - min(M, N) + 1
    # numba does not support numpy.convolve with the 'mode' argument.
    # Therefore we need to cut the unwanted data manually
    result = np.empty(shape=result_size, dtype=float)
    result = np.convolve(data_array, weights,
                         #mode='valid'
                         )[4:-4]
    return result


def get_smoothed_diff_per_partition(df, moving_average_params, dt, n=1, method='5point', partition_key='bird_name'):
    column_names_to_eval = list(moving_average_params.keys())
    unique_partitions = df[partition_key].unique()
    results_dict = {col: np.array([]) for col in column_names_to_eval}
    df_calc = df[column_names_to_eval + ['time']]
    for idx, partition_value in enumerate(unique_partitions):
        df_filter = df_calc.loc[df[partition_key] == partition_value]

        for col, ma_args in moving_average_params.items():
            # diff
            if method == '5point':
                current_diff = get_5_point_stencil_differentiation(df_filter[col].values, dt=dt, n=n)

                diff = df_filter[col].diff().values/dt
                current_diff = np.concatenate((diff[:2], current_diff, diff[-2:]))  # diff[0] = nan
            elif method == '3point':
                current_diff = get_3_point_stencil_differentiation(df_filter[col].values, dt=dt, n=n)

                diff = df_filter[col].diff().values/dt
                current_diff = np.concatenate((diff[:1], current_diff, diff[-1:]))  # diff[0] = nan
            elif method == 'spline':

                sp = UnivariateSpline(df_filter['time'].values, df_filter[col].values)
                current_diff = sp(df_filter['time'].values, nu=n)
            else:
                current_diff = df_filter[col].diff().values / dt

            # Linear Interpolation
            # (f2 - f1) / (t2-t1)
            # t2 - t1 = (index_2 - index_1) * dt = 1 * dt, but when t0 and t1 are calculated the dt would cancel out
            m = (current_diff[2] - current_diff[1])

            b = current_diff[1]
            d0 = b + m * (-1)
            current_diff[0] = d0

            # Smoothing
            if ma_args is not None:
                ma_args = ma_args.copy()
                if (ma_args is not None) and ('window_args' in ma_args.keys()):
                    window_args = ma_args.pop('window_args')
                else:
                    window_args = {}

                current_diff = pd.Series(current_diff).rolling(**ma_args).mean(**window_args).values

            results_dict[col] = np.concatenate([results_dict[col],  # Concatenate to add each bird to the right column
                                                current_diff])

    return results_dict


def get_moving_average_per_bird(df, moving_average_params, partition_key='bird_name'):
    column_names_to_eval = list(moving_average_params.keys())

    unique_partitions = df[partition_key].unique()

    moving_average_results_dict = {col: np.array([]) for col in column_names_to_eval}

    for idx, partition_value in enumerate(unique_partitions):
        df_filter = df.loc[df[partition_key] == partition_value]
        df_filter = df_filter[column_names_to_eval]

        for col, ma_args in moving_average_params.items():

            if ma_args is not None:
                ma_args = ma_args.copy()
                if (ma_args is not None) and ('window_args' in ma_args.keys()):
                    window_args = ma_args.pop('window_args')
                else:
                    window_args = {}
                current_ma = df_filter[col].rolling(**ma_args).mean(**window_args)

                moving_average_results_dict[col] = np.concatenate([moving_average_results_dict[col],
                                                                   current_ma.values])

    return moving_average_results_dict


def get_diff_per_bird(df, column_names_to_eval, partition_key='bird_name'):
    # column_names_to_eval = list(moving_average_params.keys())

    unique_partitions = df[partition_key].unique()

    diff_results_dict = {col: np.array([]) for col in column_names_to_eval}

    for idx, partition_value in enumerate(unique_partitions):
        df_filter = df.loc[df[partition_key] == partition_value]
        df_filter = df_filter[column_names_to_eval]

        for col in column_names_to_eval:
            current_diff = df_filter[col].diff()

            # Linear Interpolation
            current_diff[current_diff.index[0]] = (current_diff[current_diff.index[1]]
                                                   - (current_diff[current_diff.index[2]]
                                                      - current_diff[current_diff.index[1]]
                                                      )
                                                   )

            current_diff = current_diff.fillna(0)
            diff_results_dict[col] = np.concatenate([diff_results_dict[col],
                                                     current_diff.values])

    return diff_results_dict


def calculate_per_partition(df, function, list_of_columns_to_eval, kwargs_per_col=None, partition_key='bird_name',
                            suffix='_new'):
    if kwargs_per_col is None:
        kwargs_per_col = {col: {} for col in list_of_columns_to_eval}

    unique_partitions = df[partition_key].unique()
    df_big = pd.DataFrame()

    for idx, partition_value in enumerate(unique_partitions):
        df_partition = df[df[partition_key] == partition_value]

        for col in list_of_columns_to_eval:
            df_partition[col + suffix] = df_partition.apply(function, **kwargs_per_col[col])

        df_big = pd.concat([df_big, df_partition])

    return df_big


def get_gradient(f, X, delta=1.0, N=10, ):

    x0, y0, z0 = X
    if N % 2 == 0:
        N = N + 1
    x = np.linspace(x0 - delta, x0 + delta, N, endpoint=True)
    y = np.linspace(y0 - delta, y0 + delta, N, endpoint=True)
    z = np.linspace(z0 - delta, z0 + delta, N, endpoint=True)
    d = 2 * delta / N
    x_mg, y_mg, z_mg = np.meshgrid(x, y, z)

    f_values = f(x_mg, y_mg, z_mg)

    grad = np.gradient(f_values, d)
    N_return = N//2 + 1
    return [grad[i][N_return, N_return, N_return] for i in range(3)]


def fill_nan_with_spline(x_array, f_array, n_max=None, extrapolate='linear'):
    na_mask = np.isnan(f_array)
    spline = UnivariateSplineWrapper(x_array[~na_mask],
                                     f_array[~na_mask],
                                     )

    resampled_f_array = spline(x_array, extrapolate=extrapolate)
    return resampled_f_array, spline


def get_na_mask(*arr):
    arr = np.array(arr)
    na_mask = np.logical_not(np.isnan(arr[0]))
    for a in arr[1:]:
        current_na_mask = np.logical_not(np.isnan(a))
        na_mask = np.logical_and(na_mask, current_na_mask)

    return na_mask


def prepare_tensor_for_periodic_boundary(tensor):
    # Extend turbulence data to include periodic boundary conditions
    tensor_shape = tensor.shape
    new_dataframe = np.empty(
        shape=(tensor_shape[0],  # time
               #tensor_shape[1],  # component
               tensor_shape[1] + 1,  # X
               tensor_shape[2] + 1,  # Y
               tensor_shape[3] + 1))  # Z

    # fill 200x200x200
    new_dataframe[..., :-1, :-1, :-1] = tensor[..., :, :, :]

    # fill extra faces
    new_dataframe[..., -1, :-1, :-1] = tensor[..., 0, :, :]
    new_dataframe[..., :-1, -1, :-1] = tensor[..., :, 0, :]
    new_dataframe[..., :-1, :-1, -1] = tensor[..., :, :, 0]

    #fill extra edges
    new_dataframe[..., -1, -1, :-1] = tensor[..., 0, 0, :]
    new_dataframe[..., :-1, -1, -1] = tensor[..., :, 0, 0]
    new_dataframe[..., -1, :-1, -1] = tensor[..., 0, :, 0]

    #fill extra point
    new_dataframe[..., -1, -1, -1] = tensor[..., 0, 0, 0]

    return new_dataframe


def get_periodic_linear_interpolator(tensor, limits):
    tensor = prepare_tensor_for_periodic_boundary(tensor)
    #del turbulence_dataframe
    tensor = tensor
    tensor_shape = tensor.shape

    points = [np.linspace(l_min, l_max, tensor_shape[i], endpoint=True)
              for i, (l_min, l_max) in enumerate(limits)]

    interpolator = RegularGridInterpolator(points=points, values=tensor)
    return interpolator
