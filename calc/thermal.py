import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)


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


def get_ND_random_walk(mean, std, n_vars, n_steps=200, shape=2):

    RW_values = mean + (std * np.random.standard_normal((n_steps ** n_vars, shape))).cumsum(axis=0)
    RW_values = mean + (RW_values - RW_values.mean())

    return RW_values


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

