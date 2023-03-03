import numbers
import os
import warnings
from copy import deepcopy
from types import LambdaType, FunctionType
from typing import Iterable

import h5py
import logging

from scipy.interpolate import LinearNDInterpolator

from object.flight import BirdPoint

# with warnings.catch_warnings():
#     warnings.simplefilter('ignore', category=DeprecationWarning)
#     import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from calc.auxiliar import SplineWrapper, parse_projection_string, get_gradient, get_periodic_linear_interpolator
from calc.thermal import get_predefined_thermal_model, get_hierarchical_turbulence


logger = logging.getLogger(__name__)


class AirVelocityField:
    config = {'t_start_max': 300,
              'time_resolution': 10,
              'z_max_limit': 3000,
              'dt_to_iterate': 0.1,
              'dt_to_save': 1,
              'thermal_core_initial_position': [0, 0, 0],
              }

    def __init__(self, air_parameters, **config_args):
        self.turbulence_interpolator_parameters = None
        self.wind_index_list = []
        self.wind_var_list = []
        self.wind_time_var = False
        self._air_parameters = deepcopy(air_parameters)
        self.rho = air_parameters['rho']
        self.thermal_rotation_function = None
        self.thermal_profile_function = None
        self.wind_function = None
        self.turbulence_function = None
        self.thermal_core = None
        self.components = ['wind', 'rotation', 'thermal', 'turbulence']

        # properties

        self.config = AirVelocityField.config

        self.config.update(config_args)
        # Adding an extra resolution unit prevents issues with undefined thermal core
        # when time is close to the end of the simulation.
        self.config['t_start_max'] = self.config['t_start_max'] + self.config['time_resolution']

        self.turbulence_parameters = {}
        self.turbulence_interpolators = {}

        self.current_turbulence_time_limits = None

        self.legacy_handling()

        self.preprocessing()
        # self.get_velocity = np.vectorize(self.get_velocity)

    def legacy_handling(self):

        if 'rotation' in self._air_parameters.keys():
            self._air_parameters['thermal']['rotation'] = self._air_parameters['rotation']
            del self._air_parameters['rotation']

        if 'model' in self._air_parameters['thermal'].keys():
            self._air_parameters['thermal']['profile'] = get_predefined_thermal_model(
                self._air_parameters['thermal']['model'])

        if 'args' not in self._air_parameters['thermal']:
            self._air_parameters['thermal']['args'] = {}

        if isinstance(self._air_parameters['wind'], (list, np.ndarray)):
            self._air_parameters['wind'] = np.array(self._air_parameters['wind'])

            if self._air_parameters['wind'].ndim == 1:  # Constant wind
                self._air_parameters['wind'] = {'values': self._air_parameters['wind'],
                                                'used_vars': ''
                                                }
            elif self._air_parameters['wind'].ndim == 2:  # Z dependent by default
                z_array = np.linspace(0, self.config['z_max_limit'], self._air_parameters['wind'].shape[0])
                self._air_parameters['wind'] = {'values': self._air_parameters['wind'],
                                                'XYZT_values': z_array,
                                                'used_vars': 'z'}

    def preprocessing(self):
        self.preprocess_wind()
        self.preprocess_thermal_rotation()
        self.preprocess_thermal_profile()
        self.preprocess_turbulence()
        self.set_thermal_core_advection_time_dependent()

    def preprocess_wind(self):
        # Custom Function
        if isinstance(self._air_parameters['wind'], (LambdaType, FunctionType)):
            def wind_function(X, t=0):
                return np.concatenate([self._air_parameters['wind'](X, t), [0]], dtype=np.float32)

        # Data driven wind
        else:
            # Check which variables are used in the data driven wind
            for i, coord in enumerate(self._air_parameters['wind']['used_vars'].lower()):
                index = ['x', 'y', 'z', 't'].index(coord)

                if coord == 't':
                    self.wind_time_var = True
                else:
                    self.wind_var_list.append(coord)
                    self.wind_index_list.append(index)

            n_used_vars = (len(self.wind_index_list) + int(self.wind_time_var))

            if n_used_vars == 0:
                def wind_function(X, t=0): # Constant Wind
                    result = np.empty(shape=np.array(X).shape)
                    result[:, [0,1]] = self._air_parameters['wind']['values']
                    result[:, 2] = 0

                    return result
            elif n_used_vars == 1:
                # Unidimensional

                wind_spline = SplineWrapper(X=self._air_parameters['wind']['values'].T,
                                            y=self._air_parameters['wind']['XYZT_values'], degree=3)
                wind_spline.fit()

                def wind_function(X, t):

                    X = np.array(X)
                    n_points = X.shape[0]
                    if self.wind_time_var:
                        input_vars = np.hstack([X[:, self.wind_index_list], [t]])
                    else:
                        input_vars = X[:, self.wind_index_list]

                    return_array = np.empty((n_points, 3), dtype=np.float32)
                    wind_values = wind_spline(input_vars).reshape((2, n_points))
                    return_array[:, 0] = wind_values[0]
                    return_array[:, 1] = wind_values[1]
                    return_array[:, 2] = 0

                    return return_array
            else:  # Multidimensional
                interpolator = [LinearNDInterpolator(self._air_parameters['wind']['XYZT_values'],
                                                     self._air_parameters['wind']['values'][:, i]
                                                     ) for i in [0, 1]]

                interpolator = np.array(interpolator)

                def wind_function(X: np.ndarray, t: float = 0) -> np.ndarray:
                    X = np.array(X)
                    if self.wind_time_var:
                        input_vars = np.hstack([X[self.wind_index_list], [t]])
                    else:
                        input_vars = X[:, self.wind_index_list]
                    n_points = input_vars.shape[0]
                    return_array = np.hstack([interpolator[0](input_vars).reshape((n_points, 1)),
                                              interpolator[1](input_vars).reshape((n_points, 1)),
                                              np.zeros(shape=(n_points, 1))]
                                             ).astype(np.float32)

                    if np.any(np.isnan(return_array)):
                        pass
                    return return_array

        self.wind_function = wind_function

    def preprocess_thermal_rotation(self):
        # ====================================      ROTATION        ================================================== #
        if ('rotation' not in self._air_parameters['thermal']) or self._air_parameters['thermal']['rotation'] is None:
            thermal_rotation = lambda X, t: np.zeros(shape=(np.array(X).shape[0], 3))

        elif isinstance(self._air_parameters['thermal']['rotation'], (float, int)):
            def thermal_rotation(X, t):
                import numpy as np
                rotation_magnitude = self._air_parameters['thermal']['rotation']
                result = np.empty(shape=(X.shape[0], 3), dtype=np.float32)
                for i_x, current_X in enumerate(X):
                    x, y, z = current_X
                    # Local coordinates
                    r = np.linalg.norm([x, y])
                    theta = np.arctan2(y, x)
                    result[i_x] = [-rotation_magnitude * np.sin(theta),
                                   rotation_magnitude * np.cos(theta),
                                   0]
                return result
        else:  # elif isinstance(self._air_parameters['thermal']['rotation'], (LambdaType, FunctionType)):
            def thermal_rotation(X, t=0):
                import numpy as np
                result = np.empty(shape=(X.shape[0], 3), dtype=np.float32)
                for i_x, current_X in enumerate(X):

                    x, y, z = current_X
                    # Local coordinates
                    r = np.linalg.norm([x, y])
                    theta = np.arctan2(y, x)

                    result[i_x] = self._air_parameters['thermal']['rotation'](r, theta, z, t) + [0]
                return result

        self.thermal_rotation_function = thermal_rotation

        # ====================================      PROFILE         ================================================== #

    def preprocess_thermal_profile(self):
        if ('profile' not in self._air_parameters['thermal']) or self._air_parameters['thermal']['profile'] is None:
            thermal_profile = lambda X, t: np.zeros(shape=(np.array(X).shape[0], 3))
        else:
            def thermal_profile(X, t=0):
                import numpy as np
                result = np.empty(shape=(X.shape[0], 3), dtype=np.float32)
                for i_x, current_X in enumerate(X):
                    x_relative, y_relative, z = current_X
                    # Local coordinates

                    r = np.linalg.norm([x_relative, y_relative])
                    theta = np.arctan2(y_relative, x_relative)

                    vz = self._air_parameters['thermal']['profile'](r, theta, z, t, **self._air_parameters['thermal']['args'])
                    result[i_x] = [0, 0, vz]
                return result

        self.thermal_profile_function = thermal_profile

    def preprocess_turbulence(self):
        if 'turbulence' not in self._air_parameters.keys():
            self.turbulence_function = lambda X, t: np.zeros(shape=(np.array(X).shape[0], 3))
            self.components.remove('turbulence')
        elif not self._air_parameters['turbulence']:
            self.turbulence_function = lambda X, t: np.zeros(shape=(np.array(X).shape[0], 3))
            self.components.remove('turbulence')
        else:

            self.turbulence_parameters = deepcopy(self._air_parameters['turbulence'])
            self.turbulence_parameters['largest_spatial_scale'] = max(self.turbulence_parameters['scales'])
            self.turbulence_parameters['largest_velocity_scale'] = self.turbulence_parameters['scales'][self.turbulence_parameters['largest_spatial_scale']]

            # Normalize the scales
            if self.turbulence_parameters['normalization'] is None:
                normalization = self.turbulence_parameters['largest_velocity_scale']
            else:
                normalization = self.turbulence_parameters['normalization']

            if normalization:
                norm = np.sum(list(self.turbulence_parameters['scales'].values())) / normalization

                for spatial_scale in self.turbulence_parameters['scales'].keys():
                    self.turbulence_parameters['scales'][spatial_scale] /= norm

            with h5py.File(self.turbulence_parameters['data_path'], 'r') as hfd:
                n_time, _, n_x, n_y, n_z = hfd['data'].shape
                self.turbulence_parameters['time_limits'] = (0, n_time - 1)
                self.turbulence_parameters['n_grid'] = {'x': n_x // self.turbulence_parameters['downsampling_factor'],
                                                        'y': n_y // self.turbulence_parameters['downsampling_factor'],
                                                        'z': n_z // self.turbulence_parameters['downsampling_factor']}

            #for spatial_scale, velocity_scale in self.turbulence_parameters['scales'].items():
#
            #    self.turbulence_parameters['scales'][spatial_scale]['space_resolution'] = {
            #        coord: spatial_scale / n
            #        for coord, n in self.turbulence_parameters['n_grid'].items()
            #    }
                # TODO
                # Make this programmatically

    def reset_turbulence_function(self, t):

        if ('turbulence' not in self._air_parameters.keys()) or (self._air_parameters['turbulence'] is None):
            return
        elif not self._air_parameters['turbulence']:
            return
        else:
            del self.turbulence_interpolators

        logger.debug('resetting turbulence')
        # Fit interpolators
        largest_spatial_scale = self.turbulence_parameters['largest_spatial_scale']
        largest_velocity_scale = self.turbulence_parameters['scales'][largest_spatial_scale]

        L = {'x': largest_spatial_scale,
             'y': largest_spatial_scale,
             'z': largest_spatial_scale}

        t_min = np.floor(t).astype(int)
        t_max = int(t_min + self.turbulence_parameters['reset_period'])

        t_max = np.min([self.turbulence_parameters['time_limits'][-1],
                        t_max])

        self.turbulence_interpolator_parameters = {'spatial_scale': largest_spatial_scale,
                                                   'velocity_scale': largest_velocity_scale}
        if t_max == t_min:
            t_min = t_min - 1

        logger.debug(f'reading data file from {t_min:.1f} to {t_max:.1f}')
        with h5py.File(self.turbulence_parameters['data_path'], 'r') as hfd:
            DS = self.turbulence_parameters['downsampling_factor']
            turbulence_dataframe = hfd['data'][t_min: t_max + 1, :, ::DS, ::DS, ::DS]

        turbulence_dataframe = largest_velocity_scale * turbulence_dataframe
        logger.debug('done reading data file')
        self.current_turbulence_time_limits = [t_min, t_max]

        limits = [self.current_turbulence_time_limits,
                  [0, L['x']],
                  [0, L['y']],
                  [0, L['z']]]
        logger.debug(f'Interpolating turbulence data: spatial_grid:{self.turbulence_parameters["n_grid"]}, time: {t_max - t_min}')
        self.turbulence_interpolators = [get_periodic_linear_interpolator(turbulence_dataframe[:, i, :, :, :],
                                                                          limits=limits)
                                         for i in range(3)]
        del turbulence_dataframe
        logger.debug('Done interpolating turbulence data')

        def turbulence_function(X, t, scales=None, weighting_method=None, weighting_method_miltiplier=50,
                                velocity_component=None):
            if isinstance(velocity_component, int):
                list_of_components = [velocity_component]
            elif isinstance(velocity_component, (list, np.ndarray)):
                list_of_components = velocity_component.copy()
            else:
                list_of_components = np.arange(len(self.turbulence_interpolators))

            if weighting_method is None:
                weighting_method = self.turbulence_parameters['weighting_method']

            if not (self.current_turbulence_time_limits[0] <= t <= self.current_turbulence_time_limits[1]):
                self.reset_turbulence_function(t)

            if scales is None:
                list_of_scales = self.turbulence_parameters['scales'].keys()
            elif isinstance(scales, (numbers.Integral, numbers.Real)):
                list_of_scales = [scales]
            else:
                list_of_scales = scales

            scales_dict = {s: self.turbulence_parameters['scales'][s] for s in list_of_scales}
            interpolators_to_use = [self.turbulence_interpolators[i] for i in list_of_components]
            results = get_hierarchical_turbulence(X, t,
                                                  scales_dict=scales_dict,
                                                  list_of_interpolator=interpolators_to_use,
                                                  interpolator_parameters=self.turbulence_interpolator_parameters)

            sum = self.add_velocities(results)

            # Weighting
            if weighting_method == 'value':
                thermal_profile_value = self.thermal_profile_function(X=X, t=t)
                thermal_profile_value = thermal_profile_value[-1]
                return_value = np.abs(thermal_profile_value) * sum
            elif weighting_method == 'gradient':

                #sigma = 15
                #thermal_profile_value = self.thermal_profile_function(X=X, t=t)
                #thermal_profile_gradient = - (np.linalg.norm(X[:2]) / sigma ** 2) * thermal_profile_value
                thermal_profile_gradient = self.get_velocity_gradient(X, t, include='thermal', N=5)

                return_value = weighting_method_miltiplier * np.linalg.norm(thermal_profile_gradient) * sum

            else:
                return_value = sum

            return return_value

        self.turbulence_function = turbulence_function

    def set_thermal_core_advection_time_dependent(self):
        zt_array = np.empty((0, 2), float)
        list_of_paths = np.empty((0, 3), float)
        first_t_array = []
        first_path = []
        for t_start in range(0, self.config['t_start_max'], self.config['time_resolution']):
            current_point = np.array(self.config['thermal_core_initial_position'])
            current_path = np.array([current_point], dtype=float)
            current_t_array = np.array([t_start], dtype=float)
            logger.debug(f't={t_start}')
            t = t_start
            while current_point[-1] < self.config['z_max_limit']:
                dv = self.wind_function(X=[current_point],
                                        t=t)[0]
                if np.any(np.isnan(dv)):
                    logger.debug(f'{current_point=}, {t=}')
                    break
                dv += self.thermal_profile_function(X=np.array([0, 0, current_point[2]]).reshape((1,3)),
                                                    t=t)[0]

                dl = dv * self.config['dt_to_iterate']
                current_point = current_point + dl
                t = t + self.config['dt_to_iterate']
                if np.isclose(t % self.config['dt_to_save'], 0, atol=1e-4) or np.isclose(t % self.config['dt_to_save'], 1, atol=1e-4):
                    current_path = np.append(current_path, [current_point], axis=0)
                    current_t_array = np.append(current_t_array, [t], axis=0)

            list_of_paths = np.append(list_of_paths, current_path, axis=0)
            z_array = current_path[:, -1]

            current_zt = np.array([z_array, current_t_array]).T
            zt_array = np.append(zt_array, current_zt, axis=0)

            if len(first_t_array) == 0:
                first_t_array = current_t_array.copy()
                first_path = current_path.copy()

        # This fills the list with the instants that are not expressed in the previous iterations, e.g. thermal core
        # for z=0.1 only exists for t=0, for z=200 only exists for, say, t > 50 for a thermal with v_max = 4 m/s
        # This fills the gaps as if before t=0 everything was stationary.
        logger.debug('doing before')
        time_step = int(round(self.config['time_resolution'] / self.config['dt_to_save']))
        for (x, y, z), t_max in zip(first_path[1::time_step],
                                    first_t_array[1::time_step]):  # the first element would be repeated
            logger.debug(f't={t_max}')
            current_t_array = np.arange(0, t_max, self.config['dt_to_save'])
            current_path = np.array([[x, y, z]] * len(current_t_array))

            list_of_paths = np.append(list_of_paths, current_path, axis=0)
            z_array = current_path[:, -1]

            current_zt = np.array([z_array, current_t_array]).T
            zt_array = np.append(zt_array, current_zt, axis=0)

        logger.debug(f'using {len(zt_array)} points')
        thermal_core_interpolator = [LinearNDInterpolator(zt_array,
                                                          list_of_paths[:, i]
                                                          ) for i in [0, 1]]

        # thermal_core = SplineWrapper(current_path[:, :2].T, current_path[:, -1].T)
        # thermal_core.fit()

        def thermal_core_function(z: float, t: float = 0) -> np.ndarray:

            input_vars = [z, t]

            return_array = np.vstack([thermal_core_interpolator[0](*input_vars),
                                      thermal_core_interpolator[1](*input_vars)]
                                     ).astype(np.float32)

            if np.any(np.isnan(return_array)):
                pass
            return return_array

        self.thermal_core = thermal_core_function

    def get_thermal_core(self, z, t=0):
        if isinstance(z, Iterable):
            z = np.array(z.copy())
            negative_z_indices = np.argwhere(z < 0).flatten()
        else:
            if z < 0:
                core = self.config['thermal_core_initial_position'][:2]
                return core

        core = self.thermal_core(z, t=t).T
        if isinstance(z, Iterable):
            core[negative_z_indices] = self.config['thermal_core_initial_position'][:2]
            return core
        else:
            return core[0]

    def change_frame_of_reference(self, X, t, ground_to_air=True):
        core = self.get_thermal_core(X[:, -1], t)
        if ground_to_air:
            X_transformed = np.array([X[:, 0] - core[:, 0], X[:, 1] - core[:, 1], X[:, -1]]).T
        else:  # air to ground
            X_transformed = np.array([X[:, 0] + core[:, 0], X[:, 1] + core[:, 1], X[:, -1]]).T

        return X_transformed

    def get_velocity_field_component(self, X, t=0, component=None):

        if component == 'wind':
            return self.wind_function(X, t)

        if component == 'rotation':
            return self.thermal_rotation_function(X, t)

        if component == 'thermal':
            return self.thermal_profile_function(X, t)

        if component == 'turbulence':
            return self.turbulence_function(X, t)

    def get_velocity(self, X, t=0, include=None, exclude=None, relative_to_ground=True, return_components=False):

        X = np.array(X)
        single_point = X.ndim == 1
        if not single_point:
            n_points = X.shape[0]
        else:
            n_points = 1
            X = X.reshape((1,3))

        if exclude is None:
            exclude = set([])
        elif isinstance(exclude, str):
            exclude = {exclude}
        elif isinstance(exclude, (list, np.ndarray)):
            exclude = set(exclude)

        if include is None:
            include = set(self.components)
        elif isinstance(include, str):
            include = {include}
        elif isinstance(include, (list, np.ndarray)):
            include = set(include)

        if relative_to_ground:
            X_air = self.change_frame_of_reference(X, t, ground_to_air=True)
            X_ground = X.copy()
        else:
            X_ground = self.change_frame_of_reference(X, t, ground_to_air=False)
            X_air = X.copy()

        to_include = include - exclude
        components = {}
        if 'wind' in to_include:
            components['wind'] = self.wind_function(X_ground, t)
        else:
            components['wind'] = np.zeros(shape=(n_points, 3))

        if 'rotation' in to_include:
            components['rotation'] = self.thermal_rotation_function(X_air, t)
        else:
            components['rotation'] = np.zeros(shape=(n_points, 3))

        if 'thermal' in to_include:
            components['thermal'] = self.thermal_profile_function(X_air, t)
        else:
            components['thermal'] = np.zeros(shape=(n_points, 3))

        if 'turbulence' in to_include:
            components['turbulence'] = self.turbulence_function(X_air, t)
        else:
            components['turbulence'] = np.zeros(shape=(n_points, 3))

        result = self.add_velocities(np.array(list(components.values())).T).T
        if single_point:
            result = result[0]
            components = {k: v[0] for k, v in components.items()}
        if return_components:
            return result, components
        else:
            return result

    @staticmethod
    def add_velocities(velocities_array):
        return np.sum(velocities_array, axis=-1)  # sum on scales

    def __add__(self, bird_air: BirdPoint, dt, return_components=True):
        #assert isinstance(other, BirdPoint), "Adding is only defined between instances of AirVelocityField and BirdPoint"

        V_air, V_components = self.get_velocity(bird_air.X, bird_air.t, return_components=True)
        bird_parameters = bird_air.get_bird_properties()

        bird_ground = BirdPoint(V=bird_air.V + V_air, t=bird_air.t, A=None, X=None, bank_angle=None, **bird_parameters)

        return bird_ground, V_air, V_components

    def get_velocity_gradient(self, X, t, include, delta=1.0, N=11):

        x0, y0, z0 = X
        if N % 2 == 0:
            N = N + 1
        x = np.linspace(x0 - delta, x0 + delta, N, endpoint=True)
        y = np.linspace(y0 - delta, y0 + delta, N, endpoint=True)
        z = np.linspace(z0 - delta, z0 + delta, N, endpoint=True)
        d = 2 * delta / N
        x_mg, y_mg, z_mg = np.meshgrid(x, y, z)

        X_array = np.array(list(zip(x_mg.flatten(),
                                    y_mg.flatten(),
                                    z_mg.flatten())))

        f_values = self.get_velocity(X_array, t=t, include=include, relative_to_ground=False)[:, -1]
        f_values = f_values.reshape(x_mg.shape)
        grad = np.gradient(f_values, d)  # , axis=[0,1,2]
        N_return = N // 2 + 1
        gradient_return = [grad[i][N_return, N_return, N_return] for i in range(3)]
        return gradient_return

#
# from plotting.auxiliar import fig_ax_handler, get_cross_section_meshgrid
# from plotting.plot import plot_quiver_3d, plot_stream_interpolated, plot_contour_tri
# class AirVelocityFieldVisualization(object):
#     def __init__(self, air_velocity_obj: AirVelocityField):
#         self.air_velocity_field = air_velocity_obj
#
#     @classmethod
#     def from_air_parameters(cls, air_parameters, **config_args):
#         air_velocity_obj = AirVelocityField(air_parameters, **config_args)
#         return cls(air_velocity_obj=air_velocity_obj)
#
#     def plot_thermal_profile(self, ax, plot_type, max_rho, Z_level, t_value=0, resolution=30,
#                              include_turbulence=False, add_colorbar=True,  cross_section_type='XY',
#                              kwargs=None):
#
#         if kwargs is None:
#             kwargs = {}
#         limits = [[-max_rho, max_rho],
#                   [-max_rho, max_rho]]
#         to_include = ['thermal']
#         if include_turbulence:
#             to_include += ['turbulence']
#         ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=Z_level,
#                                                                   include=to_include, t_value=t_value,
#                                                                   cross_section_type=cross_section_type,
#                                                                   plot_type=plot_type,
#                                                                   plotting_function=2,
#                                                                   color_function=None,
#                                                                   resolution=resolution,
#                                                                   plot_kwargs=kwargs,
#                                                                   velocity_kwargs={'relative_to_ground': False},
#                                                                   add_colorbar=add_colorbar)
#         return ax, artist, cbar
#
#     def plot_thermal_core_3d(self, ax, t_value=0, resolution=30, z_max=1000, kwargs=None):
#
#         if kwargs is None:
#             kwargs = {}
#
#         Z_array = np.linspace(0.1, z_max, resolution)
#         t_array = t_value * np.ones(shape=Z_array.shape)
#
#         # ZT_array = np.stack([Z_array, t_array], axis=-1)
#         XY_core = self.air_velocity_field.get_thermal_core(z=Z_array, t=t_value)
#
#         core = np.hstack([XY_core, Z_array.reshape((Z_array.shape[0], 1))])
#
#         artist = ax.plot(xs=core[:, 0],
#                          ys=core[:, 1],
#                          zs=core[:, 2], **kwargs)
#
#         ax.set_xlabel('X (m)')
#         ax.set_ylabel('Y (m)')
#         ax.set_zlabel('Z (m)')
#
#         ax.plot(core[:, 0], core[:, 1], zs=0, zdir='z', color='k', alpha=0.3)
#
#         return ax, artist
#
#     def plot_thermal_rotation(self, ax, max_rho, Z_level, t_value, resolution,  include_turbulence=False,
#                               add_colorbar=True, kwargs=None):
#
#         if kwargs is None:
#             kwargs = {}
#
#         limits = [[-max_rho, max_rho],
#                   [-max_rho, max_rho]]
#         to_include = ['rotation']
#         if include_turbulence:
#             to_include += ['turbulence']
#         ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=Z_level,
#                                                                   include=to_include, t_value=t_value,
#                                                                   cross_section_type='XY',
#                                                                   plot_type='streamplot',
#                                                                   plotting_function=[0, 1],
#                                                                   color_function=[0, 1],
#                                                                   resolution=resolution,
#                                                                   plot_kwargs=kwargs,
#                                                                   velocity_kwargs={'relative_to_ground': False},
#                                                                   add_colorbar=add_colorbar)
#
#         return ax, artist, cbar
#
#     def plot_turbulence(self, ax, plot_type, limits, t=0, cross_section_type='XY', section_level=0, resolution=20,
#                         add_colorbar=True, plotting_function=None, color_function=None,
#                         velocity_function_kwargs=None, plot_kwargs=None):
#
#         if color_function is None:
#             color_function = [0, 1]
#         if plotting_function is None:
#             plotting_function = [0, 1]
#         if velocity_function_kwargs is None:
#             velocity_function_kwargs = {}
#         if plot_kwargs is None:
#             plot_kwargs = {}
#
#         ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=section_level,
#                                                                   include='turbulence', t_value=t,
#                                                                   cross_section_type=cross_section_type,
#                                                                   plot_type=plot_type,
#                                                                   plotting_function=plotting_function,
#                                                                   color_function=color_function,
#                                                                   resolution=resolution,
#                                                                   plot_kwargs=plot_kwargs,
#                                                                   velocity_kwargs=velocity_function_kwargs,
#                                                                   add_colorbar=add_colorbar)
#         return ax, artist, cbar
#
#     def plot_wind(self, ax, limits, t=0, section_level=0, resolution=20, cross_section_type='XZ',
#                   add_colorbar=True, kwargs=None):
#
#         if kwargs is None:
#             kwargs = {}
#
#         ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=section_level,
#                                                                   include='wind', t_value=t,
#                                                                   cross_section_type=cross_section_type,
#                                                                   plot_type='streamplot',
#                                                                   plotting_function=lambda v: [v[0], 0],
#                                                                   color_function=0,
#                                                                   resolution=resolution,
#                                                                   velocity_kwargs={'relative_to_ground': True},
#                                                                   add_colorbar=add_colorbar)
#
#         ax.set_aspect('equal')
#
#         return ax, artist, cbar
#
#     def plot_all(self, z_value=200, section_value=0, t_value=0, include_turbulence=False):
#         import matplotlib.pyplot as plt
#
#         fig = plt.figure(figsize=(12, 10), constrained_layout=True)
#         ax = [fig.add_subplot(221),
#               fig.add_subplot(222, projection='3d'),
#               fig.add_subplot(223, projection='3d'),
#               fig.add_subplot(224)]
#
#         # ==============================================================================================================
#         # ========================================      WIND      ======================================================
#         # ==============================================================================================================
#         limits = [[10, 1000],
#                   [0.1, 1000]]
#         _, art, cbar = self.plot_wind(ax=ax[0], limits=limits, t=t_value, section_level=section_value,)
#         ax[0].set_aspect('equal')
#         ax[0].set_title('Wind')
#
#
#         # ==============================================================================================================
#         # ===================================      THERMAL PROFILE       ===============================================
#         # ==============================================================================================================
#
#         _, art, cbar = self.plot_thermal_profile(ax=ax[1], plot_type='surface', max_rho=60, Z_level=z_value, t_value=t_value,
#                                                  resolution=100, include_turbulence=include_turbulence,
#                                                  add_colorbar=True,
#                                                  kwargs={'linewidth': 0}, )
#         cbar.set_label('$V_{Z}$ (m/s)')
#
#         ax[1].set_title('Thermal Profile')
#         ax[1].set_zlabel('$V_Z$ (m/s)')
#
#         # ==============================================================================================================
#         # ===================================      THERMAL CORE       ==================================================
#         # ==============================================================================================================
#         _, art = self.plot_thermal_core_3d(ax=ax[2], t_value=t_value, )
#
#         ax[2].set_title('Thermal Core')
#
#         # ==============================================================================================================
#         # ===================================      THERMAL ROTATION      ===============================================
#         # ==============================================================================================================
#         _, art, cbar = self.plot_thermal_rotation(ax=ax[3], max_rho=60, Z_level=z_value, t_value=t_value, resolution=15,
#                                                   add_colorbar=True, include_turbulence=include_turbulence)
#         ax[3].set_aspect('equal')
#         ax[3].set_title('Thermal Rotation')
#         if cbar:
#             cbar.set_label('$V_{Horizontal}$ (m/s)')
#         fig.suptitle(f't={t_value:.1f}')
#         return fig, ax
#
#     def plot_all_over_time(self, t_steps, destination_folder):
#
#         import matplotlib.pyplot as plt
#         plt.ioff()
#         n_digits = np.ceil(np.log10(max(t_steps))).astype(int)
#         try:
#             os.makedirs(destination_folder)
#         except FileExistsError:
#             pass
#
#         for t in t_steps:
#             fig, ax = self.plot_all(t_value=t)
#             filename = 't=' + ('0' * n_digits + str(t))[-n_digits:]
#             full_path = os.path.join(destination_folder, f'{filename}.png')
#             fig.savefig(full_path)
#             print(f'saved to {full_path}')
#             plt.close(fig)
#
#     def _plot_per_component_and_plot_type(self, ax, limits, section_value, include,
#                                           plot_type, t_value, plotting_function, color_function=None,
#                                           cross_section_type='XY',
#                                           resolution=15, velocity_kwargs=None, plot_kwargs=None, add_colorbar=True):
#         if velocity_kwargs is None:
#             velocity_kwargs = {}
#         if plot_kwargs is None:
#             plot_kwargs = {}
#
#         (XYZ_meshgrid,
#          plotting_vars, plotting_indices,
#          section_var, section_index) = get_cross_section_meshgrid(limits=limits,
#                                                                   cross_section_type=cross_section_type,
#                                                                   n_points=resolution,
#                                                                   section_value=section_value)
#
#         mg_shape = XYZ_meshgrid.shape
#         X_meshgrid = XYZ_meshgrid[..., plotting_indices[0]]
#         Y_meshgrid = XYZ_meshgrid[..., plotting_indices[1]]
#         Z_meshgrid = XYZ_meshgrid[..., section_index]
#
#         XYZ_flat = XYZ_meshgrid.reshape((mg_shape[0] * mg_shape[1], mg_shape[-1]))
#         velocities_array = self.air_velocity_field.get_velocity(XYZ_flat, t=t_value, include=include, **velocity_kwargs)
#
#         velocities_array = velocities_array.reshape(XYZ_meshgrid.shape)
#         #plotting_function
#
#         if isinstance(plotting_function, (FunctionType, LambdaType)):
#             plot_array = np.apply_along_axis(plotting_function, -1, velocities_array)
#         elif isinstance(plotting_function, Iterable):
#             plot_array = velocities_array[..., plotting_function]
#         else:
#             plot_array = velocities_array[..., plotting_function]
#
#         if isinstance(color_function, (FunctionType, LambdaType)):
#             color_array = np.apply_along_axis(color_function, -1, velocities_array)
#         elif np.isscalar(color_function):
#             color_array = velocities_array[..., color_function]
#         elif color_function is None:
#             color_array = None
#         else:  # Defaults to norm of the vector with indices in plotting_function,
#             # e.g., plotting_function=[0, 1] will yield the horizontal velocity
#             color_array = np.linalg.norm(velocities_array[..., color_function], axis=-1)
#
#         if plot_type == 'quiver':
#             if color_array is not None:
#                 artist = ax.quiver(X_meshgrid,
#                                    Y_meshgrid,
#                                    plot_array[..., 0],
#                                    plot_array[..., 1],
#                                    color_array,
#                                    **plot_kwargs)
#             else:
#                 artist = ax.quiver(X_meshgrid,
#                                    Y_meshgrid,
#                                    plot_array[..., 0],
#                                    plot_array[..., 1],
#                                    **plot_kwargs)
#         elif plot_type == 'contour':
#             if plot_array.ndim > 2:
#                 plot_array = np.linalg.norm(plot_array, axis=-1)
#             artist = ax.contourf(X_meshgrid,
#                                  Y_meshgrid,
#                                  plot_array,
#                                  **plot_kwargs
#                                  )
#         elif plot_type == 'streamplot':
#             artist = ax.streamplot(x=X_meshgrid,
#                                    y=Y_meshgrid,
#                                    u=plot_array[..., 0],
#                                    v=plot_array[..., 1],
#                                    color=color_array, **plot_kwargs)
#         elif plot_type == 'imshow':
#             if plot_array.ndim > 2:
#                 plot_array = np.linalg.norm(plot_array, axis=-1)
#             artist = ax.imshow(plot_array, origin='lower', extent=(limits[0][0],
#                                                                     limits[0][1],
#                                                                     limits[1][0],
#                                                                     limits[1][1]),
#                                **plot_kwargs
#                                )
#         elif plot_type == 'surface':
#             if plot_array.ndim > 2:
#                 plot_array = np.linalg.norm(plot_array, axis=-1)
#             artist = ax.plot_surface(X_meshgrid,
#                                      Y_meshgrid,
#                                      plot_array, facecolors=color_array,
#                                      cmap='viridis', **plot_kwargs)
#
#         if add_colorbar:
#             import matplotlib.pyplot as plt
#             if plot_type == 'streamplot':
#                 cbar = plt.colorbar(artist.lines, ax=ax)
#             else:
#                 cbar = plt.colorbar(artist, ax=ax)
#         else:
#             cbar = None
#
#         ax.set_xlabel(plotting_vars[0] + ' (m)')
#         ax.set_ylabel(plotting_vars[1] + ' (m)')
#         return ax, artist, cbar
