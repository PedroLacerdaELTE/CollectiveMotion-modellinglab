import numpy as np
from matplotlib import pyplot as plt

from calc.control import get_thermal_core_estimation, get_bank_angle_from_delta_V, get_orthogonal_unit_vector, \
    get_bank_angle_update_reichmann
from calc.flight import get_horizontal_velocity_from_bird_parameters, get_min_sink_rate_from_bank_angle, \
    get_radius_from_bank_angle, get_bank_angle_from_radius
from calc.geometry import get_curvature_from_trajectory, get_radius_from_curvature
from data.synthetic.auxiliar import get_control_parameters_from_yaml


class Point():
    def __init__(self, V, X=None, A=None, t=None):
        self.t = float(t)
        self.V = np.array(V)
        self.X = np.array(X) if X is not None else None
        self.A = np.array(A) if A is not None else None
        self.set_geometry()
        self.curvature = np.nan
        self.radius = np.nan

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def set_geometry(self):
        if (self.V is None) or (self.A is None):
            self.curvature = np.nan
            self.radius = np.nan
        else:
            self.curvature = get_curvature_from_trajectory(self.V[0], self.V[1], self.A[0], self.A[1])
            self.radius = get_radius_from_curvature(self.curvature)


class BirdPoint(Point):
    def __init__(self, V, X, A, t, bank_angle, mass, CL, CD, wing_area, flight_mode=None, thermalling_mode=None,
                 rho=1.225, g=9.8):
        super().__init__(V, X, A, t=t)
        self.rho = rho
        self.g = g
        self.mass = mass
        self.CL = CL
        self.CD = CD
        self.wing_area = wing_area
        try:
            self.wing_load = self.mass * self.g / self.wing_area
        except TypeError as e:
            self.wing_load = None
        self.Vh = np.linalg.norm([self.V[0], self.V[1]])
        self.Vz = self.V[-1]
        self.bank_angle = bank_angle
        self.bearing = np.arctan2(self.V[1], self.V[0])
        self.flight_mode = flight_mode
        self.thermalling_mode = thermalling_mode if thermalling_mode is not None else ''

    @classmethod
    def from_bank_angle_and_bearing(cls, bank_angle, bearing, t, X, mass, CL, CD, wing_area, flight_mode=None,
                                    thermalling_mode=None, A=None, rho=1.225, g=9.8):
        Vh = get_horizontal_velocity_from_bird_parameters(bank_angle=bank_angle,
                                                          mass=mass,
                                                          wing_area=wing_area,
                                                          CL=CL,
                                                          rho=rho)
        Vz = - get_min_sink_rate_from_bank_angle(bank_angle, mass=mass, wing_area=wing_area,
                                                 CL=CL, CD=CD, rho=rho, g=g)

        Vx, Vy = Vh * np.cos(bearing), Vh * np.sin(bearing)

        if flight_mode is None:
            flight_mode = 'initial'
            thermalling_mode = 'initial'

        return cls([Vx, Vy, Vz], X=X, A=A, t=t, bank_angle=bank_angle,
                   mass=mass, CL=CL, CD=CD, wing_area=wing_area, rho=rho, g=g,
                   flight_mode=flight_mode, thermalling_mode=thermalling_mode)

    def get_bird_properties(self):
        return {'rho': self.rho,
                'g': self.g,
                'mass': self.mass,
                'CL': self.CL,
                'CD': self.CD,
                'wing_area': self.wing_area}

    def __setattr__(self, key, value):
        self.__dict__[key] = value


class Trajectory:
    def __init__(self, position, control_parameters=None):
        if control_parameters is None:
            control_parameters = {}

        default_control_parameters = get_control_parameters_from_yaml({})
        self.control_parameters = default_control_parameters
        self.control_parameters.update(control_parameters)

        self.list_of_positions = [position]
        self.thermal_core_estimate = []
        self.last_control_time = 0

    def insert_by_bank_angle(self, bank_angle, dt, flight_mode=None, thermalling_mode=None,
                             CL=None, CD=None, mass=None, wing_area=None, rho=None, g=None):
        previous_point = self.get_last_N_point()

        t = previous_point.t + dt

        bearing = self.get_new_bearing(dt)
        bird_props = previous_point.get_bird_properties()

        flight_mode = previous_point['flight_mode'] if flight_mode is None else flight_mode
        thermalling_mode = previous_point['thermalling_mode'] if thermalling_mode is None else thermalling_mode

        CL = bird_props['CL'] if CL is None else CL
        CD = bird_props['CD'] if CD is None else CD
        mass = bird_props['mass'] if mass is None else mass
        wing_area = bird_props['wing_area'] if wing_area is None else wing_area
        rho = bird_props['rho'] if rho is None else rho
        g = bird_props['g'] if g is None else g

        bank_angle = self.bank_angle_preprocess(bank_angle, dt)
        new_point = BirdPoint.from_bank_angle_and_bearing(bank_angle, bearing, t=t, X=None, flight_mode=flight_mode,
                                                          thermalling_mode=thermalling_mode, mass=mass, CL=CL, CD=CD,
                                                          wing_area=wing_area, rho=rho, g=g)
        self.list_of_positions.append(new_point)

        self.post_insert()

    def inherit_attributes(self, bird_point):
        previous = self.get_last_N_point()
        if bird_point.CL is None:
            bird_point.CL = previous.CL
        if bird_point.CD is None:
            bird_point.CD = previous.CD
        if bird_point.mass is None:
            bird_point.mass = previous.mass
        if bird_point.wing_area is None:
            bird_point.wing_area = previous.wing_area
        if bird_point.rho is None:
            bird_point.rho = previous.rho
        if bird_point.g is None:
            bird_point.g = previous.g

    def tail(self, N):
        try:
            return_obj = self.list_of_positions[-(N + 1):]
        except IndexError as e:
            return_obj = None
        return return_obj

    def head(self, N):
        try:
            return_obj = self.list_of_positions[:N]
        except IndexError as e:
            return_obj = None
        return return_obj

    def get_last_N_point(self, N=0):
        try:
            return_obj = self.list_of_positions[-(N + 1)]
        except IndexError as e:
            return_obj = None
        return return_obj

    def get_new_bearing(self, dt):
        previous_point = self.get_last_N_point()

        # Vh_prev is not squared because it cancels out with the unit vector
        lift_force = (previous_point.CL
                      * np.sin(previous_point.bank_angle)
                      * (1 / 2)
                      * previous_point.rho
                      * previous_point.wing_area
                      * np.linalg.norm(previous_point.V[:2])
                      * np.array([-previous_point.V[1],
                                  previous_point.V[0]])
                      )

        delta_Vh_prev = lift_force / previous_point.mass * dt
        Vh_vector = np.array([previous_point.V[0], previous_point.V[1]]) + delta_Vh_prev
        return np.arctan2(Vh_vector[1], Vh_vector[0])

    def integrate(self):
        previous = self.get_last_N_point(1)
        current = self.get_last_N_point()

        current.X = previous.X + previous.V * (current.t - previous.t)

    def differentiate(self):
        current = self.get_last_N_point()
        previous = self.get_last_N_point(N=1)
        if previous is None:
            current.A = np.array([0, 0, 0])
        else:
            current.A = (current.V - previous.V) / (current.t - previous.t)

    def __getitem__(self, item):

        if item == 'thermal_core_estimate':
            return self.thermal_core_estimate

        return_array = np.array(list(map(lambda elem: elem[item], self.list_of_positions)))

        return return_array

    def insert(self, bird_point):
        self.list_of_positions.append(bird_point)
        self.post_insert()

    def insert_by_velocity(self, V, dt, flight_mode=None, thermalling_mode=None):
        last = self.get_last_N_point()
        bird_properties = last.get_bird_properties()

        t = last.t + dt
        self.list_of_positions.append(BirdPoint(V=V, X=None, A=None, t=t, bank_angle=None, flight_mode=flight_mode,
                                                thermalling_mode=thermalling_mode, **bird_properties))
        self.post_insert()

    def set_thermal_core_estimation(self, N_circles=1, N_min=50):
        if self.get_last_N_point()['flight_mode'] != 'out_of_thermal':
            estimate = get_thermal_core_estimation(X_array=self['X'][:, 0],
                                                   Y_array=self['X'][:, 1],
                                                   Z_array=self['X'][:, 2],
                                                   bearing_array=self['bearing'],
                                                   weights_array=self['V'][:, 2],
                                                   N_circles=N_circles, N_min=N_min)

            estimate['time'] = self.get_last_N_point().t
            self.thermal_core_estimate.append(estimate)

    def post_insert(self):
        last = self.get_last_N_point()

        if last.A is None:
            self.differentiate()

        if last.X is None:
            self.integrate()

        if np.isnan(last.radius):
            last.set_geometry()

    def get_flight_mode(self, N=5):
        current_point = self.get_last_N_point()
        data_to_consider = self['V'][-N:, 2]
        if np.all(data_to_consider < 0):
            flight_mode = 'out_of_thermal'
        else:
            flight_mode = 'thermalling'

        return flight_mode

    def should_explore(self):
        last_point = self.get_last_N_point()
        return last_point['t'] < self.control_parameters['exploration']['time']

    def get_control(self, dt, air_velocity_obj=None):

        last_point = self.get_last_N_point()
        if np.isclose(last_point['t'] - self.last_control_time, self.control_parameters['general_args']['period']):
            self.last_control_time = last_point['t'] + dt

            if self.should_explore():
                bank_angle_new, flight_mode, thermalling_mode = self.get_control_exploration()

            elif last_point['flight_mode'] == 'out_of_thermal':
                bank_angle_new, flight_mode = self.go_back_to_thermal()
                thermalling_mode = ''
            else:
                bank_angle_new_exploitation, flight_mode, thermalling_mode = self.get_control_exploitation()
                bank_angle_new_exploration, _, _ = self.get_control_exploration()
                ratio = self.control_parameters['general_args']['exploration_exploitation_ratio']
                bank_angle_new = bank_angle_new_exploration * ratio + (1 - ratio) * bank_angle_new_exploitation

            flight_mode = self.get_flight_mode(N=self.control_parameters['general_args']['N_out_of_thermal'])

        else:
            flight_mode = self.get_flight_mode(N=self.control_parameters['general_args']['N_out_of_thermal'])
            thermalling_mode = last_point['thermalling_mode']
            bank_angle_new = last_point['bank_angle']

        bank_angle_new = self.bank_angle_preprocess(bank_angle_new, dt)
        return bank_angle_new, flight_mode, thermalling_mode

    def get_control_exploration(self, N=5):

        alpha = self.control_parameters['exploration']['alpha_degrees'] * np.pi / 180
        last_point = self.get_last_N_point()
        bank_angle = last_point['bank_angle']

        Az = np.average([bird['A'][2] for bird in self.tail(N) if bird['A'] is not None])

        if np.isclose(Az, 0, atol=0.05):
            bank_angle_new = bank_angle
            thermalling_mode = 'exploration_keep'
        elif np.isclose(bank_angle, 0, atol=0.1):
            if Az > 0:
                bank_angle_new = 0
                thermalling_mode = 'exploration_level'
            else:
                thermalling_mode = 'exploration_bank'
                bank_angle_new = bank_angle - np.random.choice([1, -1], p=[0.5, 0.5]) * alpha * np.sign(Az)
        else:
            if Az > 0:
                bank_angle_new = bank_angle - alpha * np.abs(Az) * np.sign(bank_angle)
            else:
                bank_angle_new = bank_angle + alpha * np.abs(Az) * np.sign(bank_angle)
            thermalling_mode = 'exploration_bank_further'

        flight_mode = 'thermalling'
        return bank_angle_new, flight_mode, thermalling_mode

    def get_control_exploitation(self):

        # Set Parameters
        thermalling_bank_angle = self.control_parameters['exploitation']['thermalling_bank_angle'] * np.pi / 180
        K = self.control_parameters['exploitation']['K']

        current_point = self.get_last_N_point()
        current_position = current_point['X'][:2]
        current_velocity = current_point['V'][:2]
        bank_angle = current_point['bank_angle']
        bird_parameters = current_point.get_bird_properties()

        thermal_core_estimate = self.get_thermal_core_estimate_at_altitude(current_point['X'][-1])[:2]
        if thermal_core_estimate is None:
            return self.get_control_exploration()

        delta_position = thermal_core_estimate - current_position
        unit_current_velocity = current_velocity / np.linalg.norm(current_velocity)

        radius_ref = get_radius_from_bank_angle(bank_angle=thermalling_bank_angle, mass=bird_parameters['mass'],
                                                wing_area=bird_parameters['wing_area'], CL=bird_parameters['CL'])
        if current_point['radius'] < radius_ref:
            radius = radius_ref  # current_point['radius']
        else:
            radius = radius_ref

        # Calculations

        if np.isclose(bank_angle, 0):
            center_of_trajectory = thermal_core_estimate - (np.dot(delta_position, unit_current_velocity)
                                                            * unit_current_velocity)
            radius_vector = center_of_trajectory - current_position
            unit_centripetal_vector = radius_vector / np.linalg.norm(radius_vector)
        else:
            unit_centripetal_vector = np.sign(bank_angle) * get_orthogonal_unit_vector(current_velocity)
            radius_vector = unit_centripetal_vector * radius  # radius_ref
            center_of_trajectory = current_position + radius_vector

        is_turning_left = np.sign(np.cross(current_velocity, radius_vector) / (np.linalg.norm(current_velocity)
                                                                               * np.linalg.norm(radius_vector))
                                  )

        delta_core = thermal_core_estimate - center_of_trajectory  # CW
        unit_orthogonal_delta_core = get_orthogonal_unit_vector(delta_core)
        cos_theta = np.inner(delta_core, current_velocity) / (np.linalg.norm(current_velocity)
                                                              * np.linalg.norm(delta_core))

        scaled_distance = np.linalg.norm(delta_core) / radius

        thermal_entry_point = thermal_core_estimate - is_turning_left * unit_orthogonal_delta_core * radius_ref

        if scaled_distance < 0.5:  # Thermalling
            target_position = thermal_entry_point
            thermalling_mode = 'thermalling'
            bank_angle_new = np.sign(np.cross(current_velocity, unit_centripetal_vector)) * thermalling_bank_angle

        elif scaled_distance > 2:  # Get closer.
            bank_angle_new = self.get_bank_angle_from_go_to_point(thermal_core_estimate, debug=False)
            target_position = thermal_core_estimate
            thermalling_mode = 'getting closer'
        else:  # Approach
            if cos_theta < 0:  # First Leg and half of the second
                target_position = thermal_entry_point
                thermalling_mode = 'go_to_thermal'
                bank_angle_new = self.get_bank_angle_from_go_to_point(target_position, debug=False)
            else:  # Last Leg
                target_position = thermal_entry_point
                thermalling_mode = 'approach_thermal'
                bank_angle_new = self.get_bank_angle_from_go_to_point(target_position, debug=False)
        if self.control_parameters['general_args']['debug']:
            fig = plt.figure(figsize=(12, 12), tight_layout=True)
            ax = fig.add_subplot()  # projection='3d'
            im = ax.scatter(self['X'][:, 0], self['X'][:, 1], c=self['V'][:, 2])
            ax.quiver(self['X'][-1, 0], self['X'][-1, 1],
                      self['V'][-1, 0], self['V'][-1, 1])

            ax.arrow(*center_of_trajectory,
                     *(radius * unit_current_velocity), fc='g', ec='g', label=' projected \ncurrentvelocity')
            ax.arrow(*current_position,
                     *radius_vector, fc='r', ec='r', label='radius vector')
            ax.arrow(*center_of_trajectory,
                     *delta_core, fc='b', ec='b', label='delta core')
            # ax.arrow(*thermal_core_estimate[:2],
            #         *reference_vector, fc='b', ec='c', label='to_core')
            ax.scatter(*thermal_core_estimate, marker='x', label='estimate')
            ax.scatter(*thermal_entry_point, marker='x', c='y', label='thermal_entry_point', s=100)
            ax.scatter(*center_of_trajectory, marker='x', label='center_of_trajectory')
            ax.set_aspect('equal')
            plt.colorbar(im)
            ax.scatter(*target_position, s=36, marker='o', label='target')
            plt.tight_layout()
            plt.legend()
            fig.suptitle(
                f'bank_angle={bank_angle * 180 / np.pi:.2f}   new_bank_angle={bank_angle_new * 180 / np.pi:.2f}\n'
                f'cos = {cos_theta:.2f}\n'
                f'mode={thermalling_mode}   scaled_distance={scaled_distance:.2f}')
            plt.show(block=True)

        flight_mode = 'thermalling'
        return bank_angle_new, flight_mode, thermalling_mode

    def get_bank_angle_from_go_to_point(self, target_position, debug=False):

        last_point = self.get_last_N_point()
        current_position = last_point['X'][:2]
        current_velocity = last_point['V'][:2]
        bank_angle = last_point['bank_angle']
        bird_parameters = last_point.get_bird_properties()

        wing_area = bird_parameters['wing_area']
        CL = bird_parameters['CL']
        rho = 1.2
        delta_position = target_position - current_position
        target_velocity = delta_position / np.linalg.norm(delta_position) * np.linalg.norm(current_velocity)
        delta_velocity = target_velocity - current_velocity

        cos_theta = np.dot(current_velocity, delta_position) / (np.linalg.norm(delta_position)
                                                                * np.linalg.norm(current_velocity))
        sin_theta = np.cross(current_velocity, delta_position) / (np.linalg.norm(delta_position)
                                                                  * np.linalg.norm(current_velocity))
        K = 10
        e_f = get_orthogonal_unit_vector(current_velocity)
        if cos_theta > 0:
            #alpha = K * (1 - cos_theta) / (np.linalg.norm(delta_position) + 0.001)
            # alpha = 10 * max( 1 / (np.linalg.norm(delta_position) + 0.001), 1 / (20))
            alpha=5
            bank_angle_new = get_bank_angle_from_delta_V(bank_angle, current_velocity, delta_velocity,
                                                         alpha=alpha,
                                                         CL=CL, wing_area=wing_area, rho=rho, debug=debug)
        else:
            bank_angle_new = np.sign(sin_theta) * np.pi / 2

        if debug:
            fig = plt.figure(figsize=(12, 12))
            fig.suptitle(f'bank_angle={bank_angle * 180 / np.pi:.2f}\n'
                         f'new_bank_angle{bank_angle_new * 180 / np.pi:.2f}\n'
                        )
            ax = fig.add_subplot()  # projection='3d'
            im = ax.scatter(self['X'][:, 0], self['X'][:, 1], c=self['t'])
            #plt.colorbar(im)

            ax.arrow(self['X'][-1, 0], self['X'][-1, 1],
                     self['V'][-1, 0], self['V'][-1, 1])
            ax.arrow(*current_position,
                     *current_velocity, width=0.5, fc='r', ec='r', label='cur. velocity')
            ax.arrow(*current_position,
                     *target_velocity, width=0.5, fc='g', ec='g', label='target velocity')

            ax.scatter(*current_position, marker='x', label='cur. position')
            ax.scatter(*target_position, marker='x', label='target')
            ax.set_aspect('equal')
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

            plt.tight_layout()
            plt.show(block=True)
        return bank_angle_new

    def get_bank_angle_from_go_to_point_radius(self, target_position, debug=False):

        last_point = self.get_last_N_point()
        current_position = last_point['X'][:2]
        current_velocity = last_point['V'][:2]
        bank_angle = last_point['bank_angle']
        bird_parameters = last_point.get_bird_properties()

        wing_area = bird_parameters['wing_area']
        CL = bird_parameters['CL']
        rho = 1.2
        # Calculations

        delta_position = target_position - current_position

        cos_theta = np.dot(current_velocity, delta_position) / (np.linalg.norm(delta_position)
                                                                * np.linalg.norm(current_velocity))
        if np.isclose(cos_theta, 1, atol=0.01):
            return 0

        radius = last_point['radius']
        if np.isnan(radius):
            radius = 30
        current_velocity_unit_vector = current_velocity / np.linalg.norm(current_velocity)
        centripetal_unit_vector = np.sign(bank_angle) * get_orthogonal_unit_vector(current_velocity)


        if np.isclose(bank_angle, 0):
            center_to_target = target_position - current_position

            if (np.sign(np.dot(- current_position, delta_position)) > 0) and  cos_theta > 0:
                bank_angle_sign = - np.sign(bank_angle)
            else:
                bank_angle_sign = np.sign(bank_angle)
        else:
            radius_vector = centripetal_unit_vector * radius  # radius_ref
            center_of_trajectory = current_position + radius_vector
            center_to_target = target_position - center_of_trajectory
            #bank_angle_sign = np.sign(np.dot(radius_vector, delta_position)) * np.sign(bank_angle)

            if (np.sign(np.dot(- radius_vector, delta_position)) > 0) and  cos_theta > 0:
                bank_angle_sign = - np.sign(bank_angle)
            else:
                bank_angle_sign = np.sign(bank_angle)


        if cos_theta < 0:
            bank_angle_new = bank_angle_sign * np.pi / 2
        else:
            radial_distance_to_target = np.linalg.norm(center_to_target)
            bank_angle_new = bank_angle_sign * get_bank_angle_from_radius(radial_distance_to_target,
                                                                          CL=bird_parameters['CL'],
                                                                          CD=bird_parameters['CD'],
                                                                          mass=bird_parameters['mass'],
                                                                          wing_area=bird_parameters['wing_area'],
                                                                          rho=rho)
        return bank_angle_new

    def go_to_point(self, target_position, dt, method='delta_v', debug=False):
        if method == 'delta_v':
            bank_angle_new = self.get_bank_angle_from_go_to_point(target_position, debug=debug)
        else:
            bank_angle_new = self.get_bank_angle_from_go_to_point_radius(target_position, debug=debug)

        self.insert_by_bank_angle(bank_angle=bank_angle_new, dt=dt)

    def land(self, dt, CL=None, CD=None, mass=None, wing_area=None, ):
        current = self.get_last_N_point()
        current.X[-1] = 0
        self.insert(BirdPoint(V=[0, 0, 0], X=current.X, A=None, t=current.t,
                              bank_angle=0, CL=CL, CD=CD, mass=mass, wing_area=wing_area, flight_mode='Landed')
                    )

    def to_dataframe(self, suffix=None, include_thermal_core_estimate=True, thermal_core_repeat=None):
        import pandas as pd
        df = pd.DataFrame()
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''

        if thermal_core_repeat is None:
            thermal_core_repeat = 5

        df[f'time'] = self['t']
        df[f'X{suffix}'] = self['X'][:, 0]
        df[f'Y{suffix}'] = self['X'][:, 1]
        df[f'Z{suffix}'] = self['X'][:, 2]

        df[f'dXdT{suffix}'] = self['V'][:, 0]
        df[f'dYdT{suffix}'] = self['V'][:, 1]
        df[f'dZdT{suffix}'] = self['V'][:, 2]

        df[f'd2XdT2{suffix}'] = self['A'][:, 0]
        df[f'd2YdT2{suffix}'] = self['A'][:, 1]
        df[f'd2ZdT2{suffix}'] = self['A'][:, 2]

        df[f'bearing{suffix}'] = self['bearing']
        df[f'radius{suffix}'] = self['radius']
        df[f'bank_angle{suffix}'] = self['bank_angle']
        df[f'flight_mode{suffix}'] = self['flight_mode']
        df[f'thermalling_mode{suffix}'] = self['thermalling_mode']
        if include_thermal_core_estimate and len(self['thermal_core_estimate']) != 0:
            df_thermal_core = pd.DataFrame(self['thermal_core_estimate'])
            df_thermal_core.rename(columns={col: f'thermal_core_estimate_{col}'
                                            for col in df_thermal_core.columns},
                                   inplace=True)
            df = pd.merge_asof(df, df_thermal_core, left_on='time', right_on='thermal_core_estimate_time')

        df[f'flight_mode{suffix}'] = df[f'flight_mode{suffix}'].astype('string')
        df[f'thermalling_mode{suffix}'] = df[f'thermalling_mode{suffix}'].astype('string')

        return df

    def bank_angle_preprocess(self, bank_angle, dt):
        delta_bank_angle_max = self.control_parameters['general_args']['delta_bank_angle_max'] * np.pi / 180
        bank_angle_max = self.control_parameters['general_args']['bank_angle_max'] * np.pi / 180
        sigma_noise = self.control_parameters['general_args']['sigma_noise_degrees'] * np.pi / 180

        last_point = self.get_last_N_point()
        current_bank_angle = last_point['bank_angle']

        delta_bank_angle = (bank_angle - current_bank_angle)
        delta_bank_angle = np.clip(delta_bank_angle,
                                   -delta_bank_angle_max * self.control_parameters['general_args']['period'],
                                   delta_bank_angle_max * self.control_parameters['general_args']['period'])

        bank_angle_new = current_bank_angle + delta_bank_angle

        bank_angle_new = bank_angle_new + sigma_noise * np.random.standard_normal()
        bank_angle_new = np.clip(bank_angle_new, -bank_angle_max, bank_angle_max)

        return bank_angle_new

    def go_back_to_thermal(self):
        last_point = self.get_last_N_point()

        target_point = self.get_thermal_core_estimate_at_altitude(Z=last_point['X'][-1])
        if target_point is None:
            exploration_return = self.get_control_exploration()
            return exploration_return[:2]
        bank_angle_new = self.get_bank_angle_from_go_to_point(target_position=target_point)

        flight_mode = 'out_of_thermal'
        return bank_angle_new, flight_mode

    def get_thermal_core_estimate_at_altitude(self, Z):
        # Returns the X and Y positions of the thermal core estimate closest to the altitude Z
        if len(self.thermal_core_estimate) == 0:
            return None
        closest = min(self.thermal_core_estimate, key=lambda x: abs(x['Z_avg'] - Z))


        return np.array([closest['X_avg'], closest['Y_avg']])
