import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from calc.flight import get_radius_from_bank_angle


def get_is_in_thermal(Vz_array, N=10):
    Vz = np.array(Vz_array[-N:])
    if np.all(Vz < 0):
        return False
    else:
        return True

def go_to_thermal_core():
    pass


def get_thermal_core_estimation(X_array, Y_array, Z_array, bearing_array, weights_array, N_circles, N_min,
                                alpha=1, debug=False ):

    if len(bearing_array) < N_min:
        N = N_min
    else:
        new_circle = np.isclose(np.array(bearing_array) - bearing_array[-1], 0, atol=0.05)
        new_circle = len(bearing_array) - np.argwhere(new_circle) # Counting from the end
        new_circle = new_circle[new_circle > N_min]
        if len(new_circle):
            N = new_circle[-N_circles]
        else:
            N = N_min

    weights = np.array(weights_array[-N:])
    weights = np.exp(alpha*(weights - np.average(weights)))
    #from scipy import signal
    #kernel = signal.windows.exponential(N, 0, N/2, False)[::-1]
    #weights = weights * kernel
    X_weighted_average = np.average(X_array[-N:], weights=weights)
    Y_weighted_average = np.average(Y_array[-N:], weights=weights)
    Z_weighted_average = np.average(Z_array[-N:], weights=weights)

    X_weighted_std = np.sqrt(np.average((X_array[-N:] - X_weighted_average) ** 2, weights=weights))
    Y_weighted_std = np.sqrt(np.average((Y_array[-N:] - Y_weighted_average) ** 2, weights=weights))
    Z_weighted_std = np.sqrt(np.average((Z_array[-N:] - Z_weighted_average) ** 2, weights=weights))

    thermal_core_estimation_avg = [X_weighted_average, Y_weighted_average, Z_weighted_average]
    thermal_core_estimation_std = [X_weighted_std, Y_weighted_std, Z_weighted_std]

    if debug:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.scatter(X_array, Y_array)
        aaa = ax.scatter(X_array[-N:], Y_array[-N:], c=weights_array[-N:],
                         norm=Normalize(vmin=-2, vmax=5),
                         cmap='Spectral_r', label='track')
        ax.scatter(thermal_core_estimation_avg[0], thermal_core_estimation_avg[1], marker='x', label='weighted\naverage')

        ax.scatter([0], [0], marker='o', label='real core')
        ax.set_aspect('equal')
        plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1))
        plt.colorbar(aaa)
        plt.show(block=True)

    return {'X_avg': thermal_core_estimation_avg[0],
            'Y_avg': thermal_core_estimation_avg[1],
            'Z_avg': thermal_core_estimation_avg[2],
            'X_std': thermal_core_estimation_std[0],
            'Y_std': thermal_core_estimation_std[1],
            'Z_std': thermal_core_estimation_std[2],
            'N': N}


def get_bank_angle_update_random(bank_angle, trajectory_obj,sigma_bank_angle_degrees=5):
    # Conver to radians
    sigma_bank_angle = sigma_bank_angle_degrees * np.pi / 180
    delta_bank_angle = sigma_bank_angle * np.random.standard_normal(1)[0]
    return delta_bank_angle


def get_bank_angle_update_reichmann(bank_angle, trajectory_obj, Az=None, N=5, alpha_degrees=15):
    # Conver to radians
    if Az is None:
        Az = np.average(trajectory_obj['A'][-N:, 2])
    alpha = alpha_degrees * np.pi / 180
    if np.isclose(Az, 0, atol=0.05):
        return 0
    elif Az > 0 and np.isclose(bank_angle, 0, atol=0.1):
        return 0
    else:
        delta_bank_angle = - alpha * np.sign(Az)
        return delta_bank_angle


def get_orthogonal_unit_vector(V):
    # orthogonal unit vector such that V x unit_vector > 0
    rotation_matrix = np.array([[0, -1], [1, 0]])

    unit_vector = rotation_matrix @ V / np.linalg.norm(V)
    return unit_vector


def get_bank_angle_from_delta_V(bank_angle, current_velocity, delta_velocity, CL, wing_area, alpha=1, rho=1.225,
                                debug=False):

    e_f = get_orthogonal_unit_vector(current_velocity)
    projected_delta_V = delta_velocity @ e_f
    force = alpha * projected_delta_V
    if np.isclose(np.linalg.norm(delta_velocity), 0, atol=0.1):
        bank_angle_new = 0
    else:
        cos_theta = projected_delta_V / (np.linalg.norm(delta_velocity))

        if np.isclose(cos_theta, 1, atol=0.01):
            bank_angle_new = 0
        else:
            try:
                bank_angle_new = np.arcsin(2 * force / (CL * rho * wing_area
                                                            * np.linalg.norm(current_velocity) ** 2))
            except RuntimeWarning as e:
                #print(e)
                bank_angle_new = np.sign(projected_delta_V) * np.pi/2
    if debug:
        fig = plt.figure()
        fig.suptitle(f'{bank_angle_new * 180 / np.pi:.2g}')
        ax = fig.add_subplot()  # projection='3d'
        ax.arrow(0, 0,
                 *current_velocity,
                 fc='r', ec='r', width=0.05, length_includes_head=True, label='current_velocity')
        ax.arrow(0, 0,
                 *delta_velocity,
                 fc='b', ec='b', width=0.05, length_includes_head=True, label='delta_velocity')
        ax.arrow(0,0,
                 *(e_f * projected_delta_V), width=0.05, fc='g', ec='g', length_includes_head=True, label='projected $\\Delta V$')
        ax.set_aspect('equal')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.show(block=False)

    return bank_angle_new

def get_bank_angle_update_weighted_average(bank_angle, trajectory_obj, K, K_reichmann=1,
                                           radius_ref=20, thermalling_bank_angle=30, debug=False,
                                           air_obj=None):

    if trajectory_obj['thermal_core_estimate'][-1] is None:
        return get_bank_angle_update_reichmann(bank_angle, np.average(trajectory_obj['A'][-5:, 2]))
        #if N < 100:
        #    return get_bank_angle_update_reichmann(bank_angle, Az)

    thermal_core_estimate = trajectory_obj['thermal_core_estimate'][-1]
    thermal_core_estimate, N = np.array([thermal_core_estimate['X_avg'],
                                         thermal_core_estimate['Y_avg'],
                                         thermal_core_estimate['Z_avg']]), \
                               thermal_core_estimate['N']

    last_point = trajectory_obj.get_last_N_point()
    X_array = trajectory_obj['X'][-N:, 0]
    Y_array = trajectory_obj['X'][-N:, 1]

    current_position = last_point['X'][:2]


    current_velocity = last_point['V'][:2]
    current_acceleration_z = last_point['A'][2]
    bird_parameters = last_point.get_bird_properties()

    scale = get_radius_from_bank_angle(bank_angle=thermalling_bank_angle * np.pi / 180,
                                       mass=bird_parameters['mass'],
                                       wing_area=bird_parameters['wing_area'],
                                       CL=bird_parameters['CL'])
    if last_point['radius'] < radius_ref:
        radius = radius_ref #last_point['radius']
    else:
        radius = radius_ref

    centripetal_unit_vector = np.sign(bank_angle) * get_orthogonal_unit_vector(current_velocity)
    radius_vector = centripetal_unit_vector * radius  # radius_ref
    delta_core = thermal_core_estimate[:2] - (current_position + radius_vector) # CW

    theta = np.cross(delta_core, current_velocity) / (np.linalg.norm(delta_core)
                                                      * np.linalg.norm(current_velocity)
                                                      + 0.0001)
    theta = np.arcsin(theta)
    inner_product = np.inner(delta_core, current_velocity)

    scaled_distance = np.linalg.norm(delta_core) / scale

    target_position = thermal_core_estimate[:2] - np.sign(bank_angle) * get_orthogonal_unit_vector(delta_core) * radius_ref
    # APPROACH
    if scaled_distance > 0.5: # GO TO THERMAL
        flight_mode = 'GoToThermal'
        if inner_product > 0:
            delta_position = target_position - current_position
            target_velocity = delta_position / np.linalg.norm(delta_position) * np.linalg.norm(current_velocity)
            delta_velocity = target_velocity - current_velocity
            bank_angle_new = get_bank_angle_from_delta_V(current_velocity, delta_velocity,
                                                         wing_area=bird_parameters['wing_area'],
                                                         CL=bird_parameters['CL'])
            delta_bank_angle = K * (bank_angle_new - np.abs(bank_angle))
        else:
            delta_bank_angle = np.pi/2
    else: #THERMALLING
        flight_mode="Thermalling"
        delta_bank_angle = K / np.linalg.norm(delta_core)
        delta_bank_angle = K * (thermalling_bank_angle * np.pi / 180 - np.abs(bank_angle) )
        delta_bank_angle = delta_bank_angle - K_reichmann * current_acceleration_z

    if debug:
        bank_angle_array = trajectory_obj['bank_angle'][-N:]
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.scatter(X_array, Y_array)

        ax.scatter(target_position[0], target_position[1], marker='x', label='target position')
        aaa = ax.scatter(X_array[-N:], Y_array[-N:], c=np.abs(bank_angle_array[-N:]),
                   norm=Normalize(vmin=0, vmax=np.pi/2),
                   cmap='Spectral_r', label='track')
        #ax.scatter([X_average], [Y_average], marker='x', label='average')

        ax.quiver(current_position[0], current_position[1],
                  radius_vector[0], radius_vector[1],
                  scale=1, scale_units='x')
        ax.scatter(current_position[0] + radius_vector[0], current_position[1] + radius_vector[1],
                   marker='o', label='center of\ntrajectory')
        ax.quiver(current_position[0] + radius_vector[0], current_position[1] + radius_vector[1],
                  delta_core[0], delta_core[1],
                  scale=1, scale_units='x')
        ax.scatter(thermal_core_estimate[0], thermal_core_estimate[1],
                   marker='x', label='weighted\naverage')
        ax.quiver(thermal_core_estimate[0], thermal_core_estimate[1],
                  target_position[0] - thermal_core_estimate[0], target_position[1] - thermal_core_estimate[1],
                  scale=1, scale_units='x')

        ax.scatter(air_obj.get_thermal_core(last_point['X'][2])[0],
                   air_obj.get_thermal_core(last_point['X'][2])[1], marker='o', label='real core')
        ax.quiver(current_position[0], current_position[1],
                  current_velocity[0], current_velocity[1])
        ax.set_aspect('equal')
        plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1))
        plt.colorbar(aaa)
        fig.suptitle(f'distance = {round(scaled_distance,2)}, N = {N},'
                     f'delta = {round(delta_bank_angle * 180 / np.pi),}'
                     f'bank_angle = {round(bank_angle * 180 / np.pi)}\n'
                     f'{flight_mode}')
        plt.show(block=False)
    return delta_bank_angle


def get_new_bank_angle(trajectory_obj, control_parameters, control_type, i_last_update):
    # Convert to radians
    current_bank_angle = trajectory_obj.get_last_N_point()['bank_angle']
    bank_angle_max = control_parameters['general_args']['bank_angle_max'] * np.pi / 180
    delta_bank_angle_max = control_parameters['general_args']['delta_bank_angle_max'] * np.pi / 180
    sigma_noise_degrees =control_parameters['general_args']['sigma_noise_degrees'] * np.pi / 180

    dt = control_parameters['general_args']['period']
    stiffness = control_parameters['general_args']['stiffness']

    if control_parameters[control_type]['algorithm'] == 'reichmann':
        delta_bank_angle = get_bank_angle_update_reichmann(current_bank_angle, trajectory_obj,
                                                           **control_parameters[control_type]['args'])
    elif control_parameters[control_type]['algorithm'] == 'weighted_average':
        delta_bank_angle = get_bank_angle_update_weighted_average(current_bank_angle, trajectory_obj,
                                                                  **control_parameters[control_type]['args'])
    else:
        delta_bank_angle = get_bank_angle_update_random(current_bank_angle, trajectory_obj,
                                                        **control_parameters[control_type]['args'])


    delta_bank_angle = np.clip(delta_bank_angle, -delta_bank_angle_max, delta_bank_angle_max)
    #delta_bank_angle = delta_bank_angle_max * np.arctan( stiffness * delta_bank_angle)
    delta_bank_angle = delta_bank_angle * dt
    if current_bank_angle >= 0:
        bank_angle_new = current_bank_angle + delta_bank_angle
    else:
        bank_angle_new = current_bank_angle - delta_bank_angle

    bank_angle_new = bank_angle_new + sigma_noise_degrees * np.pi / 180 * np.random.standard_normal()
    bank_angle_new = np.clip(bank_angle_new, -bank_angle_max, bank_angle_max)

    return bank_angle_new


def get_control(t, trajectory_ground, control_parameters, i_last_update, air_obj):

    if t < control_parameters['exploration']['time']:
        # Exploration
        # control_parameters['exploration']['args'].update({'Az': np.average((np.diff(V_air_array[:, 2]) / dt )[i_last_update:]) })
        bank_angle_new = get_new_bank_angle(trajectory_ground,
                                            control_parameters,
                                            control_type='exploration',
                                            i_last_update=i_last_update)
    else:
        # Exploitation
        if control_parameters['exploitation']['algorithm'] == 'reichmann':
            control_parameters['exploitation']['args'].update({})
        elif control_parameters['exploitation']['algorithm'] == 'weighted_average':
            control_parameters['exploitation']['args'].update({'air_obj': air_obj }
                                                              )
        bank_angle_new = get_new_bank_angle(trajectory_ground,
                                            control_parameters,
                                            control_type='exploitation',  i_last_update=i_last_update)


    return bank_angle_new