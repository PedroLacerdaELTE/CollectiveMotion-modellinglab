import numbers
import time

import numpy as np
import yaml

from calc.thermal import get_ND_random_walk


def get_flock_stats(list_of_bird_parameters, return_data=False):
    if isinstance(list_of_bird_parameters, str):
        with open(list_of_bird_parameters, 'r') as f:
            list_of_bird_parameters = yaml.load(f, yaml.FullLoader)

    one_bird = list_of_bird_parameters[00]

    stats_dict = {}
    for parameter_set in one_bird:
        if not isinstance(one_bird[parameter_set], (dict, numbers.Number)):
            continue
        if isinstance(one_bird[parameter_set], dict):
            stats_dict[parameter_set] = {}
        for parameter in one_bird[parameter_set]:

            if isinstance(one_bird[parameter_set][parameter], numbers.Number):

                parameter_data = list(map(lambda x: x[parameter_set][parameter],
                                          list_of_bird_parameters))

                parameter_stats = {'mean': np.mean(parameter_data),
                                   'std': np.std(parameter_data),
                                   'N': len(parameter_data)}
                if return_data:
                    parameter_stats['sample'] = parameter_data
                stats_dict[parameter_set][parameter] = parameter_stats.copy()
            elif isinstance(one_bird[parameter_set][parameter], dict):
                stats_dict[parameter_set][parameter] = {}
                for subparameter in one_bird[parameter_set][parameter]:
                    parameter_data = list(map(lambda x: x[parameter_set][parameter][subparameter],
                                              list_of_bird_parameters))

                    parameter_stats = {'mean': np.mean(parameter_data),
                                       'std': np.std(parameter_data),
                                       'N': len(parameter_data)}
                    if return_data:
                        parameter_stats['sample'] = parameter_data

                    stats_dict[parameter_set][parameter][subparameter] = parameter_stats.copy()

    return stats_dict


DEFAULT_YAML_PATH = 'config/default/bird_generate.default.yaml'
PARAMETERS_DICT = {
    'run': ['save_folder', 'n_birds', 'dt', # in seconds,
            'dt_to_save',# in seconds
            'duration',  # in seconds,
            'noise_level', 'prefix', 'skip_landed', 'random_seed', 'inspect_before_saving', 'verbosity'],
    'initial_conditions': ['rho', 'phi', 'z', 'bank_angle', 'bearing'],
    'physical_parameters': ['CL', 'CD', 'mass', 'wing_area'],
    'control_parameters': {'general_args': ['period',
                                            'bank_angle_max',
                                            'delta_bank_angle_max',
                                            'sigma_noise_degrees',
                                            'N_out_of_thermal',
                                            'exploration_exploitation_ratio',
                                            'debug'],
                           'exploration': ['alpha_degrees', 'time'],
                           'exploitation': ['K', 'thermalling_bank_angle']
                           },
    'air': [{'thermal': ['rotation', 'profile']},
            'wind',
            'rho',
            'turbulence']
}


def get_defaults_for_air(component):
    with open(DEFAULT_YAML_PATH) as f:
        default_yaml = yaml.load(f, Loader=yaml.FullLoader)

    if component != 'thermal':
        if component in default_yaml['air']:
            return parse_air_yaml(default_yaml['air'][component])
        else:
            return None
    else:
        if component == 'rotation':
            parameters = {'radius': 20,
                          'A': 2}
        else:
            parameters = {'radius': 20,
                          'A': 3}

        if component == 'rotation':
            def function(r, theta, z, t):
                import numpy as np
                radius = parameters['radius']
                A_rotation = parameters['A']

                # K is the constant so that magnitude is A_rotation at r=radius/2
                K = 4 * A_rotation / radius ** 2
                if r > radius:
                    return [0, 0]
                else:
                    magnitude = K * r * (radius - r)

                    return [-magnitude * np.sin(theta),
                            magnitude * np.cos(theta)]
        else:
            def function(r, theta, z, t):
                import numpy as np
                A = parameters['A']
                radius = parameters['radius']

                return A * np.exp(-np.power(r, 2.) / (2 * np.power(radius, 2.)))
        return function


def get_defaults(parameter, parameter_set):
    with open(DEFAULT_YAML_PATH) as f:
        default_yaml = yaml.load(f, Loader=yaml.FullLoader)

    if parameter_set == 'run':
        return parse_bird_yaml(default_yaml['run'][parameter])
    if parameter_set == 'initial_conditions':
        return parse_bird_yaml(default_yaml['bird']['initial_conditions'][parameter])
    if parameter_set == 'physical_parameters':
        return parse_bird_yaml(default_yaml['bird']['physical_parameters'][parameter])
    if parameter_set == 'control_general_args':
        return parse_bird_yaml(default_yaml['bird']['control_parameters']['general_args'][parameter])
    if parameter_set == 'control_exploitation':
        return parse_bird_yaml(default_yaml['bird']['control_parameters']['exploitation'][parameter])
    if parameter_set == 'control_exploration':
        return parse_bird_yaml(default_yaml['bird']['control_parameters']['exploration'][parameter])
    if parameter_set == 'air':
        return get_defaults_for_air(parameter)


def parse_bird_yaml(yaml_struct, bird_idx=None):
    # CONSTANT
    if isinstance(yaml_struct, (int, float, str, type(None))):
        return yaml_struct
    # FROM LIST
    elif isinstance(yaml_struct, list):
        return yaml_struct[bird_idx]
    else:
        if 'distribution' in yaml_struct:
            # FROM RANDOM DISTRIBUTION
            try:
                distribution = getattr(np.random, yaml_struct['distribution'])
            except AttributeError as e:
                raise Exception("distribution keyword must match one distribution provided by numpy. " +
                                "Check https://numpy.org/doc/1.16/reference/routines.random.html#distributions")

            return distribution(**yaml_struct['parameters'])
        else:
            return yaml_struct


def generate_air_data(generate_yaml, shape=2):
    limits = generate_yaml['limits']
    mgs = np.meshgrid(*[np.linspace(*limits, generate_yaml['n_steps']) for limits in limits.values()])
    used_vars = list(limits.keys())
    used_vars = ''.join(used_vars)
    n_vars = len(limits)
    n_steps = generate_yaml['n_steps']

    xyzt_array = np.stack([mg.flatten() for mg in mgs], axis=-1)
    if generate_yaml['method'] == 'random_walk':
        generate_yaml.pop('method')
        values = get_ND_random_walk(generate_yaml['mean'],
                                    generate_yaml['std'],n_vars,n_steps)

    elif generate_yaml['method'] == 'random':

        if 'distribution' in generate_yaml:
            # FROM RANDOM DISTRIBUTION
            try:
                distribution = getattr(np.random, generate_yaml['distribution'])
            except AttributeError as e:
                raise Exception("distribution keyword must match one distribution provided by numpy. " +
                                "Check https://numpy.org/doc/1.16/reference/routines.random.html#distributions")

            values = distribution(size=(n_steps ** n_vars, shape), **generate_yaml['parameters'])

    return {'XYZT_values': xyzt_array, 'values': values, 'used_vars': used_vars}


def get_air_function(yaml_struct):
    if yaml_struct['function'] == 'quadratic':
        def quadratic_function(r, theta, z, t):
            import numpy as np
            radius = yaml_struct['parameters']['radius']
            A_rotation = yaml_struct['parameters']['A']

            if r > radius:
                return [0, 0]
            else:
                # K is the constant so that magnitude is A_rotation at r=radius/2
                K = 4 * A_rotation / radius ** 2
                magnitude = K * r * (radius - r)

                return [-magnitude * np.sin(theta),
                        magnitude * np.cos(theta)]

        return quadratic_function

    elif yaml_struct['function'] == 'gaussian':
        def gaussian_function(r, theta, z, t):
            import numpy as np
            A = yaml_struct['parameters']['A']
            radius = yaml_struct['parameters']['radius']

            return A * np.exp(-np.power(r, 2.) / (2 * np.power(radius, 2.)))
        return gaussian_function
    else:
        raise Exception(f"function {yaml_struct['function']} unrecognized")


def parse_air_yaml(yaml_struct):
    # CONSTANT
    if isinstance(yaml_struct, (int, float, str, type(None), list)):
        return yaml_struct
    else:
        if 'generate' in yaml_struct:
            return generate_air_data(yaml_struct['generate'])
        # METHOD - FUNCTION
        elif 'from_function' in yaml_struct:
            return get_air_function(yaml_struct['from_function'])
        elif 'from_data' in yaml_struct:
            yaml_struct['from_data']['XYZT_values'] = np.array(yaml_struct['from_data']['XYZT_values'])
            yaml_struct['from_data']['values'] = np.array(yaml_struct['from_data']['values'])
            return yaml_struct['from_data']
        else:
            return yaml_struct


def get_initial_condition_from_yaml(initial_conditions_yaml, i=None):
    initial_conditions = {}
    for coord in PARAMETERS_DICT['initial_conditions']:

        if coord in initial_conditions_yaml:
            initial_conditions[coord] = parse_bird_yaml(initial_conditions_yaml[coord], i)
        else:
            initial_conditions[coord] = get_defaults(coord, 'initial_conditions')

    initial_conditions['bank_angle'] *= np.pi / 180

    return initial_conditions


def get_physical_parameters_from_yaml(physical_parameters_yaml, i=None):
    physical_parameters = {}
    for parameter in PARAMETERS_DICT['physical_parameters']:
        if parameter in physical_parameters_yaml:
            physical_parameters[parameter] = parse_bird_yaml(physical_parameters_yaml[parameter], i)
        else:
            physical_parameters[parameter] = get_defaults(parameter, 'physical_parameters')

    return physical_parameters


def get_control_parameters_from_yaml(control_parameters_yaml, i=None):
    list_parameter_sets = list(PARAMETERS_DICT['control_parameters'].keys())
    control_parameters = {parameter_set: {} for parameter_set in list_parameter_sets}
    # GENERAL ARGS

    for parameter_set in list_parameter_sets:
        if parameter_set in control_parameters_yaml:
            for parameter in PARAMETERS_DICT['control_parameters'][parameter_set]:
                if parameter in control_parameters_yaml[parameter_set]:
                    control_parameters[parameter_set][parameter] = parse_bird_yaml(
                        control_parameters_yaml[parameter_set][parameter], i)
                else:
                    control_parameters[parameter_set][parameter] = get_defaults(parameter, f'control_{parameter_set}')
        else:
            for parameter in PARAMETERS_DICT['control_parameters'][parameter_set]:
                control_parameters[parameter_set][parameter] = get_defaults(parameter, f'control_{parameter_set}')

    return control_parameters


def get_air_parameters_from_yaml(air_yaml):
    air_parameters = {}

    for component in PARAMETERS_DICT['air']:
        if not isinstance(component, dict):  # This is a proxy for thermal
            if component not in air_yaml:
                air_parameters[component] = get_defaults_for_air(component)
            else:
                air_parameters[component] = parse_air_yaml(air_yaml[component])
        else:
            air_parameters['thermal'] = {}

            for parameter in PARAMETERS_DICT['air'][0]['thermal']:
                if parameter not in air_yaml['thermal']:
                    air_parameters['thermal'][parameter] = get_defaults_for_air(parameter)
                else:
                    air_parameters['thermal'][parameter] = parse_air_yaml(air_yaml['thermal'][parameter])

    return air_parameters


def get_run_parameters_from_yaml(run_yaml):
    run_parameters = {}

    for parameter in PARAMETERS_DICT['run']:

        if parameter in run_yaml:
            run_parameters[parameter] = parse_bird_yaml(run_yaml[parameter])
        else:
            run_parameters[parameter] = get_defaults(parameter, 'run')

    if ('random_seed' in run_parameters) and (run_parameters['random_seed'] is not None):
        np.random.seed(run_parameters['random_seed'])
    else:
        random_seed = int(time.time())
        np.random.seed(random_seed)
        run_parameters['random_seed'] = random_seed

    return run_parameters


def get_bird_parameters_from_yaml(yaml_dict=None, i=None):

    if yaml_dict is None:
        bird_yaml = {'initial_conditions': {},
                     'physical_parameters': {},
                     'control_parameters': {}}
    else:
        bird_yaml = yaml_dict['bird']

    initial_conditions = get_initial_condition_from_yaml(bird_yaml['initial_conditions'], i=i)
    physical_parameters = get_physical_parameters_from_yaml(bird_yaml['physical_parameters'], i=i)
    control_parameters = get_control_parameters_from_yaml(bird_yaml['control_parameters'], i=i)

    return initial_conditions, physical_parameters, control_parameters