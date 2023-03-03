import numpy as np
import pandas as pd
import os
import dill as pickle
import yaml


def load_decomposition_data(path, list_of_files=None, iteration=None):
    if iteration is not None:
        if np.isscalar(iteration):
            iteration = [iteration]
    if list_of_files is None:
        list_of_files = ['iterations', 'bins', 'thermal_core', 'vz_max', 'aerodynamic_parameters', 'splines']

    df_dict = {}
    for key in list_of_files:
        try:
            with open(os.path.join(path, f'{key}.pkl'), 'rb') as f:
                df_dict[key] = pickle.load(f)
                if isinstance(df_dict[key], dict):
                    df_dict[key] = pd.DataFrame.from_dict([df_dict[key]], orient='columns')
                if iteration is not None:
                    df_dict[key] = df_dict[key][df_dict[key]['iteration'].isin(iteration)]
        except FileNotFoundError:
            pass

    with open(os.path.join(path, 'decomposition_args.yml'), 'r') as f:
        import yaml

        df_dict['decomposition_args'] = yaml.load(f, yaml.FullLoader)

    return df_dict


def load_synthetic_data(path, list_of_object=None):

    if list_of_object is None:
        list_of_object = ['air_parameters.pkl',
                          'air_velocity_field.pkl',
                          'data.csv',
                          'data_real.csv',
                          'synthetic_run_params.pkl',
                          'bird_parameters.yml',
                          'bird_parameters.pkl'
                          ]
    return_dict = {}
    for file in list_of_object:
        if not os.path.exists(os.path.join(path, file)):
            continue
        filetype = file.split('.')[-1]
        file_name = '.'.join(file.split('.')[:-1])
        if filetype == 'pkl':
            with open(os.path.join(path, file), 'rb') as f:
                return_dict[file_name] = pickle.load(f)
        elif filetype == 'csv':
            return_dict[file_name] = pd.read_csv(os.path.join(path, file))
        else:
            try:
                with open(os.path.join(path, file), 'r') as f:
                    return_dict[file_name] = yaml.load(f, Loader=yaml.FullLoader)
            except:
                continue

    return return_dict


def load_synthetic_and_decomposed(path_to_decomposition, list_of_files=None, input_folder=None, iteration=None):

    decomposition_dict = load_decomposition_data(path_to_decomposition, list_of_files=list_of_files,
                                                 iteration=iteration)

    decomposition_args = decomposition_dict['decomposition_args']
    if input_folder is None:
        path_to_synthetic_data = decomposition_args['run_parameters']['input_folder']
    else:
        path_to_synthetic_data = input_folder

    synthetic_data_dict = load_synthetic_data(path_to_synthetic_data, list_of_object=list_of_files)

    return synthetic_data_dict, decomposition_dict
