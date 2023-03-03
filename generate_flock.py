import datetime
import logging
import os
import pprint
import sys
import time
from shutil import copyfile, SameFileError

import dill as pickle
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.auxiliar import downsample_dataframe
from data.synthetic.generate import get_synthetic_bird
from data.synthetic.auxiliar import get_air_parameters_from_yaml, get_run_parameters_from_yaml, \
    get_bird_parameters_from_yaml
from misc.auxiliar import config_logger, flatten_dict, sanitize_dict_for_yaml
from object.air import AirVelocityField
#, AirVelocityFieldVisualization
from object.flight import BirdPoint
#from plotting.plot import inspect_flock


def main(path_to_yaml):
    with open(path_to_yaml, 'r') as file:
        import yaml
        parameters_dict = yaml.load(file, Loader=yaml.FullLoader)

    run_parameters = get_run_parameters_from_yaml(parameters_dict['run'])
    n_characters = int(np.ceil(np.log10(run_parameters['n_birds'])))
    air_parameters = get_air_parameters_from_yaml(parameters_dict['air'])

    run_time = datetime.datetime.isoformat(datetime.datetime.now()).split('.')[0]
    if run_parameters['save_folder']:
        destination_folder = os.path.join(run_parameters['save_folder'], f'{run_time}_{run_parameters["prefix"]}')
        try:
            os.makedirs(destination_folder)
        except FileExistsError as e:
            pass
    else:
        destination_folder = ''

    #verbosity_level = logging.WARNING
    logger = logging.getLogger()

    config_logger(logger, output_dir=destination_folder,
                  verbosity=run_parameters['verbosity'], log_to_file=bool(run_parameters['save_folder']))

    air_velocity_field_obj = AirVelocityField(air_parameters=air_parameters,
                                              t_start_max=run_parameters['duration']
                                                          + AirVelocityField.config['time_resolution'])
    #air_velocity_field_vis = AirVelocityFieldVisualization(air_velocity_field_obj)
    #air_velocity_field_vis.plot_all()
    #plt.show(block=True)
    start_time = time.time()
    synthetic_bird_parameters = []
    df = pd.DataFrame()
    i = 0
    while i < run_parameters['n_birds']:
        bird_name = ('0' * n_characters + str(i))
        bird_name = 'sb_' + bird_name[-n_characters:]


        init_condition, physical_parameters, control_parameters = get_bird_parameters_from_yaml(parameters_dict, i)

        core = air_velocity_field_obj.get_thermal_core(init_condition['z'], 0)
        init_condition['x'] = float(core[0] + init_condition['rho'] * np.cos(init_condition['phi']))
        init_condition['y'] = float(core[1] + init_condition['rho'] * np.sin(init_condition['phi']))

        bank_angle_init = init_condition['bank_angle']
        bearing_init = init_condition['bearing']


        logger.info(bird_name)

        for line in pprint.pformat(init_condition).split('\n'):
            logger.log(level=logging.DEBUG, msg=line)
        for line in pprint.pformat(physical_parameters).split('\n'):
            logger.log(level=logging.DEBUG, msg=line)
        for line in pprint.pformat(control_parameters).split('\n'):
            logger.log(level=logging.DEBUG, msg=line)

        bird_air_init = BirdPoint.from_bank_angle_and_bearing(bank_angle=bank_angle_init,
                                                              bearing=bearing_init,
                                                              X=[init_condition['x'],
                                                                 init_condition['y'],
                                                                 init_condition['z'],
                                                                 ],
                                                              A=[0, 0, 0], t=0,
                                                              CL=physical_parameters['CL'],
                                                              CD=physical_parameters['CD'],
                                                              mass=physical_parameters['mass'],
                                                              wing_area=physical_parameters['wing_area'])

        (current_bird, is_landed) = get_synthetic_bird(duration=run_parameters['duration'],
                                                       dt=run_parameters['dt'],
                                                       air_velocity_field_obj=air_velocity_field_obj,
                                                       bird_air_init=bird_air_init,
                                                       control_parameters=control_parameters,
                                                       debug=False
                                                       )
        if is_landed and run_parameters['skip_landed']:
            logger.info('skipping...')
            continue
        else:
            current_bird['bird_name'] = bird_name
            current_bird['bird_name'] = current_bird['bird_name'].astype('string')
            synthetic_bird_parameters.append({'bird_name': bird_name,
                                              'initial_conditions': init_condition,
                                              'physical_parameters': physical_parameters,
                                              'control_parameters': control_parameters})

            df = pd.concat([df, current_bird])
            i = i + 1

    # =======================================            SAVING             ======================================= #

    if run_parameters['inspect_before_saving']:
        #_, _, _, _, exclude_list = inspect_flock(df, X_col='X', Y_col='Y', Z_col='Z', time_col='time',
        #                                         bird_label_col='bird_name', color=None,
        #                                         air_velocity_field=air_velocity_field_obj)
        all_birds = map(lambda elem: elem['bird_name'], synthetic_bird_parameters)
        birds_to_keep = list(all_birds) #  [bird for bird in all_birds if bird not in exclude_list]

        synthetic_bird_parameters = list(filter(lambda x: x['bird_name'] in birds_to_keep, synthetic_bird_parameters))
        df = df[df['bird_name'].isin(birds_to_keep)]

    df_output = df[['bird_name', 'time', 'X', 'Y', 'Z']].copy()
    df_output = downsample_dataframe(df_output, round(run_parameters['dt_to_save'] / run_parameters['dt']),
                                     partition_key='bird_name')
    for coord in ['X', 'Y', 'Z']:
        df_output[coord] = df_output[coord] \
                           + run_parameters['noise_level'] * np.random.standard_normal(df_output[coord].count())

    if run_parameters['save_folder']:
        try:
            copyfile(sys.argv[1], os.path.join(destination_folder, 'parameters.yml'))
        except SameFileError:
            pass

        with open(os.path.join(destination_folder, 'air_velocity_field.pkl'), 'wb') as f:
            pickle.dump(air_velocity_field_obj, f)
        with open(os.path.join(destination_folder, 'air_parameters.pkl'), 'wb') as f:
            pickle.dump(air_parameters, f)

        with open(os.path.join(destination_folder, 'bird_parameters.yml'), 'w') as f:
            import yaml
            yaml.dump(list(map(sanitize_dict_for_yaml, synthetic_bird_parameters)), f, default_flow_style=False)

        synthetic_bird_parameters = [flatten_dict(d) for d in synthetic_bird_parameters]
        df_synthetic_bird_parameters = pd.DataFrame(synthetic_bird_parameters)
        df_synthetic_bird_parameters.to_csv(os.path.join(destination_folder, 'bird_parameters.csv'),
                                            index=False, sep=',')

        df.to_csv(os.path.join(destination_folder, 'data_real.csv'), index=False, sep=',')
        df_output.to_csv(os.path.join(destination_folder, 'data.csv'), index=False, sep=',')

        logger.info(f'saved to {destination_folder}')

    run_duration = time.time() - start_time
    logger.info(f'this took {round(run_duration/60.,1)}')


if __name__ == '__main__':
    main(sys.argv[1])
