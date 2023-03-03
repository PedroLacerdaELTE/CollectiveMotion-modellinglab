from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from calc.geometry import get_curvature_from_trajectory
from data.get_data import load_synthetic_data
from misc.auxiliar import flatten_dict, sanitize_dict_for_yaml


def std_from_IQR(s, axis=0):
    return (np.nanquantile(s, 0.75, axis=axis) - np.nanquantile(s, 0.25, axis=axis)) / 1.35


def get_flock_parameter_stats_from_bird_parameters(df_bird_parameters):
    flock_stats = {}
    df_dict = {}
    df_bird_parameters = df_bird_parameters[[col for col in list(df_bird_parameters.columns) if 'debug' not in col]]
    for parameter_type in ['control_parameters', 'initial_conditions', 'physical_parameters']:
        current_df = df_bird_parameters[['bird_name'] + [col for col in list(df_bird_parameters.columns) if parameter_type in col]].copy()
        current_df.rename(columns={col: col.replace(f'{parameter_type}.', '') for col in current_df.columns},
                          inplace=True)
        current_df.set_index('bird_name', inplace=True)
        current_df.sort_index(inplace=True)

        flock_stats[parameter_type] = current_df.describe().drop('count').to_dict()
        df_dict[parameter_type] = current_df

    multi_level_columns = reduce(lambda a, b: a + b,
                                 [[(param_type, col) for col in param_dict.keys()] for param_type, param_dict in
                                  df_dict.items()])

    df_param = pd.DataFrame(columns=pd.MultiIndex.from_tuples(multi_level_columns, names=['parameter_type',
                                                                                          'parameter']))
    for param_type, param_dict in df_dict.items():
        df_param[param_type] = param_dict

    return flock_stats, df_param


def get_bird_stats_from_bird_trajectories(df, cols_to_aggregate=None):

    if cols_to_aggregate is None:
        cols_to_aggregate = ['dXdT_air_real',
                             'dYdT_air_real',
                             'dZdT_air_real',
                             'dXdT_air_wind_real',
                             'dYdT_air_wind_real',
                             'dZdT_air_wind_real',
                             'dXdT_air_rotation_real',
                             'dYdT_air_rotation_real',
                             'dZdT_air_rotation_real',
                             'dXdT_air_thermal_real',
                             'dYdT_air_thermal_real',
                             'dZdT_air_thermal_real',
                             'curvature_bird_real',
                             'curvature',
                             'X_thermal_real',
                             'Y_thermal_real',
                             'rho_thermal_real',
                             'phi_thermal_real',
                             ]
    df_grouped = df.groupby('bird_name').agg(**{f'{col}_{stat.__name__}': (col, stat)
                                                for col, stat in product(cols_to_aggregate,
                                                                         [np.nanmean,
                                                                          np.nanstd,
                                                                          np.nanmedian,
                                                                          np.nanmax,
                                                                          np.nanmin,
                                                                          std_from_IQR])})

    return df_grouped


def get_flock_stats_from_bird_trajectories(df, cols_to_aggregate=None):

    if cols_to_aggregate is None:
        cols_to_aggregate = ['dXdT_air_real',
                             'dYdT_air_real',
                             'dZdT_air_real',
                             'dXdT_air_wind_real',
                             'dYdT_air_wind_real',
                             'dZdT_air_wind_real',
                             'dXdT_air_rotation_real',
                             'dYdT_air_rotation_real',
                             'dZdT_air_rotation_real',
                             'dXdT_air_thermal_real',
                             'dYdT_air_thermal_real',
                             'dZdT_air_thermal_real',
                             'curvature_bird_real',
                             'curvature',
                             'X_thermal_real',
                             'Y_thermal_real',
                             'rho_thermal_real',
                             'phi_thermal_real',
                             ]

    df_stats = pd.DataFrame()
    for col in cols_to_aggregate:
        current_data = df.replace(to_replace=np.inf,
                                    value=np.nan)[col]
        for stat in [np.nanmean, np.nanstd, np.nanmedian, std_from_IQR, np.nanmax, np.nanmin]:
            df_stats.loc[col, stat.__name__] = stat(current_data, axis=0)
        df_stats.loc[col, 'percentile_1'] = np.nanpercentile(current_data,
                                                              1,
                                                              axis=0)
        df_stats.loc[col, 'percentile_99'] = np.nanpercentile(current_data,
                                                              99,
                                                              axis=0)
    return df_stats


synt_data = load_synthetic_data('synthetic_data/really_constant_wind')

df_real = synt_data['data_real']
air_velocity_field = synt_data['air_velocity_field']
param = synt_data['bird_parameters']
flatten_params = [flatten_dict(e) for e in param]

df_parameters = pd.DataFrame(flatten_params)


f_stats, dd = get_flock_parameter_stats_from_bird_parameters(df_parameters)

list_of_birds = df_parameters['bird_name']

for para_type in dd.columns.get_level_values(0).unique():
    current_df = dd[para_type]
    n_plots = len(current_df.columns)
    n_rows = np.ceil(np.sqrt(n_plots)).astype(int)
    n_cols = np.ceil(n_plots / n_rows).astype(int)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(19,12), constrained_layout=True)
    fig.suptitle(para_type)
    ax = ax.flatten()
    for i, current_col in enumerate(current_df.columns):
        ax[i].hist(current_df[current_col], bins=len(list_of_birds), align='left')
        ax[i].set_title(current_col)
        mu = f_stats[para_type][current_col]['mean']
        std = f_stats[para_type][current_col]['std']
        ax[i].axvline(mu, c='r')
        ax[i].axvline(mu + std, c='r', ls='--')
        ax[i].axvline(mu - std, c='r', ls='--')
        #ax[i].set_xticklabels([e.get_label() for e in ax[i].get_xticklabels()], rotation=90)
    plt.show(block=True)


list_of_cols = ['dXdT_air_real',
                'dYdT_air_real',
                'dZdT_air_real',
                #'dXdT_air_wind_real',
                #'dYdT_air_wind_real',
                #'dZdT_air_wind_real',
                'dXdT_air_thermal_real',
                'dYdT_air_thermal_real',
                'dZdT_air_thermal_real',
                #'curvature_bird_real',
                #'radius',
                'radius_abs',
                #'curvature',
                'curvature_abs',
                #'rho_thermal_real',
                ]

df_real['radius_abs'] = np.abs(df_real['radius'])
df_real['curvature_abs'] = np.abs(df_real['curvature'])
df_real['curvature_bird_real'] = df_real['curvature']
aaa = get_flock_stats_from_bird_trajectories(df_real, cols_to_aggregate=list_of_cols)

fig, ax = plt.subplots(1,1)

for i_col, col in enumerate(list_of_cols):
    print(i_col, col)
    ax.violinplot(df_real.loc[df_real[col].between(aaa.loc[col, 'percentile_1'],aaa.loc[col, 'percentile_99']), col
                  ].values, vert=False, positions=[i_col],
                  showmedians=True)

ax.set_yticks(np.arange(len(list_of_cols)),
              labels=list_of_cols)
ax.legend()
plt.show(block=True)