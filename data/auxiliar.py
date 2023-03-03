import pandas as pd
import yaml


def downsample_dataframe(df, factor=10, partition_key='bird_name'):
    df_return = pd.DataFrame(columns=df.columns)

    unique_partitions = df[partition_key].unique()

    for partition in unique_partitions:
        df_partition = df[df[partition_key] == partition]
        df_partition = df_partition.iloc[::factor]
        df_return = pd.concat([df_return, df_partition])

    df_return = df_return.reset_index(drop=True)
    return df_return


def get_args_from_yaml_with_default(path_to_yaml, default_parameters):
    with open(path_to_yaml, 'r') as f:
        yaml_dict = yaml.load(f, yaml.FullLoader)
    parameter_dict = default_parameters.copy()

    for parameter_set in default_parameters.keys():
        if parameter_set in yaml_dict:
            if yaml_dict[parameter_set] is not None:
                parameter_dict[parameter_set].update(yaml_dict[parameter_set])
            else:
                parameter_dict[parameter_set] = yaml_dict[parameter_set]
    return parameter_dict

