"""Python script to provide general utility functions."""
import os
import re
import numpy as np
import torch
from torch import optim
import pandas as pd
import scipy
import json


def get_common_constants_for_index(params, idx):
    common_constants = params[params['run']]['common_constants']
    constants_by_index = {}
    constants_by_index['E'] = common_constants['E_true'][idx]
    constants_by_index['A'] = common_constants['A_true'][idx]
    constants_by_index['n'] = common_constants['n_true'][idx]
    constants_by_index['C1'] = common_constants['C1_true'][idx]
    constants_by_index['gam1'] = common_constants['gam1_true'][idx]
    constants_by_index['kX1'] = common_constants['kX1_true'][idx]
    constants_by_index['C2'] = common_constants['C2_true'][idx]
    constants_by_index['gam2'] = common_constants['gam2_true'][idx]
    constants_by_index['kX2'] = common_constants['kX2_true'][idx]
    constants_by_index['R0'] = common_constants['R0_true'][idx]
    constants_by_index['Q1'] = common_constants['Q1_true'][idx]
    constants_by_index['b1'] = common_constants['b1_true'][idx]
    constants_by_index['KR1'] = common_constants['KR1_true'][idx]

    return constants_by_index


def load_data_based_on_strain_rate(matlab_data_path, test_data: dict,
                                   pinn_script_to_run: str):

    data = scipy.io.loadmat(matlab_data_path)

    if pinn_script_to_run == 'stress_relaxation':
        test_data['t_data'] = torch.from_numpy(np.reshape(data['t'], (-1, 1)))

    test_data['sig_data'] = torch.from_numpy(np.reshape(data['sig'], (-1, 1)))

    if pinn_script_to_run != 'stress_relaxation':
        test_data['eps_tot_data'] = torch.from_numpy(
            np.reshape(data['eps_tot'], (-1, 1)))

    test_data['R_data'] = torch.from_numpy(np.reshape(data['R'], (-1, 1)))
    test_data['X1_data'] = torch.from_numpy(np.reshape(data['X1'], (-1, 1)))
    test_data['X2_data'] = torch.from_numpy(np.reshape(data['X2'], (-1, 1)))
    test_data['eps_p_data'] = torch.from_numpy(
        np.reshape(data['eps_in'], (-1, 1)))

    print(f"Data loaded from {matlab_data_path} into test_data dictionary.")


def load_parameters_from_json(json_file):
    with open(json_file, 'r') as f:
        params_dict = json.load(f)
    return params_dict


def convert_numerical_strings_to_numbers(params):
    for key, value in params.items():
        if isinstance(value, str):
            numerical_pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
            if re.match(numerical_pattern, value):
                try:
                    params[key] = float(value)
                except ValueError:
                    print(
                        f"Unable to convert {value} to a numerical value for key {key}."
                    )

        elif isinstance(value, dict):
            convert_numerical_strings_to_numbers(value)


def return_optimizer(params, pinn):
    pinn_script_to_run = params['run']
    selected_optimizer = params['common_parameters']['optimizer']
    optimizer_params = params[pinn_script_to_run][selected_optimizer]

    if selected_optimizer == 'adam':
        optimizer = optim.Adam(list(pinn.approximate_solution.parameters()),
                               lr=optimizer_params['learning_rate'],
                               weight_decay=optimizer_params['weight_decay'])
    elif selected_optimizer == 'lbfgs':
        optimizer = optim.LBFGS(
            list(pinn.approximate_solution.parameters()),
            lr=float(optimizer_params['learning_rate']),
            max_iter=optimizer_params['max_iter'],
            max_eval=optimizer_params['max_eval'],
            tolerance_grad=optimizer_params['tolerance_grad'],
            tolerance_change=optimizer_params['tolerance_change'],
            line_search_fn=optimizer_params['line_search_fn'])
    else:
        raise ValueError(
            'Unknown optimizer chosen. Please check the config.json file.')

    return optimizer


def load_pinn_data_from_csv(params):
    pinn_scripts_to_run = ['monotonic', 'stress_relaxation', 'cyclic']
    device = torch.device(params['common_parameters']['device'])
    for pinn_script_to_run in pinn_scripts_to_run:
        pinn_specific_params = params[pinn_script_to_run]
        data_path = pinn_specific_params['data_path']
        data_dict = {}

        if os.path.exists(data_path):
            data = pd.read_csv(data_path, header=None)
            for i in range(len(data)):
                data_list = data.iloc[i].values.tolist()
                formatted_list = format_data_into_torch_tensor(
                    data_list[1:], device)
                data_dict[data_list[0]] = formatted_list

            params[pinn_script_to_run]['data'] = data_dict
            print(
                f"Data loaded for {pinn_script_to_run} pinn from {data_path}.")

        else:
            raise FileNotFoundError(
                f"Data file {data_path} not found. Please check the data dir.")


def format_data_into_torch_tensor(data_list, device):
    formatted_list = [x for x in data_list if not pd.isna(x)]
    formatted_list = [
        formatted_list[i:i + 2] for i in range(0, len(formatted_list), 2)
    ]

    return torch.tensor(formatted_list).to(device)


def load_common_constants(params):
    common_constants_path = params['common_parameters']['data_path']
    common_constants_dict = {}

    if os.path.exists(common_constants_path):
        common_constants = pd.read_csv(common_constants_path, header=None)
        for i in range(len(common_constants)):
            common_constants_list = common_constants.iloc[i].values.tolist()
            common_constants_dict[
                common_constants_list[0]] = common_constants_list[1:]

        print(f"Common constants loaded from {common_constants_path}.")
        return common_constants_dict
    else:
        raise FileNotFoundError(
            f"Common constants file {common_constants_path} not found. Please check the data dir."
        )
