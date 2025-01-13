"""Main file"""
import os
import time
import numpy as np
import torch
import neptune
from utils.general_functions import return_optimizer, load_pinn_data_from_csv, \
    load_common_constants, load_parameters_from_json
from utils.plotting import plot_individual_losses, plot_train_loss
from utils.test import test_monotonic_pinn, test_cyclic_pinn, test_stress_relaxation_pinn
from pinn_scripts.pinns_monotonic import Monotonic
from pinn_scripts.pinns_stress_relaxation import StressRelaxation
# from pinn_scripts.pinns_cyclic import Cyclic


def run_monotonic(params):
    #ensure that 'run' in config.json is 'monotonic'
    load_pinn_data_from_csv(params)
    common_constants = load_common_constants(params)
    params[params['run']]['common_constants'] = common_constants
    expt_name = (
        f"mono_{params['current_time']}_"
        f"{params['common_parameters']['neural_network']['n_hidden_layers']}_"
        f"{params['common_parameters']['neural_network']['neurons']}")
    ''''''
    print('Initialised Neptune logger')
    neptune_logger = neptune.init_run(
        project="constitutive-model-calibration/monotonic",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        name=expt_name)

    pinn = Monotonic(params, neptune_logger)
    neptune_logger['parameters'] = params
    optimizer = return_optimizer(params, pinn)
    start_time = time.time()
    if params['load_model']:
        print('Load trained model')
        pinn.load_model()
    else:
        print('Start from random initial weight and bias')
    #always do the training part before plotting
    history1, loss_11, loss_21, loss_31, loss_41 = pinn.fit(
        num_epochs=params['monotonic']['num_epochs'],
        optimizer=optimizer,
        verbose=False)
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f}')
    neptune_logger.stop()

    if params['save_model']:
        pinn.save_model()

    plot_train_loss(params, history1)
    plot_individual_losses(params, loss_11, loss_21, loss_31, loss_41)
    test_monotonic_pinn(params, pinn)


def run_cyclic(params):
    #ensure that 'run' in config.json is 'cyclic'
    load_pinn_data_from_csv(params)
    common_constants = load_common_constants(params)
    params[params['run']]['common_constants'] = common_constants
    pinn = Cyclic(params)
    optimizer = return_optimizer(params, pinn)
    start_time = time.time()
    if params['load_model']:
        print('Load trained model')
        pinn.load_model()
    else:
        print('Start from random initial weight and bias')
    history1, loss_11, loss_21, loss_31, loss_41 = pinn.fit(
        num_epochs=params['cyclic']['num_epochs'],
        optimizer=optimizer,
        verbose=False)

    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f}')

    if params['save_model']:
        pinn.save_model()

    plot_train_loss(params, history1)
    plot_individual_losses(params, loss_11, loss_21, loss_31, loss_41)
    test_cyclic_pinn(params, pinn)


def run_stress_relaxation(params):
    #ensure that 'run' in config.json is 'stress_relaxation'
    load_pinn_data_from_csv(params)
    common_constants = load_common_constants(params)
    params[params['run']]['common_constants'] = common_constants
    pinn = StressRelaxation(params, mono_params=params)
    optimizer = return_optimizer(params, pinn)
    start_time = time.time()

    if params['load_model']:
        print('Load trained model')
        pinn.load_model()
    else:
        print('Start from random initial weight and bias')

    history1, loss_11, loss_21, loss_31, loss_41 = pinn.fit(
        num_epochs=params['stress_relaxation']['num_epochs'],
        optimizer=optimizer,
        verbose=False)

    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f}')

    if params['save_model']:
        pinn.save_model()

    plot_train_loss(params, history1)
    plot_individual_losses(params, loss_11, loss_21, loss_31, loss_41)
    test_stress_relaxation_pinn(params, pinn)

    return


def run_pinn(config):
    params = load_parameters_from_json(config)
    pinn_script_to_run = params['run']
    common_params = params['common_parameters']

    torch.manual_seed(common_params['seed'])
    np.random.seed(common_params['seed'])
    torch.autograd.set_detect_anomaly(common_params['detect_anomaly'])

    params['current_time'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    common_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if pinn_script_to_run == 'monotonic':
        run_monotonic(params)
    elif pinn_script_to_run == 'cyclic':
        run_cyclic(params)
    elif pinn_script_to_run == 'stress_relaxation':
        run_stress_relaxation(params)
    else:
        raise ValueError(
            'Unknown PINN script to run. Please check the config.json file.')


if __name__ == '__main__':
    config_file = './config.json'
    run_pinn(config_file)
