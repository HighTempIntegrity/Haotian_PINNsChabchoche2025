import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from utils.general_functions import get_common_constants_for_index, load_data_based_on_strain_rate
from utils.generate_data import generate_hard_constraints_for_monotonic, generate_monotonic_plot_data, \
    generate_hard_constraints_for_stress_relaxation, generate_stress_relaxation_plot_data
from utils.generate_test_data import generate_monotonic_test_data, generate_stress_relaxation_test_data
from utils.normalization import denormalize_output
from utils.plotting import plot_monotonic_learning_result, \
    plot_stress_relaxation_learning_result


def test_monotonic_pinn(params, pinn):
    input_int = (pinn.soboleng.draw(10000).to(
        torch.device(params['common_parameters']['device']))) * 2 - 1

    indices = params['common_parameters']['idx']
    strain_rates = params['common_parameters']['strain_rates']

    for index in indices:
        for i, strain_rate in enumerate(strain_rates):
            test_data = {}
            test_data['strain_rate'] = strain_rate
            test_data['index'] = index

            matlab_data_path = 'data_new_benchmark_monotonic_' + str(
                index +
                1) + params['common_parameters']['strain_rate_matlab_data'][i]
            matlab_data_path = os.path.join(
                params['common_parameters']['matlab_data_folder'],
                matlab_data_path)

            load_data_based_on_strain_rate(matlab_data_path, test_data,
                                           'monotonic')

            constants_by_index = get_common_constants_for_index(params, index)

            test_data['x_test'], test_data['eps_tot'], test_data[
                'eps_tot_rate'] = generate_monotonic_test_data(
                    params, constants_by_index, input_int, strain_rate, index)

            generate_hard_constraints_for_monotonic(params, constants_by_index,
                                                    test_data)

            y_pred = pinn.predict(test_data['x_test'])
            test_data['y_pred'] = denormalize_output(
                y_pred, params[params['run']]['data']['result_extrema']).to(
                    torch.float64)

            generate_monotonic_plot_data(params, constants_by_index, test_data)
            plot_monotonic_learning_result(params, test_data)


def test_cyclic_pinn(params, pinn):
    # work in progress - not complete yet
    pass


def test_stress_relaxation_pinn(params, pinn):
    input_int = (pinn.soboleng.draw(10000).to(
        torch.device(params['common_parameters']['device']))) * 2 - 1

    indices = params['common_parameters']['idx']
    strain_rates = params['common_parameters']['strain_rates']

    for index in indices:
        for i, strain_rate in enumerate(strain_rates):
            test_data = {}
            test_data['strain_rate'] = strain_rate
            test_data['index'] = index

            matlab_data_path = 'data_new_benchmark_stress_relaxation_' + str(
                index +
                1) + params['common_parameters']['strain_rate_matlab_data'][i]
            matlab_data_path = os.path.join(
                params['common_parameters']['matlab_data_folder'],
                matlab_data_path)

            load_data_based_on_strain_rate(matlab_data_path, test_data,
                                           'stress_relaxation')

            constants_by_index = get_common_constants_for_index(params, index)

            test_data['x_test'], test_data[
                't'] = generate_stress_relaxation_test_data(
                    params,
                    test_data,
                    constants_by_index,
                    input_int,
                    strain_rate,
                    index,
                    pinn,
                )

            generate_hard_constraints_for_stress_relaxation(test_data)

            y_pred = pinn.predict(
                test_data['x_test']) * test_data['hard_constraint_linear']
            test_data['y_pred'] = denormalize_output(
                y_pred, params[params['run']]['data']['result_extrema']).to(
                    torch.float64)

            generate_stress_relaxation_plot_data(
                params,
                constants_by_index,
                test_data,
            )
            plot_stress_relaxation_learning_result(params, test_data)
