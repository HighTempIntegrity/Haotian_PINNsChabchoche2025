import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from utils.normalization import normalize_A_n, normalize_input_linear, normalize_input_log, denormalize_input_linear, \
    denormalize_t, normalize_A_kX1, normalize_A_kX2, normalize_A_KR, normalize_initial, normalize_initial_X2
from pinn_scripts.pinns_stress_relaxation import StressRelaxation


def generate_monotonic_test_data(params, constants_by_index, input_int,
                                 strain_rate, index):
    x_test = input_int.clone().detach()
    domain_extrema = params[params['run']]['data']['domain_extrema']
    domain_extrema_log = params[params['run']]['data']['domain_extrema_log']
    x_test[:, 0] = torch.sort(x_test[:, 0], 0)[0].reshape(-1, )
    input_linear = torch.tensor([
        0, constants_by_index['E'], constants_by_index['n'],
        constants_by_index['C1'], constants_by_index['gam1'],
        constants_by_index['C2'], constants_by_index['gam2'],
        constants_by_index['R0'], constants_by_index['Q1'],
        constants_by_index['b1']
    ],
                                device=torch.device(
                                    params['common_parameters']['device']))
    input_log = torch.tensor([
        strain_rate, constants_by_index['kX1'], constants_by_index['kX2'],
        constants_by_index['KR1']
    ],
                             dtype=torch.float64,
                             device=torch.device(
                                 params['common_parameters']['device']))

    x_test[:, 1:10] = normalize_input_linear(input_linear, domain_extrema)
    x_test[:, 11:15] = normalize_input_log(input_log, domain_extrema_log)
    x_test[:, 10] = normalize_A_n(
        torch.tensor([constants_by_index['A']], dtype=torch.float64),
        constants_by_index['n'])

    x_test.requires_grad = True
    eps_tot = denormalize_input_linear(x_test[:, :10], domain_extrema)[:, 0]
    eps_tot_rate = input_log[0]

    print(
        f"Test data generated for strain rate {strain_rate} and index {index}."
    )

    return x_test, eps_tot, eps_tot_rate


def generate_stress_relaxation_test_data(
    params,
    test_data,
    constants_by_index,
    input_int,
    strain_rate,
    index,
    sr_pinn: StressRelaxation,
):

    device = torch.device(params['common_parameters']['device'])
    x_test = input_int.clone().detach()
    x_test[:, 0] = torch.sort(x_test[:, 0], 0)[0].reshape(-1, )

    domain_extrema = params[params['run']]['data']['domain_extrema']
    domain_extrema_log = params[params['run']]['data']['domain_extrema_log']
    initial_extrema = params[params['run']]['data']['initial_extrema']

    n = constants_by_index['n']
    A = constants_by_index['A']
    A_tensor = torch.tensor([A], dtype=torch.float64, device=device)

    input_linear = torch.tensor([
        constants_by_index['E'], n, constants_by_index['C1'],
        constants_by_index['gam1'], constants_by_index['C2'],
        constants_by_index['gam2'], constants_by_index['R0'],
        constants_by_index['Q1'], constants_by_index['b1']
    ],
                                device=device)
    input_log = torch.tensor([
        strain_rate, constants_by_index['kX1'], constants_by_index['kX2'],
        constants_by_index['KR1']
    ],
                             dtype=torch.float64,
                             device=device)

    x_test[:, 1:10] = normalize_input_linear(input_linear, domain_extrema)
    x_test[:, 11:15] = normalize_input_log(input_log, domain_extrema_log)
    x_test[:, 12] = normalize_A_kX1(
        torch.tensor([constants_by_index['kX1']],
                     dtype=torch.float64,
                     device=device), A_tensor)
    x_test[:, 13] = normalize_A_kX2(
        torch.tensor([constants_by_index['kX2']],
                     dtype=torch.float64,
                     device=device), A_tensor)
    x_test[:, 14] = normalize_A_KR(
        torch.tensor([constants_by_index['KR1']],
                     dtype=torch.float64,
                     device=device), A_tensor)
    x_test[:, 10] = normalize_A_n(A_tensor, n)

    t = denormalize_t(x_test[:, 0], device)

    x_pre_ini = x_test[:, :15].clone()
    x_pre_ini[:, 0] = 1.0

    monotonic_pinn = sr_pinn.pinn_mono

    sig_ini, R_ini, X1_ini, X2_ini = monotonic_pinn.prediction_end(
        x_pre_ini, constants_by_index['kX1'], constants_by_index['kX2'],
        constants_by_index['KR1'])

    input_initial = torch.tensor([(test_data['sig_data'][0] - sig_ini[0]),
                                  (test_data['R_data'][0] - R_ini[0]),
                                  (test_data['X1_data'][0] - X1_ini[0]),
                                  (test_data['X2_data'][0] - X2_ini[0])],
                                 device=device,
                                 dtype=torch.float64)

    x_test[:, 15:18] = normalize_initial(input_initial[:3], initial_extrema)
    x_test[:, 18] = normalize_initial_X2(input_initial[3],
                                         test_data['sig_data'][0],
                                         test_data['R_data'][0],
                                         test_data['X1_data'][0], X2_ini[0], A,
                                         n, device).detach()
    x_test.requires_grad = True

    print(
        f'Test data generated for strain rate {strain_rate} and index {index}.'
    )

    return x_test, t
