import os
import sys
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from utils.normalization import normalize_A_n, normalize_input_linear, normalize_input_log, denormalize_input_linear


def generate_hard_constraints_for_stress_relaxation(test_data):
    x_test = test_data['x_test']
    linear_constraint = (x_test[:, 0] + 1.)
    hard_constraint_linear = torch.cat([
        linear_constraint.reshape(-1, 1),
        linear_constraint.reshape(-1, 1),
        linear_constraint.reshape(-1, 1),
        linear_constraint.reshape(-1, 1)
    ], 1)

    test_data['hard_constraint_linear'] = hard_constraint_linear

    print("Hard constraint - linear - generated for stress relaxation.")


def generate_hard_constraints_for_monotonic(params, constants_by_index,
                                            test_data):

    eps_tot = test_data['eps_tot']
    eps_tot_rate = test_data['eps_tot_rate']

    domain_extrema = params[params['run']]['data']['domain_extrema']
    hard_constraint_other = (eps_tot.reshape(-1, ) /
                             (domain_extrema[0, 1] - domain_extrema[0, 0]))
    t = eps_tot / eps_tot_rate
    sigma_el = eps_tot * constants_by_index['E']
    R_el = constants_by_index['R0'] * torch.exp(-constants_by_index['KR1'] * t)
    X1_el = 0.
    X2_el = 0.
    X_el = X1_el + X2_el
    a0 = sigma_el - X_el
    a00 = abs(a0) - R_el
    INP = (0.5 * (a00 + torch.sqrt(a00**2))) / (domain_extrema[0, 1] *
                                                constants_by_index['E'])
    hard_constraint_eps_p = 2 * (torch.tanh(2 * INP)**2)

    test_data['hard_constraint_other'] = hard_constraint_other
    test_data['hard_constraint_eps_p'] = hard_constraint_eps_p

    print("Hard constraints - other and eps_p - generated.")


def generate_monotonic_plot_data(params, constants_by_index, test_data):
    x_test = test_data['x_test']
    y_pred = test_data['y_pred']
    eps_tot = test_data['eps_tot']
    eps_tot_rate = test_data['eps_tot_rate']
    hard_constraint_other = test_data['hard_constraint_other']
    hard_constraint_eps_p = test_data['hard_constraint_eps_p']

    test_data['eps_p_pred'] = y_pred[:, 0] * hard_constraint_eps_p
    R_pred = y_pred[:, 1] * hard_constraint_other
    X1_pred = y_pred[:, 2] * hard_constraint_other
    X2_pred = y_pred[:, 3] * hard_constraint_other

    R_pred = R_pred + constants_by_index['R0']
    X1_pred = X1_pred
    X2_pred = X2_pred

    test_data['R_pred'] = R_pred
    test_data['X1_pred'] = X1_pred
    test_data['X2_pred'] = X2_pred
    test_data['sig_pred'] = (eps_tot -
                             test_data['eps_p_pred']) * constants_by_index['E']

    test_data['X'] = test_data['X1_pred'] + test_data['X2_pred']
    domain_extrema = params[params['run']]['data']['domain_extrema']

    eps_p_eps = torch.autograd.grad(
        test_data['eps_p_pred'].sum(), x_test, create_graph=True)[0][:, 0] * (
            1 / (domain_extrema[0, 1] - domain_extrema[0, 0]))
    R_eps = torch.autograd.grad(
        R_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
            1 / (domain_extrema[0, 1] - domain_extrema[0, 0]))
    X1_eps = torch.autograd.grad(
        X1_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
            1 / (domain_extrema[0, 1] - domain_extrema[0, 0]))
    X2_eps = torch.autograd.grad(
        X2_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
            1 / (domain_extrema[0, 1] - domain_extrema[0, 0]))

    eps_p_t = (eps_p_eps * eps_tot_rate)
    eps_p_eq_t = abs(eps_p_t)

    R_t = (R_eps * eps_tot_rate)
    X1_t = (X1_eps * eps_tot_rate)
    X2_t = (X2_eps * eps_tot_rate)

    print("Calculated gradients for eps_p, R, X1, X2.")

    A = constants_by_index['A']
    n = constants_by_index['n']
    C1 = constants_by_index['C1']
    gam1 = constants_by_index['gam1']
    kX1 = constants_by_index['kX1']
    C2 = constants_by_index['C2']
    gam2 = constants_by_index['gam2']
    kX2 = constants_by_index['kX2']
    Q1 = constants_by_index['Q1']
    b1 = constants_by_index['b1']
    KR1 = constants_by_index['KR1']

    loss0_vec = 1000 * ((-X1_t + torch.sqrt(X1_t**2)) +
                        (-X2_t + torch.sqrt(X2_t**2)))
    test_data['loss0'] = torch.mean(loss0_vec**params['monotonic']['power'])
    a1 = test_data['sig_pred'].reshape(-1, ) - test_data['X'].reshape(-1, )
    a2 = a1 * torch.tanh(a1 / 0.001) - R_pred.reshape(-1, )
    a3 = A * (0.5 * (a2 + torch.sqrt(0.001**2 + a2**2)))**n * torch.sign(a1)
    test_data['loss1'] = (eps_p_t.reshape(-1, ) - a3)
    test_data['mean1'] = torch.mean(abs(
        test_data['loss1'])).cpu().detach().numpy()
    test_data['loss2'] = (
        R_t.reshape(-1, ) - Q1 * eps_p_eq_t +
        b1 * R_pred.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) +
        KR1 * R_pred.reshape(-1, ))
    test_data['mean2'] = torch.mean(abs(
        test_data['loss2'])).cpu().detach().numpy()
    test_data['loss3'] = (X1_t.reshape(-1, ) - C1 * eps_p_t +
                          gam1 * eps_p_eq_t * X1_pred.reshape(-1, ) +
                          kX1 * X1_pred.reshape(-1, ))
    test_data['mean3'] = torch.mean(abs(
        test_data['loss3'])).cpu().detach().numpy()
    test_data['loss4'] = (X2_t.reshape(-1, ) - C2 * eps_p_t +
                          gam2 * eps_p_eq_t * X2_pred.reshape(-1, ) +
                          kX2 * X2_pred.reshape(-1, ))
    test_data['mean4'] = torch.mean(abs(
        test_data['loss4'])).cpu().detach().numpy()

    print(
        "Calculated losses - 0, 1, 2, 3, 4 and their means for the test data - monotonic."
    )


def generate_stress_relaxation_plot_data(
    params,
    constants_by_index,
    test_data,
):

    y_pred = test_data['y_pred']
    x_test = test_data['x_test']
    t = test_data['t']
    device = torch.device(params['common_parameters']['device'])

    eps_p_pred = y_pred[:, 0]
    R_pred = y_pred[:, 1]
    X1_pred = y_pred[:, 2]
    X2_pred = y_pred[:, 3]

    R_data = test_data['R_data']
    X1_data = test_data['X1_data']
    X2_data = test_data['X2_data']
    sig_data = test_data['sig_data']
    t_data = test_data['t_data']

    R_pred = R_pred + R_data[0]
    X1_pred = X1_pred + X1_data[0]
    X2_pred = X2_pred + X2_data[0]

    test_data['R_pred'] = R_pred
    test_data['X1_pred'] = X1_pred
    test_data['X2_pred'] = X2_pred
    test_data['eps_p_pred'] = eps_p_pred

    E = constants_by_index['E']
    A = constants_by_index['A']
    C1 = constants_by_index['C1']
    gam1 = constants_by_index['gam1']
    kX1 = constants_by_index['kX1']
    C2 = constants_by_index['C2']
    gam2 = constants_by_index['gam2']
    kX2 = constants_by_index['kX2']
    Q1 = constants_by_index['Q1']
    b1 = constants_by_index['b1']
    KR1 = constants_by_index['KR1']

    test_data['sig_pred'] = sig_data[0] - eps_p_pred * E

    X = X1_pred + X2_pred

    ub = torch.log10(torch.tensor([900. + 1e-3], device=device))
    lb = torch.log10(torch.tensor([1e-3], device=device))

    eps_p_t = torch.autograd.grad(
        eps_p_pred.sum(), x_test,
        create_graph=True)[0][:, 0] * (2 /
                                       (ub - lb)) / ((t + 1e-3) * np.log(10))
    R_t = torch.autograd.grad(R_pred.sum(), x_test,
                              create_graph=True)[0][:, 0] * (2 / (ub - lb)) / (
                                  (t + 1e-3) * np.log(10))
    X1_t = torch.autograd.grad(
        X1_pred.sum(), x_test,
        create_graph=True)[0][:, 0] * (2 /
                                       (ub - lb)) / ((t + 1e-3) * np.log(10))
    X2_t = torch.autograd.grad(
        X2_pred.sum(), x_test,
        create_graph=True)[0][:, 0] * (2 /
                                       (ub - lb)) / ((t + 1e-3) * np.log(10))

    eps_p_eq_t = abs(eps_p_t)

    one_tensor = torch.tensor([1.], device=device)
    weight = torch.min(t.detach(), one_tensor) * torch.max(
        torch.log(t.detach()), one_tensor)

    a1 = test_data['sig_pred'].reshape(-1, ) - X.reshape(-1, )
    a2 = a1 * torch.tanh(a1 / 0.001) - R_pred.reshape(-1, )
    a3 = A * (0.5 * (a2 + torch.sqrt(0.001**2 + a2**2)))**n * torch.sign(a1)
    test_data['loss1'] = (eps_p_t.reshape(-1, ) - a3) * weight
    test_data['mean1'] = torch.mean(abs(
        test_data['loss1'])).cpu().detach().numpy()

    test_data['loss2'] = (R_t.reshape(-1, ) - Q1 * eps_p_eq_t +
                          b1 * R_pred.reshape(-1, ) * eps_p_eq_t.reshape(-1, )
                          + KR1 * R_pred.reshape(-1, )) * weight
    test_data['mean2'] = torch.mean(abs(
        test_data['loss2'])).cpu().detach().numpy()

    test_data['loss3'] = (X1_t.reshape(-1, ) - C1 * eps_p_t +
                          gam1 * eps_p_eq_t * X1_pred.reshape(-1, ) +
                          kX1 * X1_pred.reshape(-1, )) * weight
    test_data['mean3'] = torch.mean(abs(
        test_data['loss3'])).cpu().detach().numpy()

    test_data['loss4'] = (X2_t.reshape(-1, ) - C2 * eps_p_t +
                          gam2 * eps_p_eq_t * X2_pred.reshape(-1, ) +
                          kX2 * X2_pred.reshape(-1, )) * weight
    test_data['mean4'] = torch.mean(abs(
        test_data['loss4'])).cpu().detach().numpy()

    test_data['t_plot'] = torch.log10(t + 1e-3)
    test_data['t_data_plot'] = torch.log10(t_data + 1e-3)

    print(
        'Calculated losses - 1, 2, 3, 4 and their means for the test data - stress relaxation.'
    )
