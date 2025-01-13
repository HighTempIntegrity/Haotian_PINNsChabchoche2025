import torch
from PINNs_monotonic_class_beta_range import Pinns_monotonic as PINN_mono
from PINNs_stress_relaxation_class_beta_range import Pinns_stress as PINN_stress
from PINNs_cyclic_class_beta_range import Pinns_cyclic as PINN_cyclic
from Beta_parameters import cal_parameter_from_beta
import math
import numpy as np
import betas_normalization
import time as tm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pinn_mono = PINN_mono(100)
pinn_mono.load_model()

pinn_stress = PINN_stress(100)
pinn_stress.load_model()

pinn_cyclic = PINN_cyclic(100)
pinn_cyclic.load_model()

torch.autograd.set_detect_anomaly(False)


def cal_loss_hold_beta(cycle, log_beta, beta_upper, beta_lower, temp_K, sig_save_stress, t_save_stress, sig_save_monotonic, eps_save_monotonic,
                       sig_save_cyclic, eps_save_cyclic, strain_rate_save_monotonic, strain_rate_save_stress,
                       strain_rate_save_cyclic, sig_mean_save, end_save_monotonic, end_save_stress, end_save_cyclic,
                       ranges, log_index, linear_index, p):
    ranges_log = torch.log10(ranges[log_index, :])
    ranges_linear = ranges[linear_index, :]

    # calculate model parameters from beta

    beta = log_beta.clone()
    beta[log_index] = betas_normalization.denormalize_log_betas(log_beta[log_index], ranges_log)
    beta[linear_index] = betas_normalization.denormalize_linear_betas(log_beta[linear_index], ranges_linear)

    ## weight of loss terms for elastic region
    weight = torch.ones((temp_K.shape[0]), device=device)
    indices_weight = list(range(0, 30)) + list(range(100, 130)) + list(range(200, 230)) + \
                     list(range(300, 330)) + list(range(400, 430)) + list(range(500, 530)) + \
                     list(range(600, 630)) + list(range(700, 730)) + list(range(800, 830)) + \
                     list(range(900, 930)) + list(range(1000, 1030)) + list(range(1100, 1130)) + list(range(1200, 1230))

    # weight[indices_weight] = 1.2

    E, A, n, C1, gam1, kX1, C2, gam2, kX2, Q1, b1, KR1, R0 = cal_parameter_from_beta(beta, temp_K.reshape(-1, ))
    model_para = torch.zeros((temp_K.shape[0], 14), device=device)
    input_linear = torch.cat((E.reshape(-1, 1), n.reshape(-1, 1), C1.reshape(-1, 1), gam1.reshape(-1, 1),
                              C2.reshape(-1, 1), gam2.reshape(-1, 1), R0.reshape(-1, 1), Q1.reshape(-1, 1),
                              b1.reshape(-1, 1)), 1)
    input_log = torch.cat((strain_rate_save_monotonic.reshape(-1, 1), A.reshape(-1, 1), kX1.reshape(-1, 1),
                           kX2.reshape(-1, 1), KR1.reshape(-1, 1)), 1)
    model_para[:, 9:] = pinn_mono.normalize_input_log(input_log.clone())
    model_para[:, :9] = pinn_stress.normalize_input_linear(input_linear.clone())

    ub = torch.mean(
        torch.max(model_para.clone().reshape(-1, ) - beta_upper.reshape(-1, ), torch.zeros([model_para.clone().reshape(-1, ).shape[0]]).to(device)) ** 2)
    lb = torch.mean(
        torch.min(model_para.clone().reshape(-1, ) - beta_lower.reshape(-1, ), torch.zeros([model_para.clone().reshape(-1, ).shape[0]]).to(device)) ** 2)

    loss_main = 0
    # first loading ramp
    x_test = torch.zeros((sig_save_stress.shape[0], 15), device=device)
    x_test[:, 1:] = model_para[:, :].clone()
    x_test[:, 0] = eps_save_monotonic.reshape(-1, ) / 0.005 - 1

    x_test_end = x_test.clone()
    x_test_end[:, 0] = end_save_monotonic.reshape(-1, ) / 0.005 - 1

    t, eps_tot, eps_p, sig, R, X1, X2 = pinn_mono.prediction(x_test)
    t_end, eps_tot_end, eps_p_end, sig_end, R_end, X1_end, X2_end = pinn_mono.prediction(x_test_end)

    loss_main = loss_main + torch.mean(
        ((sig.reshape(-1, ) - sig_save_monotonic.reshape(-1, )) * weight.reshape(-1, )) ** p / sig_mean_save.reshape(
            -1, ))
    indices = torch.arange(99, sig_save_stress.shape[0] + 1, 100, device=device)
    sig_pred_end1 = sig[indices]
    sig_data_end1 = sig_save_monotonic[indices]

    for i in range(3):
        if cycle == 1 and i == 1:
            break
        ## stress relaxation
        x_test = torch.zeros((sig_save_stress.shape[0], 15), device=device)
        x_test[:, 1:] = model_para[:, :].clone()
        input_log = torch.cat((strain_rate_save_stress[:, 2 * i].reshape(-1, 1), A.reshape(-1, 1), kX1.reshape(-1, 1),
                               kX2.reshape(-1, 1), KR1.reshape(-1, 1)), 1)
        x_test[:, 10] = pinn_mono.normalize_input_log(input_log.clone())[:, 0]
        x_test[:, 0] = pinn_stress.normalize_t(t_save_stress[:, 2 * i])
        x_test_end = x_test.clone()
        x_test_end[:, 0] = pinn_stress.normalize_t(end_save_stress[:, 2 * i].reshape(-1, ))

        t, eps_tot, eps_p, sig, R, X1, X2 = pinn_stress.prediction(x_test, sig_end, R_end, X1_end, X2_end)
        t_end, eps_tot_end, eps_p_end, sig_end, R_end, X1_end, X2_end = pinn_stress.prediction(x_test_end, sig_end,
                                                                                               R_end, X1_end, X2_end)
        R_end = torch.max(R_end, torch.ones(1, device=device))

        loss_main = loss_main + torch.mean(
            (sig.reshape(-1, ) - sig_save_stress[:, 2 * i].reshape(-1, )) ** p / sig_mean_save.reshape(-1, ))

        ## reverse loading ramp
        x_test = torch.zeros((sig_save_stress.shape[0], 15), device=device)
        x_test[:, 1:] = model_para[:, :].clone()
        input_log = torch.cat((strain_rate_save_cyclic[:, 2 * i].reshape(-1, 1), A.reshape(-1, 1), kX1.reshape(-1, 1),
                               kX2.reshape(-1, 1), KR1.reshape(-1, 1)), 1)
        x_test[:, 10] = pinn_mono.normalize_input_log(input_log.clone())[:, 0]
        x_test[:, 0] = eps_save_cyclic[:, 2 * i].reshape(-1, ) / 0.01 - 1
        x_test_end = x_test.clone()
        x_test_end[:, 0] = end_save_cyclic[:, 2 * i].reshape(-1, ) / 0.01 - 1

        # t, eps_tot, eps_p, sig, R, X1, X2 = pinn_cyclic.prediction(x_test[300:, :], -sig_end[300:], R_end[300:], -X1_end[3500:], -X2_end[3500:])
        t, eps_tot, eps_p, sig, R, X1, X2 = pinn_cyclic.prediction(x_test, -sig_end, R_end, -X1_end, -X2_end)
        t_end, eps_tot_end, eps_p_end, sig_end, R_end, X1_end, X2_end = pinn_cyclic.prediction(x_test_end, -sig_end,
                                                                                               R_end, -X1_end, -X2_end)
        R_end = torch.max(R_end, torch.ones(1, device=device))

        loss_main = loss_main + torch.mean(
            ((sig.reshape(-1, ) - sig_save_cyclic[:, 2 * i].reshape(-1, )) * weight.reshape(
                -1, )) ** p / sig_mean_save.reshape(-1, ))

        ## second stress relaxation
        x_test = torch.zeros((sig_save_stress.shape[0], 15), device=device)
        x_test[:, 1:] = model_para[:, :]
        input_log = torch.cat(
            (strain_rate_save_stress[:, 2 * i + 1].reshape(-1, 1), A.reshape(-1, 1), kX1.reshape(-1, 1),
             kX2.reshape(-1, 1), KR1.reshape(-1, 1)), 1)
        x_test[:, 10] = pinn_mono.normalize_input_log(input_log.clone())[:, 0]
        x_test[:, 0] = pinn_stress.normalize_t(t_save_stress[:, 2 * i + 1])
        x_test_end = x_test.clone()
        x_test_end[:, 0] = pinn_stress.normalize_t(end_save_stress[:, 2 * i + 1].reshape(-1, ))

        t, eps_tot, eps_p, sig, R, X1, X2 = pinn_stress.prediction(x_test, sig_end, R_end, X1_end, X2_end)
        t_end, eps_tot_end, eps_p_end, sig_end, R_end, X1_end, X2_end = pinn_stress.prediction(x_test_end, sig_end,
                                                                                               R_end, X1_end, X2_end)
        R_end = torch.max(R_end, torch.ones(1, device=device))

        loss_main = loss_main + torch.mean(
            (sig.reshape(-1, ) - sig_save_stress[:, 2 * i + 1].reshape(-1, )) ** p / sig_mean_save.reshape(-1, ))

        ## second reverse loading ramp
        x_test = torch.zeros((sig_save_stress.shape[0], 15), device=device)
        x_test[:, 1:] = model_para[:, :]
        input_log = torch.cat(
            (strain_rate_save_cyclic[:, 2 * i + 1].reshape(-1, 1), A.reshape(-1, 1), kX1.reshape(-1, 1),
             kX2.reshape(-1, 1), KR1.reshape(-1, 1)), 1)
        x_test[:, 10] = pinn_mono.normalize_input_log(input_log.clone())[:, 0]
        x_test[:, 0] = eps_save_cyclic[:, 2 * i + 1].reshape(-1, ) / 0.01 - 1
        x_test_end = x_test.clone()
        x_test_end[:, 0] = end_save_cyclic[:, 2 * i + 1].reshape(-1, ) / 0.01 - 1
        t, eps_tot, eps_p, sig, R, X1, X2 = pinn_cyclic.prediction(x_test, -sig_end, R_end, -X1_end, -X2_end)
        t_end, eps_tot_end, eps_p_end, sig_end, R_end, X1_end, X2_end = pinn_cyclic.prediction(x_test_end, -sig_end,
                                                                                               R_end,
                                                                                               -X1_end, -X2_end)
        R_end = torch.max(R_end, torch.ones(1, device=device))

        loss_main = loss_main + torch.mean(
            ((sig.reshape(-1, ) - sig_save_cyclic[:, 2 * i + 1].reshape(-1, )) * weight.reshape(
                -1, )) ** p / sig_mean_save.reshape(-1, ))

        if i == 0:
            sig_pred_end2 = sig[indices]
            sig_data_end2 = sig_save_cyclic[indices, 2 * i + 1]
        if i == 2:
            sig_pred_end3 = sig[indices]
            sig_data_end3 = sig_save_cyclic[indices, 2 * i + 1]

    PINNs_end_local1 = sig_pred_end2.reshape(-1, ) - sig_pred_end1.reshape(-1, )
    data_end_local1 = sig_data_end2.reshape(-1, ) - sig_data_end1.reshape(-1, )
    loss_end_local1 = torch.mean((PINNs_end_local1 - data_end_local1) ** 2)  # / torch.mean(abs(sig_data.reshape(-1, )))
    loss_end = 0.1 * loss_end_local1
    if cycle == 2:
        PINNs_end_local2 = sig_pred_end3.reshape(-1, ) - sig_pred_end1.reshape(-1, )
        data_end_local2 = sig_data_end3.reshape(-1, ) - sig_data_end1.reshape(-1, )
        loss_end_local2 = torch.mean(
            (PINNs_end_local2 - data_end_local2) ** 2)  # / torch.mean(abs(sig_data.reshape(-1, )))
        loss_end = loss_end + 0.2 * loss_end_local2


    penalty_range = 100 * (ub + lb)

    # loss = torch.mean(((sig_pred - sig_data) / (abs(sig_data) + 0.01)) ** 2) + 100000000 * (ub + lb)

    loss = loss_main + penalty_range

    return loss, loss_main


def fit(num_epochs, optimizer, cycle, verbose, beta, beta_upper, beta_lower, temp_K, sig_save_stress, t_save_stress, sig_save_monotonic,
        eps_save_monotonic, sig_save_cyclic, eps_save_cyclic, strain_rate_save_monotonic, strain_rate_save_stress,
        strain_rate_save_cyclic, sig_mean_save, end_save_monotonic, end_save_stress, end_save_cyclic, ranges, log_index,
        linear_index, p):
    history = list()
    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose:
            print("################################ ", epoch, " ################################")

        def closure():
            # start_time = tm.time()
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            loss, loss_main = cal_loss_hold_beta(cycle, beta, beta_upper, beta_lower, temp_K, sig_save_stress, t_save_stress,
                                                 sig_save_monotonic, eps_save_monotonic, sig_save_cyclic,
                                                 eps_save_cyclic, strain_rate_save_monotonic, strain_rate_save_stress,
                                                 strain_rate_save_cyclic, sig_mean_save, end_save_monotonic,
                                                 end_save_stress, end_save_cyclic, ranges, log_index, linear_index, p)
            loss.backward()
            history.append(loss_main.item())
            # elapsed = tm.time() - start_time
            # print('Training time: %.5f' % (elapsed))
            # print('Loss: ', history[-1])
            return loss

        # print(model_para)
        optimizer.step(closure=closure)
        # scheduler.step()
        if history[-1] == math.inf or math.isnan(history[-1]):
            break
        # if history[-1] < 1e-4: break

    print('Final Loss: ', history[-1])
    return history


def cal_beta_limit(par_true, temp_K):
    E, A, n, C1, gam1, kX1, C2, gam2, kX2, Q1, b1, KR1, R0 = cal_parameter_from_beta(par_true, temp_K.reshape(-1, ))
    model_para_upper = torch.zeros((temp_K.shape[0], 14), device=device)
    model_para_lower = torch.zeros((temp_K.shape[0], 14), device=device)
    ones = torch.ones(temp_K.shape[0], device=device)
    input_linear_upper = torch.cat(
        ((E + 1e4).reshape(-1, 1), (n + 2).reshape(-1, 1), (C1 + 2e3).reshape(-1, 1), (gam1 + 5).reshape(-1, 1),
         (C2 + 1e3).reshape(-1, 1), (gam2 + 1e2).reshape(-1, 1), (R0 + 15).reshape(-1, 1),
         (Q1 + 100).reshape(-1, 1), (b1 + 1).reshape(-1, 1)), 1)
    input_log = torch.cat(
        (ones.reshape(-1, 1), A.reshape(-1, 1), kX1.reshape(-1, 1), kX2.reshape(-1, 1), KR1.reshape(-1, 1)), 1)

    model_para_upper[:, 9:] = pinn_mono.normalize_input_log(input_log.clone())
    model_para_upper[:, :9] = pinn_stress.normalize_input_linear(input_linear_upper.clone())
    model_para_upper[:, 9] = 1.
    model_para_upper[:, 10] = model_para_upper[:, 10] + 3 / (
                pinn_mono.domain_extrema_log[1, 1] - pinn_mono.domain_extrema_log[1, 0])
    model_para_upper[:, 11] = model_para_upper[:, 11] + 2 / (
                pinn_mono.domain_extrema_log[2, 1] - pinn_mono.domain_extrema_log[2, 0])
    model_para_upper[:, 12] = model_para_upper[:, 12] + 1 / (
                pinn_mono.domain_extrema_log[3, 1] - pinn_mono.domain_extrema_log[3, 0])
    model_para_upper[:, 13] = model_para_upper[:, 13] + 1 / (
                pinn_mono.domain_extrema_log[4, 1] - pinn_mono.domain_extrema_log[4, 0])

    input_linear_lower = torch.cat(
        ((E - 1e4).reshape(-1, 1), (n - 2).reshape(-1, 1), (C1 - 2e3).reshape(-1, 1), (gam1 - 5).reshape(-1, 1),
         (C2 - 1e3).reshape(-1, 1), (gam2 - 1e2).reshape(-1, 1), (R0 - 15).reshape(-1, 1),
         (Q1 - 100).reshape(-1, 1), (b1 - 1).reshape(-1, 1)), 1)

    model_para_lower[:, 9:] = pinn_mono.normalize_input_log(input_log.clone())
    model_para_lower[:, :9] = pinn_stress.normalize_input_linear(input_linear_lower.clone())
    model_para_lower[:, 9] = -1.
    model_para_lower[:, 10] = model_para_lower[:, 10] - 3 / (
            pinn_mono.domain_extrema_log[1, 1] - pinn_mono.domain_extrema_log[1, 0])
    model_para_lower[:, 11] = model_para_lower[:, 11] - 2 / (
            pinn_mono.domain_extrema_log[2, 1] - pinn_mono.domain_extrema_log[2, 0])
    model_para_lower[:, 12] = model_para_lower[:, 12] - 1 / (
            pinn_mono.domain_extrema_log[3, 1] - pinn_mono.domain_extrema_log[3, 0])
    model_para_lower[:, 13] = model_para_lower[:, 13] - 1 / (
            pinn_mono.domain_extrema_log[4, 1] - pinn_mono.domain_extrema_log[4, 0])

    upper = torch.max(torch.min(model_para_upper, torch.ones(1, device=device)), -torch.ones(1, device=device))
    lower = torch.max(torch.min(model_para_lower, torch.ones(1, device=device)), -torch.ones(1, device=device))

    return upper, lower
