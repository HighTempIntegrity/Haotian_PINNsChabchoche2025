"""Normalization and denormalization functions for input and output data."""

import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)


def normalize_output(tens, result_extrema):
    return (tens - result_extrema[:, 0]) / (result_extrema[:, 1] -
                                            result_extrema[:, 0])


def denormalize_output(tens, result_extrema):
    return tens * (result_extrema[:, 1] -
                   result_extrema[:, 0]) + result_extrema[:, 0]


def normalize_input_linear(tens, domain_extrema):
    return (tens - domain_extrema[:, 0]) / (domain_extrema[:, 1] -
                                            domain_extrema[:, 0])


def denormalize_input_linear(tens, domain_extrema):
    return tens * (domain_extrema[:, 1] -
                   domain_extrema[:, 0]) + domain_extrema[:, 0]


def normalize_input_log(tens, domain_extrema_log):
    return (torch.log10(tens) - domain_extrema_log[:, 0]) / (
        domain_extrema_log[:, 1] - domain_extrema_log[:, 0])


def denormalize_input_log(tens, domain_extrema_log):
    return 10**(tens * (domain_extrema_log[:, 1] - domain_extrema_log[:, 0]) +
                domain_extrema_log[:, 0])


#pylint: disable=C0103
def normalize_A_n(tens, n):
    log_A = torch.log10(tens)
    mid = -2 * n
    ub = -2 * n + 10
    return (log_A - mid) / (ub - mid)


def denormalize_A_n(tens, n):
    mid = -2 * n
    ub = -2 * n + 10
    return 10**(tens * (ub - mid) + mid)


def normalize_t(tens, device):
    ub = torch.log10(torch.tensor([900. + 1e-3], device=device))
    lb = torch.log10(torch.tensor([1e-3], device=device))
    mid = (ub + lb) / 2
    return (torch.log10(tens + 1e-3) - mid) / (ub - mid)


def denormalize_t(tens, device):
    ub = torch.log10(torch.tensor([900. + 1e-3], device=device))
    lb = torch.log10(torch.tensor([1e-3], device=device))
    mid = (ub + lb) / 2
    return 10**(tens * (ub - mid) + mid) - 1e-3


def normalize_initial(tens, initial_extrema):
    return (tens - initial_extrema[:, 0]) / (initial_extrema[:, 1] -
                                             initial_extrema[:, 0])


def denormalize_initial(tens, initial_extrema):
    return tens * (initial_extrema[:, 1] -
                   initial_extrema[:, 0]) + initial_extrema[:, 0]


def normalize_X0_Cgam(tens, domain_extrema_X0, C, gam, kX, eps_tot_rate):
    multi = tens / (C / (gam + kX / eps_tot_rate))
    return (multi - domain_extrema_X0[:, 0]) / (domain_extrema_X0[:, 1] -
                                                domain_extrema_X0[:, 0])


def denormalize_X0_Cgam(tens, domain_extrema_X0, C, gam, kX, eps_tot_rate):
    multi = tens * (domain_extrema_X0[:, 1] -
                    domain_extrema_X0[:, 0]) + domain_extrema_X0[:, 0]
    return multi * (C / (gam + kX / eps_tot_rate))


def normalize_el0(tens, X0, R0, E, A, n, eps_tot_rate):
    lb = (X0 - R0 - (eps_tot_rate / A)**(1 / n)) / E
    ub = X0 / E
    mid = (lb + ub) / 2
    return (tens - mid) / (ub - mid)


def denormalize_el0(tens, X0, R0, E, A, n, eps_tot_rate):
    lb = (X0 - R0 - (eps_tot_rate / A)**(1 / n)) / E
    ub = X0 / E
    mid = (lb + ub) / 2
    return tens * (ub - mid) + mid


def normalize_A_kX1(tens, A):
    device = A.device
    log_A = torch.log10(A)
    lb = torch.min(
        torch.max(0.4 * log_A + 6, torch.tensor([-18], device=device)),
        torch.tensor([0], device=device))
    ub = torch.min(0.4 * log_A + 14, torch.tensor([2], device=device))
    mid = (lb + ub) / 2
    return (torch.log10(tens) - mid) / (ub - mid)


def denormalize_A_kX1(tens, A):
    device = A.device
    log_A = torch.log10(A)
    lb = torch.min(
        torch.max(0.4 * log_A + 6, torch.tensor([-18], device=device)),
        torch.tensor([0], device=device))
    ub = torch.min(0.4 * log_A + 14, torch.tensor([2], device=device))
    mid = (lb + ub) / 2
    return 10**(tens * (ub - mid) + mid)


def normalize_A_kX2(tens, A):
    device = A.device
    log_A = torch.log10(A)
    lb = torch.max(0.1 * log_A - 1, torch.tensor([-8], device=device))
    ub = torch.min(0.1 * log_A + 1, torch.tensor([0], device=device))
    mid = (lb + ub) / 2
    return (torch.log10(tens) - mid) / (ub - mid)


def denormalize_A_kX2(tens, A):
    device = A.device
    log_A = torch.log10(A)
    lb = torch.max(0.1 * log_A - 1, torch.tensor([-8], device=device))
    ub = torch.min(0.1 * log_A + 1, torch.tensor([0], device=device))
    mid = (lb + ub) / 2
    return 10**(tens * (ub - mid) + mid)


def normalize_A_KR(tens, A):
    device = A.device
    log_A = torch.log10(A)
    lb = torch.tensor([-7.5], device=device)
    ub = torch.min(0.1 * log_A - 0.5, torch.tensor([-2.5], device=device))
    mid = (lb + ub) / 2
    aa = torch.log10(tens)
    return (torch.log10(tens) - mid) / (ub - mid)


def denormalize_A_KR(tens, A):
    device = A.device
    log_A = torch.log10(A)
    lb = torch.tensor([-7.5], device=device)
    ub = torch.min(0.1 * log_A - 0.5, torch.tensor([-2.5], device=device))
    mid = (lb + ub) / 2
    return 10**(tens * (ub - mid) + mid)


def normalize_initial_X2(tens, sig, R, X1, X2, A, n, device):
    lb = torch.max((sig - X1 - R - (0.011 / A)**(1 / n) - X2),
                   torch.tensor([-40.], device=device))
    ub = torch.min((sig - X1 - R - X2), torch.tensor([40.], device=device))
    mid = (ub + lb) / 2
    aa = (tens - mid) / (ub - mid)
    return (tens - mid) / (ub - mid)


def denormalize_initial_X2(tens, sig, R, X1, X2, A, n, device):
    lb = torch.max((sig - X1 - R - (0.011 / A)**(1 / n) - X2),
                   torch.tensor([-40.], device=device))
    ub = torch.min((sig - X1 - R - X2), torch.tensor([40.], device=device))
    mid = (ub + lb) / 2
    return tens * (ub - mid) + mid
