# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:41:51 2023

@author: Haotian
"""

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import os
from torch.utils.data import DataLoader
from Common import NeuralNet
from PINNs_monotonic_class_newrange import Pinns_monotonic as PINN_mono
# from PINNs_6_128_toEhsan_4 import Pinns_monotonic as PINN_mono
import scipy.io
import time
import matplotlib as mpl
import pandas as pd

# mpl.use('Qt5Agg')  # activate this only if you run the code locally and want to see the plot on your screen
torch.autograd.set_detect_anomaly(False)

# Set random seed for reproducibility
seed = 1283
torch.manual_seed(seed)
np.random.seed(seed)

#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
if device == 'cuda':
    print(torch.cuda.get_device_name())

# define hyperparameters for NNs
n_hidden_layers = 6
neurons = 128
print('hidden layers:', n_hidden_layers)
print('neurons:', neurons)
p = 2

pinn_mono = PINN_mono(100)
pinn_mono.load_model()


class Pinns_stress:
    def __init__(self, n_int_):
        self.n_int = n_int_  # number of drawed interior points

        self.domain_extrema = torch.tensor([[1.4e5, 1.9e5],  # E [MPa]
                                            [20., 32.],  # n  [-]
                                            [6.95e5, 7.1e5],  # C1  [MPa]
                                            [8.65e3, 8.67e3],  # gam1 [-]
                                            [1.34e5, 1.38e5],  # C2  [MPa]
                                            [1000., 1500.],  # gam2 [-]
                                            [78., 150.],  # R0  [MPa]
                                            [360., 700.],  # Q1  [MPa]
                                            [6.5, 12.],  # b1 [-]
                                            ]).to(device)

        self.domain_extrema_log = torch.tensor(
            [[np.log10(0.001), np.log10(0.01)],  # total strain rate [1/s] - to be extended
             [np.log10(1e-8), np.log10(1e2)],  # kX1 [1/s] log10
             [np.log10(1e-5), np.log10(1e-0)],  # kX2 [1/s] log10
             [np.log10(1e-6), np.log10(1e-2)]  # kR1 [1/s] log10
             ]).to(device)

        self.initial_extrema = torch.tensor([[0, 40],  # initial stress
                                             [0, 7.],  # initial R
                                             [0, 10.],  # initial X1
                                             ]).to(device)

        # Extrema of the result domain
        self.result_extrema = torch.tensor([[0., 5e-4],  # Delta eps_p [-] allows negative plastic strains
                                            [0., 1.],  # Delta R [MPa]
                                            [0., 10.],  # Delta X1 [MPa]
                                            [0., 30.]  # Delta X2 [MPa]
                                            ]).to(device)

        # need to include A and n also here to the list?

        # F Dense NN to approximate the solution
        self.approximate_solution = NeuralNet(input_dimension=19, output_dimension=4,
                                              n_hidden_layers=n_hidden_layers,
                                              neurons=neurons,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=128).to(device)

    # load saved NN parameters if need for further training
    def load_model(self):
        modelname = '20240506_233520_PDE_loss_learning_regular_6_128_final_X0_model.pth'  # kX2 0.1 KR1 0.1 # new benchmark # new initial value
        # path = os.path.join(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\model_save', modelname)
        path = os.path.join(r'/cluster/home/haotxu/PINNs/model_save', modelname)
        self.approximate_solution.load_state_dict(torch.load(path))

    ################################################################################################
    # Normalization for the four outputs
    def normalize_output(self, tens):
        # return torch.atan((tens - self.result_extrema[:, 0]) / (self.result_extrema[:, 1] - self.result_extrema[:, 0])) * (2/np.pi)
        # target = (tens - self.result_extrema[:, 0]) / (self.result_extrema[:, 1] - self.result_extrema[:, 0])
        # return torch.log(target + torch.sqrt(target**2 + 1))
        return (tens - self.result_extrema[:, 0]) / (self.result_extrema[:, 1] - self.result_extrema[:, 0])

    def denormalize_output(self, tens):
        # target = torch.sinh(tens)
        return tens * (self.result_extrema[:, 1] - self.result_extrema[:, 0]) + self.result_extrema[:, 0]
        # return target * (self.result_extrema[:, 1] - self.result_extrema[:, 0]) + self.result_extrema[:, 0]
        # return torch.tan(tens * (np.pi / 2)) *

    # Normalization for the inputs with linear function
    def normalize_input_linear(self, tens):
        return (tens - self.domain_extrema[:, 0]) / (self.domain_extrema[:, 1] - self.domain_extrema[:, 0])

    def denormalize_input_linear(self, tens):
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Normalization for the inputs with log function
    def normalize_input_log(self, tens):
        return (torch.log10(tens) - self.domain_extrema_log[:, 0]) / (
                self.domain_extrema_log[:, 1] - self.domain_extrema_log[:, 0])

    def denormalize_input_log(self, tens):
        return 10 ** (
                tens * (self.domain_extrema_log[:, 1] - self.domain_extrema_log[:, 0]) + self.domain_extrema_log[:, 0])

    # Normalization for A according to the value of n
    def normalize_A_n(self, tens, n):
        log_A = torch.log10(tens)
        mid = -2 * n
        ub = -2 * n + 10
        return (log_A - mid) / (ub - mid)

    def denormalize_A_n(self, tens, n):
        mid = -2 * n
        ub = -2 * n + 10
        return 10 ** (tens * (ub - mid) + mid)

    def normalize_A_kX1(self, tens, A):
        log_A = torch.log10(A)
        lb = torch.min(torch.max(0.4 * log_A + 6, torch.tensor([-18], device=device)),
                       torch.tensor([0], device=device))  # np.log10(13.)  #
        ub = torch.min(0.4 * log_A + 14, torch.tensor([2], device=device))  # np.log10(14.)  #
        mid = (lb + ub) / 2
        return (torch.log10(tens) - mid) / (ub - mid)

    def denormalize_A_kX1(self, tens, A):
        log_A = torch.log10(A)
        lb = torch.min(torch.max(0.4 * log_A + 6, torch.tensor([-18], device=device)),
                       torch.tensor([0], device=device))  # np.log10(13.)  #
        ub = torch.min(0.4 * log_A + 14, torch.tensor([2], device=device))  # np.log10(14.)  #
        mid = (lb + ub) / 2
        return 10 ** (tens * (ub - mid) + mid)

    def normalize_A_kX2(self, tens, A):
        log_A = torch.log10(A)
        lb = torch.max(0.1 * log_A - 1, torch.tensor([-8], device=device))  # np.log10(0.096)  #
        ub = torch.min(0.1 * log_A + 1, torch.tensor([0], device=device))  # np.log10(0.097)  #
        mid = (lb + ub) / 2
        return (torch.log10(tens) - mid) / (ub - mid)

    def denormalize_A_kX2(self, tens, A):
        log_A = torch.log10(A)
        lb = torch.max(0.1 * log_A - 1, torch.tensor([-8], device=device))  # np.log10(0.096)  #
        ub = torch.min(0.1 * log_A + 1, torch.tensor([0], device=device))  # np.log10(0.097)  #
        mid = (lb + ub) / 2
        return 10 ** (tens * (ub - mid) + mid)

    def normalize_A_KR(self, tens, A):
        log_A = torch.log10(A)
        lb = torch.tensor([-7.5], device=device)
        ub = torch.min(0.1 * log_A - 0.5, torch.tensor([-2.5], device=device))
        mid = (lb + ub) / 2
        aa = torch.log10(tens)
        return (torch.log10(tens) - mid) / (ub - mid)

    def denormalize_A_KR(self, tens, A):
        log_A = torch.log10(A)
        lb = torch.tensor([-7.5], device=device)
        ub = torch.min(0.1 * log_A - 0.5, torch.tensor([-2.5], device=device))
        mid = (lb + ub) / 2
        return 10 ** (tens * (ub - mid) + mid)

    def normalize_t(self, tens):
        ub = torch.log10(torch.tensor([900. + 1e-3], device=device))
        lb = torch.log10(torch.tensor([1e-3], device=device))
        mid = (ub + lb) / 2
        return (torch.log10(tens + 1e-3) - mid) / (ub - mid)

    def denormalize_t(self, tens):
        ub = torch.log10(torch.tensor([900. + 1e-3], device=device))
        lb = torch.log10(torch.tensor([1e-3], device=device))
        mid = (ub + lb) / 2
        return 10 ** (tens * (ub - mid) + mid) - 1e-3

    def normalize_initial(self, tens):
        return (tens - self.initial_extrema[:, 0]) / (self.initial_extrema[:, 1] - self.initial_extrema[:, 0])

    def denormalize_initial(self, tens):
        return tens * (self.initial_extrema[:, 1] - self.initial_extrema[:, 0]) + self.initial_extrema[:, 0]

    def normalize_initial_X2(self, tens, sig, R, X1, X2, A, n):
        lb = torch.max((sig - X1 - R - (0.011 / A) ** (1 / n) - X2), torch.tensor([-40.], device=device))
        ub = torch.min((sig - X1 - R - X2), torch.tensor([40.], device=device))
        mid = (ub + lb) / 2
        aa = (tens - mid) / (ub - mid)
        return (tens - mid) / (ub - mid)

    def denormalize_initial_X2(self, tens, sig, R, X1, X2, A, n):
        lb = torch.max((sig - X1 - R - (0.011 / A) ** (1 / n) - X2), torch.tensor([-40.], device=device))
        ub = torch.min((sig - X1 - R - X2), torch.tensor([40.], device=device))
        mid = (ub + lb) / 2
        return tens * (ub - mid) + mid

    ################################################################################################
    def prediction_end(self, x_test):
        x_test = x_test.clone().detach()
        x_test[:, 0] = 1.
        x_test_plus = torch.zeros([x_test.shape[0], 4], device=device)
        x_test = torch.cat((x_test, x_test_plus), 1)
        linear_input = self.denormalize_input_linear(x_test[:, 1:10])

        E = linear_input[:, 0]
        n = linear_input[:, 1]
        A = self.denormalize_A_n(x_test[:, 10].to(torch.float64), n).reshape(-1, )


        x_pre_ini = x_test[:, :15].clone().detach()
        x_pre_ini[:, 0] = 1.
        sig_ini, R_ini, X1_ini, X2_ini = pinn_mono.prediction_end(x_pre_ini)

        input_initial = torch.tensor([0, 0, 0, 0], device=device)
        x_test[:, 15:18] = self.normalize_initial(input_initial[:3])
        x_test[:, 18] = self.normalize_initial_X2(input_initial[3], sig_ini, R_ini, X1_ini, X2_ini, A,
                                                  n).detach()

        linear_constraint = (x_test[:, 0] + 1.)
        hard_constraint_linear = torch.cat(
            [linear_constraint.reshape(-1, 1), linear_constraint.reshape(-1, 1), linear_constraint.reshape(-1, 1),
             linear_constraint.reshape(-1, 1)], 1)

        y_pred = self.approximate_solution(x_test) * hard_constraint_linear

        y_pred = self.denormalize_output(y_pred)

        eps_p_pred = y_pred[:, 0].to(torch.float64)
        R_pred = y_pred[:, 1].to(torch.float64)
        X1_pred = y_pred[:, 2].to(torch.float64)
        X2_pred = y_pred[:, 3].to(torch.float64)

        X1 = X1_pred + X1_ini
        X2 = X2_pred + X2_ini
        R = R_pred + R_ini

        sig = sig_ini - eps_p_pred * E

        return sig, R, X1, X2
