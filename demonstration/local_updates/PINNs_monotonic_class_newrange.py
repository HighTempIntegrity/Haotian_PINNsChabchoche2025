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
import scipy.io
import time
import matplotlib as mpl

# mpl.use('Qt5Agg')  # activate this only if you run the code locally and want to see the plot on your screen
torch.autograd.set_detect_anomaly(False)

# Set random seed for reproducibility
seed = 128
torch.manual_seed(seed)
np.random.seed(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define hyperparameters for NNs
n_hidden_layers = 6
neurons = 128

p = 2


class Pinns_monotonic:
    def __init__(self, n_int_):
        self.n_int = n_int_  # number of drawed interior points
        # Extrema of the solution domain (t) in [0,1.5]

        self.domain_extrema = torch.tensor([[0.00375, 0.0075],  # total strain dimension []     [mean upperBound]
                                            [1.4e5, 1.9e5],  # E [MPa]
                                            [20., 32.],  # n  [-]
                                            [6.95e5, 7.1e5],  # C1  [MPa]
                                            [8.65e3, 8.67e3],  # gam1 [-]
                                            [1.34e5, 1.38e5],  # C2  [MPa]
                                            [1000., 1500.],  # gam2 [-]
                                            [75.5, 150.],  # R0  [MPa]
                                            [360., 700.],  # Q1  [MPa]
                                            [6.5, 12.],  # b1 [-]
                                            ]).to(device)

        self.domain_extrema_log = torch.tensor(
            [[np.log10(0.001), np.log10(0.01)],  # total strain rate [1/s] - to be extended
             [np.log10(1e-8), np.log10(1e2)],  # kX1 [1/s] log10
             [np.log10(1e-5), np.log10(1e-0)],  # kX2 [1/s] log10
             [np.log10(1e-6), np.log10(1e-2)]  # kR1 [1/s] log10
             ]).to(device)


        # Extrema of the result domain

        self.result_extrema = torch.tensor([[0., 5e-3],  # Delta eps_p [-] allows negative plastic strains
                                            [0., 5.],  # Delta R [MPa]
                                            [0., 300.],  # Delta X1 [MPa]
                                            [0., 300.]  # Delta X2 [MPa]
                                            ]).to(device)
                                            
                                            
                                            # need to include A and n also here to the list? 

        # F Dense NN to approximate the solution
        self.approximate_solution = NeuralNet(input_dimension=15, output_dimension=4,
                                              n_hidden_layers=n_hidden_layers,
                                              neurons=neurons,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=128).to(device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=15)

    # load saved NN parameters if need for further training
    def load_model(self):
        modelname = '20240617_144703_PDE_loss_learning_regular_6_128_X0_model.pth' #new range R0
        # modelname = '20240516_105225_PDE_loss_learning_regular_6_128_X0_model.pth'
        # path = os.path.join(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\model_save', modelname)
        path = os.path.join(r'/cluster/home/haotxu/PINNs/model_save', modelname)
        self.approximate_solution.load_state_dict(torch.load(path))

    ################################################################################################
    # Normalization for the four outputs
    def normalize_output(self, tens):
        return (tens - self.result_extrema[:, 0]) / (self.result_extrema[:, 1] - self.result_extrema[:, 0])

    def denormalize_output(self, tens):
        return tens * (self.result_extrema[:, 1] - self.result_extrema[:, 0]) + self.result_extrema[:, 0]

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
                tens * (self.domain_extrema_log[:, 1] - self.domain_extrema_log[:, 0]) + self.domain_extrema_log[:,
                                                                                         0])


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

    def prediction(self, x_train):

        linear_input = self.denormalize_input_linear(x_train[:, :10])
        log_input = self.denormalize_input_log(x_train[:, 11:15])

        E = linear_input[:, 1]
        n = linear_input[:, 2]
        A = self.denormalize_A_n(x_train[:, 10].to(torch.float64), n).reshape(-1, )
        KR1 = self.denormalize_A_KR(x_train[:, 14].to(torch.float64), A).reshape(-1, )

        R0 = linear_input[:, 7].to(torch.float64)
        eps_tot_rate = log_input[:, 0]

        X10 = torch.tensor([0.], device=device)
        X20 = torch.tensor([0.], device=device)

        eps_tot = linear_input[:, 0]
        linear_constraint = (eps_tot.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        t = eps_tot / eps_tot_rate
        sigma_el = eps_tot * E
        R_el = R0 * torch.exp(- KR1 * t)
        X1_el = 0.
        X2_el = 0.
        X_el = X1_el + X2_el
        a0 = sigma_el - X_el
        a00 = abs(a0) - R_el
        INP = (0.5 * (a00 + torch.sqrt(a00 ** 2))) / (self.domain_extrema[0, 1] * E)
        hard_constraint_eps_p = 2 * (torch.tanh(2 * INP) ** 2)
        hard_constraint_other = linear_constraint

        y_pred = self.denormalize_output(self.approximate_solution(x_train))

        # add hard constraints
        eps_p = y_pred[:, 0].to(torch.float64) * hard_constraint_eps_p
        R = y_pred[:, 1].to(torch.float64) * hard_constraint_other
        X1 = y_pred[:, 2].to(torch.float64) * hard_constraint_other
        X2 = y_pred[:, 3].to(torch.float64) * hard_constraint_other
        del y_pred
        X1 = X1 + X10
        X2 = X2 + X20
        R = R + R0
        sig = (eps_tot - eps_p) * E
        return eps_tot, eps_p, sig, R, X1, X2

    def prediction_end(self, x_train):
        x_train = x_train.clone().detach()
        x_train[:, 0] = 1.
        linear_input = self.denormalize_input_linear(x_train[:, :10])
        log_input = self.denormalize_input_log(x_train[:, 11:15])

        E = linear_input[:, 1]
        n = linear_input[:, 2]
        A = self.denormalize_A_n(x_train[:, 10].to(torch.float64), n).reshape(-1, )
        KR1 = self.denormalize_A_KR(x_train[:, 14].to(torch.float64), A).reshape(-1, )

        R0 = linear_input[:, 7].to(torch.float64)
        eps_tot_rate = log_input[:, 0]

        X10 = torch.tensor([0.], device=device)
        X20 = torch.tensor([0.], device=device)

        eps_tot = self.domain_extrema[0, 1]
        linear_constraint = (eps_tot.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        t = eps_tot / eps_tot_rate
        sigma_el = (eps_tot) * E
        R_el = R0 * torch.exp(- KR1 * t)
        X1_el = 0.
        X2_el = 0.
        X_el = X1_el + X2_el
        a0 = sigma_el - X_el
        a00 = abs(a0) - R_el
        INP = (0.5 * (a00 + torch.sqrt(a00 ** 2))) / (self.domain_extrema[0, 1] * E)
        hard_constraint_eps_p = 2 * (torch.tanh(2 * INP) ** 2)
        hard_constraint_other = linear_constraint

        y_pred = self.denormalize_output(self.approximate_solution(x_train))

        # add hard constraints
        eps_p = y_pred[:, 0].to(torch.float64) * hard_constraint_eps_p
        R = y_pred[:, 1].to(torch.float64) * hard_constraint_other
        X1 = y_pred[:, 2].to(torch.float64) * hard_constraint_other
        X2 = y_pred[:, 3].to(torch.float64) * hard_constraint_other
        del y_pred
        X1 = X1 + X10
        X2 = X2 + X20
        R = R + R0
        sig = (eps_tot - eps_p) * E
        return sig, R, X1, X2



