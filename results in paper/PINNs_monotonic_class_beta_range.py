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
import pandas as pd
from Beta_parameters import cal_parameter_from_beta

# mpl.use('Qt5Agg')  # activate this only if you run the code locally and want to see the plot on your screen
torch.autograd.set_detect_anomaly(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.path.dirname(__file__)
# define hyperparameters for NNs
n_hidden_layers = 3
neurons = 64

class Pinns_monotonic:
    def __init__(self, n_int_):
        self.n_int = n_int_  # number of drawed interior points
        self.batch_size = self.n_int
        # Extrema of the solution domain (t) in [0,1.5]

        self.domain_extrema = torch.tensor([[0.005, 0.01],  # total strain dimension []     [mean upperBound]
                                            [1.4e5, 1.9e5],  # E [MPa]
                                            [20., 32.],  # n  [-]
                                            [6.95e5, 7.1e5],  # C1  [MPa]
                                            [8.65e3, 8.67e3],  # gam1 [-]
                                            [1.34e5, 1.38e5],  # C2  [MPa]
                                            [1000., 1600.],  # gam2 [-]
                                            [75.5, 150.],  # R0  [MPa]
                                            [360., 700.],  # Q1  [MPa]
                                            [6.5, 12.],  # b1 [-]
                                            ]).to(device)


        self.domain_extrema_log = torch.tensor(
                [[np.log10(0.001), np.log10(0.011)],  # total strain rate [1/s] - to be extended
                 [np.log10(1e-40), np.log10(1e-5)],  # A [-] log10
                 [np.log10(1e-9), np.log10(3e1)],  # kX1 [1/s] log10
                 [np.log10(1e-5), np.log10(1e-0)],  # kX2 [1/s] log10
                 [np.log10(1e-6), np.log10(1e-2)]  # kR1 [1/s] log10
                 ]).to(device)



        # Extrema of the result domain

        self.result_extrema = torch.tensor([[0., 5e-3],  # Delta eps_p [-] allows negative plastic strains
                                            [0., 5.],  # Delta R [MPa]
                                            [0., 300.],  # Delta X1 [MPa]
                                            [0., 300.]  # Delta X2 [MPa]
                                            ]).to(device)


        # F Dense NN to approximate the solution
        self.approximate_solution = NeuralNet(input_dimension=15, output_dimension=4,
                                              n_hidden_layers=n_hidden_layers,
                                              neurons=neurons,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=128).to(device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=15+1)

    # load saved NN parameters if need for further training
    def load_model(self):
        modelname = '20241030_151038_PDE_loss_3_64_monotonic_model.pth'
        path = os.path.join(current_dir + r'\model_save', modelname)
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

    def prediction(self, x_test):
        # Denormalization of inputs
        linear_input = self.denormalize_input_linear(x_test[:, :10])
        log_input = self.denormalize_input_log(x_test[:, 10:15])

        eps_tot = linear_input[:, 0]
        E = linear_input[:, 1]

        KR1 = log_input[:, 4]

        R0 = linear_input[:, 7].to(torch.float64)
        eps_tot_rate = log_input[:, 0]

        del linear_input, log_input

        # Calculation of the turning point of elastic domain
        linear_constraint = (eps_tot.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        t = eps_tot / eps_tot_rate
        sigma_el = (eps_tot) * E
        R_el = R0 * torch.exp(- KR1 * t)
        X1_el = 0.  # X10 * torch.exp(- kX1 * t)
        X2_el = 0.  # X20 * torch.exp(- kX2 * t)
        X_el = X1_el + X2_el
        a0 = sigma_el - X_el
        a00 = abs(a0) - R_el
        INP = (0.5 * (a00 + torch.sqrt(a00 ** 2))) / (self.domain_extrema[0, 1] * E)
        hard_constraint_eps_p = 2 * (torch.tanh(2 * INP) ** 2)  # * torch.tanh(100000 * linear_constraint)
        hard_constraint_other = linear_constraint

        # Calculation of outputs and denormalization
        y_pred = self.denormalize_output(self.approximate_solution(x_test))

        # add hard constraints
        eps_p = y_pred[:, 0].to(torch.float64) * hard_constraint_eps_p
        R = y_pred[:, 1].to(torch.float64) * hard_constraint_other
        X1 = y_pred[:, 2].to(torch.float64) * hard_constraint_other
        X2 = y_pred[:, 3].to(torch.float64) * hard_constraint_other
        del y_pred
        X1 = X1
        X2 = X2
        R = R + R0
        sig = (eps_tot - eps_p) * E
        return t, eps_tot, eps_p, sig, R, X1, X2


