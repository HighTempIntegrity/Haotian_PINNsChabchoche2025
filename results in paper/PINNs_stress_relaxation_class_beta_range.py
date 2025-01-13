# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:41:51 2023

@author: Haotian
"""

import numpy as np
import torch
import os
from Common import NeuralNet

# mpl.use('Qt5Agg')  # activate this only if you run the code locally and want to see the plot on your screen
torch.autograd.set_detect_anomaly(False)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.path.dirname(__file__)

# define hyperparameters for NNs
n_hidden_layers = 3
neurons = 256
p = 2



class Pinns_stress:
    def __init__(self, n_int_):
        self.n_int = n_int_  # number of drawed interior points

        self.domain_extrema = torch.tensor([[1.4e5, 1.9e5],  # E [MPa]
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

        self.initial_extrema = torch.tensor([[400., 800.],  # initial stress
                                             [76., 150.],  # initial R
                                             [100., 200.],  # initial X1
                                             [200., 400.],  # initial X2
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
        modelname = '20241030_155000_PDE_loss_learning_regular_3_256_third_X0_model.pth'
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
                tens * (self.domain_extrema_log[:, 1] - self.domain_extrema_log[:, 0]) + self.domain_extrema_log[:, 0])


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

    def denormalize_initial_X2(self, tens, sig, R, X1, X2, A, n):
        lb = torch.max((sig - X1 - R - (0.011 / A) ** (1 / n) - X2), torch.tensor([-40.], device=device))
        ub = torch.min((sig - X1 - R - X2), torch.tensor([40.], device=device))
        mid = (ub + lb) / 2
        return tens * (ub - mid) + mid

    ################################################################################################
    def prediction(self, x_test, sig_pre, R_pre, X1_pre, X2_pre):
        x_test_plus = torch.zeros([x_test.shape[0], 4], device=device)
        x_test = torch.cat((x_test, x_test_plus), 1)

        linear_input = self.denormalize_input_linear(x_test[:, 1:10])
        t = self.denormalize_t(x_test[:, 0])
        E = linear_input[:, 0]

        input_initial = torch.cat([sig_pre.reshape(-1, 1), R_pre.reshape(-1, 1), X1_pre.reshape(-1, 1), X2_pre.reshape(-1, 1)], 1)

        x_test[:, 15:] = self.normalize_initial(input_initial)

        eps_tot = torch.zeros(x_test.shape[0], device=device)

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

        X1 = X1_pred + X1_pre
        X2 = X2_pred + X2_pre
        R = R_pred + R_pre

        sig = sig_pre + (eps_tot - eps_p_pred) * E

        return t, eps_tot, eps_p_pred, sig, R, X1, X2
