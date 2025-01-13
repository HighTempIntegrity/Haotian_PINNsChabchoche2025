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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
if device == 'cuda':
    print(torch.cuda.get_device_name())

current_dir = os.path.dirname(__file__)
Benchmarks = np.loadtxt(current_dir + r'\data\Benchmarks.txt', dtype=np.float64, delimiter=',')
Benchmarks = torch.tensor(Benchmarks, dtype=torch.float64)

E_true = Benchmarks[0, :]  # Elastic modulus
A_true = Benchmarks[1, :]  # Viscosity parameter
n_true = Benchmarks[2, :]  # Viscosity exponent
# Kinematic hardening
C1_true = Benchmarks[3, :]  # Linear hardening "CX1"
gam1_true = Benchmarks[4, :]  # Dynamic recovery "gammaX1"
kX1_true = Benchmarks[5, :]  # Static recovery
C2_true = Benchmarks[6, :]  # Linear hardening "CX2"
gam2_true = Benchmarks[7, :]  # Dynamic recovery "gammaX2"
kX2_true = Benchmarks[8, :]  # Static recovery
# Isotropic hardening
R0_true = Benchmarks[9, :]  # Initial yield surface
Q1_true = Benchmarks[10, :]  # Linear hardening "CR"
b1_true = Benchmarks[11, :]  # Dynamic recovery "gammaR"
KR1_true = Benchmarks[12, :]  # Static recovery

## beta values from reference for LPBF Hastelloy X
Par = np.array([1.746270443e05,2.243801741e-03,245.0381457,1.993972806e-01,6.316767047e-02,1.204509005e03,1.117509005e03,
                8.510121916, 2.081156209e01,8.302192727e-03,8.388734383e02,3.663942147e-83,8.379167369,2.126480998e01,
                9.198109814e-03,8.660381500e02, 8.332538299e05,1.495990142e-01,8.073018609e03,2.898935861e03,
                1.181854251e05,1.355441925e01,5.053413638e-02,1.860377041e05, 2.480267858e-01,7.125301256e03,
                1.073468090e01,2.823841517e02,1.571568938e-02,1.202807068e03,6.961872471e02,1.121622191e-03,
                1.898530554e02,9.1584068397,2.526706819e-02,2.2440000e-03,5.000000000e-08,5.731891732e03,1.117509005e03,
                8.833701977,9.822449538,4.787523269e+02,1.289033897e-01,2.122267609])

# weight of four PDE loss terms (hyperparameters)
R_t_PDE_W = 0.01
X1_t_PDE_W = 1.
X2_t_PDE_W = 1.
eps_p_t_PDE_W = 1e-4

print('weight of eps_p:', eps_p_t_PDE_W, ', weight of R:', R_t_PDE_W, ', weight of X1:', X1_t_PDE_W, ', weight of X2:', X2_t_PDE_W)

# define hyperparameters for NNs
n_hidden_layers = 3
neurons = 64
p = 2
seed = 1238
## ensemble training for 5 different seed
for i in range(5):
    # Use date+time for naming
    time_now = time.localtime()
    time_now = time.strftime('%Y%m%d_%H%M%S', time_now)
    print(time_now)
    # Set random seed for reproducibility
    seed = seed + 10
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('hidden layers:', n_hidden_layers)
    print('neurons:', neurons)
    print('L', p, 'norm')
    print('random seed:', seed)

    class Pinns_monotonic:
        def __init__(self, n_int_):
            self.n_int = n_int_  # number of drawed interior points
            self.batch_size = self.n_int
            # Extrema of the solution domain (t) in [0,1.5]

            self.domain_extrema = torch.tensor([[0.005, 0.01],  # total strain dimension []     [mean upperBound]
                                                [1.4e5, 1.9e5],  # E [MPa]
                                                [20., 32.],  # n  [-]
                                                [6.95e5, 7.1e5],  # CX1  [MPa]
                                                [8.65e3, 8.67e3],  # gamX1 [-]
                                                [1.34e5, 1.38e5],  # CX2  [MPa]
                                                [1000., 1600.],  # gamX2 [-]
                                                [75.5, 150.],  # R0  [MPa]
                                                [360., 700.],  # CR  [MPa]
                                                [6.5, 12.],  # gamR [-]
                                                ]).to(device)


            self.domain_extrema_log = torch.tensor(
                [[np.log10(0.001), np.log10(0.011)],  # total strain rate [1/s] - to be extended
                 [np.log10(1e-40), np.log10(1e-5)],  # A [-] log10
                 [np.log10(1e-9), np.log10(3e1)],  # kX1 [1/s] log10
                 [np.log10(1e-5), np.log10(1e-0)],  # kX2 [1/s] log10
                 [np.log10(1e-6), np.log10(1e-2)]  # kR [1/s] log10
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
                                                  retrain_seed=seed).to(device)

            # Generator of Sobol sequences
            self.soboleng = torch.quasirandom.SobolEngine(dimension=15+1)

            # Training sets S_sb, S_tb, S_int as torch dataloader
            self.training_set_int = self.assemble_datasets()

        # load saved NN parameters if need for further training
        def load_model(self):
            modelname = '20241018_143736_PDE_loss_3_64_monotonic_model.pth'  
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

        def plot_learning_result(self, input_int, name, idx, strain_rate):
            global E_true, A_true, n_true, C1_true, gam1_true, kX1_true, C2_true, gam2_true, kX2_true, R0_true, Q1_true, b1_true, KR1_true
            E = E_true[idx]
            A = A_true[idx]
            n = n_true[idx]
            C1 = C1_true[idx]
            gam1 = gam1_true[idx]
            kX1 = kX1_true[idx]
            C2 = C2_true[idx]
            gam2 = gam2_true[idx]
            kX2 = kX2_true[idx]
            R0 = R0_true[idx]
            Q1 = Q1_true[idx]
            b1 = b1_true[idx]
            KR1 = KR1_true[idx]

            # load data and true solution from .mat file (from data_1 to data_7 -- 7 sets of different material parameters)
            if strain_rate == 0.01:
                data = scipy.io.loadmat(current_dir + r'\data\data_beta_range2_benchmark_monotonic_' + str(idx + 1) + '_rate_01.mat')
            elif strain_rate == 0.002:
                data = scipy.io.loadmat(current_dir + r'\data\data_beta_range2_benchmark_monotonic_' + str(idx + 1) + '_rate_002.mat')
            elif strain_rate == 0.0005:
                data = scipy.io.loadmat(current_dir + r'\data\data_beta_range2_benchmark_monotonic_' + str(idx + 1) + '_rate_0005.mat')
            elif strain_rate == 0.0001:
                data = scipy.io.loadmat(current_dir + r'\data\data_beta_range2_benchmark_monotonic_' + str(idx + 1) + '_rate_0001.mat')

            sig_data = torch.from_numpy(np.reshape(data['sig'], (-1, 1))).to(device)  # stress
            eps_tot_data = torch.from_numpy(np.reshape(data['eps_tot'], (-1, 1))).to(device)  # total strain
            R_data = torch.from_numpy(np.reshape(data['R'], (-1, 1))).to(device)  # R
            X1_data = torch.from_numpy(np.reshape(data['X1'], (-1, 1))).to(device)  # X1
            X2_data = torch.from_numpy(np.reshape(data['X2'], (-1, 1))).to(device)  # X2
            eps_p_data = torch.from_numpy(np.reshape(data['eps_in'], (-1, 1))).to(device)  # viscoplastic strain # eps_tot_data - sig_data / E + el0 #

            x_test = input_int[:, :15].detach().to(device)
            x_test[:, 0] = torch.sort(x_test[:, 0], 0)[0].reshape(-1, )

            input_linear = torch.tensor([0, E, n, C1, gam1, C2, gam2, R0, Q1, b1], device=device)
            input_log = torch.tensor([strain_rate, A, kX1, kX2, KR1], dtype=torch.float64, device=device)

            x_test[:, 1:10] = self.normalize_input_linear(input_linear)[1:10]
            x_test[:, 10:] = self.normalize_input_log(input_log)

            x_test.requires_grad = True
            eps_tot = self.denormalize_input_linear(x_test[:, :10])[:, 0]
            eps_tot_rate = input_log[0]

            # Calculation of the turning point of elastic domain
            linear_constraint = (eps_tot.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
            t = eps_tot / eps_tot_rate
            sigma_el = (eps_tot) * E
            R_el = R0 * torch.exp(- KR1 * t)  # torch.exp(torch.log(R0) - KR1 * t)
            X1_el = 0.
            X2_el = 0.
            X_el = X1_el + X2_el
            a0 = sigma_el - X_el
            a00 = abs(a0) - R_el
            INP = (0.5 * (a00 + torch.sqrt(a00 ** 2))) / (self.domain_extrema[0, 1] * E)
            hard_constraint_eps_p = 2 * (torch.tanh(2 * INP) ** 2)
            hard_constraint_other = linear_constraint


            y_pred = self.denormalize_output(self.approximate_solution(x_test))

            eps_p_pred = y_pred[:, 0] * hard_constraint_eps_p
            R_pred = y_pred[:, 1].to(torch.float64) * hard_constraint_other
            X1_pred = y_pred[:, 2].to(torch.float64) * hard_constraint_eps_p
            X2_pred = y_pred[:, 3].to(torch.float64) * hard_constraint_eps_p
            eps_tot_plot = eps_tot

            R_pred = R_pred + R0
            X1_pred = X1_pred
            X2_pred = X2_pred

            sig_pred = (eps_tot - eps_p_pred) * E

            X = X1_pred + X2_pred

            eps_tot_rate = strain_rate

            eps_p_eps = torch.autograd.grad(eps_p_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
                        1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
            R_eps = torch.autograd.grad(R_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
                    1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
            X1_eps = torch.autograd.grad(X1_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
                    1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
            X2_eps = torch.autograd.grad(X2_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
                    1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))

            eps_p_t = (eps_p_eps * eps_tot_rate)
            eps_p_eq_t = abs(eps_p_t)

            R_t = (R_eps * eps_tot_rate)
            X1_t = (X1_eps * eps_tot_rate)
            X2_t = (X2_eps * eps_tot_rate)

            loss0_vec = 1000 * ((-X1_t + torch.sqrt(X1_t ** 2)) + (-X2_t + torch.sqrt(X2_t ** 2)))
            loss0 = torch.mean(loss0_vec ** p)
            a1 = sig_pred.reshape(-1, ) - X.reshape(-1, )
            a2 = a1 * torch.tanh(a1 / 0.001) - R_pred.reshape(-1, )
            a3 = A * (0.5 * (a2 + torch.sqrt(0.001 ** 2 + a2 ** 2))) ** n * torch.sign(
                a1)
            loss1 = (eps_p_t.reshape(-1, ) - a3)
            mean1 = torch.mean(abs(loss1)).cpu().detach().numpy()
            loss2 = (R_t.reshape(-1, ) - Q1 * eps_p_eq_t + b1 * R_pred.reshape(-1, ) * eps_p_eq_t.reshape(
                -1, ) + KR1 * R_pred.reshape(-1, ))
            mean2 = torch.mean(abs(loss2)).cpu().detach().numpy()
            loss3 = (X1_t.reshape(-1, ) - C1 * eps_p_t + gam1 * eps_p_eq_t * X1_pred.reshape(
                -1, ) + kX1 * X1_pred.reshape(-1, ))
            mean3 = torch.mean(abs(loss3)).cpu().detach().numpy()
            loss4 = (X2_t.reshape(-1, ) - C2 * eps_p_t + gam2 * eps_p_eq_t * X2_pred.reshape(
                -1, ) + kX2 * X2_pred.reshape(-1, ))
            mean4 = torch.mean(abs(loss4)).cpu().detach().numpy()

            fig = plt.figure(figsize=(16, 8))

            ax1 = plt.subplot(1, 3, 1)
            ax1.set_ylabel('stress (MPa)')
            ax1.set_xlabel('log10(time) (s)')
            ax1.plot(eps_tot_plot.cpu().detach().numpy(), sig_pred.cpu().detach().numpy(), ms=0.5, label="stress_predict")
            ax1.plot(eps_tot_data.cpu().detach().numpy(), sig_data.cpu().detach().numpy(), ms=0.5, label="stress_true")
            ax1.legend(loc='upper left')

            ax3 = plt.subplot(2, 3, 2)
            ax3.set_ylabel('eps_p')
            ax3.plot(eps_tot.cpu().detach().numpy(), eps_p_pred.cpu().detach().numpy(), ms=0.5,
                     label="viscoplastic strain predict")
            ax3.plot(eps_tot_data.cpu().detach().numpy(), eps_p_data.cpu().detach().numpy(), ms=0.5,
                     label="viscoplastic strain true")
            ax3.legend(loc='upper left')
            color = 'tab:red'
            ax12 = ax3.twinx()
            ax12.plot(eps_tot.cpu().detach().numpy(), loss1.cpu().detach().numpy(), ms=0.5, label="Loss1", color=color)
            ax12.set_ylabel('loss1', color=color)
            ax12.legend(loc='upper right')
            ax12.tick_params(axis='y', labelcolor=color)
            plt.ylim((-10 * mean1, 10 * mean1))

            ax4 = plt.subplot(2, 3, 5, sharex=ax3)
            ax4.set_xlabel('log10(time) (s)')
            ax4.set_ylabel('R (MPa)')
            ax4.plot(eps_tot.cpu().detach().numpy(), R_pred.cpu().detach().numpy(), ms=0.5, label="R_predict")
            ax4.plot(eps_tot_data.cpu().detach().numpy(), R_data.cpu().detach().numpy(), ms=0.5, label="R_true")
            ax4.legend(loc='upper left')
            color = 'tab:red'
            ax42 = ax4.twinx()
            ax42.plot(eps_tot.cpu().reshape(-1, ).detach().numpy(), loss2.cpu().reshape(-1, ).detach().numpy(), ms=0.5,
                      label="Loss2", color=color)
            ax42.set_ylabel('loss2', color=color)
            ax42.legend(loc='upper right')
            ax42.tick_params(axis='y', labelcolor=color)
            plt.ylim((-10 * mean2, 10 * mean2))

            ax5 = plt.subplot(2, 3, 3)
            ax5.set_ylabel('X1 (MPa)')
            ax5.plot(eps_tot.cpu().detach().numpy(), X1_pred.cpu().detach().numpy(), ms=0.5, label="X1_predict")
            ax5.plot(eps_tot_data.cpu().detach().numpy(), X1_data.cpu().detach().numpy(), ms=0.5, label="X1_true")
            ax5.legend(loc='upper left')
            color = 'tab:red'
            ax52 = ax5.twinx()
            ax52.plot(eps_tot.cpu().detach().numpy(), loss3.cpu().detach().numpy(), ms=0.5, label="Loss3", color=color)
            ax52.set_ylabel('loss3', color=color)
            ax52.legend(loc='upper right')
            ax52.tick_params(axis='y', labelcolor=color)
            plt.ylim((-10 * mean3, 10 * mean3))

            ax6 = plt.subplot(2, 3, 6, sharex=ax5)
            ax6.set_xlabel('log10(time) (s)')
            ax6.set_ylabel('X2 (MPa)')
            ax6.plot(eps_tot.cpu().detach().numpy(), X2_pred.cpu().detach().numpy(), ms=0.5, label="X2_predict")
            ax6.plot(eps_tot_data.cpu().detach().numpy(), X2_data.cpu().detach().numpy(), ms=0.5, label="X2_true")
            ax6.legend(loc='upper left')

            color = 'tab:red'
            ax62 = ax6.twinx()
            ax62.plot(eps_tot.cpu().detach().numpy(), loss4.cpu().detach().numpy(), ms=0.5, label="Loss4", color=color)
            ax62.set_ylabel('loss4', color=color)
            ax62.legend(loc='upper right')
            ax62.tick_params(axis='y', labelcolor=color)
            plt.ylim((-10 * mean4, 10 * mean4))
            fig.tight_layout()

            plotname = time_now + '_Benchmark_' + str(idx + 1) + '_rate_' + str(strain_rate) + '_NNstructure_' + str(
                n_hidden_layers) + '_' + str(neurons) + '.png'
            path = os.path.join(current_dir + r'\plot_fit', plotname)
            # plt.savefig(path, dpi=300, bbox_inches='tight')

            return sig_pred, sig_data, eps_tot_plot, eps_tot_data

        ################################################################################################
        #  Function returning the input-output tensor required to assemble the training set corresponding to the interior domain where the PDE is enforced
        def add_interior_points(self):
            global Par
            ## generate the collocation points in the training range ##
            input_int_pre = (self.soboleng.draw(self.n_int).to(device)) * 2 - 1
            temp_K = (input_int_pre[:, 15] + 1) * 510
            E, A, n, C1, gam1, kX1, C2, gam2, kX2, Q1, b1, KR1, R0 = cal_parameter_from_beta(Par, temp_K.double().reshape(-1, ))
            E = E + input_int_pre[:, 1] * 1.5e4
            n = n + input_int_pre[:, 2] * 3
            C1 = C1 + input_int_pre[:, 3] * 2.5e3
            gam1 = gam1 + input_int_pre[:, 4] * 7
            C2 = C2 + input_int_pre[:, 5] * 1.5e3
            gam2 = gam2 + input_int_pre[:, 6] * 1.5e2
            R0 = R0 + input_int_pre[:, 7] * 20
            Q1 = Q1 + input_int_pre[:, 8] * 150
            b1 = b1 + input_int_pre[:, 9] * 1.5

            A = 10 ** (torch.log10(A) + input_int_pre[:, 11] * 5)
            kX1 = 10 ** (torch.log10(kX1) + input_int_pre[:, 12] * 3)
            kX2 = 10 ** (torch.log10(kX2) + input_int_pre[:, 13] * 1.5)
            KR1 = 10 ** (torch.log10(KR1) + input_int_pre[:, 14] * 1.5)
            input_linear = torch.cat((input_int_pre[:, 0].reshape(-1, 1), E.reshape(-1, 1), n.reshape(-1, 1),
                                      C1.reshape(-1, 1), gam1.reshape(-1, 1), C2.reshape(-1, 1), gam2.reshape(-1, 1),
                                      R0.reshape(-1, 1), Q1.reshape(-1, 1), b1.reshape(-1, 1)), 1)
            input_log = torch.cat([input_int_pre[:, 10].reshape(-1, 1), A.reshape(-1, 1), kX1.reshape(-1, 1),
                                   kX2.reshape(-1, 1), KR1.reshape(-1, 1)], 1)
            ones = torch.ones(1, device=device)
            input_int = input_int_pre[:, :15].clone().detach()
            input_int[:, 1:10] = self.normalize_input_linear(input_linear)[:, 1:]
            input_int[:, 11:] = self.normalize_input_log(input_log)[:, 1:]
            input_int = torch.min(torch.max(input_int.clone().detach(), -1 * ones), ones)
            output_int = torch.zeros((input_int.shape[0], 1)).to(device)
            return input_int, output_int

        # Function returning the training sets S_int as dataloader
        def assemble_datasets(self):
            input_int, output_int = self.add_interior_points()
            print('batch size:', self.batch_size)
            training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.batch_size,
                                          shuffle=True)
            return training_set_int

        ################################################################################################
        def compute_pde_residual(self, x_train_, verbose):
            # weight for each loss term
            global eps_p_t_PDE_W, R_t_PDE_W, X1_t_PDE_W, X2_t_PDE_W
            x_train_.requires_grad = True

            # Denormalization of inputs
            linear_input = self.denormalize_input_linear(x_train_[:, :10])
            log_input = self.denormalize_input_log(x_train_[:, 10:15])

            eps_tot = linear_input[:, 0]
            E = linear_input[:, 1]
            n = linear_input[:, 2]
            A = log_input[:, 1]
            C1 = linear_input[:, 3]
            gam1 = linear_input[:, 4]
            kX1 = log_input[:, 2]
            C2 = linear_input[:, 5]
            gam2 = linear_input[:, 6]
            kX2 = log_input[:, 3]
            Q1 = linear_input[:, 8]
            b1 = linear_input[:, 9]
            KR1 = log_input[:, 4]
            R0 = linear_input[:, 7].to(torch.float64)
            eps_tot_rate = log_input[:, 0]
            del linear_input, log_input

            # Calculation of the turning point of elastic domain
            linear_constraint = (eps_tot.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
            t = eps_tot/eps_tot_rate
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

            # Calculation of outputs and denormalization
            y_pred = self.denormalize_output(self.approximate_solution(x_train_))

            # add hard constraints
            eps_p = y_pred[:, 0].to(torch.float64) * hard_constraint_eps_p #
            R = y_pred[:, 1].to(torch.float64) * hard_constraint_other
            X1 = y_pred[:, 2].to(torch.float64) * hard_constraint_eps_p
            X2 = y_pred[:, 3].to(torch.float64) * hard_constraint_eps_p
            del y_pred
            X1 = X1
            X2 = X2
            R = R + R0


            X = X1 + X2

            sig = (eps_tot - eps_p) * E

            Delta_eps_p_eps = torch.autograd.grad(eps_p.sum(), x_train_, create_graph=True)[0][:, 0] * (1 / (
                    self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
            Delta_R_eps = torch.autograd.grad(R.sum(), x_train_, create_graph=True)[0][:, 0] * (
                    1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
            Delta_X1_eps = torch.autograd.grad(X1.sum(), x_train_, create_graph=True)[0][:, 0] * (
                    1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
            Delta_X2_eps = torch.autograd.grad(X2.sum(), x_train_, create_graph=True)[0][:, 0] * (
                    1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))

            del x_train_
            eps_p_t = (Delta_eps_p_eps * eps_tot_rate)
            eps_p_eq_t = abs(eps_p_t)

            R_t = (Delta_R_eps * eps_tot_rate)
            X1_t = (Delta_X1_eps * eps_tot_rate)
            X2_t = (Delta_X2_eps * eps_tot_rate)


            # to put higher weight for small strains
            weight_vec = 1. / (0.5 + (0.5 * eps_tot.reshape(-1, )) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))  # * (t.reshape(-1, ) + 1.)

            weight = weight_vec.clone().detach() * torch.sqrt(0.01 / eps_tot_rate.clone().detach())
            weight_loss3 = weight
            weight_loss4 = weight

            # Loss 1
            Smooth_factor = 0.001
            a1 = sig.reshape(-1, ) - X.reshape(-1, )
            a2 = a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(-1, )
            a3 = A * (0.5 * (a2 + torch.sqrt(a2 ** 2))) ** n * torch.tanh(a1 / Smooth_factor)

            # make it such that after the training, it prints the individual values of loss0,loss n and max/min for ANN outputs
            loss1_vec = (eps_p_t.reshape(-1, ) - a3) * weight.reshape(-1, ) / eps_p_t_PDE_W
            loss1 = torch.mean(loss1_vec ** p)
            loss1_rec = loss1.detach()

            # loss 2
            loss2_vec = (R_t.reshape(-1, ) - Q1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) + b1.reshape(-1, ) * R.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) + KR1.reshape(-1, ) * R.reshape(-1, )) / R_t_PDE_W #(abs(R_t.reshape(-1, ).clone().detach() + 1e-2))#(abs(R.reshape(-1, ).clone().detach()) + R_t_PDE_W) #
            loss2_w = loss2_vec * weight.reshape(-1, )
            loss2 = torch.mean(loss2_w ** p)
            loss2_rec = loss2.detach()

            # loss 3
            loss3_vec = (X1_t.reshape(-1, ) - C1.reshape(-1, ) * eps_p_t.reshape(-1, ) + gam1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) * X1.reshape(-1, ) + kX1.reshape(-1, ) * X1.reshape(-1, )) / X1_t_PDE_W  #(abs(X1_t.reshape(-1, ).clone().detach() + 1e0)) # (abs(X1.reshape(-1, ).clone().detach()) + X1_t_PDE_W) #
            loss3_w = loss3_vec * weight_loss3.reshape(-1, )
            loss3 = torch.mean(loss3_w ** p)
            loss3_rec = loss3.detach()

            # loss 4
            loss4_vec = (X2_t.reshape(-1, ) - C2.reshape(-1, ) * eps_p_t.reshape(-1, ) + gam2.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) * X2.reshape(-1, ) + kX2.reshape(-1, ) * X2.reshape(-1, )) / X2_t_PDE_W  #(abs(X2_t.reshape(-1, ).clone().detach() + 1e0))# (abs(X2.reshape(-1, ).clone().detach()) + X2_t_PDE_W) #
            loss4_w = loss4_vec * weight_loss4.reshape(-1, )
            loss4 = torch.mean(loss4_w ** p)
            loss4_rec = loss4.detach()

            # print the loss terms
            if verbose:
                print('loss1:', loss1_rec.cpu().detach().item(), 'loss2:', loss2_rec.cpu().detach().item(), 'loss3:',
                      loss3_rec.cpu().detach().item(), 'loss4:', loss4_rec.cpu().detach().item())

            loss = torch.log10(loss1 + loss2 + loss3 + loss4)
            loss_rec = loss.detach()

            # print total loss
            if verbose:
                print('Loss: ', loss_rec.item())
            loss0_rec = 0
            return loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec



        # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
        def compute_loss(self, inp_train_int, verbose):
            loss_pde, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_pde_residual(inp_train_int, verbose)
            loss = loss_pde
            return loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec

        ################################################################################################
        def fit(self, num_epochs, optimizer, verbose):
            history = list()
            loss_1 = list()
            loss_2 = list()
            loss_3 = list()
            loss_4 = list()
            loss_0 = list()
            # Loop over epochs
            for epoch in range(num_epochs):
                if verbose: print("################################ ", epoch, " ################################")

                for j, (inp_train_int, u_train_int) in enumerate(self.training_set_int):
                    def closure():
                        optimizer.zero_grad()
                        loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_loss(inp_train_int,
                                                                                                       verbose=verbose)
                        loss.backward()
                        loss_1.append(loss1_rec.item())
                        loss_2.append(loss2_rec.item())
                        loss_3.append(loss3_rec.item())
                        loss_4.append(loss4_rec.item())
                        loss_0.append(loss0_rec.item())
                        history.append(loss_rec.item())
                        # print('Loss: ', history[-1])

                        return loss

                    optimizer.step(closure=closure)

            print('Final Loss: ', history[-1], 'loss1:', loss_1[-1], 'loss2:', loss_2[-1], 'loss3:', loss_3[-1], 'loss4:',
                      loss_4[-1], 'loss0:', loss_0[-1])

            return history, loss_1, loss_2, loss_3, loss_4


    n_int = 300000
    print('number of callocation points:', n_int)
    pinn = Pinns_monotonic(n_int)
    # load trained NN for further training
    '''
    pinn.load_model()
    print("Load model: Transfer learning")
    '''

    # opt_type = "ADAM"
    opt_type = "LBFGS"
    n_epochs1 = 1
    if opt_type == "ADAM":
        optimizer_ = optim.Adam(list(pinn.approximate_solution.parameters()), lr=0.01, weight_decay=0.)
    elif opt_type == "LBFGS":
        optimizer_ = optim.LBFGS(list(pinn.approximate_solution.parameters()), lr=float(0.5),
                                 max_iter=1000000,
                                 max_eval=1000000,
                                 tolerance_grad=1e-12,
                                 tolerance_change=1e-12,
                                 line_search_fn='strong_wolfe')

    start_time = time.time()

    # what about loss0
    history1, loss_11, loss_21, loss_31, loss_41 = pinn.fit(num_epochs=n_epochs1, optimizer=optimizer_, verbose=False)
    n_epochs1 = len(history1)

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    # save the trained model
    modelname = time_now + '_PDE_loss_' + str(n_hidden_layers) + '_' + str(neurons) + '_monotonic_model.pth'
    modelname = time_now + '_PDE_loss_' + str(n_hidden_layers) + '_' + str(neurons) + '_monotonic_model_wo_hardconstraint.pth'
    path = os.path.join(current_dir + r'\model_save', modelname)
    torch.save(pinn.approximate_solution.cpu().state_dict(), path)  # saving model !!!! activate it if you want to save the model
    pinn.approximate_solution = pinn.approximate_solution.to(device)

    # plot the curve of total loss
    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(history1) + 1), history1, label="Train Loss")
    plt.legend()
    plotname = time_now + '_PDE_loss_learning_regular_' + str(n_hidden_layers) + '_' + str(
        neurons) + '_X0_loss.png'
    path = os.path.join(current_dir + r'\plot_loss', plotname)  # path of saving
    plt.savefig(path) # save the plot if you want

    # plot the curve of each PDE loss terms
    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(loss_11) + 1).reshape(-1, 1), np.log10(loss_11).reshape(-1, 1), label="Loss1")
    plt.plot(np.arange(1, len(loss_21) + 1).reshape(-1, 1), np.log10(loss_21).reshape(-1, 1), label="Loss2")
    plt.plot(np.arange(1, len(loss_31) + 1).reshape(-1, 1), np.log10(loss_31).reshape(-1, 1), label="Loss3")
    plt.plot(np.arange(1, len(loss_41) + 1).reshape(-1, 1), np.log10(loss_41).reshape(-1, 1), label="Loss4")
    plt.legend()
    plotname = time_now + '_PDE_loss_learning_seperate_regular_' + str(
        n_hidden_layers) + '_' + str(neurons) + '_X0_loss.png'
    path = os.path.join(current_dir + r'\plot_loss', plotname)  # path of saving
    #plt.savefig(path) # save the plot if you want
    '''
    input_int = (pinn.soboleng.draw(30000).to(device)) * 2 - 1
    for j in range(0, 8):
        for strain_rate in [0.01, 0.002, 0.0005, 0.0001]:#[0.005, 0.001, 0.0002]:#
            pinn.plot_learning_result(input_int, name='end', idx=j, strain_rate=strain_rate)
    '''
