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
from PINNs_monotonic_class_beta_range import Pinns_monotonic as PINN_mono
# from PINNs_monotonic_class_newrange import Pinns_monotonic as PINN_mono
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
C1_true = Benchmarks[3, :]  # Linear hardening
gam1_true = Benchmarks[4, :]  # Dynamic recovery
kX1_true = Benchmarks[5, :]  # Static recovery
C2_true = Benchmarks[6, :]  # Linear hardening
gam2_true = Benchmarks[7, :]  # Dynamic recovery
kX2_true = Benchmarks[8, :]  # Static recovery
# Isotropic hardening
R0_true = Benchmarks[9, :]  # Initial yield surface
Q1_true = Benchmarks[10, :]  # Linear hardening
b1_true = Benchmarks[11, :]  # Dynamic recovery
KR1_true = Benchmarks[12, :]  # Static recovery

Par = np.array([1.746270443e05,2.243801741e-03,245.0381457,1.993972806e-01,6.316767047e-02,1.204509005e03,1.117509005e03,8.510121916, 2.081156209e01,8.302192727e-03,8.388734383e02,3.663942147e-83,8.379167369,2.126480998e01,9.198109814e-03,8.660381500e02, 8.332538299e05,1.495990142e-01,8.073018609e03,2.898935861e03,1.181854251e05,1.355441925e01,5.053413638e-02,1.860377041e05, 2.480267858e-01,7.125301256e03,1.073468090e01,2.823841517e02,1.571568938e-02,1.202807068e03,6.961872471e02,1.121622191e-03, 1.898530554e02,9.1584068397,2.526706819e-02,2.2440000e-03,5.000000000e-08,5.731891732e03,1.117509005e03,8.833701977,9.822449538,4.787523269e+02,1.289033897e-01,2.122267609])

# print('one over eps_tot_rate for weight')
# weight of four PDE loss terms (hyperparameters)

R_t_PDE_W = 0.02
X1_t_PDE_W = 4.
X2_t_PDE_W = 1.
eps_p_t_PDE_W = 1e-5
print('weight of eps_p:', eps_p_t_PDE_W, ', weight of R:', R_t_PDE_W, ', weight of X1:', X1_t_PDE_W, ', weight of X2:', X2_t_PDE_W)


pinn_mono = PINN_mono(100)
pinn_mono.load_model()

# define hyperparameters for NNs
n_hidden_layers = 3
neurons = 256
p = 2
seed = 123
for i in range(1):
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

    class Pinns:
        def __init__(self, n_int_, stage):
            self.n_int = n_int_  # number of drawed interior points
            self.batch_size = self.n_int
            self.stage = stage
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
                                                 [75.5, 150.],  # initial R
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
                                                  retrain_seed=seed).to(device)
    
            # Generator of Sobol sequences
            self.soboleng = torch.quasirandom.SobolEngine(dimension=19+1)
    
            # Training sets S_sb, S_tb, S_int as torch dataloader
            self.training_set_int = self.assemble_datasets()
    
        # load saved NN parameters if need for further training
        def load_model(self):
            modelname = '20241020_101213_3_256_third_stress_relaxation_model.pth'
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
    
        # Normalization for A according to the value of n
    
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
        
        def normalize_t_linear(self, tens):
            ub = torch.tensor([900.], device=device)
            lb = torch.tensor([0.], device=device)
            mid = (ub + lb) / 2
            return (tens - mid) / (ub - mid)
    
        def denormalize_t_linear(self, tens):
            ub = torch.tensor([900.], device=device)
            lb = torch.tensor([0.], device=device)
            mid = (ub + lb) / 2
            return tens * (ub - mid) + mid
    
        def normalize_initial(self, tens):
            return (tens - self.initial_extrema[:, 0]) / (self.initial_extrema[:, 1] - self.initial_extrema[:, 0])
    
        def denormalize_initial(self, tens):
            return tens * (self.initial_extrema[:, 1] - self.initial_extrema[:, 0]) + self.initial_extrema[:, 0]
    
        def denormalize_initial_X2(self, tens, sig, R, X1, X2, A, n):
            lb = torch.max((sig - X1 - R - (0.011 / A) ** (1 / n) - X2), torch.tensor([-20.], device=device))
            ub = torch.min((sig - X1 - R - X2), torch.tensor([20.], device=device))
            mid = (ub + lb) / 2
            return tens * (ub - mid) + mid
    
        def plot_learning_result(self, input_int, name, idx, strain_rate, end):
            global E_true, A_true, n_true, C1_true, gam1_true, kX1_true, C2_true, gam2_true, kX2_true, R0_true, Q1_true, b1_true, KR1_true
            E = E_true[idx]
            A = A_true[idx]
            n = n_true[idx]
            C1 = C1_true[idx]
            gam1 = gam1_true[idx]
            C2 = C2_true[idx]
            gam2 = gam2_true[idx]
            R0 = R0_true[idx]
            Q1 = Q1_true[idx]
            b1 = b1_true[idx]
            kX1 = kX1_true[idx]
            kX2 = kX2_true[idx]
            KR1 = KR1_true[idx]
    
            rate_name = str(strain_rate)
            # load data and true solution from .mat file (from data_1 to data_7 -- 7 sets of different material parameters)
            if end == 'end':
                if strain_rate == 0.01:
                    data = scipy.io.loadmat(current_dir + r'\data\data_beta_range2_benchmark_stress_relaxation_' + str(idx + 1) + '_rate_01_frommono.mat')
                elif strain_rate == 0.002:
                    data = scipy.io.loadmat(current_dir + r'\data\data_beta_range2_benchmark_stress_relaxation_' + str(idx + 1) + '_rate_002_frommono.mat')
                elif strain_rate == 0.0005:
                    data = scipy.io.loadmat(current_dir + r'\data\data_beta_range2_benchmark_stress_relaxation_' + str(idx + 1) + '_rate_0005_frommono.mat')
                elif strain_rate == 0.0001:
                    data = scipy.io.loadmat(current_dir + r'\data\data_beta_range2_benchmark_stress_relaxation_' + str(idx + 1) + '_rate_0001_frommono.mat')
            if end == 'middle':
                if strain_rate == 0.01:
                    data = scipy.io.loadmat(current_dir + r'\data\data_beta_range_02_benchmark_stress_relaxation_' + str(idx + 1) + '_rate_01_frommono.mat')
                elif strain_rate == 0.002:
                    data = scipy.io.loadmat(current_dir + r'\data\data_beta_range_02_benchmark_stress_relaxation_' + str(idx + 1) + '_rate_002_frommono.mat')
                elif strain_rate == 0.0005:
                    data = scipy.io.loadmat(current_dir + r'\data\data_beta_range_02_benchmark_stress_relaxation_' + str(idx + 1) + '_rate_0005_frommono.mat')
                elif strain_rate == 0.0001:
                    data = scipy.io.loadmat(current_dir + r'\data\data_beta_range_02_benchmark_stress_relaxation_' + str(idx + 1) + '_rate_0001_frommono.mat')

    
            sig_data = torch.from_numpy(np.reshape(data['sig'], (-1, 1))).to(device)  # stress
            t_data = torch.from_numpy(np.reshape(data['t'], (-1, 1))).to(device)  # total strain
            R_data = torch.from_numpy(np.reshape(data['R'], (-1, 1))).to(device)  # R
            X1_data = torch.from_numpy(np.reshape(data['X1'], (-1, 1))).to(device)  # X1
            X2_data = torch.from_numpy(np.reshape(data['X2'], (-1, 1))).to(device)  # X2
            eps_p_data = torch.from_numpy(np.reshape(data['eps_in'], (-1, 1))).to(
                device)  # viscoplastic strain # eps_tot_data - sig_data / E + el0 #
    
            x_test = input_int[:, :19].clone().to(device)
            x_test[:, 0] = torch.sort(x_test[:, 0], 0)[0].reshape(-1, )
    
            input_linear = torch.tensor([E, n, C1, gam1, C2, gam2, R0, Q1, b1], device=device)
            input_log = torch.tensor([strain_rate, A, kX1, kX2, KR1], dtype=torch.float64, device=device)
            input_initial = torch.tensor([sig_data[0], R_data[0], X1_data[0], X2_data[0]], dtype=torch.float64,
                                         device=device)
    
            x_test[:, 1:10] = self.normalize_input_linear(input_linear)
            x_test[:, 10:15] = self.normalize_input_log(input_log)
            ''''''
            x_test[:, 15:] = self.normalize_initial(input_initial)

            x_pre_ini = x_test[:, :15].clone().detach()
            x_pre_ini[:, 0] = 1.
            x_test.requires_grad = True
    
            t = self.denormalize_t(x_test[:, 0])

    
            # Calculation of the turning point of elastic domain
            # linear_constraint = (t.reshape(-1, ) / 450.)
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
    
            R_pred = R_pred + R_data[0]
            X1_pred = X1_pred + X1_data[0]
            X2_pred = X2_pred + X2_data[0]
    
            sig_pred = sig_data[0] - eps_p_pred * E
    
            X = X1_pred + X2_pred
    
            ub = torch.log10(torch.tensor([900. + 1e-3], device=device))
            lb = torch.log10(torch.tensor([1e-3], device=device))
            mid = (ub + lb) / 2
    
            eps_p_t = torch.autograd.grad(eps_p_pred.sum(), x_test, create_graph=True)[0][:, 0] * (2 / (ub - lb)) / (
                    (t + 1e-3) * np.log(10))
            R_t = torch.autograd.grad(R_pred.sum(), x_test, create_graph=True)[0][:, 0] * (2 / (ub - lb)) / (
                    (t + 1e-3) * np.log(10))
            X1_t = torch.autograd.grad(X1_pred.sum(), x_test, create_graph=True)[0][:, 0] * (2 / (ub - lb)) / (
                    (t + 1e-3) * np.log(10))
            X2_t = torch.autograd.grad(X2_pred.sum(), x_test, create_graph=True)[0][:, 0] * (2 / (ub - lb)) / (
                    (t + 1e-3) * np.log(10))
    
            eps_p_eq_t = abs(eps_p_t)
    
            weight_raw = (torch.tanh(
                10 * (t.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]) - 0.2)) + 1) * 2 + 1
            one_tensor = torch.tensor([1.], device=device)
            weight = torch.min(t.detach(), one_tensor) * torch.max(torch.log(t.detach()),
                                                                   one_tensor)  # torch.tensor([1.], device=device)  #
    
            a1 = sig_pred.reshape(-1, ) - X.reshape(-1, )
            a2 = a1 * torch.tanh(a1 / 0.001) - R_pred.reshape(-1, )
            a3 = A * (0.5 * (a2 + torch.sqrt(0.001 ** 2 + a2 ** 2))) ** n * torch.sign(
                a1)
            loss1 = (eps_p_t.reshape(-1, ) - a3) * weight  # * ((t + 1e-3) * np.log(10))
            mean1 = torch.mean(abs(loss1)).cpu().detach().numpy()
            loss2 = (R_t.reshape(-1, ) - Q1 * eps_p_eq_t + b1 * R_pred.reshape(-1, ) * eps_p_eq_t.reshape(
                -1, ) + KR1 * R_pred.reshape(-1, )) * weight
            mean2 = torch.mean(abs(loss2)).cpu().detach().numpy()
            loss3 = (X1_t.reshape(-1, ) - C1 * eps_p_t + gam1 * eps_p_eq_t * X1_pred.reshape(
                -1, ) + kX1 * X1_pred.reshape(-1, )) * weight
            mean3 = torch.mean(abs(loss3)).cpu().detach().numpy()
            loss4 = (X2_t.reshape(-1, ) - C2 * eps_p_t + gam2 * eps_p_eq_t * X2_pred.reshape(
                -1, ) + kX2 * X2_pred.reshape(-1, )) * weight
            mean4 = torch.mean(abs(loss4)).cpu().detach().numpy()

            '''
            t_plot = t
            t_data_plot = t_data
            '''
            t_plot = torch.log10(t + 1e-3)
            t_data_plot = torch.log10(t_data + 1e-3)

            fig = plt.figure(figsize=(16, 8))
    
            # plt.grid(True, which="both", ls=":")
            ax1 = plt.subplot(2, 3, 1)
            ax1.set_ylabel('stress (MPa)')
            ax1.set_xlabel('log10(time) (s)')
            ax1.plot(t_plot.cpu().detach().numpy(), sig_pred.cpu().detach().numpy(), ms=0.5, label="stress_predict")
            ax1.plot(t_data_plot.cpu().detach().numpy(), sig_data.cpu().detach().numpy(), ms=0.5, label="stress_true")
            ax1.legend(loc='upper right')
    
            ax2 = plt.subplot(2, 3, 4)
            ax2.set_ylabel('stress (MPa)')
            ax2.set_xlabel('time (s)')
            ax2.plot(t.cpu().detach().numpy(), sig_pred.cpu().detach().numpy(), ms=0.5, label="stress_predict")
            ax2.plot(t_data.cpu().detach().numpy(), sig_data.cpu().detach().numpy(), ms=0.5, label="stress_true")
            ax2.legend(loc='upper right')
    
            ax3 = plt.subplot(2, 3, 2)
            ax3.set_ylabel('eps_p')
            ax3.plot(t_plot.cpu().detach().numpy(), eps_p_pred.cpu().detach().numpy(), ms=0.5,
                     label="viscoplastic strain predict")
            ax3.plot(t_data_plot.cpu().detach().numpy(), eps_p_data.cpu().detach().numpy(), ms=0.5,
                     label="viscoplastic strain true")
            ax3.legend(loc='upper left')
            color = 'tab:red'
            ax12 = ax3.twinx()
            ax12.plot(t_plot.cpu().detach().numpy(), loss1.cpu().detach().numpy(), ms=0.5, label="Loss1", color=color)
            ax12.set_ylabel('loss1', color=color)
            ax12.legend(loc='upper right')
            ax12.tick_params(axis='y', labelcolor=color)
            plt.ylim((-10 * mean1, 10 * mean1))
    
            ax4 = plt.subplot(2, 3, 5, sharex=ax3)
            ax4.set_xlabel('log10(time) (s)')
            ax4.set_ylabel('R (MPa)')
            ax4.plot(t_plot.cpu().detach().numpy(), R_pred.cpu().detach().numpy(), ms=0.5, label="R_predict")
            ax4.plot(t_data_plot.cpu().detach().numpy(), R_data.cpu().detach().numpy(), ms=0.5, label="R_true")
            ax4.legend(loc='upper left')
            color = 'tab:red'
            ax42 = ax4.twinx()
            ax42.plot(t_plot.cpu().reshape(-1, ).detach().numpy(), loss2.cpu().reshape(-1, ).detach().numpy(), ms=0.5,
                      label="Loss2", color=color)
            ax42.set_ylabel('loss2', color=color)
            ax42.legend(loc='upper right')
            ax42.tick_params(axis='y', labelcolor=color)
            plt.ylim((-10 * mean2, 10 * mean2))
    
            ax5 = plt.subplot(2, 3, 3)
            ax5.set_ylabel('X1 (MPa)')
            ax5.plot(t_plot.cpu().detach().numpy(), X1_pred.cpu().detach().numpy(), ms=0.5, label="X1_predict")
            ax5.plot(t_data_plot.cpu().detach().numpy(), X1_data.cpu().detach().numpy(), ms=0.5, label="X1_true")
            ax5.legend(loc='upper left')
            color = 'tab:red'
            ax52 = ax5.twinx()
            ax52.plot(t_plot.cpu().detach().numpy(), loss3.cpu().detach().numpy(), ms=0.5, label="Loss3", color=color)
            ax52.set_ylabel('loss3', color=color)
            ax52.legend(loc='upper right')
            ax52.tick_params(axis='y', labelcolor=color)
            plt.ylim((-10 * mean3, 10 * mean3))
    
            ax6 = plt.subplot(2, 3, 6, sharex=ax5)
            ax6.set_xlabel('log10(time) (s)')
            ax6.set_ylabel('X2 (MPa)')
            ax6.plot(t_plot.cpu().detach().numpy(), X2_pred.cpu().detach().numpy(), ms=0.5, label="X2_predict")
            ax6.plot(t_data_plot.cpu().detach().numpy(), X2_data.cpu().detach().numpy(), ms=0.5, label="X2_true")
            ax6.legend(loc='upper left')
    
            color = 'tab:red'
            ax62 = ax6.twinx()
            ax62.plot(t_plot.cpu().detach().numpy(), loss4.cpu().detach().numpy(), ms=0.5, label="Loss4", color=color)
            ax62.set_ylabel('loss4', color=color)
            ax62.legend(loc='upper right')
            ax62.tick_params(axis='y', labelcolor=color)
            plt.ylim((-10 * mean4, 10 * mean4))
            fig.tight_layout()
    
            plotname = time_now + '_' + end + '_Benchmark_' + str(
                idx + 1) + '_rate_' + rate_name + '_NNstructure_' + str(
                n_hidden_layers) + '_' + str(neurons) + '.png'
            path = os.path.join(current_dir + r'\plot_fit', plotname)
            plt.savefig(path, dpi=100, bbox_inches='tight')
    
        # def plot_random_result(self, input_int):
        ################################################################################################
        #  Function returning the input-output tensor required to assemble the training set corresponding to the interior domain where the PDE is enforced
        def add_interior_points(self):
            global Par
            if self.stage == 1:
                a = 0.1
                b = 0.1
            elif self.stage == 2:
                a = 1
                b = 0.1
            else:
                a = 1
                b = 1
            input_int_pre = ((self.soboleng.draw(self.n_int).to(device)) * 2 - 1)
            temp_K = (input_int_pre[:, 19] + 1) * 510
            E, A, n, C1, gam1, kX1, C2, gam2, kX2, Q1, b1, KR1, R0 = cal_parameter_from_beta(Par, temp_K.double().reshape(-1, ))
            E = E + input_int_pre[:, 1] * 1.5e4 * a
            n = n + input_int_pre[:, 2] * 3 * a
            C1 = C1 + input_int_pre[:, 3] * 2.5e3 * a
            gam1 = gam1 + input_int_pre[:, 4] * 7 * a
            C2 = C2 + input_int_pre[:, 5] * 1.5e3 * a
            gam2 = gam2 + input_int_pre[:, 6] * 1.5e2 * a
            R0 = R0 + input_int_pre[:, 7] * 20 * a
            Q1 = Q1 + input_int_pre[:, 8] * 150 * a
            b1 = b1 + input_int_pre[:, 9] * 1.5 * a

            A = 10 ** (torch.log10(A) + input_int_pre[:, 11] * 5 * a)
            kX1 = 10 ** (torch.log10(kX1) + input_int_pre[:, 12] * 3 * a)
            kX2 = 10 ** (torch.log10(kX2) + input_int_pre[:, 13] * 1.5 * a)
            KR1 = 10 ** (torch.log10(KR1) + input_int_pre[:, 14] * 1.5 * a)
            input_linear = torch.cat((E.reshape(-1, 1), n.reshape(-1, 1),
                                      C1.reshape(-1, 1), gam1.reshape(-1, 1), C2.reshape(-1, 1), gam2.reshape(-1, 1),
                                      R0.reshape(-1, 1), Q1.reshape(-1, 1), b1.reshape(-1, 1)), 1)
            input_log = torch.cat([input_int_pre[:, 10].reshape(-1, 1), A.reshape(-1, 1), kX1.reshape(-1, 1),
                                   kX2.reshape(-1, 1), KR1.reshape(-1, 1)], 1)
            ones = torch.ones(1, device=device)
            input_int_mono = input_int_pre[:, :15].clone().detach()
            input_int_mono[:, 1:10] = self.normalize_input_linear(input_linear)
            input_int_mono[:, 11:] = self.normalize_input_log(input_log)[:, 1:]
            input_int_mono = torch.min(torch.max(input_int_mono.clone().detach(), -1 * ones), ones)

            linear_input = self.denormalize_input_linear(input_int_mono[:, 1:10])
            log_input = self.denormalize_input_log(input_int_mono[:, 10:15])
            n_ini = linear_input[:, 1]
            A_ini = log_input[:, 1]
            x_pre_ini = input_int_mono.clone().detach()
            x_pre_ini[:, 0] = 1.
            t, eps_tot, eps_p, sig_ini_mono, R_ini_mono, X1_ini_mono, X2_ini_mono = pinn_mono.prediction(x_pre_ini)

            sig_ini = torch.max(sig_ini_mono + 20. * input_int_pre[:, 15] * b, torch.tensor([0.], device=device))
            R_ini = torch.max(R_ini_mono + 10. * input_int_pre[:, 16] * b, torch.tensor([1.], device=device))
            X1_ini = torch.max(X1_ini_mono + 10. * input_int_pre[:, 17] * b, torch.tensor([0.], device=device))
            X2_ini_old = X2_ini_mono.clone().detach()
            X2_ini_plus = self.denormalize_initial_X2(input_int_pre[:, 18], sig_ini, R_ini, X1_ini, X2_ini_old,
                                                      A_ini, n_ini)
            X2_ini = torch.max(X2_ini_old + X2_ini_plus, torch.tensor([0.], device=device))

            input_initial = torch.cat(
                (sig_ini.reshape(-1, 1), R_ini.reshape(-1, 1), X1_ini.reshape(-1, 1), X2_ini.reshape(-1, 1)), 1)
            input_int = input_int_pre[:, :19].clone()
            input_int[:, :15] = input_int_mono  # .clone().detach()
            input_int[:, 15:] = self.normalize_initial(input_initial)  # .clone().detach()
            
            input_int_t = (input_int[:int(input_int.shape[0]/2), 0].clone() + 1.) / 90 - 1.
            input_int[:int(input_int.shape[0]/2), 0] = input_int_t.clone()

            output_int = torch.zeros((input_int.shape[0], 1)).to(device)
            return input_int, output_int
    
        # Function returning the training sets S_int as dataloader
        def assemble_datasets(self):
            input_int, output_int = self.add_interior_points()
            print('batch_size:', self.batch_size)
            training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.batch_size, shuffle=True)
            return training_set_int
    
    
        ################################################################################################
        def compute_pde_residual(self, x_train, verbose):
            # weight for each loss term
            global eps_p_t_PDE_W, R_t_PDE_W, X1_t_PDE_W, X2_t_PDE_W
    
            x_train_ = x_train.clone().detach()
            x_train_.requires_grad = True
            del x_train
            # Denormalization of inputs
            linear_input = self.denormalize_input_linear(x_train_[:, 1:10])
            log_input = self.denormalize_input_log(x_train_[:, 10:15])
            initial_input = self.denormalize_initial(x_train_[:, 15:19])
    
            t = self.denormalize_t(x_train_[:, 0])
            #t = self.denormalize_t_linear(x_train_[:, 0])
            E = linear_input[:, 0]
            n = linear_input[:, 1]
            A = log_input[:, 1]
            C1 = linear_input[:, 2]
            gam1 = linear_input[:, 3]
            kX1 = log_input[:, 2]
            C2 = linear_input[:, 4]
            gam2 = linear_input[:, 5]
            kX2 = log_input[:, 3]
            Q1 = linear_input[:, 7]
            b1 = linear_input[:, 8]
            KR1 = log_input[:, 4]
            eps_tot_rate = log_input[:, 0]
            sig_ini = initial_input[:, 0]
            R_ini = initial_input[:, 1]
            X1_ini = initial_input[:, 2]
            X2_ini = initial_input[:, 3]
            del linear_input, log_input, initial_input
    
            # Calculation of the turning point of elastic domain
            # linear_constraint = (t.reshape(-1, ) / 450.)
            linear_constraint = (x_train_[:, 0] + 1.)
            hard_constraint_linear = torch.cat(
                [linear_constraint.reshape(-1, 1), linear_constraint.reshape(-1, 1), linear_constraint.reshape(-1, 1),
                 linear_constraint.reshape(-1, 1)], 1)
    
            # Calculation of outputs and denormalization
            y_pred = self.approximate_solution(x_train_) * hard_constraint_linear
            y_pred = self.denormalize_output(y_pred)
            # add hard constraints
            eps_p = y_pred[:, 0].to(torch.float64)
            R = y_pred[:, 1].to(torch.float64)  # * hard_constraint_other
            X1 = y_pred[:, 2].to(torch.float64)  # * hard_constraint_other
            X2 = y_pred[:, 3].to(torch.float64)  # * hard_constraint_other
            del y_pred
    
            X1 = X1 + X1_ini
            X2 = X2 + X2_ini
            R = R + R_ini
            X = X1 + X2
            sig = sig_ini - eps_p * E
            
    
            ub = torch.log10(torch.tensor([900. + 1e-3], device=device))
            lb = torch.log10(torch.tensor([1e-3], device=device))
            mid = (ub + lb) / 2
    
            eps_p_t = torch.autograd.grad(eps_p.sum(), x_train_, create_graph=True)[0][:, 0] * (2 / (ub - lb)) / ((t + 1e-3) * np.log(10))
            R_t = torch.autograd.grad(R.sum(), x_train_, create_graph=True)[0][:, 0] * (2 / (ub - lb)) / ((t + 1e-3) * np.log(10))
            X1_t = torch.autograd.grad(X1.sum(), x_train_, create_graph=True)[0][:, 0] * (2 / (ub - lb)) / ((t + 1e-3) * np.log(10))
            X2_t = torch.autograd.grad(X2.sum(), x_train_, create_graph=True)[0][:, 0] * (2 / (ub - lb)) / ((t + 1e-3) * np.log(10))
            # del x_train_
            eps_p_eq_t = abs(eps_p_t)
    
            # to put higher weight at the end
           
            one_tensor = torch.tensor([1.], device=device)
            weight = one_tensor #torch.min(t.clone().detach() + 1e-2, one_tensor) * torch.max(torch.log(t.clone().detach()), one_tensor) #* torch.sqrt(0.01 / eps_tot_rate.clone().detach()) 
            weight_loss2 = weight.clone().detach()
            weight_loss3 = weight.clone().detach()
            weight_loss4 = weight.clone().detach()
    
            # Loss 1
            Smooth_factor = 0.0001
            a1 = sig.reshape(-1, ) - X.reshape(-1, )
            a2 = a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(-1, )
            loss1_right = (A * (0.5 * (a2 + torch.sqrt(a2 ** 2 + Smooth_factor ** 2))) ** n) * torch.tanh(
                a1 / Smooth_factor)
            loss1_left = eps_p_t.reshape(-1, )
    
            loss0_vec = torch.max(eps_p_t, torch.tensor([0.], device=device))
            loss0 = 1000 * torch.mean(loss0_vec)
            loss0_rec = loss0.detach()
    

            loss1_abs_vec = (loss1_left - loss1_right) / (eps_p_t_PDE_W)  # * ((t + 1e-3) * np.log(10))
            loss1_per_vec = (loss1_left - loss1_right) / (abs(loss1_left) + 1e-4)
            loss1_abs_w = loss1_abs_vec * weight.reshape(-1, )
            weight_per = weight * torch.max(torch.log10(t.detach()), torch.tensor([0.], device=device))
            loss1_per_w = loss1_per_vec * weight_per  # / 10
           
    
            loss1_abs = torch.mean(loss1_abs_w ** p)  # [400:]
            loss1_per = torch.mean(loss1_per_w ** p)
    
            loss1 = loss1_abs #loss1_per  #+ 
            loss1_rec = loss1.detach()
    
            # loss 2
            loss2_left = R_t.reshape(-1, )
            loss2_right = Q1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) - b1.reshape(-1, ) * R.reshape(
                -1, ) * eps_p_eq_t.reshape(-1, ) - KR1.reshape(-1, ) * R.reshape(-1, )
    
            loss2_abs_vec = (loss2_left - loss2_right) / (R_t_PDE_W)  # * ((torch.mean(R0)/R0) * ((t + 1e-3) * np.log(10))
            loss2_abs_w = loss2_abs_vec * weight_loss2.reshape(-1, )
            loss2_abs = torch.mean(loss2_abs_w ** p)  # [400:]
    
            loss2_per_vec = (loss2_left - loss2_right) / (abs(loss2_left) + 0.1)
            loss2_per_w = loss2_per_vec * weight_per  # / 10
            loss2_per = torch.mean(loss2_per_w ** p)
    
            loss2 = loss2_abs  #loss2_per  # + 
            loss2_rec = loss2.detach()
           
            # loss 3
            loss3_left = X1_t.reshape(-1, )
            loss3_right = C1.reshape(-1, ) * eps_p_t.reshape(-1, ) - gam1.reshape(-1, ) * eps_p_eq_t.reshape(
                -1, ) * X1.reshape(-1, ) - kX1.reshape(-1, ) * X1.reshape(-1, )
    
            loss3_abs_vec = (loss3_left - loss3_right) / (X1_t_PDE_W)  # * (torch.mean(E)/E)
            loss3_abs_w = loss3_abs_vec * weight_loss3.reshape(-1, )
            loss3_abs = torch.mean(loss3_abs_w ** p)  # [400:]
    
            loss3_per_vec = (loss3_left - loss3_right) / (abs(loss3_left) + 50.)
            loss3_per_w = loss3_per_vec * weight_per  # / 1000
            loss3_per = torch.mean(loss3_per_w ** p)
    
            loss3 = loss3_abs  #loss3_per  # + 
            loss3_rec = loss3.detach()
    
           
    
            # loss 4
            loss4_left = X2_t.reshape(-1, )
            loss4_right = C2.reshape(-1, ) * eps_p_t.reshape(-1, ) - gam2.reshape(
                -1, ) * eps_p_eq_t.reshape(-1, ) * X2.reshape(-1, ) - kX2.reshape(-1, ) * X2.reshape(-1, )
    
            loss4_abs_vec = (loss4_left - loss4_right) / (X2_t_PDE_W)  # * (torch.mean(E)/E)
            loss4_abs_w = loss4_abs_vec * weight_loss4.reshape(-1, )
            loss4_abs = torch.mean(loss4_abs_w ** p)  # [400:]
    
            loss4_per_vec = (loss4_left - loss4_right) / (abs(loss4_left) + 10.)
            loss4_per_w = loss4_per_vec * weight_per  # / 100
            loss4_per = torch.mean(loss4_per_w ** p)
    
            loss4 = loss4_abs  #loss4_per  # + 
            loss4_rec = loss4.detach()
           
            # print the loss terms
            if verbose:
                print('loss1:', loss1_rec.cpu().detach().item(), 'loss2:', loss2_rec.cpu().detach().item(), 'loss3:',
                      loss3_rec.cpu().detach().item(), 'loss4:', loss4_rec.cpu().detach().item(), 'loss0:',
                      loss0_rec.cpu().detach().item())
    
            loss_R = 100000 * torch.mean(torch.max(-R.reshape(-1, ), torch.zeros([R.reshape(-1, ).shape[0]]).to(device)) ** 2)
    
    
            loss = torch.log10(loss1 + loss2 + loss3 + loss4)  #  + loss_R
            loss_rec = loss.detach()
    
            # print total loss
            if verbose:
                print('Loss: ', loss_rec.item())
            return loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec
    
        # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
        def compute_loss(self, inp_train_int, verbose):
    
            loss_pde, loss_pde_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_pde_residual(inp_train_int, verbose)
    
            loss = loss_pde
            return loss, loss_pde_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec
    
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
                        loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_loss(inp_train_int, verbose=verbose)
                        loss.backward()
                        loss_1.append(loss1_rec.item())
                        loss_2.append(loss2_rec.item())
                        loss_3.append(loss3_rec.item())
                        loss_4.append(loss4_rec.item())
                        loss_0.append(loss0_rec.item())
                        history.append(loss_rec.item())
                        # print('Loss_pde: ', history[-1])
    
                        return loss
    
                    optimizer.step(closure=closure)
    
            print('Final Loss: ', history[-1], 'loss1:', loss_1[-1], 'loss2:', loss_2[-1], 'loss3:', loss_3[-1], 'loss4:',
                  loss_4[-1], 'loss0:', loss_0[-1])
    
            return history, loss_1, loss_2, loss_3, loss_4
    
    
    n_int = 300000
    # n_int = 100
    print('number of collocation points:', n_int)
    start_time = time.time()
    for q in range(3):
        pinn = Pinns(n_int, 1 + q)
        # load trained NN for further training
        if q == 0:
            flag_train = 'first'
        elif q == 1:
            flag_train = 'second'
        else:
            flag_train = 'third'
        print(flag_train)
        if q != 0:
            pinn.approximate_solution = save_model

        # pinn.load_model()
        n_epochs = 1
        optimizer_ = optim.LBFGS(list(pinn.approximate_solution.parameters()), lr=float(0.5),
                                 max_iter=1000000,
                                 max_eval=1000000,
                                 tolerance_grad=1e-12,
                                 tolerance_change=1e-12,
                                 line_search_fn='strong_wolfe')

        

        # what about loss0
        history1, loss_11, loss_21, loss_31, loss_41 = pinn.fit(num_epochs=n_epochs, optimizer=optimizer_, verbose=False)
        n_epochs1 = len(history1)

        

        # save the trained model
        modelname = time_now + '_' + str(n_hidden_layers) + '_' + str(neurons) + '_' + flag_train + '_stress_relaxation_model.pth'
        modelname = time_now + '_' + str(n_hidden_layers) + '_' + str(neurons) + '_' + flag_train + '_stress_relaxation_model_linear_t.pth'
        # path = os.path.join(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\model_save', modelname)  # path for saving the model
        path = os.path.join(current_dir + r'\model_save', modelname)
        torch.save(pinn.approximate_solution.cpu().state_dict(), path)
        pinn.approximate_solution = pinn.approximate_solution.to(device)
        save_model = pinn.approximate_solution

    
    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))
    '''
    input_int = ((pinn.soboleng.draw(10000).to(device)) * 2 - 1)
    for i in range(0, 8):
        for strain_rate in [0.01, 0.002, 0.0005, 0.0001]:#[0.002, 0.0005]:#[0.005, 0.001, 0.0002]:#
            pinn.plot_learning_result(input_int, name='first', idx=i, strain_rate=strain_rate, end='middle')
    for i in range(0, 8):
        for strain_rate in [0.01, 0.002, 0.0005, 0.0001]:#[0.002, 0.0005]:#[0.005, 0.001, 0.0002]:#
            pinn.plot_learning_result(input_int, name='first', idx=i, strain_rate=strain_rate, end='end')
    '''
    # plt.show(block=True)  # use this only when you run the code locally to show the plots on your screen
