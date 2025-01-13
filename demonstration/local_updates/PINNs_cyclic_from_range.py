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

from PINNs_monotonic_class_newrange import Pinns_monotonic as PINN_mono

pinn_mono = PINN_mono(100)
pinn_mono.load_model()

from PINNs_stress_relaxation_class import Pinns_stress as PINN_stress

pinn_stress = PINN_stress(100)
pinn_stress.load_model()

mpl.use('Qt5Agg')  # activate this only if you run the code locally and want to see the plot on your screen
torch.autograd.set_detect_anomaly(False)

# Set random seed for reproducibility
seed = 128
torch.manual_seed(seed)
np.random.seed(seed)

# Use date+time for naming
time_now = time.localtime()
time_now = time.strftime('%Y%m%d_%H%M%S', time_now)
print(time_now)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
if device == 'cuda':
    print(torch.cuda.get_device_name())

# Benchmarks = pd.read_excel(r'/cluster/home/haotxu/PINNs/data/New_benchmarks.xlsx', header=None)  # dtype=np.float64,
Benchmarks = pd.read_excel(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\New_benchmarks.xlsx', header=None)  # dtype=np.float64,
Benchmarks = Benchmarks.iloc[:, 1:].astype('float64')
Benchmarks = torch.tensor(Benchmarks.values, dtype=torch.float64)

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

# weight of four PDE loss terms (hyperparameters)
R_t_PDE_W = 0.005
X1_t_PDE_W = 0.5
X2_t_PDE_W = 0.3
eps_p_t_PDE_W = 1e-4

# define hyperparameters for NNs
n_hidden_layers = 6
neurons = 128

print('hidden layers:', n_hidden_layers)
print('neurons:', neurons)
p = 2


class Pinns:
    def __init__(self, n_int_):
        self.n_int = n_int_  # number of drawed interior points

        self.domain_extrema = torch.tensor([[0.0075, 0.015],  # total strain dimension []     [mean upperBound]
                                            [1.4e5, 1.9e5],  # E [MPa]
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

        self.initial_extrema = torch.tensor([[-400., 0.],  # initial stress
                                             [76., 150.],  # initial R
                                             [-100., 0.],  # initial X1
                                             [-200., 0.],  # initial X2
                                             ]).to(device)

        # Extrema of the result domain
        self.result_extrema = torch.tensor([[0., 5e-3],  # Delta eps_p [-] allows negative plastic strains
                                            [0., 5],  # Delta R [MPa]
                                            [0., 300.],  # Delta X1 [MPa]
                                            [0., 300.]  # Delta X2 [MPa]
                                            ]).to(device)

        # need to include A and n also here to the list?

        # F Dense NN to approximate the solution
        self.approximate_solution = NeuralNet(input_dimension=19, output_dimension=4,
                                              n_hidden_layers=n_hidden_layers,
                                              neurons=neurons,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=128).to(device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=19)

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_int = self.assemble_datasets()

    # load saved NN parameters if need for further training
    def load_model(self):
        modelname = '20240516_190837_PDE_loss_learning_regular_6_128_first_X0_model.pth'  #before transfer
        # modelname = '20240517_103835_PDE_loss_learning_regular_6_128_final_X0_model.pth'  # after transfer

        path = os.path.join(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\model_save', modelname)
        # path = os.path.join(r'/cluster/home/haotxu/PINNs/model_save', modelname)
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

    # Normalization for X0 terms
    def normalize_initial(self, tens):
        return (tens - self.initial_extrema[:, 0]) / (self.initial_extrema[:, 1] - self.initial_extrema[:, 0])

    def denormalize_initial(self, tens):
        return tens * (self.initial_extrema[:, 1] - self.initial_extrema[:, 0]) + self.initial_extrema[:, 0]


    def denormalize_initial_X2(self, tens, sig, R, X1, X2, A, n):
        lb = torch.max((sig - X1 - R - (0.011 / A) ** (1 / n) - X2), torch.tensor([-40.], device=device))
        ub = torch.min((sig - X1 - R - X2), torch.tensor([40.], device=device))
        mid = (ub + lb) / 2
        return tens * (ub - mid) + mid

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

    def plot_learning_result(self, input_int, name, idx, strain_rate, from_flag):
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
        rate_name = str(strain_rate)
        # load data and true solution from .mat file (from data_1 to data_7 -- 7 sets of different material parameters)
        if from_flag == 1:
            if strain_rate == 0.005:
                data = scipy.io.loadmat(
                    r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_new_benchmark_cyclic_' + str(idx + 1) + '_rate_005_frommono.mat')
            elif strain_rate == 0.001:
                data = scipy.io.loadmat(
                    r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_new_benchmark_cyclic_' + str(idx + 1) + '_rate_001_frommono.mat')
            elif strain_rate == 0.0002:
                data = scipy.io.loadmat(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_new_benchmark_cyclic_' + str(
                    idx + 1) + '_rate_0002_frommono.mat')
        if from_flag == -1:
            if strain_rate == 0.005:
                data = scipy.io.loadmat(
                    r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_new_benchmark_cyclic_' + str(idx + 1) + '_rate_005_fromstress.mat')
            elif strain_rate == 0.001:
                data = scipy.io.loadmat(
                    r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_new_benchmark_cyclic_' + str(idx + 1) + '_rate_001_fromstress.mat')
            elif strain_rate == 0.0002:
                data = scipy.io.loadmat(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_new_benchmark_cyclic_' + str(
                    idx + 1) + '_rate_0002_fromstress.mat')

        sig_data = torch.from_numpy(np.reshape(data['sig'], (-1, 1))).to(device)  # stress
        eps_tot_data = torch.from_numpy(np.reshape(data['eps_tot'], (-1, 1))).to(device)  # total strain
        R_data = torch.from_numpy(np.reshape(data['R'], (-1, 1))).to(device)  # R
        X1_data = torch.from_numpy(np.reshape(data['X1'], (-1, 1))).to(device)  # X1
        X2_data = torch.from_numpy(np.reshape(data['X2'], (-1, 1))).to(device)  # X2
        eps_p_data = torch.from_numpy(np.reshape(data['eps_in'], (-1, 1))).to(
            device)  # viscoplastic strain # eps_tot_data - sig_data / E + el0 #



        x_test = input_int.clone().to(device)
        x_test[:, 0] = torch.sort(x_test[:, 0], 0)[0].reshape(-1, )

        input_linear = torch.tensor([0, E, n, C1, gam1, C2, gam2, R0, Q1, b1], device=device)
        input_log = torch.tensor([strain_rate, kX1, kX2, KR1], dtype=torch.float64, device=device)
        input_initial = torch.tensor([sig_data[0], R_data[0], X1_data[0], X2_data[0]], dtype=torch.float64, device=device)

        x_test[:, 1:10] = self.normalize_input_linear(input_linear)[1:10]
        x_test[:, 11:15] = self.normalize_input_log(input_log)
        x_test[:, 12] = self.normalize_A_kX1(torch.tensor([kX1], dtype=torch.float64, device=device),
                                             torch.tensor([A], dtype=torch.float64, device=device))
        x_test[:, 13] = self.normalize_A_kX2(torch.tensor([kX2], dtype=torch.float64, device=device),
                                             torch.tensor([A], dtype=torch.float64, device=device))
        x_test[:, 14] = self.normalize_A_KR(torch.tensor([KR1], dtype=torch.float64, device=device),
                                            torch.tensor([A], dtype=torch.float64, device=device))
        x_test[:, 10] = self.normalize_A_n(torch.tensor([A], dtype=torch.float64, device=device), n)
        x_test[:, 15:] = self.normalize_initial(input_initial)


        x_test.requires_grad = True

        eps_tot = self.denormalize_input_linear(x_test[:, :10])[:, 0]
        eps_tot_rate = strain_rate

        # Calculation of the turning point of elastic domain
        linear_constraint = (eps_tot.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        hard_constraint_linear = torch.cat(
            [linear_constraint.reshape(-1, 1), linear_constraint.reshape(-1, 1), linear_constraint.reshape(-1, 1),
             linear_constraint.reshape(-1, 1)], 1)

        y_pred = self.approximate_solution(x_test) * hard_constraint_linear

        y_pred = self.denormalize_output(y_pred)

        eps_p_pred = y_pred[:, 0].to(torch.float64)
        R_pred = y_pred[:, 1].to(torch.float64)
        X1_pred = y_pred[:, 2].to(torch.float64)
        X2_pred = y_pred[:, 3].to(torch.float64)

        eps_tot_plot = eps_tot

        R_pred = R_pred + R_data[0]
        X1_pred = X1_pred + X1_data[0]
        X2_pred = X2_pred + X2_data[0]

        sig_pred = sig_data[0] + (eps_tot - eps_p_pred) * E

        X = X1_pred + X2_pred

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
        ax1.set_xlabel('total strain')
        ax1.plot(eps_tot_plot.cpu().detach().numpy(), sig_pred.cpu().detach().numpy(), ms=0.5, label="stress_predict")
        ax1.plot(eps_tot_data.cpu().detach().numpy(), sig_data.cpu().detach().numpy(), ms=0.5, label="stress_true")
        ax1.legend(loc='upper left')

        ax3 = plt.subplot(2, 3, 2)
        ax3.set_ylabel('eps_p')
        ax3.plot(eps_tot_plot.cpu().detach().numpy(), eps_p_pred.cpu().detach().numpy(), ms=0.5,
                 label="viscoplastic strain predict")
        ax3.plot(eps_tot_data.cpu().detach().numpy(), eps_p_data.cpu().detach().numpy(), ms=0.5,
                 label="viscoplastic strain true")
        ax3.legend(loc='upper left')
        color = 'tab:red'
        ax12 = ax3.twinx()
        ax12.plot(eps_tot_plot.cpu().detach().numpy(), loss1.cpu().detach().numpy(), ms=0.5, label="Loss1", color=color)
        ax12.set_ylabel('loss1', color=color)
        ax12.legend(loc='upper right')
        ax12.tick_params(axis='y', labelcolor=color)
        plt.ylim((-10 * mean1, 10 * mean1))

        ax4 = plt.subplot(2, 3, 5, sharex=ax3)
        ax4.set_xlabel('total strain')
        ax4.set_ylabel('R (MPa)')
        ax4.plot(eps_tot_plot.cpu().detach().numpy(), R_pred.cpu().detach().numpy(), ms=0.5, label="R_predict")
        ax4.plot(eps_tot_data.cpu().detach().numpy(), R_data.cpu().detach().numpy(), ms=0.5, label="R_true")
        ax4.legend(loc='upper left')
        color = 'tab:red'
        ax42 = ax4.twinx()
        ax42.plot(eps_tot_plot.cpu().reshape(-1, ).detach().numpy(), loss2.cpu().reshape(-1, ).detach().numpy(), ms=0.5,
                  label="Loss2", color=color)
        ax42.set_ylabel('loss2', color=color)
        ax42.legend(loc='upper right')
        ax42.tick_params(axis='y', labelcolor=color)
        plt.ylim((-10 * mean2, 10 * mean2))

        ax5 = plt.subplot(2, 3, 3)
        ax5.set_ylabel('X1 (MPa)')
        ax5.plot(eps_tot_plot.cpu().detach().numpy(), X1_pred.cpu().detach().numpy(), ms=0.5, label="X1_predict")
        ax5.plot(eps_tot_data.cpu().detach().numpy(), X1_data.cpu().detach().numpy(), ms=0.5, label="X1_true")
        ax5.legend(loc='upper left')
        color = 'tab:red'
        ax52 = ax5.twinx()
        ax52.plot(eps_tot_plot.cpu().detach().numpy(), loss3.cpu().detach().numpy(), ms=0.5, label="Loss3", color=color)
        ax52.set_ylabel('loss3', color=color)
        ax52.legend(loc='upper right')
        ax52.tick_params(axis='y', labelcolor=color)
        plt.ylim((-10 * mean3, 10 * mean3))

        ax6 = plt.subplot(2, 3, 6, sharex=ax5)
        ax6.set_xlabel('total strain')
        ax6.set_ylabel('X2 (MPa)')
        ax6.plot(eps_tot_plot.cpu().detach().numpy(), X2_pred.cpu().detach().numpy(), ms=0.5, label="X2_predict")
        ax6.plot(eps_tot_data.cpu().detach().numpy(), X2_data.cpu().detach().numpy(), ms=0.5, label="X2_true")
        ax6.legend(loc='upper left')

        color = 'tab:red'
        ax62 = ax6.twinx()
        ax62.plot(eps_tot_plot.cpu().detach().numpy(), loss4.cpu().detach().numpy(), ms=0.5, label="Loss4", color=color)
        ax62.set_ylabel('loss4', color=color)
        ax62.legend(loc='upper right')
        ax62.tick_params(axis='y', labelcolor=color)
        plt.ylim((-10 * mean4, 10 * mean4))
        fig.tight_layout()

        ''''''
        plotname = time_now + '_Benchmark_' + str(idx + 1) + '_rate_' + rate_name + '_NNstructure_' + str(
            n_hidden_layers) + '_' + str(neurons) + '_flag_' + str(from_flag) + '.png'
        path = os.path.join(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\plot_fit', plotname)
        # path = os.path.join(r'/cluster/home/haotxu/PINNs/plot_fit', plotname)
        plt.savefig(path, dpi=300, bbox_inches='tight')

        return sig_pred, sig_data, eps_tot_plot, eps_tot_data

    ################################################################################################
    #  Function returning the input-output tensor required to assemble the training set corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int_pre = (self.soboleng.draw(self.n_int).to(device)) * 2 - 1

        linear_input = self.denormalize_input_linear(input_int_pre[:, :10])
        n = linear_input[:, 2].reshape(-1, )
        A = self.denormalize_A_n(input_int_pre[:, 10].to(torch.float64), n).reshape(-1, )

        x_pre_ini1 = input_int_pre[:int(self.n_int/2), :15]
        x_pre_ini2 = input_int_pre[int(self.n_int/2):, :15]
        sig_ini_mono, R_ini_mono, X1_ini_mono, X2_ini_mono = pinn_mono.prediction_end(x_pre_ini1)
        sig_ini_stress, R_ini_stress, X1_ini_stress, X2_ini_stress = pinn_stress.prediction_end(x_pre_ini2)

        sig_ini = torch.max(torch.cat((sig_ini_mono, sig_ini_stress)) + 40. * input_int_pre[:, 15], torch.tensor([0.], device=device))
        R_ini = torch.max(torch.cat((R_ini_mono, R_ini_stress)) + 20. * input_int_pre[:, 16], torch.tensor([2.], device=device))
        X1_ini = torch.max(torch.cat((X1_ini_mono, X1_ini_stress)) + 20. * input_int_pre[:, 17], torch.tensor([0.], device=device))
        X2_ini = torch.cat((X2_ini_mono, X2_ini_stress))
        X2_ini_plus = self.denormalize_initial_X2(input_int_pre[:, 17], sig_ini, R_ini, X1_ini, X2_ini, A, n)
        X2_ini = torch.max(X2_ini + X2_ini_plus, torch.tensor([0.], device=device))

        input_initial = torch.cat((-sig_ini.reshape(-1, 1), R_ini.reshape(-1, 1), -X1_ini.reshape(-1, 1), -X2_ini.reshape(-1, 1)), 1)
        input_int_pre[:, 15:] = self.normalize_initial(input_initial)

        input_int = input_int_pre.clone().detach()
        output_int = torch.zeros((input_int.shape[0], 1)).to(device)
        return input_int, output_int

    # Function returning the training sets S_int as dataloader
    def assemble_datasets(self):
        input_int, output_int = self.add_interior_points()
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int,
                                      shuffle=False)
        return training_set_int

    ################################################################################################
    def compute_pde_residual(self, x_train, verbose):
        # weight for each loss term
        global eps_p_t_PDE_W, R_t_PDE_W, X1_t_PDE_W, X2_t_PDE_W

        x_train_ = x_train.clone()
        x_train_.requires_grad = True
        del x_train
        # Denormalization of inputs
        linear_input = self.denormalize_input_linear(x_train_[:, :10])
        log_input = self.denormalize_input_log(x_train_[:, 11:15])
        initial_input = self.denormalize_initial(x_train_[:, 15:19])

        eps_tot = linear_input[:, 0]
        E = linear_input[:, 1]
        n = linear_input[:, 2]
        A = self.denormalize_A_n(x_train_[:, 10].to(torch.float64), n).reshape(-1, )
        C1 = linear_input[:, 3]
        gam1 = linear_input[:, 4]
        kX1 = self.denormalize_A_kX1(x_train_[:, 12].to(torch.float64), A).reshape(-1, )  # log_input[:, 1]
        C2 = linear_input[:, 5]
        gam2 = linear_input[:, 6]
        kX2 = self.denormalize_A_kX2(x_train_[:, 13].to(torch.float64), A).reshape(-1, )  # log_input[:, 2]
        Q1 = linear_input[:, 8]
        b1 = linear_input[:, 9]
        KR1 = self.denormalize_A_KR(x_train_[:, 14].to(torch.float64), A).reshape(-1, )  # log_input[:, 3]

        eps_tot_rate = log_input[:, 0]

        sig_ini = initial_input[:, 0]
        R_ini = initial_input[:, 1]
        X1_ini = initial_input[:, 2]
        X2_ini = initial_input[:, 3]
        del linear_input, log_input, initial_input



        # Calculation of the turning point of elastic domain
        linear_constraint = (eps_tot.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))

        hard_constraint_other = linear_constraint  # * (torch.tanh(2 * INP) ** 2 + 0.001)
        hard_constraint_linear = torch.cat(
            [linear_constraint.reshape(-1, 1), linear_constraint.reshape(-1, 1), linear_constraint.reshape(-1, 1),
             linear_constraint.reshape(-1, 1)], 1)

        # Calculation of outputs and denormalization
        y_pred = self.approximate_solution(x_train_) * hard_constraint_linear

        y_pred = self.denormalize_output(y_pred)
        # add hard constraints
        eps_p = y_pred[:, 0].to(torch.float64)
        # me = torch.mean(abs(eps_p))
        R = y_pred[:, 1].to(torch.float64)
        X1 = y_pred[:, 2].to(torch.float64)
        X2 = y_pred[:, 3].to(torch.float64)
        del y_pred

        X1 = X1 + X1_ini
        X2 = X2 + X2_ini
        R = R + R_ini

        sig = sig_ini + (eps_tot - eps_p) * E
        X = X1 + X2

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

        # Loss 0 is the penulty of pure elastic case
        loss0_vec = eps_tot.reshape(-1, ) - eps_p.reshape(-1, ) - 0.002
        loss0_vec = 100000000 * 0.5 * (loss0_vec + torch.sqrt(loss0_vec ** 2))
        loss0 = torch.mean(loss0_vec ** p)
        loss0_rec = loss0.detach()

        # to put higher weight for small strains
        weight = 1. / (0.5 + (0.5 * eps_tot.reshape(-1, )) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        # weight = 0. / (0.5 + (0.5 * eps_tot.reshape(-1, )) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0])) + 1.
        weight_raw = (-torch.tanh(
            10 * (eps_tot.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]) - 0.6)) + 1) * 3 + 1

        # Loss 1
        Smooth_factor = 0.0001
        a1 = sig.reshape(-1, ) - X.reshape(-1, )
        a2 = a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(
            -1, )  # a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(-1, )   # what is the range of overstress
        a3 = (A * (0.5 * (a2 + torch.sqrt(a2 ** 2 + Smooth_factor ** 2))) ** n) * torch.tanh(a1 / Smooth_factor)
        a4 = (A * (0.5 * (a2 + torch.sqrt(a2 ** 2))) ** n)

        # flag = (torch.tanh((a4 - 1e-5)/1e-5) + 1) / 2
        # flag = torch.tensor([1.], device=device)
        flag = (torch.tanh(a2) + 1) / 2

        loss1_vec = (eps_p_t.reshape(-1, ) - flag * a3) / (eps_p_t_PDE_W)
        loss1_w = loss1_vec * weight.reshape(-1, )
        # loss1_vec = (eps_p_eq_t.reshape(-1, ) / A) ** (1/n) - 0.5 * (a2 + torch.sqrt(a2 ** 2 + Smooth_factor ** 2))

        # to damp the effect of big loss values at the begining of training
        loss1 = torch.mean(loss1_w ** p)  # [400:]
        # loss1 = torch.log10(1 + torch.mean(loss1_vec ** p))
        loss1_rec = loss1.detach()

        # loss 2
        # loss2_vec = (R_t.reshape(-1, ) - Q1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) + b1.reshape(-1, ) * R.reshape(
        #     -1, ) * eps_p_eq_t.reshape(-1, ) + KR1.reshape(-1, ) * R.reshape(-1, )) / R_t_PDE_W
        loss2_vec = (R_t.reshape(-1, ) - flag * (
                    Q1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) - b1.reshape(-1, ) * R.reshape(
                -1, ) * eps_p_eq_t.reshape(-1, )) + KR1.reshape(-1, ) * R.reshape(-1, )) / (
                        R_t_PDE_W)  # * ((torch.mean(R0)/R0)
        loss2_w = loss2_vec * weight.reshape(-1, )
        loss2 = torch.mean(loss2_w ** p)  # [400:]
        loss2_rec = loss2.detach()

        # loss 3
        loss3_vec = (X1_t.reshape(-1, ) - flag * (C1.reshape(-1, ) * eps_p_t.reshape(-1, ) - gam1.reshape(
            -1, ) * eps_p_eq_t.reshape(-1, ) * X1.reshape(-1, )) + kX1.reshape(-1, ) * X1.reshape(-1, )) / (
                        X1_t_PDE_W)  # * (torch.mean(E)/E)
        # loss3_vec = (X1_t.reshape(-1, ) - C1.reshape(-1, ) * eps_p_t.reshape(-1, ) + gam1.reshape(
        #     -1, ) * eps_p_eq_t.reshape(-1, ) * X1.reshape(-1, ) + kX1.reshape(-1, ) * X1.reshape(-1, )) / X1_t_PDE_W
        loss3_w = loss3_vec * weight.reshape(-1, )
        # a = loss3_w ** p
        loss3 = torch.mean(loss3_w ** p)  # [400:]
        loss3_rec = loss3.detach()

        # loss 4
        loss4_vec = (X2_t.reshape(-1, ) - flag * (C2.reshape(-1, ) * eps_p_t.reshape(-1, ) - gam2.reshape(
            -1, ) * eps_p_eq_t.reshape(-1, ) * X2.reshape(-1, )) + kX2.reshape(-1, ) * X2.reshape(-1, )) / (
                        X2_t_PDE_W)  # * (torch.mean(E)/E)
        # loss4_vec = (X2_t.reshape(-1, ) - C2.reshape(-1, ) * eps_p_t.reshape(-1, ) + gam2.reshape(
        #     -1, ) * eps_p_eq_t.reshape(-1, ) * X2.reshape(-1, ) + kX2.reshape(-1, ) * X2.reshape(-1, )) / X2_t_PDE_W
        loss4_w = loss4_vec * weight.reshape(-1, )
        loss4 = torch.mean(loss4_w ** p)  # [400:]
        loss4_rec = loss4.detach()

        # print the loss terms
        if verbose:
            print('loss1:', loss1_rec.cpu().detach().item(), 'loss2:', loss2_rec.cpu().detach().item(), 'loss3:',
                  loss3_rec.cpu().detach().item(), 'loss4:', loss4_rec.cpu().detach().item(), 'loss0:',
                  loss0_rec.cpu().detach().item())

        # loss = ((loss1 + loss2 + loss3 + loss4 + loss0) / 5) # check the log function here
        # loss = torch.log10(loss1 + loss2 + loss3 + loss4 + loss0) # check the log function here
        loss = torch.log10(loss1 + loss2 + loss3 + loss4)  # check the log function here
        # loss = (loss1 + loss2 + loss3 + loss4) / 4  # check the log function here
        # loss = torch.log10(torch.mean(loss_data ** p))
        loss_rec = loss.detach()

        # print total loss
        if verbose:
            print('Loss: ', loss_rec.item())
        return loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_int, verbose):
        loss_pde, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_pde_residual(
            inp_train_int, verbose)
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
                    loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_loss(
                        inp_train_int,
                        verbose=verbose)
                    loss.backward()
                    loss_1.append(loss1_rec.item())
                    loss_2.append(loss2_rec.item())
                    loss_3.append(loss3_rec.item())
                    loss_4.append(loss4_rec.item())
                    loss_0.append(loss0_rec.item())
                    history.append(loss_rec.item())
                    print('Loss: ', history[-1])

                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1], 'loss1:', loss_1[-1], 'loss2:', loss_2[-1], 'loss3:', loss_3[-1], 'loss4:',
              loss_4[-1], 'loss0:', loss_0[-1])

        return history, loss_1, loss_2, loss_3, loss_4


n_int = 50000

pinn = Pinns(n_int)
# load trained NN for further training
flag_train = 'first'

pinn.load_model()
flag_train = 'final'

# opt_type = "ADAM"
opt_type = "LBFGS"
n_epochs1 = 1
if opt_type == "ADAM":
    optimizer_ = optim.Adam(list(pinn.approximate_solution.parameters()), lr=0.00001, weight_decay=0.)
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
modelname = time_now + '_PDE_loss_learning_regular_' + str(n_hidden_layers) + '_' + str(
    neurons) + '_' + flag_train + '_X0_model.pth'
path = os.path.join(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\model_save', modelname)
# path = os.path.join(r'/cluster/home/haotxu/PINNs/model_save', modelname)
torch.save(pinn.approximate_solution.cpu().state_dict(),
           path)  # saving model !!!! activate it if you want to save the model
pinn.approximate_solution = pinn.approximate_solution.to(device)

# plot the curve of total loss
plt.figure()
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(history1) + 1), history1, label="Train Loss")
plt.legend()
plotname = time_now + '_PDE_loss_learning_regular_' + str(n_hidden_layers) + '_' + str(
    neurons) + '_X0_loss.png'
# path = os.path.join(r'/cluster/home/haotxu/PINNs/plot_loss', plotname)
path = os.path.join(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\plot_loss', plotname)
plt.savefig(path)  # save the plot if you want

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
# path = os.path.join(r'/cluster/home/haotxu/PINNs/plot_loss', plotname)  # path of saving
path = os.path.join(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\plot_loss', plotname)
plt.savefig(path)  # save the plot if you want

input_int = (pinn.soboleng.draw(100000).to(device)) * 2 - 1
for i in range(7):
    for strain_rate in [0.005, 0.001, 0.0002]:
        pinn.plot_learning_result(input_int, name='end', idx=i, strain_rate=strain_rate, from_flag=1)
        pinn.plot_learning_result(input_int, name='end', idx=i, strain_rate=strain_rate, from_flag=-1)
plt.show(block=True) # use this only when you run the code locally to show the plots on your screen
