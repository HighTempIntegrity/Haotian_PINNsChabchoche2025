# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:41:51 2023

@author: Haotian
"""

import os
import sys
import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch import optim
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from models.network import NeuralNet
from utils.normalization import normalize_input_linear, normalize_input_log
from utils.normalization import denormalize_input_linear, denormalize_input_log, denormalize_output
from utils.normalization import normalize_X0_Cgam, denormalize_X0_Cgam
from utils.normalization import normalize_el0, denormalize_el0
from utils.normalization import normalize_A_n, denormalize_A_n

# mpl.use('Qt5Agg')  # activate this only if you run the code locally and want to see the plot on your screen
torch.autograd.set_detect_anomaly(False)
mpl.use('Qt5Agg')

# Set random seed for reproducibility
seed = 128
torch.manual_seed(seed)
np.random.seed(seed)

# Use date+time for naming
time_now = time.localtime()
time_now = time.strftime('%Y%m%d_%H%M%S', time_now)
print(time_now)
print('new x0 range')
print('eps_tot input')
print('with strain rate range')
print('plot 63 figures')
print('new collocation points')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
if device == 'cuda':
    print(torch.cuda.get_device_name())

E_true = [
    1.733202524017848e+05, 1.719250429988526e+05, 1.685154224701322e+05,
    1.608032458881426e+05, 1.433591734082915e+05, 1.276015067185787e+05,
    1.039026514462645e+05
]  # Elastic modulus
A_true = [
    7.271967187774671e-68, 1.223036548454673e-58, 2.841141305251292e-48,
    6.600034910325641e-38, 1.533202897617428e-27, 2.336825755956020e-22,
    3.561664683901490e-17
]  # Viscosity parameter
n_true = [
    3.097637504571841e+01, 3.052924289231759e+01, 2.800343670376871e+01,
    2.010721603401106e+01, 1.257707996491060e+01, 1.101319566195635e+01,
    1.031361490831761e+01
]  # Viscosity exponent
# Kinematic hardening
C1_true = [
    7.039581899135466e+05, 7.010757215318991e+05, 6.977602579615910e+05,
    6.943616316132719e+05, 6.908777564897652e+05, 6.891031844916017e+05,
    6.873064942701949e+05
]  # Linear hardening
gam1_true = [
    8.638076287898486e+03, 8.642456613149870e+03, 8.647386204050341e+03,
    8.652324144143302e+03, 8.657270447569681e+03, 8.659746739958495e+03,
    8.662225128494367e+03
]  # Dynamic recovery
kX1_true = [
    1.213437339458799e-17, 9.784290653571963e-14, 2.398107573943819e-09,
    5.877682096546933e-05, 1.302209338992991e+00, 1.278599928169617e+01,
    1.354921798869957e+01
]  # Static recovery
C2_true = [
    1.379438862752224e+05, 1.367273026669160e+05, 1.353236014916855e+05,
    1.338799416681737e+05, 1.323951857081455e+05, 1.316370323828586e+05,
    1.308681637429038e+05
]  # Linear hardening
gam2_true = [
    5.146608338936772e+02, 5.282558324999578e+02, 5.649853015340607e+02,
    6.572176538786182e+02, 8.888247814173294e+02, 1.113843280641131e+03,
    1.470419635751522e+03
]  # Dynamic recovery
kX2_true = [
    8.227905586075923e-08, 1.349513542842455e-06, 3.126971074556235e-05,
    7.208438853093072e-04, 1.486336260264507e-02, 4.969795091030596e-02,
    9.684343028613533e-02
]  # Static recovery
C3_true = [0, 0, 0, 0, 0, 0, 0]
gam3_true = [0, 0, 0, 0, 0, 0, 0]
kX3_true = [0, 0, 0, 0, 0, 0, 0]
# Isotropic hardening
R0_true = [
    1.380808629279288e+02, 1.369357706397548e+02, 1.334209679579912e+02,
    1.233422134498845e+02, 9.444121910339426e+01, 6.367160889994643e+01,
    1.156714239639362e+01
]  # Initial yield surface
Q1_true = [
    6.924912041144943e+02, 6.867484362852281e+02, 6.691213017952751e+02,
    6.185752036807515e+02, 4.736334358591429e+02, 3.193203473680843e+02,
    5.801053235322929e+01
]  # Linear hardening
b1_true = [
    2.123070481555192e+00, 2.128459769590981e+00, 2.183741598316607e+00,
    2.732563727449043e+00, 8.181112307858712e+00, 2.121264119480846e+01,
    6.227273882447193e+01
]  # Dynamic recovery
KR1_true = [
    5.945735735198818e-214, 9.464111101007290e-176, 7.934764066644871e-133,
    6.652550896894482e-90, 5.577536151555349e-47, 1.614988699857690e-25,
    4.676237732571239e-04
]  # Static recovery

# weight of four PDE loss terms (hyperparameters)
R_t_PDE_W = 0.05
X1_t_PDE_W = 5.
X2_t_PDE_W = 5.
eps_p_t_PDE_W = 1e-4

# define hyperparameters for NNs
n_hidden_layers = 6
neurons = 128

print('hidden layers:', n_hidden_layers)
print('neurons:', neurons)
p = 2


class Cyclic:

    def __init__(self, params):
        self.params = params
        self.num_interior_points = self.params['cyclic']['num_interior_points']
        self.device = torch.device(self.params['common_parameters']['device'])

        self.domain_extrema = self.params['cyclic']['data']['domain_extrema']
        self.domain_extrema_log = self.params['cyclic']['data'][
            'domain_extrema_log']
        self.domain_extrema_X0 = self.params['cyclic']['data'][
            'domain_extrema_X0']
        self.result_extrema = self.params['cyclic']['data']['result_extrema']
        neural_network_params = self.params['common_parameters'][
            'neural_network']

        self.approximate_solution = NeuralNet(
            input_dimension=self.params['cyclic']['input_dimension'],
            output_dimension=self.params['cyclic']['output_dimension'],
            n_hidden_layers=neural_network_params['n_hidden_layers'],
            neurons=neural_network_params['neurons'],
            regularization_param=neural_network_params['regularization'],
            regularization_exp=neural_network_params['regularization_exp'],
            retrain_seed=neural_network_params['retrain_seed']).to(self.device)

        self.soboleng = torch.quasirandom.SobolEngine(dimension=18)

        self.training_set_int = self.assemble_datasets()

    # load saved NN parameters if need for further training
    def load_model(self):
        model_name = self.params['cyclic']['load_model_name']
        model_save_path = self.params['common_parameters']['model_save_path']
        full_model_path = os.path.join(model_save_path, model_name)
        self.approximate_solution.load_state_dict(torch.load(full_model_path))

    ################################################################################################
    def plot_learning_result(self, input_int, name, idx, state, strain_rate):
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
        if strain_rate == 0.005:
            data = scipy.io.loadmat(
                r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_'
                + str(idx + 1) + '_rate_005_state_' + str(state) + '.mat')
        elif strain_rate == 0.001:
            data = scipy.io.loadmat(
                r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_'
                + str(idx + 1) + '_rate_001_state_' + str(state) + '.mat')
        elif strain_rate == 0.0002:
            data = scipy.io.loadmat(
                r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_'
                + str(idx + 1) + '_rate_0002_state_' + str(state) + '.mat')

        sig_data = torch.from_numpy(np.reshape(data['sig'],
                                               (-1, 1))).to(device)  # stress
        eps_tot_data = torch.from_numpy(np.reshape(
            data['eps_tot'], (-1, 1))).to(device)  # total strain
        R_data = torch.from_numpy(np.reshape(data['R'],
                                             (-1, 1))).to(device)  # R
        X1_data = torch.from_numpy(np.reshape(data['X1'],
                                              (-1, 1))).to(device)  # X1
        X2_data = torch.from_numpy(np.reshape(data['X2'],
                                              (-1, 1))).to(device)  # X2
        eps_p_data = torch.from_numpy(np.reshape(data['eps_in'], (-1, 1))).to(
            device
        )  # viscoplastic strain # eps_tot_data - sig_data / E + el0 #

        if state == 1:
            x10 = 1.
            x20 = 1.
            el0 = 1.
        elif state == 2:
            x10 = 0.
            x20 = 0.
            el0 = 0.
        elif state == 3:
            x10 = -1.
            x20 = -1.
            el0 = -1.

        x_test = input_int.clone().to(device)
        x_test[:, 0] = torch.sort(x_test[:, 0], 0)[0].reshape(-1, )

        input_linear = torch.tensor([0, E, n, C1, gam1, C2, gam2, R0, Q1, b1],
                                    device=device)
        input_log = torch.tensor([strain_rate, kX1, kX2, KR1],
                                 dtype=torch.float64,
                                 device=device)

        x_test[:, 1:10] = normalize_input_linear(input_linear,
                                                 self.domain_extrema)[1:10]
        x_test[:, 11:15] = normalize_input_log(input_log,
                                               self.domain_extrema_log)
        x_test[:, 15:17] = torch.tensor([x10, x20], device=device)
        x_test[:, 17] = normalize_A_n(
            torch.tensor([A], dtype=torch.float64, device=device), n)
        x_test[:, 10] = torch.tensor([el0], device=device)
        x_test = x_test.clone()
        x_test.requires_grad = True

        linear_input = denormalize_input_linear(x_test[:, :10],
                                                self.domain_extrema)
        log_input = denormalize_input_log(x_test[:, 11:15],
                                          self.domain_extrema_log)

        eps_tot = linear_input[:, 0]
        E = linear_input[:, 1]
        n = linear_input[:, 2]
        A = denormalize_A_n(x_test[:, 17].to(torch.float64), n).reshape(-1, )
        C1 = linear_input[:, 3]
        gam1 = linear_input[:, 4]
        kX1 = log_input[:, 1]
        C2 = linear_input[:, 5]
        gam2 = linear_input[:, 6]
        kX2 = log_input[:, 2]
        Q1 = linear_input[:, 8]
        b1 = linear_input[:, 9]
        KR1 = log_input[:, 3]

        R0 = linear_input[:, 7].to(torch.float64)
        eps_tot_rate = log_input[:, 0]

        C = torch.cat([C1.reshape(-1, 1), C2.reshape(-1, 1)], 1)
        gam = torch.cat([gam1.reshape(-1, 1), gam2.reshape(-1, 1)], 1)
        kX = torch.cat([kX1.reshape(-1, 1), kX2.reshape(-1, 1)], 1)
        strain_rate = torch.cat(
            [eps_tot_rate.reshape(-1, 1),
             eps_tot_rate.reshape(-1, 1)], 1)
        X0 = denormalize_X0_Cgam(x_test[:, 15:17], self.domain_extrema_X0, C,
                                 gam, kX, strain_rate)
        X10 = X0[:, 0].to(torch.float64)
        X20 = X0[:, 1].to(torch.float64)
        X_0 = X10 + X20
        eps_e_0 = denormalize_el0(x_test[:, 10], X_0, R0, E, A, n,
                                  eps_tot_rate)
        del C, X0, X_0, gam, linear_input, log_input

        # Calculation of the turning point of elastic domain
        linear_constraint = (
            eps_tot.reshape(-1, ) /
            (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        hard_constraint_linear = torch.cat([
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1)
        ], 1)

        y_pred = self.approximate_solution(x_test) * hard_constraint_linear

        y_pred = denormalize_output(y_pred, self.result_extrema)

        eps_p_pred = y_pred[:, 0].to(torch.float64)
        R_pred = y_pred[:, 1].to(torch.float64)
        X1_pred = y_pred[:, 2].to(torch.float64)
        X2_pred = y_pred[:, 3].to(torch.float64)

        eps_tot_plot = eps_tot

        R_pred = R_pred + R0
        X1_pred = X1_pred + X10
        X2_pred = X2_pred + X20

        sig_pred = (eps_e_0 + eps_tot - eps_p_pred) * E

        X = X1_pred + X2_pred

        eps_p_eps = torch.autograd.grad(
            eps_p_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
                1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        R_eps = torch.autograd.grad(
            R_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
                1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        X1_eps = torch.autograd.grad(
            X1_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
                1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        X2_eps = torch.autograd.grad(
            X2_pred.sum(), x_test, create_graph=True)[0][:, 0] * (
                1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))

        eps_p_t = (eps_p_eps * eps_tot_rate)
        eps_p_eq_t = abs(eps_p_t)

        R_t = (R_eps * eps_tot_rate)
        X1_t = (X1_eps * eps_tot_rate)
        X2_t = (X2_eps * eps_tot_rate)

        a1 = sig_pred.reshape(-1, ) - X.reshape(-1, )
        a2 = a1 * torch.tanh(a1 / 0.001) - R_pred.reshape(-1, )
        a3 = A * (0.5 *
                  (a2 + torch.sqrt(0.001**2 + a2**2)))**n * torch.sign(a1)
        loss1 = (eps_p_t.reshape(-1, ) - a3)
        mean1 = torch.mean(abs(loss1)).cpu().detach().numpy()

        loss2 = (R_t.reshape(-1, ) - Q1 * eps_p_eq_t +
                 b1 * R_pred.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) +
                 KR1 * R_pred.reshape(-1, ))
        mean2 = torch.mean(abs(loss2)).cpu().detach().numpy()
        loss3 = (X1_t.reshape(-1, ) - C1 * eps_p_t +
                 gam1 * eps_p_eq_t * X1_pred.reshape(-1, ) +
                 kX1 * X1_pred.reshape(-1, ))
        mean3 = torch.mean(abs(loss3)).cpu().detach().numpy()
        loss4 = (X2_t.reshape(-1, ) - C2 * eps_p_t +
                 gam2 * eps_p_eq_t * X2_pred.reshape(-1, ) +
                 kX2 * X2_pred.reshape(-1, ))
        mean4 = torch.mean(abs(loss4)).cpu().detach().numpy()
        '''
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.plot(eps_tot_plot.cpu().detach().numpy(), eps_p_t.reshape(-1, 1).cpu().detach().numpy(), ms=0.5,
                 label="plastic potential left")
        plt.legend()

        plt.figure()
        plt.plot(eps_tot_plot.cpu().detach().numpy(), a3.reshape(-1, 1).cpu().detach().numpy(), ms=0.5,
                 label="plastic potential right")
        plt.legend()
        # plt.show(block=True)
        '''
        fig = plt.figure(figsize=(16, 8))
        # fig, ax
        plt.grid(True, which="both", ls=":")
        ax1 = plt.subplot(1, 3, 1)
        ax1.set_ylabel('stress (MPa)')
        ax1.set_xlabel('total strain')
        ax1.plot(eps_tot_plot.cpu().detach().numpy(),
                 sig_pred.cpu().detach().numpy(),
                 ms=0.5,
                 label="stress_predict")
        ax1.plot(eps_tot_data.cpu().detach().numpy(),
                 sig_data.cpu().detach().numpy(),
                 ms=0.5,
                 label="stress_true")
        ax1.legend(loc='upper left')

        ax3 = plt.subplot(2, 3, 2)
        ax3.set_ylabel('eps_p')
        ax3.plot(eps_tot_plot.cpu().detach().numpy(),
                 eps_p_pred.cpu().detach().numpy(),
                 ms=0.5,
                 label="viscoplastic strain predict")
        ax3.plot(eps_tot_data.cpu().detach().numpy(),
                 eps_p_data.cpu().detach().numpy(),
                 ms=0.5,
                 label="viscoplastic strain true")
        ax3.legend(loc='upper left')
        color = 'tab:red'
        ax12 = ax3.twinx()
        ax12.plot(eps_tot_plot.cpu().detach().numpy(),
                  loss1.cpu().detach().numpy(),
                  ms=0.5,
                  label="Loss1",
                  color=color)
        ax12.set_ylabel('loss1', color=color)
        ax12.legend(loc='upper right')
        ax12.tick_params(axis='y', labelcolor=color)
        plt.ylim((-10 * mean1, 10 * mean1))

        ax4 = plt.subplot(2, 3, 5, sharex=ax3)
        ax4.set_xlabel('total strain')
        ax4.set_ylabel('R (MPa)')
        ax4.plot(eps_tot_plot.cpu().detach().numpy(),
                 R_pred.cpu().detach().numpy(),
                 ms=0.5,
                 label="R_predict")
        ax4.plot(eps_tot_data.cpu().detach().numpy(),
                 R_data.cpu().detach().numpy(),
                 ms=0.5,
                 label="R_true")
        ax4.legend(loc='upper left')
        color = 'tab:red'
        ax42 = ax4.twinx()
        ax42.plot(eps_tot_plot.cpu().reshape(-1, ).detach().numpy(),
                  loss2.cpu().reshape(-1, ).detach().numpy(),
                  ms=0.5,
                  label="Loss2",
                  color=color)
        ax42.set_ylabel('loss2', color=color)
        ax42.legend(loc='upper right')
        ax42.tick_params(axis='y', labelcolor=color)
        plt.ylim((-10 * mean2, 10 * mean2))

        ax5 = plt.subplot(2, 3, 3)
        ax5.set_ylabel('X1 (MPa)')
        ax5.plot(eps_tot_plot.cpu().detach().numpy(),
                 X1_pred.cpu().detach().numpy(),
                 ms=0.5,
                 label="X1_predict")
        ax5.plot(eps_tot_data.cpu().detach().numpy(),
                 X1_data.cpu().detach().numpy(),
                 ms=0.5,
                 label="X1_true")
        ax5.legend(loc='upper left')
        color = 'tab:red'
        ax52 = ax5.twinx()
        ax52.plot(eps_tot_plot.cpu().detach().numpy(),
                  loss3.cpu().detach().numpy(),
                  ms=0.5,
                  label="Loss3",
                  color=color)
        ax52.set_ylabel('loss3', color=color)
        ax52.legend(loc='upper right')
        ax52.tick_params(axis='y', labelcolor=color)
        plt.ylim((-10 * mean3, 10 * mean3))

        ax6 = plt.subplot(2, 3, 6, sharex=ax5)
        ax6.set_xlabel('total strain')
        ax6.set_ylabel('X2 (MPa)')
        ax6.plot(eps_tot_plot.cpu().detach().numpy(),
                 X2_pred.cpu().detach().numpy(),
                 ms=0.5,
                 label="X2_predict")
        ax6.plot(eps_tot_data.cpu().detach().numpy(),
                 X2_data.cpu().detach().numpy(),
                 ms=0.5,
                 label="X2_true")
        ax6.legend(loc='upper left')

        color = 'tab:red'
        ax62 = ax6.twinx()
        ax62.plot(eps_tot_plot.cpu().detach().numpy(),
                  loss4.cpu().detach().numpy(),
                  ms=0.5,
                  label="Loss4",
                  color=color)
        ax62.set_ylabel('loss4', color=color)
        ax62.legend(loc='upper right')
        ax62.tick_params(axis='y', labelcolor=color)
        plt.ylim((-10 * mean4, 10 * mean4))
        fig.tight_layout()
        plt.show(block=True)

        plotname = time_now + '_Benchmark_' + str(idx + 1) + '_state_' + str(
            state) + '_rate_' + rate_name + '_NNstructure_' + str(
                n_hidden_layers) + '_' + str(neurons) + '.png'
        path = os.path.join(
            r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\plot_fit',
            plotname)
        # path = os.path.join(r'/cluster/home/haotxu/PINNs/plot_fit', plotname)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        return sig_pred, sig_data, eps_tot_plot, eps_tot_data

    ################################################################################################
    #  Function returning the input-output tensor required to assemble the training set corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = (self.soboleng.draw(
            self.num_interior_points).to(device)) * 2 - 1
        output_int = torch.zeros((input_int.shape[0], 1)).to(device)
        return input_int, output_int

    # Function returning the training sets S_int as dataloader
    def assemble_datasets(self):
        input_int, output_int = self.add_interior_points()
        training_set_int = DataLoader(torch.utils.data.TensorDataset(
            input_int, output_int),
                                      batch_size=self.num_interior_points,
                                      shuffle=False)
        return training_set_int

    ################################################################################################
    def compute_pde_residual(self, x_train, verbose):
        # weight for each loss term
        global eps_p_t_PDE_W, R_t_PDE_W, X1_t_PDE_W, X2_t_PDE_W
        ''''''
        global E_true, A_true, n_true, C1_true, gam1_true, kX1_true, C2_true, gam2_true, kX2_true, R0_true, Q1_true, b1_true, KR1_true
        idx = 1

        # data = scipy.io.loadmat(r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\PINNs\File from Patrik\Viscoplastic_Chaboche_Model\data_' + str(idx + 1) + '_X10_-30_X20_-60_el_-001_rate_0002.mat')
        '''
        t_data = torch.from_numpy(np.reshape(data['t'], (-1, 1))).to(device)  # stress
        sig_data = torch.from_numpy(np.reshape(data['sig'], (-1, 1))).to(device)  # stress
        eps_tot_data = torch.from_numpy(np.reshape(data['eps_tot'], (-1, 1))).to(device)  # total strain
        R_data = torch.from_numpy(np.reshape(data['R'], (-1, 1))).to(device)  # R
        X1_data = torch.from_numpy(np.reshape(data['X1'], (-1, 1))).to(device)  # X1
        X2_data = torch.from_numpy(np.reshape(data['X2'], (-1, 1))).to(device)  # X2
        eps_p_data = torch.from_numpy(np.reshape(data['eps_in'], (-1, 1))).to(device)
        '''
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

        x10 = 0
        x20 = 0
        el0 = 0

        x_test = x_train.clone().to(device)
        a11 = x_test[0, :]
        x_test[:, 0] = torch.sort(x_test[:, 0], 0)[0].reshape(-1, )
        '''
        x_test = torch.zeros((eps_tot_data.shape[0], 18), device=device)
        a11 = x_test[0, :]
        x_test[:, 0] = eps_tot_data.reshape(-1, ) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]) - 1.
        '''
        input_linear = torch.tensor([0, E, n, C1, gam1, C2, gam2, R0, Q1, b1],
                                    device=device)
        input_log = torch.tensor([0.0002, kX1, kX2, KR1],
                                 dtype=torch.float64,
                                 device=device)
        input_X0 = torch.tensor([x10, x20], device=device)
        C = torch.tensor([C1, C2], device=device)
        gam = torch.tensor([gam1, gam2], device=device)
        kX = torch.tensor([kX1, kX2], device=device)
        strain_rate_tensor = torch.tensor([0.0002, 0.0002], device=device)

        x_test[:, 1:10] = normalize_input_linear(input_linear,
                                                 self.domain_extrema)[1:10]
        x_test[:, 11:15] = normalize_input_log(input_log,
                                               self.domain_extrema_log)
        x_test[:, 15:17] = normalize_X0_Cgam(input_X0, self.domain_extrema_X0,
                                             C, gam, kX, strain_rate_tensor)
        x_test[:, 17] = normalize_A_n(
            torch.tensor([A], dtype=torch.float64, device=device), n)

        eps_e = torch.tensor([el0], device=device)
        eps_tot_rate = input_log[0]
        X_0 = torch.tensor([x10 + x20], device=device)
        eps_e_0 = normalize_el0(eps_e, X_0, R0, E, A, n, eps_tot_rate)
        x_test[:, 10] = eps_e_0
        x_train_ = x_test.clone()

        # x_train_ = x_train.clone()
        # x_train_[:, 0] = torch.sort(x_train_[:, 0], 0)[0].reshape(-1, )
        x_train_.requires_grad = True
        del x_train
        # Denormalization of inputs
        linear_input = denormalize_input_linear(x_train_[:, :10],
                                                self.domain_extrema)
        log_input = denormalize_input_log(x_train_[:, 11:15],
                                          self.domain_extrema_log)

        eps_tot = linear_input[:, 0]
        E = linear_input[:, 1]
        n = linear_input[:, 2]
        A = denormalize_A_n(x_train_[:, 17].to(torch.float64), n).reshape(-1, )
        C1 = linear_input[:, 3]
        gam1 = linear_input[:, 4]
        kX1 = log_input[:, 1]
        C2 = linear_input[:, 5]
        gam2 = linear_input[:, 6]
        kX2 = log_input[:, 2]
        Q1 = linear_input[:, 8]
        b1 = linear_input[:, 9]
        KR1 = log_input[:, 3]

        R0 = linear_input[:, 7].to(torch.float64)
        eps_tot_rate = log_input[:, 0]

        C = torch.cat([C1.reshape(-1, 1), C2.reshape(-1, 1)], 1)
        gam = torch.cat([gam1.reshape(-1, 1), gam2.reshape(-1, 1)], 1)
        kX = torch.cat([kX1.reshape(-1, 1), kX2.reshape(-1, 1)], 1)
        strain_rate = torch.cat(
            [eps_tot_rate.reshape(-1, 1),
             eps_tot_rate.reshape(-1, 1)], 1)
        X0 = denormalize_X0_Cgam(x_train_[:, 15:17], self.domain_extrema_X0, C,
                                 gam, kX, strain_rate)
        X10 = X0[:, 0].to(torch.float64)
        X20 = X0[:, 1].to(torch.float64)
        X_0 = X10 + X20
        eps_e_0 = denormalize_el0(x_train_[:, 10], X_0, R0, E, A, n,
                                  eps_tot_rate)
        del C, X0, X_0, gam, linear_input, log_input

        # Calculation of the turning point of elastic domain
        linear_constraint = (
            eps_tot.reshape(-1, ) /
            (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))

        hard_constraint_other = linear_constraint  #* (torch.tanh(2 * INP) ** 2 + 0.001)
        hard_constraint_linear = torch.cat([
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1)
        ], 1)

        # Calculation of outputs and denormalization
        y_pred = self.approximate_solution(x_train_) * hard_constraint_linear
        '''
        y_true_raw = torch.cat((eps_p_data, R_data-R0.reshape(-1, 1), X1_data-x10, X2_data-x20), 1)
        y_true = self.normalize_output(y_true_raw)
        loss_data = y_pred.reshape(-1, ) - y_true.reshape(-1, )
        '''
        y_pred = denormalize_output(y_pred, self.result_extrema)
        # add hard constraints
        eps_p = y_pred[:, 0].to(torch.float64)
        # me = torch.mean(abs(eps_p))
        R = y_pred[:, 1].to(torch.float64)  # * hard_constraint_other
        X1 = y_pred[:, 2].to(torch.float64)  # * hard_constraint_other
        X2 = y_pred[:, 3].to(torch.float64)  # * hard_constraint_other
        del y_pred

        X1 = X1 + X10
        X2 = X2 + X20
        R = R + R0
        '''
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.plot(eps_tot_data.cpu().detach().numpy(), X2.reshape(-1, 1).cpu().detach().numpy(), ms=0.5,
                 label="PINN")
        plt.plot(eps_tot_data.cpu().detach().numpy(), X2_data.reshape(-1, 1).cpu().detach().numpy(), ms=0.5,
                 label="True")
        plt.legend()
        plt.show(block=True)
        '''
        X = X1 + X2

        sig = (eps_e_0 + eps_tot - eps_p) * E

        Delta_eps_p_eps = torch.autograd.grad(
            eps_p.sum(), x_train_, create_graph=True)[0][:, 0] * (
                1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        Delta_R_eps = torch.autograd.grad(
            R.sum(), x_train_, create_graph=True)[0][:, 0] * (
                1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        Delta_X1_eps = torch.autograd.grad(
            X1.sum(), x_train_, create_graph=True)[0][:, 0] * (
                1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        Delta_X2_eps = torch.autograd.grad(
            X2.sum(), x_train_, create_graph=True)[0][:, 0] * (
                1 / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        del x_train_
        eps_p_t = (Delta_eps_p_eps * eps_tot_rate)
        eps_p_eq_t = abs(eps_p_t)

        R_t = (Delta_R_eps * eps_tot_rate)
        X1_t = (Delta_X1_eps * eps_tot_rate)
        X2_t = (Delta_X2_eps * eps_tot_rate)

        # Loss 0 is the penulty of pure elastic case
        loss0_vec = eps_tot.reshape(-1, ) - eps_p.reshape(-1, ) - 0.002
        loss0_vec = 100000000 * 0.5 * (loss0_vec + torch.sqrt(loss0_vec**2))
        loss0 = torch.mean(loss0_vec**p)
        loss0_rec = loss0.detach()

        # to put higher weight for small strains
        weight = 1. / (0.5 + (0.5 * eps_tot.reshape(-1, )) /
                       (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        # weight = 0. / (0.5 + (0.5 * eps_tot.reshape(-1, )) / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0])) + 1.
        weight_raw = (-torch.tanh(
            10 *
            (eps_tot.reshape(-1, ) /
             (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]) - 0.6)) +
                      1) * 3 + 1
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(eps_tot.cpu().detach().numpy(),
                    weight_raw.reshape(-1, 1).cpu().detach().numpy(),
                    s=0.5,
                    label="Weight")
        plt.legend()
        plt.show(block=True)

        # Loss 1
        Smooth_factor = 0.0001
        a1 = sig.reshape(-1, ) - X.reshape(-1, )
        a2 = a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(
            -1,
        )  # a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(-1, )   # what is the range of overstress
        a3 = (A *
              (0.5 *
               (a2 + torch.sqrt(a2**2 + Smooth_factor**2)))**n) * torch.tanh(
                   a1 / Smooth_factor)
        a4 = (A * (0.5 * (a2 + torch.sqrt(a2**2)))**n)

        # flag = (torch.tanh((a4 - 1e-5)/1e-5) + 1) / 2
        flag = torch.tensor([1.], device=device)
        # flag = (torch.tanh(a2) + 1) / 2
        '''
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(a4.cpu().detach().numpy(), flag.reshape(-1, 1).cpu().detach().numpy(), s=0.5,
                 label="Flag")
        plt.set_xlabel('total strain')
        plt.legend()
        plt.show(block=True)
        '''
        loss1_vec = (eps_p_t.reshape(-1, ) - flag * a3) / (eps_p_t_PDE_W)
        loss1_w = loss1_vec * weight.reshape(-1, )
        # loss1_vec = (eps_p_eq_t.reshape(-1, ) / A) ** (1/n) - 0.5 * (a2 + torch.sqrt(a2 ** 2 + Smooth_factor ** 2))

        # to damp the effect of big loss values at the begining of training
        loss1 = torch.mean(loss1_w**p)  # [400:]
        # loss1 = torch.log10(1 + torch.mean(loss1_vec ** p))
        loss1_rec = loss1.detach()

        # loss 2
        # loss2_vec = (R_t.reshape(-1, ) - Q1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) + b1.reshape(-1, ) * R.reshape(
        #     -1, ) * eps_p_eq_t.reshape(-1, ) + KR1.reshape(-1, ) * R.reshape(-1, )) / R_t_PDE_W
        loss2_vec = (
            R_t.reshape(-1, ) - flag *
            (Q1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) -
             b1.reshape(-1, ) * R.reshape(-1, ) * eps_p_eq_t.reshape(-1, )) +
            KR1.reshape(-1, ) * R.reshape(-1, )) / (R_t_PDE_W
                                                    )  # * ((torch.mean(R0)/R0)
        loss2_w = loss2_vec * weight.reshape(-1, )
        loss2 = torch.mean(loss2_w**p)  # [400:]
        loss2_rec = loss2.detach()

        # loss 3
        loss3_vec = (X1_t.reshape(-1, ) - flag *
                     (C1.reshape(-1, ) * eps_p_t.reshape(-1, ) - gam1.reshape(
                         -1, ) * eps_p_eq_t.reshape(-1, ) * X1.reshape(-1, )) +
                     kX1.reshape(-1, ) * X1.reshape(-1, )) / (
                         X1_t_PDE_W)  # * (torch.mean(E)/E)
        # loss3_vec = (X1_t.reshape(-1, ) - C1.reshape(-1, ) * eps_p_t.reshape(-1, ) + gam1.reshape(
        #     -1, ) * eps_p_eq_t.reshape(-1, ) * X1.reshape(-1, ) + kX1.reshape(-1, ) * X1.reshape(-1, )) / X1_t_PDE_W
        loss3_w = loss3_vec * weight.reshape(-1, )
        # a = loss3_w ** p
        loss3 = torch.mean(loss3_w**p)  # [400:]
        loss3_rec = loss3.detach()

        # loss 4
        loss4_vec = (X2_t.reshape(-1, ) - flag *
                     (C2.reshape(-1, ) * eps_p_t.reshape(-1, ) - gam2.reshape(
                         -1, ) * eps_p_eq_t.reshape(-1, ) * X2.reshape(-1, )) +
                     kX2.reshape(-1, ) * X2.reshape(-1, )) / (
                         X2_t_PDE_W)  # * (torch.mean(E)/E)
        # loss4_vec = (X2_t.reshape(-1, ) - C2.reshape(-1, ) * eps_p_t.reshape(-1, ) + gam2.reshape(
        #     -1, ) * eps_p_eq_t.reshape(-1, ) * X2.reshape(-1, ) + kX2.reshape(-1, ) * X2.reshape(-1, )) / X2_t_PDE_W
        loss4_w = loss4_vec * weight.reshape(-1, )
        loss4 = torch.mean(loss4_w**p)  # [400:]
        loss4_rec = loss4.detach()
        '''
        save_file = torch.cat((E.reshape(-1, 1), torch.log10(A).reshape(-1, 1), n.reshape(-1, 1), C1.reshape(-1, 1), gam1.reshape(-1, 1), torch.log10(kX1).reshape(-1, 1), C2.reshape(-1, 1), gam2.reshape(-1, 1), torch.log10(kX2).reshape(-1, 1), Q1.reshape(-1, 1), b1.reshape(-1, 1), torch.log10(KR1).reshape(-1, 1), R0.reshape(-1, 1), X10.reshape(-1, 1), X20.reshape(-1, 1), eps_e_0.reshape(-1, 1), torch.log10(eps_tot_rate).reshape(-1, 1), eps_p.reshape(-1, 1), R.reshape(-1, 1), X1.reshape(-1, 1), X2.reshape(-1, 1), loss1_vec.reshape(-1, 1), loss2_vec.reshape(-1, 1), loss3_vec.reshape(-1, 1), loss4_vec.reshape(-1, 1)), 1)
        np.savetxt(time_now + '_corelation matrix.txt', save_file.cpu().detach().numpy(), delimiter=',')
        del save_file
        '''
        # print the loss terms
        if verbose:
            print('loss1:',
                  loss1_rec.cpu().detach().item(), 'loss2:',
                  loss2_rec.cpu().detach().item(), 'loss3:',
                  loss3_rec.cpu().detach().item(), 'loss4:',
                  loss4_rec.cpu().detach().item(), 'loss0:',
                  loss0_rec.cpu().detach().item())

        # loss = ((loss1 + loss2 + loss3 + loss4 + loss0) / 5) # check the log function here
        # loss = torch.log10(loss1 + loss2 + loss3 + loss4 + loss0) # check the log function here
        loss = torch.log10(loss1 + loss2 + loss3 +
                           loss4)  # check the log function here
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
            if verbose:
                print("################################ ", epoch,
                      " ################################")

            for j, (inp_train_int,
                    u_train_int) in enumerate(self.training_set_int):

                def closure():
                    optimizer.zero_grad()
                    loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_loss(
                        inp_train_int, verbose=verbose)
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

        print('Final Loss: ', history[-1], 'loss1:', loss_1[-1], 'loss2:',
              loss_2[-1], 'loss3:', loss_3[-1], 'loss4:', loss_4[-1], 'loss0:',
              loss_0[-1])

        return history, loss_1, loss_2, loss_3, loss_4


num_interior_points = 50000

pinn = Cyclic(num_interior_points)
print('range of Xi_0:', pinn.domain_extrema_X0.cpu().detach().numpy())
print('range of strain_rate:',
      (10**pinn.domain_extrema_log).cpu().detach().numpy()[0, :])
# load trained NN for further training
flag_train = 'first'

pinn.load_model()
flag_train = 'final'

input_int = (pinn.soboleng.draw(10000).to(device)) * 2 - 1
pinn.plot_learning_result(input_int,
                          name='end',
                          idx=6,
                          state=2,
                          strain_rate=0.005)
'''
pinn.plot_learning_result(input_int, name='end', idx=3, x10=0, x20=0, el0=0)

pinn.plot_learning_result(input_int, name='end', idx=3, x10=-5, x20=-10, el0=0)
pinn.plot_learning_result(input_int, name='end', idx=3, x10=-20, x20=-50, el0=-0.001)
pinn.plot_learning_result(input_int, name='end', idx=3, x10=-30, x20=-60, el0=-0.001)

for i in range(7):
    pinn.plot_learning_result(input_int, name='end', idx=i, x10=0, x20=0, el0=0)
    # pinn.plot_learning_result(input_int, name='end', idx=i, x10=-5, x20=-10, el0=0)
    pinn.plot_learning_result(input_int, name='end', idx=i, x10=-20, x20=-50, el0=-0.001)
    pinn.plot_learning_result(input_int, name='end', idx=i, x10=-30, x20=-60, el0=-0.001)
'''
plt.show(block=True)

# random initialize the NN parameters
# pinn.approximate_solution.init_xavier()

# opt_type = "ADAM"
opt_type = "LBFGS"
n_epochs1 = 1
if opt_type == "ADAM":
    optimizer_ = optim.Adam(list(pinn.approximate_solution.parameters()),
                            lr=0.00001,
                            weight_decay=0.)
elif opt_type == "LBFGS":
    optimizer_ = optim.LBFGS(list(pinn.approximate_solution.parameters()),
                             lr=float(0.1),
                             max_iter=1000000,
                             max_eval=1000000,
                             tolerance_grad=1e-12,
                             tolerance_change=1e-12,
                             line_search_fn='strong_wolfe')

start_time = time.time()

# what about loss0
history1, loss_11, loss_21, loss_31, loss_41 = pinn.fit(num_epochs=n_epochs1,
                                                        optimizer=optimizer_,
                                                        verbose=False)
n_epochs1 = len(history1)

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))

input_int = (pinn.soboleng.draw(100000).to(device)) * 2 - 1
y_test = pinn.approximate_solution(input_int)
print('max and min and mean of eps_p:',
      torch.max(y_test[:, 0]).item(),
      torch.min(y_test[:, 0]).item(),
      torch.mean(abs(y_test[:, 0])).item(), 'max and min and mean of R:',
      torch.max(y_test[:, 1]).item(),
      torch.min(y_test[:, 1]).item(),
      torch.mean(abs(y_test[:, 1])).item(), 'max and min and mean of X1:',
      torch.max(y_test[:, 2]).item(),
      torch.min(y_test[:, 2]).item(),
      torch.mean(abs(y_test[:, 2])).item(), 'max and min and mean of X2:',
      torch.max(y_test[:, 3]).item(),
      torch.min(y_test[:, 3]).item(),
      torch.mean(abs(y_test[:, 3])).item())
# save the trained model
modelname = time_now + '_PDE_loss_learning_regular_' + str(
    n_hidden_layers) + '_' + str(neurons) + '_' + flag_train + '_X0_model.pth'
path = os.path.join(
    r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\model_save',
    modelname)  # path for saving the model
# path = os.path.join(r'/cluster/home/haotxu/PINNs/model_save', modelname)
torch.save(pinn.approximate_solution.cpu().state_dict(),
           path)  # saving model !!!! activate it if you want to save the model
pinn.approximate_solution = pinn.approximate_solution.to(device)

# plot the curve of total loss
plt.figure()
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(history1) + 1), history1, label="Train Loss")
plt.legend()
plotname = time_now + '_PDE_loss_learning_regular_' + str(
    n_hidden_layers) + '_' + str(neurons) + '_X0_loss.png'
# path = os.path.join(r'/cluster/home/haotxu/PINNs/plot_loss', plotname)  # path of saving
path = os.path.join(
    r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\plot_loss',
    plotname)  # path of saving
plt.savefig(path)  # save the plot if you want

# plot the curve of each PDE loss terms
plt.figure()
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1,
                   len(loss_11) + 1).reshape(-1, 1),
         np.log10(loss_11).reshape(-1, 1),
         label="Loss1")
plt.plot(np.arange(1,
                   len(loss_21) + 1).reshape(-1, 1),
         np.log10(loss_21).reshape(-1, 1),
         label="Loss2")
plt.plot(np.arange(1,
                   len(loss_31) + 1).reshape(-1, 1),
         np.log10(loss_31).reshape(-1, 1),
         label="Loss3")
plt.plot(np.arange(1,
                   len(loss_41) + 1).reshape(-1, 1),
         np.log10(loss_41).reshape(-1, 1),
         label="Loss4")
plt.legend()
plotname = time_now + '_PDE_loss_learning_seperate_regular_' + str(
    n_hidden_layers) + '_' + str(neurons) + '_X0_loss.png'
# path = os.path.join(r'/cluster/home/haotxu/PINNs/plot_loss', plotname)  # path of saving
path = os.path.join(
    r'C:\Users\Haotian\OneDrive - ETH Zurich\ETH PhD\Code\Chaboche_PINNS\plot_loss',
    plotname)  # path of saving
plt.savefig(path)  # save the plot if you want

for i in range(7):
    for j in range(3):
        pinn.plot_learning_result(input_int,
                                  name='end',
                                  idx=i,
                                  state=j + 1,
                                  strain_rate=0.0002)

plt.show(
    block=True
)  # use this only when you run the code locally to show the plots on your screen
