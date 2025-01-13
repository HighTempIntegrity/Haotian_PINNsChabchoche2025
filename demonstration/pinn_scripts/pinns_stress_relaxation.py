# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:41:51 2023

@author: Haotian
"""

import os
import sys
import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch import optim
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from models.network import NeuralNet
from pinn_scripts.pinns_monotonic import Monotonic
from utils.normalization import normalize_input_linear, denormalize_input_linear
from utils.normalization import normalize_input_log, denormalize_input_log
from utils.normalization import normalize_t, denormalize_t
from utils.normalization import normalize_A_n, denormalize_A_n
from utils.normalization import normalize_initial, denormalize_initial
from utils.normalization import denormalize_output, normalize_output


class StressRelaxation:

    def __init__(
        self,
        params,
        mono_params=None,
        neptune_logger=None,
    ):
        self.params = params
        if neptune_logger is not None:
            self.neptune_logger = neptune_logger
        else:
            self.neptune_logger = None
        self.device = torch.device(self.params['common_parameters']['device'])
        self.num_interior_points = self.params['stress_relaxation'][
            'num_interior_points']

        self.domain_extrema = (
            self.params['stress_relaxation']['data']['domain_extrema']).to(
                self.device)
        self.domain_extrema_log = (
            self.params['stress_relaxation']['data']['domain_extrema_log']).to(
                self.device)

        self.initial_extrema = (
            self.params['stress_relaxation']['data']['initial_extrema']).to(
                self.device)

        self.result_extrema = (
            self.params['stress_relaxation']['data']['result_extrema']).to(
                self.device)

        neural_network_params = self.params['common_parameters'][
            'neural_network']

        self.approximate_solution = NeuralNet(
            input_dimension=self.params['stress_relaxation']
            ['input_dimension'],
            output_dimension=self.params['stress_relaxation']
            ['output_dimension'],
            n_hidden_layers=neural_network_params['n_hidden_layers'],
            neurons=neural_network_params['neurons'],
            regularization_param=neural_network_params['regularization'],
            regularization_exp=neural_network_params['regularization_exp'],
            retrain_seed=neural_network_params['retrain_seed']).to(self.device)

        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.params['stress_relaxation']['input_dimension'])

        self.training_set_int = self.assemble_datasets()

        if mono_params is not None:
            mono_params['monotonic']['num_interior_points'] = 100
            self.pinn_mono = Monotonic(params=mono_params)
            self.pinn_mono.load_model()

    def load_model(self):
        model_name = self.params['stress_relaxation']['load_model_name']
        model_save_path = self.params['common_parameters']['model_save_path']
        full_model_path = os.path.join(model_save_path, model_name)
        self.approximate_solution.load_state_dict(torch.load(full_model_path))

        print('Model loaded successfully')

    def save_model(self):
        neural_network_params = self.params['common_parameters'][
            'neural_network']
        model_name = 'stress_relaxation_' + self.params['current_time'] + str(
            neural_network_params['n_hidden_layers']) + str(
                neural_network_params['neurons']) + '.pth'
        model_save_path = self.params['common_parameters']['model_save_path']
        full_model_path = os.path.join(model_save_path, model_name)
        torch.save(self.approximate_solution.state_dict(), full_model_path)

    def predict(self, x_test):
        return self.approximate_solution.forward(x_test)

    ################################################################################################
    #  Function returning the input-output tensor required to assemble the training set corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int_0 = (self.soboleng.draw(self.num_interior_points).to(
            self.device)) * 2 - 1

        input_int_1 = self.soboleng.draw(int(self.num_interior_points / 2)).to(
            self.device)
        input_int_1[:, 1:] = input_int_1[:, 1:] * 2 - 1
        input_int_1[:, 0] = normalize_t(input_int_1[:, 0] * 900 + 1e-3,
                                        self.device)

        input_int = input_int_0  #torch.cat((input_int_0, input_int_1), 0)

        output_int = torch.zeros((input_int.shape[0], 1)).to(self.device)
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
    def compute_data_residual(self, x_data, y_data):
        # linear_constraint = (t.reshape(-1, ) / 450.)
        linear_constraint = (x_data[:, 0] + 1.)
        hard_constraint_linear = torch.cat([
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1)
        ], 1)
        y_pred = self.approximate_solution(x_data) * hard_constraint_linear

        loss_data = (torch.mean(
            ((y_pred.reshape(-1, ) - y_data.reshape(-1, )))**2))

        return torch.log10(loss_data)

    ################################################################################################
    def compute_pde_residual(self, x_train, verbose, idx):
        # weight for each loss term

        E = self.params['stress_relaxation']['common_constants']['E_true'][idx]
        A = self.params['stress_relaxation']['common_constants']['A_true'][idx]
        n = self.params['stress_relaxation']['common_constants']['n_true'][idx]
        C1 = self.params['stress_relaxation']['common_constants']['C1_true'][
            idx]
        gam1 = self.params['stress_relaxation']['common_constants'][
            'gam1_true'][idx]
        kX1 = self.params['stress_relaxation']['common_constants']['kX1_true'][
            idx]
        C2 = self.params['stress_relaxation']['common_constants']['C2_true'][
            idx]
        gam2 = self.params['stress_relaxation']['common_constants'][
            'gam2_true'][idx]

        kX2 = self.params['stress_relaxation']['common_constants']['kX2_true'][
            idx]
        R0 = self.params['stress_relaxation']['common_constants']['R0_true'][
            idx]
        Q1 = self.params['stress_relaxation']['common_constants']['Q1_true'][
            idx]
        b1 = self.params['stress_relaxation']['common_constants']['b1_true'][
            idx]
        KR1 = self.params['stress_relaxation']['common_constants']['KR1_true'][
            idx]

        x_test = x_train.clone().to(self.device)
        input_linear = torch.tensor([E, n, C1, gam1, C2, gam2, R0, Q1, b1],
                                    device=self.device)
        input_log = torch.tensor([0.005, kX1, kX2, KR1],
                                 dtype=torch.float64,
                                 device=self.device)

        # x_test[:, 1:10] = self.normalize_input_linear(input_linear)
        x_test[:, 8:10] = normalize_input_linear(input_linear,
                                                 self.domain_extrema)[7:]
        x_test[:, 12:15] = normalize_input_log(
            input_log, self.domain_extrema_log)[1:]  #+ x_test[:, 12:15] * 0.01
        # x_test[:, 11:15] = self.normalize_input_log(input_log)
        x_test[:, 10] = normalize_A_n(
            torch.tensor([A], dtype=torch.float64, device=self.device), n)

        x_train_ = x_test.clone()

        x_train_.requires_grad = True
        del x_train
        # Denormalization of inputs
        linear_input = denormalize_input_linear(x_train_[:, 1:10],
                                                self.domain_extrema)
        log_input = denormalize_input_log(x_train_[:, 11:15],
                                          self.domain_extrema_log)
        initial_input = denormalize_initial(x_train_[:, 15:19],
                                            self.initial_extrema)

        t = denormalize_t(x_train_[:, 0], self.device)
        E = linear_input[:, 0]
        n = linear_input[:, 1]
        A = denormalize_A_n(x_train_[:, 10].to(torch.float64), n).reshape(-1, )
        C1 = linear_input[:, 2]
        gam1 = linear_input[:, 3]
        kX1 = log_input[:, 1]
        C2 = linear_input[:, 4]
        gam2 = linear_input[:, 5]
        kX2 = log_input[:, 2]
        Q1 = linear_input[:, 7]
        b1 = linear_input[:, 8]
        KR1 = log_input[:, 3]

        x_pre_ini = x_train_[:, :15].clone().detach()
        x_pre_ini[:, 0] = 1.

        sig_ini, R_ini, X1_ini, X2_ini = self.pinn_mono.prediction_end(
            x_pre_ini)
        sig_ini = sig_ini.detach() + initial_input[:, 0]
        R_ini = R_ini.detach() + initial_input[:, 1]
        X1_ini = X1_ini.detach() + initial_input[:, 2]
        X2_ini = X2_ini.detach() + initial_input[:, 3]

        # Calculation of the turning point of elastic domain
        # linear_constraint = (t.reshape(-1, ) / 450.)
        linear_constraint = (x_train_[:, 0] + 1.)
        hard_constraint_linear = torch.cat([
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1),
            linear_constraint.reshape(-1, 1)
        ], 1)

        # Calculation of outputs and denormalization
        y_pred = self.approximate_solution(x_train_) * hard_constraint_linear
        y_pred = denormalize_output(y_pred, self.result_extrema)
        # add hard constraints
        eps_p = y_pred[:, 0].to(torch.float64)
        R = y_pred[:, 1].to(torch.float64)  # * hard_constraint_other
        X1 = y_pred[:, 2].to(torch.float64)  # * hard_constraint_other
        X2 = y_pred[:, 3].to(torch.float64)  # * hard_constraint_other
        del y_pred

        ub = torch.log10(torch.tensor([900. + 1e-3], device=self.device))
        lb = torch.log10(torch.tensor([1e-3], device=self.device))

        eps_p_t = torch.autograd.grad(
            eps_p.sum(), x_train_, create_graph=True)[0][:, 0] * (
                2 / (ub - lb)) / ((t + 1e-3) * np.log(10))
        R_t = torch.autograd.grad(
            R.sum(), x_train_, create_graph=True)[0][:, 0] * (
                2 / (ub - lb)) / ((t + 1e-3) * np.log(10))
        X1_t = torch.autograd.grad(
            X1.sum(), x_train_, create_graph=True)[0][:, 0] * (
                2 / (ub - lb)) / ((t + 1e-3) * np.log(10))
        X2_t = torch.autograd.grad(
            X2.sum(), x_train_, create_graph=True)[0][:, 0] * (
                2 / (ub - lb)) / ((t + 1e-3) * np.log(10))
        # del x_train_
        eps_p_eq_t = abs(eps_p_t)

        X1 = X1 + X1_ini
        X2 = X2 + X2_ini
        R = R + R_ini
        X = X1 + X2
        sig = sig_ini - eps_p * E

        one_tensor = torch.tensor([1.], device=self.device)
        weight = torch.min(t.detach(), one_tensor) * torch.max(
            torch.log(t.detach()), one_tensor
        )  #(t.detach() + 1.)  #(t.detach() + 1.)  #* (10 ** -(log_input.detach()[:, 2] - 1.))  #torch.tensor([1.], device=device)  #
        weight_loss2 = torch.min(t.detach(), one_tensor) * torch.max(
            torch.log(t.detach()), one_tensor
        )  #(t.detach() + 1.) #* (10 ** -(log_input.detach()[:, 1] - 1.))
        weight_loss3 = torch.min(t.detach(), one_tensor) * torch.max(
            torch.log(t.detach()), one_tensor
        )  #(t.detach() + 1.) #* (10 ** (-1.5 * (log_input.detach()[:, 2] - 1.)))
        weight_loss4 = torch.min(t.detach(), one_tensor) * torch.max(
            torch.log(t.detach()), one_tensor
        )  #(t.detach() + 1.) #* (10 ** -(log_input.detach()[:, 3] - 1.))

        # Loss 1
        Smooth_factor = 0.0001
        a1 = sig.reshape(-1, ) - X.reshape(-1, )
        a2 = a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(-1, )
        loss1_right = (A * (0.5 * (a2 + torch.sqrt(a2**2 + Smooth_factor**2)))
                       **n) * torch.tanh(a1 / Smooth_factor)
        loss1_left = eps_p_t.reshape(-1, )

        loss0_vec = torch.max(eps_p_t, torch.tensor([0.], device=self.device))
        loss0 = 1000 * torch.mean(loss0_vec)
        loss0_rec = loss0.detach()

        eps_p_t_PDE_W = self.params['stress_relaxation']['loss_weights'][
            'eps_p_t_PDE_W']
        R_t_PDE_W = self.params['stress_relaxation']['loss_weights'][
            'R_t_PDE_W']
        X1_t_PDE_W = self.params['stress_relaxation']['loss_weights'][
            'X1_t_PDE_W']
        X2_t_PDE_W = self.params['stress_relaxation']['loss_weights'][
            'X2_t_PDE_W']
        p = self.params['stress_relaxation']['power']

        loss1_abs_vec = (loss1_left - loss1_right) / (
            eps_p_t_PDE_W)  #  * ((t + 1e-3) * np.log(10))
        loss1_per_vec = (loss1_left - loss1_right) / (loss1_left + 1e-20)
        loss1_abs_w = loss1_abs_vec * weight.reshape(-1, )
        weight_per = torch.max(
            -torch.log(loss1_left**2 + loss1_right**2 + 1e-20),
            torch.tensor([0.], device=self.device))
        loss1_per_w = loss1_per_vec * weight_per
        loss1_per_w = loss1_per_w / 2000
        loss1_abs = torch.mean(loss1_abs_w**p)
        loss1_per = torch.mean(loss1_per_w**p)

        loss1 = loss1_abs + loss1_per
        loss1_rec = loss1.detach()

        # loss 2
        loss2_left = R_t.reshape(-1, )
        loss2_right = Q1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) - b1.reshape(
            -1, ) * R.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) - KR1.reshape(
                -1, ) * R.reshape(-1, )

        loss2_vec = (loss2_left - loss2_right) / (R_t_PDE_W)
        loss2_w = loss2_vec * weight_loss2.reshape(-1, )
        loss2 = torch.mean(loss2_w**p)
        loss2_rec = loss2.detach()

        # loss 3
        loss3_left = X1_t.reshape(-1, )
        loss3_right = C1.reshape(-1, ) * eps_p_t.reshape(-1, ) - gam1.reshape(
            -1, ) * eps_p_eq_t.reshape(-1, ) * X1.reshape(-1, ) - kX1.reshape(
                -1, ) * X1.reshape(-1, )

        loss3_vec = (loss3_left - loss3_right) / (X1_t_PDE_W)
        loss3_w = loss3_vec * weight_loss3.reshape(-1, )
        loss3 = torch.mean(loss3_w**p)
        loss3_rec = loss3.detach()

        # loss 4
        loss4_left = X2_t.reshape(-1, )
        loss4_right = C2.reshape(-1, ) * eps_p_t.reshape(-1, ) - gam2.reshape(
            -1, ) * eps_p_eq_t.reshape(-1, ) * X2.reshape(-1, ) - kX2.reshape(
                -1, ) * X2.reshape(-1, )

        loss4_vec = (loss4_left - loss4_right) / (X2_t_PDE_W)
        loss4_w = loss4_vec * weight_loss4.reshape(-1, )
        loss4 = torch.mean(loss4_w**p)
        loss4_rec = loss4.detach()
        # print the loss terms
        if verbose:
            print('loss1:',
                  loss1_rec.cpu().detach().item(), 'loss2:',
                  loss2_rec.cpu().detach().item(), 'loss3:',
                  loss3_rec.cpu().detach().item(), 'loss4:',
                  loss4_rec.cpu().detach().item(), 'loss0:',
                  loss0_rec.cpu().detach().item())

        loss = torch.log10(loss1 + loss2 + loss3 + loss4)
        loss_rec = loss.detach()

        # print total loss
        if verbose:
            print('Loss: ', loss_rec.item())
        return loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_int, x_data, y_data, verbose):

        loss_pde, loss_pde_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_pde_residual(
            inp_train_int, verbose, idx_pde=6)

        loss_data = self.compute_data_residual(x_data, y_data)

        loss = loss_pde  #+ loss_pde_2 + loss_pde_3 + loss_pde_4 + loss_pde_5 + loss_pde_6 + loss_pde_7 #+ 1 *
        return loss, loss_data, loss_pde_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec

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

            for j, ((inp_train_int, u_train_int),
                    (x_data, y_data)) in enumerate(
                        zip(self.training_set_int, self.training_set_data)):

                def closure():
                    optimizer.zero_grad()
                    loss, loss_data, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_loss(
                        inp_train_int,
                        x_data,
                        y_data,
                        verbose=verbose,
                    )
                    loss.backward()
                    loss_1.append(loss1_rec.item())
                    loss_2.append(loss2_rec.item())
                    loss_3.append(loss3_rec.item())
                    loss_4.append(loss4_rec.item())
                    loss_0.append(loss0_rec.item())
                    history.append(loss_rec.item())
                    print('Loss_pde: ', history[-1], 'Loss_data',
                          loss_data.item())

                    self.neptune_logger['losses/final_loss'].log(history[-1])
                    self.neptune_logger['losses/loss_0'].log(loss_0[-1])
                    self.neptune_logger['losses/loss_1'].log(loss_1[-1])
                    self.neptune_logger['losses/loss_2'].log(loss_2[-1])
                    self.neptune_logger['losses/loss_3'].log(loss_3[-1])
                    self.neptune_logger['losses/loss_4'].log(loss_4[-1])

                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1], 'loss1:', loss_1[-1], 'loss2:',
              loss_2[-1], 'loss3:', loss_3[-1], 'loss4:', loss_4[-1], 'loss0:',
              loss_0[-1])

        return history, loss_1, loss_2, loss_3, loss_4
