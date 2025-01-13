# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:41:51 2023

@author: Haotian
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from models.network import NeuralNet
from utils.normalization import denormalize_output, normalize_input_linear
from utils.normalization import denormalize_input_linear, normalize_input_log
from utils.normalization import denormalize_input_log, normalize_A_n, denormalize_A_n

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)



class Monotonic:
    """Monotonic PINN class"""

    def __init__(self, params, neptune_logger=None):
        self.params = params
        if neptune_logger is not None:
            self.neptune_logger = neptune_logger
        else:
            self.neptune_logger = None
        self.device = torch.device(self.params['common_parameters']['device'])
        self.num_interior_points = self.params['monotonic'][
            'num_interior_points']

        self.domain_extrema = (
            self.params['monotonic']['data']['domain_extrema']).to(self.device)
        self.domain_extrema_log = self.params['monotonic']['data'][
            'domain_extrema_log'].to(self.device)
        self.result_extrema = self.params['monotonic']['data'][
            'result_extrema'].to(self.device)

        neural_network_params = self.params['common_parameters'][
            'neural_network']

        self.approximate_solution = NeuralNet(
            input_dimension=self.params['monotonic']['input_dimension'],
            output_dimension=self.params['monotonic']['output_dimension'],
            n_hidden_layers=neural_network_params['n_hidden_layers'],
            neurons=neural_network_params['neurons'],
            regularization_param=neural_network_params['regularization'],
            regularization_exp=neural_network_params['regularization_exp'],
            retrain_seed=neural_network_params['retrain_seed']).to(self.device)

        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.params['monotonic']['input_dimension'])

        self.training_set_int = self.assemble_datasets()

        print('Monotonic PINN initialized')

    def load_model(self):
        model_name = self.params['monotonic']['load_model_name']
        model_save_path = self.params['common_parameters']['model_save_path']
        full_model_path = os.path.join(model_save_path, model_name)
        self.approximate_solution.load_state_dict(torch.load(full_model_path))

        print('Model loaded successfully')

    def save_model(self):
        neural_network_params = self.params['common_parameters'][
            'neural_network']
        model_name = (f"monotonic_{self.params['current_time']}_"
                      f"{neural_network_params['n_hidden_layers']}_"
                      f"{neural_network_params['neurons']}.pth")
        model_save_path = self.params['common_parameters']['model_save_path']
        full_model_path = os.path.join(model_save_path, model_name)
        torch.save(self.approximate_solution.state_dict(), full_model_path)

        print('Model saved successfully')

    def predict(self, x_test):
        return self.approximate_solution.forward(x_test)

    def add_interior_points(self):
        input_int = (self.soboleng.draw(self.num_interior_points).to(
            self.device)) * 2 - 1
        output_int = torch.zeros((input_int.shape[0], 1)).to(self.device)
        return input_int, output_int

    def assemble_datasets(self):
        input_int, output_int = self.add_interior_points()
        training_set_int = DataLoader(torch.utils.data.TensorDataset(
            input_int, output_int),
                                      batch_size=self.num_interior_points,
                                      shuffle=False)
        return training_set_int

    def compute_pde_residual(self, x_train_, verbose):
        # weight for each loss term

        eps_p_t_PDE_W = self.params['monotonic']['loss_weights'][
            'eps_p_t_PDE_W']
        R_t_PDE_W = self.params['monotonic']['loss_weights']['R_t_PDE_W']
        X1_t_PDE_W = self.params['monotonic']['loss_weights']['X1_t_PDE_W']
        X2_t_PDE_W = self.params['monotonic']['loss_weights']['X2_t_PDE_W']
        p = self.params['monotonic']['power']

        x_train_.requires_grad = True

        # Denormalization of inputs
        linear_input = denormalize_input_linear(x_train_[:, :10],
                                                self.domain_extrema)
        log_input = denormalize_input_log(x_train_[:, 11:15],
                                          self.domain_extrema_log)

        eps_tot = linear_input[:, 0]
        E = linear_input[:, 1]
        n = linear_input[:, 2]
        A = denormalize_A_n(x_train_[:, 10].to(torch.float64), n).reshape(-1, )
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

        del linear_input, log_input

        # Calculation of the turning point of elastic domain
        linear_constraint = (
            eps_tot.reshape(-1, ) /
            (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        t = eps_tot / eps_tot_rate
        sigma_el = (eps_tot) * E
        R_el = R0 * torch.exp(-KR1 * t)
        X1_el = 0.  #X10 * torch.exp(- kX1 * t)
        X2_el = 0.  #X20 * torch.exp(- kX2 * t)
        X_el = X1_el + X2_el
        a0 = sigma_el - X_el
        a00 = abs(a0) - R_el
        INP = (0.5 *
               (a00 + torch.sqrt(a00**2))) / (self.domain_extrema[0, 1] * E)
        hard_constraint_eps_p = 2 * (torch.tanh(
            2 * INP)**2)  #* torch.tanh(100000 * linear_constraint)
        hard_constraint_other = linear_constraint  #* (torch.tanh(2 * INP) ** 2 + 0.001)

        # Calculation of outputs and denormalization
        y_pred = denormalize_output(
            self.approximate_solution.forward(x_train_), self.result_extrema)

        # add hard constraints
        eps_p = y_pred[:, 0].to(torch.float64) * hard_constraint_eps_p
        eps_p = eps_p
        R = y_pred[:, 1].to(torch.float64) * hard_constraint_other
        X1 = y_pred[:, 2].to(torch.float64) * hard_constraint_other
        X2 = y_pred[:, 3].to(torch.float64) * hard_constraint_other
        del y_pred
        X1 = X1
        X2 = X2
        R = R + R0

        X = X1 + X2

        sig = (eps_tot - eps_p) * E

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
        '''
        loss0_vec = eps_tot.reshape(-1, ) - eps_p.reshape(-1, ) - 0.002
        loss0_vec = 100000000 * 0.5 * (loss0_vec + torch.sqrt(loss0_vec ** 2))
        '''
        loss0_vec = 1000 * ((-X1_t + torch.sqrt(X1_t**2)) +
                            (-X2_t + torch.sqrt(X2_t**2)))
        loss0 = torch.mean(loss0_vec**p)
        loss0_rec = loss0.detach()

        # to put higher weight for small strains
        weight_vec = 1. / (
            0.5 + (0.5 * eps_tot.reshape(-1, )) /
            (self.domain_extrema[0, 1] - self.domain_extrema[0, 0])
        )  # * (t.reshape(-1, ) + 1.)
        '''
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(eps_tot.cpu().detach().numpy(), weight_vec.reshape(-1, 1).cpu().detach().numpy(), s=0.5,
                    label="Weight")
        plt.legend()
        plt.show(block=True)
        '''
        # weight = 1 + eps_tot.detach() / (self.domain_extrema[0, 1] - self.domain_extrema[0, 0])
        weight = weight_vec.detach()
        weight_loss3 = weight  # * (C1.detach().mean()/gam1.detach().mean()) / (C1.detach()/gam1.detach())
        weight_loss4 = weight  # * (C2.detach().mean()/gam2.detach().mean()) / (C2.detach()/gam2.detach())

        # Loss 1
        Smooth_factor = 0.001
        a1 = sig.reshape(-1, ) - X.reshape(-1, )
        a2 = a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(
            -1,
        )  # a1 * torch.tanh(a1 / Smooth_factor) - R.reshape(-1, )   # what is the range of overstress
        a3 = A * (0.5 *
                  (a2 + torch.sqrt(a2**2)))**n * torch.tanh(a1 / Smooth_factor)

        # loss0 = loss0 + 0.0005 / (0.000005 + torch.mean(a3))
        loss0_rec = loss0.detach()

        # make it such that after the training, it prints the individual values of loss0,loss n and max/min for ANN outputs
        # also in slides, talk about switching to hard constrain for initial BC
        # loss1_vec = (torch.log10(1 + abs(eps_p_t.reshape(-1, ))) - torch.log10(1 + a3)) * weight.reshape(-1, ) / eps_p_t_PDE_W
        loss1_vec = (eps_p_t.reshape(-1, ) - a3) * weight.reshape(
            -1, ) / eps_p_t_PDE_W
        # to damp the effect of big loss values at the begining of training
        loss1 = torch.mean(loss1_vec**p)
        loss1_rec = loss1.detach()

        # loss 2
        loss2_vec = (
            R_t.reshape(-1, ) - Q1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) +
            b1.reshape(-1, ) * R.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) +
            KR1.reshape(-1, ) * R.reshape(-1, )) / R_t_PDE_W
        loss2_w = loss2_vec * weight.reshape(-1, )
        loss2 = torch.mean(loss2_w**p)
        loss2_rec = loss2.detach()

        # loss 3
        loss3_vec = (
            X1_t.reshape(-1, ) - C1.reshape(-1, ) * eps_p_t.reshape(-1, ) +
            gam1.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) * X1.reshape(-1, ) +
            kX1.reshape(-1, ) * X1.reshape(-1, )) / X1_t_PDE_W
        loss3_w = loss3_vec * weight_loss3.reshape(-1, )
        # a = loss3_w ** p
        loss3 = torch.mean(loss3_w**p)
        loss3_rec = loss3.detach()

        # loss 4
        loss4_vec = (
            X2_t.reshape(-1, ) - C2.reshape(-1, ) * eps_p_t.reshape(-1, ) +
            gam2.reshape(-1, ) * eps_p_eq_t.reshape(-1, ) * X2.reshape(-1, ) +
            kX2.reshape(-1, ) * X2.reshape(-1, )) / X2_t_PDE_W
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

        # loss = ((loss1 + loss2 + loss3 + loss4 + loss0) / 5) # check the log function here
        # loss = (loss1 + loss2 + loss3 + loss4 + loss0) / 5  # check the log function here
        loss = torch.log10(loss1 + loss2 + loss3 +
                           loss4)  # check the log function here
        # loss = (loss1 + loss2 + loss3 + loss4) / 4  # check the log function here
        loss_rec = loss.detach()

        # print total loss
        if verbose:
            print('Loss: ', loss_rec.item())
        return loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec

    def prediction_end(self, x_train, kX1, kX2, KR1):
        linear_input = denormalize_input_linear(x_train[:, :10],
                                                self.domain_extrema)
        log_input = denormalize_input_log(x_train[:, 11:15],
                                          self.domain_extrema_log)
        log_input[:, 1] = kX1
        log_input[:, 2] = kX2
        log_input[:, 3] = KR1
        x_train[:, 11:15] = normalize_input_log(log_input,
                                                self.domain_extrema_log)

        E = linear_input[:, 1]

        R0 = linear_input[:, 7].to(torch.float64)
        eps_tot_rate = log_input[:, 0]

        X10 = torch.tensor([0.], device=self.device)
        X20 = torch.tensor([0.], device=self.device)

        eps_tot = self.domain_extrema[0, 1]
        linear_constraint = (
            eps_tot.reshape(-1, ) /
            (self.domain_extrema[0, 1] - self.domain_extrema[0, 0]))
        t = eps_tot / eps_tot_rate
        sigma_el = (eps_tot) * E
        R_el = R0 * torch.exp(-KR1 * t)
        X1_el = 0.
        X2_el = 0.
        X_el = X1_el + X2_el
        a0 = sigma_el - X_el
        a00 = abs(a0) - R_el
        INP = (0.5 *
               (a00 + torch.sqrt(a00**2))) / (self.domain_extrema[0, 1] * E)
        hard_constraint_eps_p = 2 * (torch.tanh(2 * INP)**2)
        hard_constraint_other = linear_constraint

        y_pred = denormalize_output(self.approximate_solution(x_train),
                                    self.result_extrema)

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

    def compute_loss(self, inp_train_int, verbose):
        loss_pde, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec = self.compute_pde_residual(
            inp_train_int, verbose)
        loss = loss_pde
        return loss, loss_rec, loss1_rec, loss2_rec, loss3_rec, loss4_rec, loss0_rec

    def fit(self, num_epochs, optimizer, verbose):
        history = []
        loss_1 = []
        loss_2 = []
        loss_3 = []
        loss_4 = []
        loss_0 = []
        print('Start training')
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

                    # Plot to Neptune monotonic project
                    self.neptune_logger["losses/final_loss"].log(history[-1])
                    self.neptune_logger["losses/loss0"].log(loss_0[-1])
                    self.neptune_logger["losses/loss1"].log(loss_1[-1])
                    self.neptune_logger["losses/loss2"].log(loss_2[-1])
                    self.neptune_logger["losses/loss3"].log(loss_3[-1])
                    self.neptune_logger["losses/loss4"].log(loss_4[-1])

                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1], 'loss1:', loss_1[-1], 'loss2:',
              loss_2[-1], 'loss3:', loss_3[-1], 'loss4:', loss_4[-1], 'loss0:',
              loss_0[-1])

        return history, loss_1, loss_2, loss_3, loss_4
