""" Neural network model for the solution of the PDE """
import os
import sys
import torch.nn
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(78)

# pylint: disable=E1101


class NeuralNet(torch.nn.Module):
    """Class for the neural network module."""

    def __init__(self, input_dimension, output_dimension, n_hidden_layers,
                 neurons, regularization_param, regularization_exp,
                 retrain_seed):

        super().__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = torch.nn.Tanh()  #nn.ReLU() #nn.Tanh()
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()
        # self.init_zeros()

        self.input_layer = torch.nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(self.neurons, self.neurons)
            for _ in range(n_hidden_layers)
        ])
        self.output_layer = torch.nn.Linear(self.neurons,
                                            self.output_dimension)

    def forward(self, x):
        """ The forward function performs the set of affine 
        and non-linear transformations defining the network 
        (see equation above)"""

        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        """ Initialize the weights of the network 
        using the Xavier initialization scheme."""
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(
                    m
            ) == torch.nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = torch.nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        """ Compute the regularization term of the loss function."""
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param,
                                                 self.regularization_exp)
        return self.regularization_param * reg_loss


def fit(model, training_set, num_epochs, optimizer, p, verbose=True):
    history = []

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose:
            print("################################ ", epoch,
                  " ################################")

        running_loss = list([0])

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):

            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                u_pred_ = model(x_train_)
                # Item 1. below
                loss = torch.mean(
                    (u_pred_.reshape(-1, ) - u_train_.reshape(-1, ))**
                    p) + model.regularization()
                # Item 2. below
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item()
                return loss

            # Item 3. below
            optimizer.step(closure=closure)

        if verbose:
            print('Loss: ', (running_loss[0] / len(training_set)))

        history.append(running_loss[0])

    return history
