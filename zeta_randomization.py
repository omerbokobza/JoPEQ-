import gc
import sys
from statistics import mean
import time
import torch
from configurations import args_parser
from tqdm import tqdm
import utils
import models
import federated_utils
from torchinfo import summary
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython import display
import torch.nn as nn
from torchmetrics import MeanSquaredError as MSE
from quantization import LatticeQuantization


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_hidden=400):
        super(PyTorchMLP, self).__init__()
        self.layer1 = torch.nn.Linear(100, num_hidden)
        self.layer2 = torch.nn.Linear(num_hidden, 1)
        self.num_hidden = num_hidden
        self.relu = nn.ReLU() ## The Activation FunctionSSSS
    def forward(self, inp):
        inp = inp.reshape([-1, 100])
        first_layer = self.relu(self.layer1(inp))
        return self.relu(self.layer2(first_layer))


def train_model(model, learning_rate=1e-4):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # built-in L2
    # Adam for our parameter updates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # built-in L2
    train_acc = []
    epochs = []
    sum_alpha = 0
    avg_alpha = 0
    vec_alpha = []
    loss_vec = []
    # Training
    for t in range(300):
        # Divide data into mini batches

        for i in range(0, 100):
            # Feed forward to get the logits
            local_weights_orig = torch.rand(100)

            # local_weights_orig = torch.transpose(local_weights_orig, 1, 0)
            alpha = abs(model(local_weights_orig))
            local_weights = federated_utils.JoPEQ(args, alpha)(local_weights_orig)
            # print(type(local_weights), local_weights.shape, type(local_weights_orig), local_weights_orig.shape)
            # Compute the training loss and accuracy
            loss = criterion(local_weights.reshape(-1, 100), local_weights_orig.reshape(-1, 100))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute training accuracy
        sum_alpha += alpha
        avg_alpha = sum_alpha/t
        print("[EPOCH]: %i, [LOSS]: %.6f, [Alpha]: %.3f, [AVG_ALPHA]: %.3f" % (
        t, loss.item(), alpha, avg_alpha))
        display.clear_output(wait=True)

        # Save error on each epoch
        epochs.append(t)
        #train_acc.append(acc)
        vec_alpha.append(avg_alpha)
        loss_vec.append((loss.item()))

    # plotting
    plt.figure()
    plt.title("Average Alpha vs Epochs for R = {fname}, with privacy = {bool}".format(fname = args.R, bool = args.privacy))
    plt.plot(epochs, vec_alpha, label="Average Alpha")
    plt.xlabel("Epoch")
    plt.ylabel("Avg_Alpha")
    plt.legend(loc='best')
    plt.show()

    # plt.figure()
    # plt.title("Average Alpha vs LOSS")
    # plt.plot(loss_vec,vec_alpha)
    # plt.xlabel("LOSS")
    # plt.ylabel("Avg_alpha")
    # plt.show()


def randomize_zeta():
    local_weights_orig = torch.rand((100, 100))
    vec = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]
    for alpha in vec:
        mechanism = federated_utils.JoPEQ(args, alpha)
        local_weights = mechanism(local_weights_orig)
        loss = mean_squared_error(local_weights, local_weights_orig)
        print(f"The loss for alpha = {alpha} is {loss}")


if __name__ == '__main__':
    args = args_parser()
    args.R = 6
    args.privacy = True
    pytorchmlp = PyTorchMLP()
    train_model(pytorchmlp)

    # randomize_zeta()
    print("end of zeta simulation")



