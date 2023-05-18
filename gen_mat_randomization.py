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


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_hidden1=400, num_hidden2 = 300, num_hidden3 = 500, output_dim = 2):
        super(PyTorchMLP, self).__init__()
        self.output_dim = output_dim
        self.layer1 = torch.nn.Linear(100, num_hidden1)
        self.layer2 = torch.nn.Linear(num_hidden1, num_hidden2)
        self.layer3 = torch.nn.Linear(num_hidden2, num_hidden3)
        self.layer4 = torch.nn.Linear(num_hidden3, output_dim**2)
        self.relu = nn.LeakyReLU() ## The Activation FunctionSSSS
        self.sigmoid = torch.tanh #nn.Sigmoid()
    def forward(self, inp):
        inp = inp.reshape([-1, 100])
        first_layer = self.relu(self.layer1(inp))
        second_layer = self.relu(self.layer2(first_layer))
        third_layer = self.relu(self.layer3(second_layer))
        forth_layer = self.sigmoid(self.layer4(third_layer))
        return torch.reshape(forth_layer, [self.output_dim, self.output_dim])

def train_model(model, learning_rate=1e-4 , dim = 2):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # built-in L2
    # Adam for our parameter updates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # built-in L2
    train_acc = []
    epochs = []
    sum_hex = torch.zeros([dim, dim])
    # avg_hex = torch.zeros(self)
    vec_alpha = []

    loss_vec = []
    # Training
    for t in range(1 , 120):
        # Divide data into mini batches

        for i in range(0, 100):
            # Feed forward to get the logits
            local_weights_orig = torch.rand(100)

            hex_mat = model(local_weights_orig)
            alpha = 1 ### change it #####
            # hex_mat = torch.reshape(hex_mat, [2, 2])
            # hex_mat = hex_mat.detach().numpy()
            mechanism = federated_utils.JoPEQ(args, alpha, hex_mat)
            local_weights = mechanism(local_weights_orig)
            # print(local_weights.detach())
            local_weights.requires_grad_(requires_grad=True)
            local_weights_orig.requires_grad_(requires_grad=True)
            # Compute the training loss and accuracy
            if type(mechanism) == federated_utils.JoPEQ:
                overloading = mechanism.quantizer.print_overloading_vec()
            loss = criterion(local_weights.reshape(-1, 100), local_weights_orig.reshape(-1, 100)) #+ overloading

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute training accuracy
        overloading = 0
        sum_hex += hex_mat
        avg_hex = sum_hex/t
        avg_alpha = alpha
        print("[EPOCH]: %i, [LOSS]: %.6f, [Alpha]: %.3f, [AVG_ALPHA]: %.3f, [Overloading]: %.3f, [Avg_hex_mat]:" % (
        t, loss.item(), alpha, avg_alpha, overloading))
        print(avg_hex) #/torch.linalg.det(hex_mat).to(torch.float32).to(args.device) + 0.00001)#?
        display.clear_output(wait=True)

        # Save error on each epoch
        epochs.append(t)
        #train_acc.append(acc)
        vec_alpha.append(avg_alpha)
        loss_vec.append((loss.item()))

    # plotting

    plt.figure()
    plt.title("Loss vs Epochs")
    plt.plot(epochs, loss_vec, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    # plt.figure()
    # plt.title("Average Alpha vs LOSS")
    # plt.plot(loss_vec,vec_alpha)
    # plt.xlabel("LOSS")
    # plt.ylabel("Avg_alpha")
    # plt.show()

if __name__ == '__main__':
    args = args_parser()
    args.privacy = False
    args.R = 4
    args.lattice_dim = 4
    pytorchmlp11 = PyTorchMLP(output_dim=args.lattice_dim)
    train_model(pytorchmlp11,dim=args.lattice_dim)

    # randomize_zeta()
    print("end of zeta simulation")

##     !!!!!!!!!!!!!!!!   if we run this file, we have to remove the hex_mat line in init (JOPEQ) !!!!!!!!!!!!              ##


