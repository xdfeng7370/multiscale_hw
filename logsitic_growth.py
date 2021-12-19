# test averaging
# original equation dx/dt = epsilon(x(1-x)+sin t)
# averaging equation dy/dt = epsilon(y(1-y))

import torch
import torch.nn as nn
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from math import pi
import numpy as np
from scipy.integrate import odeint
import math

class Block(nn.Module):
    """
    Implementation of the block used in the Deep Ritz
    Paper
    Parameters:
    in_N  -- dimension of the input
    width -- number of nodes in the interior middle layer
    out_N -- dimension of the output
    phi   -- activation function used
    """

    def __init__(self, in_N, width, out_N):
        super(Block, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x))))
        # return self.phi(self.L2(self.phi(self.L1(x)))) + x


class drrnn(nn.Module):
    """
    drrnn -- Deep Ritz Residual Neural Network
    Implements a network with the architecture used in the
    deep ritz method paper
    Parameters:
        in_N  -- input dimension
        out_N -- output dimension
        m     -- width of layers that form blocks
        depth -- number of blocks to be stacked
        phi   -- the activation function
    """

    def __init__(self, in_N, m, out_N, depth=2):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()
        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                         grad_outputs=torch.ones_like(output),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]


def function_origin(x, t):
    return epsilon * (x * (1-x) + math.sin(t))


def function_averaging(x, t):
    return epsilon * (x * (1-x))


def main():
    global epsilon
    epsilon = 0.05

    epochs = 4000
    in_N = 1
    m = 100
    out_N = 1

    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    model_aver = drrnn(in_N, m, out_N).to(device)
    optimizer_aver = optim.Adam(model_aver.parameters(), lr=1e-3)
    StepLR_aver = torch.optim.lr_scheduler.StepLR(optimizer_aver, step_size=1000, gamma=0.5)

    loss_log = []
    loss_log_aver = []
    tt = time.time()
    for epoch in range(epochs+1):
        xr = torch.rand(128, 1) * 80
        xb = torch.tensor([[0.]])
        xr = xr.to(device)
        xb = xb.to(device)
        # generate the data set
        xr.requires_grad_()
        output_r = model(xr)
        output_b = model(xb)
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss_r = torch.mean(torch.square(grads - epsilon * (output_r * (1-output_r) + torch.sin(xr))))
        loss_b = 100 * torch.mean(torch.square(output_b - 2.))
        loss = loss_r + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()
        loss_log.append(loss.item())

        output_r_aver = model_aver(xr)
        output_b_aver = model_aver(xb)
        grads_aver = autograd.grad(outputs=output_r_aver, inputs=xr,
                                   grad_outputs=torch.ones_like(output_r_aver),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss_r_aver = torch.mean(torch.square(grads_aver - epsilon * output_r_aver * (1-output_r_aver)))
        loss_b_aver = 100 * torch.mean(torch.square(output_b_aver - 2.))
        loss_aver = loss_r_aver + loss_b_aver

        optimizer_aver.zero_grad()
        loss_aver.backward()
        optimizer_aver.step()
        StepLR_aver.step()
        loss_log_aver.append(loss_aver.item())

        if epoch % 100 == 0:
            print('epoch: {:d}, loss: {:.4e}, loss_aver: {:.4e},time: {:.4e}'.format(epoch, loss.item(), loss_aver.item(), time.time()-tt))
            tt = time.time()

    with torch.no_grad():
        N0 = 1000
        x_test = torch.linspace(0, 80, N0 + 1).reshape(-1, 1)
        x_pred = model(x_test).detach()
        x_pred_aver = model_aver(x_test).detach()
        reference_origin = odeint(function_origin, torch.tensor([2.]), x_test.flatten())
        reference_aver = odeint(function_averaging, torch.tensor([2.]), x_test.flatten())

    plt.figure()
    plt.plot(x_test, x_pred, '--r', label='origin predicted')
    plt.plot(x_test, reference_origin, '-k', label='origin reference')
    plt.title('$\epsilon=$'+str(epsilon))
    plt.tight_layout()
    plt.legend()
    plt.savefig('logistic_growth_origin.pdf')
    plt.figure()
    plt.plot(x_test, x_pred_aver, '--r', label='averaging predicted')
    plt.plot(x_test, reference_aver, '-k', label='averaging reference')
    plt.title('$\epsilon=$'+str(epsilon))
    plt.tight_layout()
    plt.legend()
    plt.savefig('logistic_growth_aver.pdf')

    plt.figure()
    plt.plot(x_test, x_pred, '--r', label='origin predicted')
    plt.plot(x_test, x_pred_aver, '-k', label='averaging predicted')
    plt.legend()
    plt.title('$\epsilon=$'+str(epsilon))
    plt.tight_layout()
    plt.savefig('logistic_growth_aver_orig.pdf')

if __name__ == '__main__':
    main()
