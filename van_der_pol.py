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
    x1, x2 = x
    return [x2, -x1 - epsilon * (1-x1**2) * x2]


def function_averaging(x, t):
    x1, x2 = x
    return [0.5*epsilon*x1*(1-0.25*x1**2), 0.]


def main():
    global epsilon
    epsilon = 0.05

    epochs = 40000
    in_N = 1
    m = 100
    out_N_origin = 2
    out_N_aver = 1
    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N_origin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    model_aver = drrnn(in_N, m, out_N_aver).to(device)
    optimizer_aver = optim.Adam(model_aver.parameters(), lr=1e-3)
    StepLR_aver = torch.optim.lr_scheduler.StepLR(optimizer_aver, step_size=1000, gamma=0.9)

    loss_log = []
    loss_log_aver = []
    tt = time.time()
    for epoch in range(epochs+1):
        tr = torch.rand(128, 1) * 20
        tb = torch.tensor([[0.]])
        tr = tr.to(device)
        tb = tb.to(device)
        # generate the data set
        tr.requires_grad_()
        tb.requires_grad_()
        output_r = model(tr)[:, 0:1]
        output_r_grad = model(tr)[:, 1:2]
        output_b = model(tb)
        grads = autograd.grad(outputs=output_r, inputs=tr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        grads2 = autograd.grad(outputs=output_r_grad, inputs=tr,
                               grad_outputs=torch.ones_like(output_r_grad),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss_r = torch.mean(torch.square(grads2 + epsilon * (1-output_r ** 2) * output_r_grad + output_r) + torch.square(grads - output_r_grad))
        loss_b = 100 * torch.mean(torch.square(output_b[:, 0:1] - 0.)+torch.square(output_b[:, 1:2] - 1.))
        loss = loss_r + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()
        loss_log.append(loss.item())

        output_r_aver = model_aver(tr)
        output_b_aver = model_aver(tb)
        grads_aver = autograd.grad(outputs=output_r_aver, inputs=tr,
                                   grad_outputs=torch.ones_like(output_r_aver),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss_r_aver = torch.mean(torch.square(grads_aver - 0.5 * epsilon * output_r_aver * (1-0.25*output_r_aver**2)))
        loss_b_aver = 100 * torch.mean(torch.square(output_b_aver - 1.))
        loss_aver = loss_r_aver + loss_b_aver

        optimizer_aver.zero_grad()
        loss_aver.backward()
        optimizer_aver.step()
        StepLR_aver.step()
        loss_log_aver.append(loss_aver.item())

        if epoch % 100 == 0:
            print('epoch: {:d}, loss: {:.4e}, loss_aver: {:.4e},time: {:.4e}'.format(epoch, loss.item(), loss_aver.item(), time.time()-tt))
            tt = time.time()

    t = torch.linspace(0, 20*pi, 10001).reshape(-1, 1)
    t.requires_grad_()
    output = model(t)
    pred_origin1, pred_origin2 = output[:, 0:1], output[:, 1:2]
    plt.figure()
    plt.plot(pred_origin1.detach(), pred_origin2.detach(), label='origin pred')
    plt.tight_layout()
    plt.xlabel('$z$')
    plt.ylabel('$\dot{z}$')
    plt.legend()
    plt.savefig('pred_origin.pdf')

    pred_aver_r = model_aver(t).detach()
    t = t.detach()
    plt.figure()
    plt.plot(pred_aver_r * torch.sin(t.reshape(-1, 1)), pred_aver_r * torch.cos(t.reshape(-1, 1)), label='aver pred')
    plt.tight_layout()
    plt.xlabel('$z$')
    plt.ylabel('$\dot{z}$')
    plt.legend()
    plt.savefig('pred_aver.pdf')

    t = torch.linspace(0, 20 * pi, 10001)
    initial_state = torch.tensor([0., 1.])
    initial_aver = [torch.linalg.norm(initial_state), torch.acos(initial_state[1]/torch.linalg.norm(initial_state))]
    reference_origin = odeint(function_origin, initial_state, t)
    reference_aver = odeint(function_averaging, initial_aver, t)

    plt.figure()
    plt.scatter(initial_state[0], initial_state[1], s=25)

    plt.plot(reference_origin[:, 0:1], reference_origin[:, 1:2], label='reference origin')
    plt.plot(torch.from_numpy(reference_aver[:, 0:1]) * torch.sin(t.reshape(-1, 1)), torch.from_numpy(reference_aver[:, 0:1]) * torch.cos(t.reshape(-1, 1)), label='reference aver')
    plt.title('$\epsilon$='+str(epsilon))
    plt.legend()
    plt.tight_layout()

    plt.ylabel('$\dot{z}$')
    plt.xlabel('$z$')
    plt.savefig('reference.pdf')
if __name__ == '__main__':
    main()
