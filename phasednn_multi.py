# test equation
# u'' - \lambda ^2 u = -(lambda^2+ mu^2)sin(mu * x)
# u(-1)=u(1)=0
# exact solution: u(x) = sin(mu*x)-sin(mu)/sinh(lambda)*sinh(lambda*x)
# test standard fully-connected neural network

import torch
import torch.nn as nn
from torch import optim, autograd
from matplotlib import pyplot as plt
import time
import math
from math import pi
import numpy as np

lamb = 3
mu = 250


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


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def function_u_exact(x):
    return - math.sin(mu) / math.sinh(lamb) * torch.sinh(lamb * x) + torch.sin(mu * x)


def function_f(x):
    return -(lamb ** 2 + mu ** 2) * torch.sin(mu * x)


def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                         grad_outputs=torch.ones_like(output),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]


def main():

    epochs = 10000
    in_N = 1
    m = 100
    out_N = 1

    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    print(model)

    loss_log = []
    tt = time.time()
    for epoch in range(epochs+1):
        xr = torch.rand(128, 1) * 2. - 1.
        xb = torch.where(torch.rand(128, 1) > 0.5, torch.ones(128, 1), - torch.ones(128, 1))
        xr = xr.to(device)
        xb = xb.to(device)
        # generate the data set
        xr.requires_grad_()
        output_r = model(xr)
        output_b = model(xb)
        exact_b = function_u_exact(xb)
        f_value = function_f(xr)
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        grads2 = gradients(xr, grads[:, 0: 1])[:, 0: 1]
        loss_r = torch.mean(torch.square(grads2 - lamb ** 2 * output_r - f_value))
        loss_r = torch.mean(loss_r)
        loss_b = 500 * torch.mean(torch.abs(output_b - exact_b))
        loss = loss_r + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()
        loss_log.append(loss.item())
        if epoch % 100 == 0:
            print('epoch: {:d}, loss: {:.4e}, loss_r: {:.4e}, loss_b: {:.4e}, time: {:.4e}'.format(epoch, loss.item(), loss_r.item(), loss_b.item(), time.time()-tt))
            tt = time.time()
    torch.save(loss_log, 'loss_log')
    path = 'ex.pth'
    torch.save(model.state_dict(), path)
    with torch.no_grad():
        N0 = 1000
        N_test = 10000
        x_test = torch.linspace(0, 1, N_test + 1).reshape(-1, 1)
        x_pred = model(x_test).detach()
        x_exact = function_u_exact(x_test)
        err = torch.linalg.norm(x_pred - x_exact) / torch.linalg.norm(x_pred)
        x_data = torch.linspace(0, 1, N0 + 1).reshape(-1, 1)
        x_data = x_data.to(device)
        pred = model(x_data).detach()
        exact = function_u_exact(x_data)
    print('L2 err: {:.4e}'.format(err))
    plt.figure()
    plt.plot(x_data, pred, '--r', label='pred')
    plt.plot(x_data, exact, '-k', label='exact')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pred_result.pdf')

    plt.figure()
    plt.plot(x_data, torch.abs(pred - exact), label='abs err')
    plt.legend()
    plt.tight_layout()
    plt.savefig('abs.pdf')

    plt.figure()
    plt.plot(loss_log)
    plt.legend(['loss'])
    plt.semilogy()
    plt.tight_layout()
    plt.savefig('loss.pdf')
    plt.show()


if __name__ == '__main__':
    main()
