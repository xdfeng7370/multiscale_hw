# poisson equation
# \delta u(x) = f(x)
# u(0)=u(1)=0
# exact solution: u(x) = sin(2 * pi * x) + 0.1 * sin(50 * pi * x)
# test fourier feature fully-connected neural network

import torch
import torch.nn as nn
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from math import pi
import numpy as np

beta = 50


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

    def __init__(self, in_N, m, out_N, depth=1):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()
        self.register_buffer('B1', torch.randn(int(m/2), 1) * 10.)
        self.register_buffer('B2', torch.randn(int(m - int(m/2)), 1) * 1.)
        # list for holding all the blocks
        self.stack1 = nn.ModuleList()
        self.stack2 = nn.ModuleList()
        assert m % 2 == 0, 'm must be even!'
        # add middle blocks to list
        for i in range(depth):
            self.stack1.append(Block(m, m, m))
            self.stack2.append(Block(m, m, m))

        # add output linear layer
        self.fc = nn.Linear(int(2*m), out_N)

    def forward(self, x):
        # first layer
        y1 = torch.cat((torch.sin(torch.matmul(x, self.B1.T)*2*pi), torch.cos(2*pi * torch.matmul(x, self.B1.T))), -1)
        x1 = self.stack1[0](y1)
        y2 = torch.cat((torch.sin(torch.matmul(x, self.B2.T)*2*pi), torch.cos(2*pi * torch.matmul(x, self.B2.T))), -1)
        x2 = self.stack2[0](y2)
        x = self.fc(torch.cat((x1, x2), -1))
        return x


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def function_u_exact(x):
    return torch.sin(2 * pi * x) + 0.1 * torch.sin(beta * pi * x)


def function_f(x):
    return -(4*pi**2)*torch.sin(2*pi*x) - 0.1 * (beta ** 2 * pi ** 2) * torch.sin(beta * pi * x)


def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                         grad_outputs=torch.ones_like(output),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]


def main():

    epochs = 10000
    in_N = 1
    m = 200
    out_N = 1

    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N).to(device)
    # model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    print(model)

    loss_log_multi = []
    tt = time.time()
    for epoch in range(epochs+1):
        xr = torch.rand(128, 1)
        xb = torch.where(torch.rand(128, 1) > 0.5, torch.ones(128, 1), torch.zeros(128, 1))
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
        loss_r = torch.mean(torch.square(grads2 - f_value))
        loss_b = 500 * torch.mean(torch.square(output_b - exact_b))
        loss = loss_r + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()
        loss_log_multi.append(loss.item())
        if epoch % 100 == 0:
            print('epoch: {:d}, loss: {:.4e}, loss_r: {:.4e}, loss_b: {:.4e}, time: {:.4e}'.format(epoch, loss.item(), loss_r.item(), loss_b.item(), time.time()-tt))
            tt = time.time()
    torch.save(loss_log_multi, 'loss_log_multi')
    PATH = 'poisson_multi.pth'
    torch.save(model.state_dict(), PATH)
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
    print('Multi L2 err: {:.4e}'.format(err))
    plt.figure()
    plt.plot(x_data, pred, '--r', label='pred')
    plt.plot(x_data, exact, '-k', label='exact')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pred_result_multi.pdf')

    plt.figure()
    plt.plot(x_data, torch.abs(pred - exact), label='abs err')
    plt.legend()
    plt.tight_layout()
    plt.savefig('abs_multi.pdf')

    plt.figure()
    plt.plot(loss_log_multi)
    plt.legend(['multi loss'])
    plt.semilogy()
    plt.tight_layout()
    plt.savefig('multi_loss.pdf')

    plt.show()


if __name__ == '__main__':
    main()
