# heat equation
# \delta u(x) / (500 * pi ** 2) = u_t
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

sigma1 = 1.
sigma2 = 30.
beta = 50.


def plot_heat(value, title, extent=np.array([0, 1, 0, 1])):
    if torch.is_tensor(value):
        value = value.detach().numpy()

    plt.figure()
    ax = plt.subplot()
    im = ax.imshow(value, extent=extent)
    ax.set_title(title, fontsize=16)
    ax.invert_yaxis()
    # label_x = [i for i in range(5)] * (extent[1] - extent[0]) + extent[0]
    # label_y = [i for i in range(5)] * (extent[3] - extent[2]) + extent[2]
    plt.xticks([extent[0], (extent[0]+extent[1])/2, extent[1]], fontsize=16)
    plt.yticks([extent[2], (extent[2]+extent[3])/2, extent[3]], fontsize=16)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    cb = plt.colorbar(im, cax=cax)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.tick_params(labelsize=16)
    cb.ax.yaxis.get_offset_text().set_fontsize(16)
    cb.update_ticks()
    plt.tight_layout()


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
        self.L2 = nn.Linear(width, width)
        self.L3 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L3(self.phi(self.L2(self.phi(self.L1(x))))))
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

    def __init__(self, in_N, m, depth=1):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m

        self.depth = depth
        self.phi = nn.Tanh()
        self.register_buffer('B1', torch.randn(int(m/2), 1) * sigma1)
        self.register_buffer('B2', torch.randn(int(m - int(m/2)), 1) * sigma2)
        # list for holding all the blocks
        self.stack1 = nn.ModuleList()
        self.stack2 = nn.ModuleList()
        assert m % 2 == 0, 'm must be even!'
        # add middle blocks to list
        for i in range(depth):
            self.stack1.append(Block(m, m, m))
            self.stack2.append(Block(m, m, m))

    def forward(self, x):
        # first layer
        y1 = torch.cat((torch.sin(torch.matmul(x, self.B1.T)*2*pi), torch.cos(2*pi * torch.matmul(x, self.B1.T))), -1)
        x1 = self.stack1[0](y1)
        y2 = torch.cat((torch.sin(torch.matmul(x, self.B2.T)*2*pi), torch.cos(2*pi * torch.matmul(x, self.B2.T))), -1)
        x2 = self.stack2[0](y2)
        return x1, x2


class Model(nn.Module):
    def __init__(self, in_N=1, m=40, depth=1):
        super(Model, self).__init__()
        self.drnn1 = drrnn(in_N, m, depth)
        self.drnn2 = drrnn(in_N, m, depth)
        self.fc = nn.Linear(int(4*m**2), 1)
        self.m = m

    def forward(self, x, t):
        x1, x2 = self.drnn1(x)
        t1, t2 = self.drnn2(t)
        value1 = (torch.matmul(x1[:, :, None], t1[:, None, :])).reshape(-1, int(self.m**2))
        value2 = (torch.matmul(x2[:, :, None], t1[:, None, :])).reshape(-1, int(self.m**2))

        value3 = (torch.matmul(x1[:, :, None], t2[:, None, :])).reshape(-1, int(self.m**2))
        value4 = (torch.matmul(x2[:, :, None], t2[:, None, :])).reshape(-1, int(self.m**2))
        value = self.fc(torch.cat((value1, value2, value3, value4), -1))
        return value


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def function_u_exact(x, t):
    return torch.exp(-t) * torch.sin(beta * pi * x)


def function_f(x, t):
    return 0


def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                         grad_outputs=torch.ones_like(output),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]


def main():

    epochs = 40000
    m = 100

    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = Model(m=m).to(device)
    # model.apply(weights_init)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    # print(model)
    #
    # loss_log_multi = []
    # tt = time.time()
    # for epoch in range(epochs+1):
    #     xr = torch.rand(128, 1)
    #     tr = torch.rand(128, 1)
    #     xi = torch.rand(128, 1)
    #     ti = torch.zeros_like(xi)
    #     xb = torch.where(torch.rand(128, 1) > 0.5, torch.ones(128, 1), torch.zeros(128, 1))
    #     tb = torch.rand(128, 1)
    #     xr = xr.to(device)
    #     tr = tr.to(device)
    #     xi = xi.to(device)
    #     ti = ti.to(device)
    #     xb = xb.to(device)
    #     tb = tb.to(device)
    #
    #     # generate the data set
    #     xr.requires_grad_()
    #     tr.requires_grad_()
    #     output_r = model(xr, tr)
    #     output_b = model(xb, tb)
    #     output_i = model(xi, ti)
    #     exact_b = function_u_exact(xb, tb)
    #     exact_i = function_u_exact(xi, ti)
    #
    #     grads_t = autograd.grad(outputs=output_r, inputs=tr,
    #                             grad_outputs=torch.ones_like(output_r),
    #                             create_graph=True, retain_graph=True, only_inputs=True)[0]
    #     grads_x = autograd.grad(outputs=output_r, inputs=xr,
    #                             grad_outputs=torch.ones_like(output_r),
    #                             create_graph=True, retain_graph=True, only_inputs=True)[0]
    #
    #     grads2 = gradients(xr, grads_x[:, 0: 1])[:, 0: 1]
    #     loss_r = torch.mean(torch.square(grads2/(beta ** 2 * pi**2)-grads_t))
    #     loss_b = torch.mean(torch.square(output_b - exact_b)) * 200
    #     loss_i = torch.mean(torch.square(output_i - exact_i)) * 50
    #     loss = loss_r + loss_b + loss_i
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     StepLR.step()
    #     loss_log_multi.append(loss.item())
    #     if epoch % 100 == 0:
    #         Nx = 200
    #         Nt = 20
    #         test_x = torch.linspace(0, 1, Nx+1)
    #         test_t = torch.linspace(0, 1, Nt+1)
    #         test_X, test_T = torch.meshgrid([test_x, test_t])
    #         grid = torch.cat((test_X.flatten().reshape(-1, 1), test_T.flatten().reshape(-1, 1)), 1)
    #         test_exact = function_u_exact(grid[:, 0:1], grid[:, 1:2])
    #         test_pred = model(grid[:, 0:1], grid[:, 1:2]).detach()
    #         test_err = torch.linalg.norm(test_pred - test_exact) / torch.linalg.norm(test_exact)
    #
    #         print('epoch: {:d}, loss: {:.4e}, loss_r: {:.4e}, loss_b: {:.4e}, loss_i: {:.4e}, time: {:.4e}, test err: {:.4e}'.format(epoch, loss.item(), loss_r.item(), loss_b.item(), loss_i.item(), time.time()-tt, test_err))
    #         tt = time.time()
    # torch.save(loss_log_multi, 'heat_loss_log_multi')
    # PATH = 'heat_multi.pth'
    # torch.save(model.state_dict(), PATH)
    state_dict = torch.load('heat_multi.pth')
    model.load_state_dict(state_dict)
    with torch.no_grad():
        Nx = 501
        Nt = 101
        test_x = torch.linspace(0, 1, Nx)
        test_t = torch.linspace(0, 1, Nt)
        test_X, test_T = torch.meshgrid([test_x, test_t])
        grid = torch.cat((test_X.flatten().reshape(-1, 1), test_T.flatten().reshape(-1, 1)), 1)
        test_exact = function_u_exact(grid[:, 0:1], grid[:, 1:2]).reshape(Nx, Nt)
        test_pred = model(grid[:, 0:1], grid[:, 1:2]).detach().reshape(Nx, Nt)
        test_err = torch.linalg.norm(test_pred - test_exact) / torch.linalg.norm(test_exact)
    print('Heat Multi L2 err: {:.4e}'.format(test_err))
    # plt.figure()
    # plt.plot(x_data, pred, '--r', label='pred')
    # plt.plot(x_data, exact, '-k', label='exact')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('heat_pred_result_multi.pdf')

    # plt.figure()
    # plt.plot(x_data, torch.abs(pred - exact), label='abs err')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('abs_multi.pdf')

    loss_log_multi = torch.load('heat_loss_log_multi')
    plt.figure()
    plt.plot(loss_log_multi)
    plt.legend(['multi loss'])
    plt.semilogy()
    plt.tight_layout()
    plt.savefig('multi_loss.pdf')

    plot_heat(test_exact, 'Exact')
    plot_heat(test_pred, 'Pred')
    plot_heat(torch.abs(test_exact - test_pred), 'Abs err')
    plt.show()


if __name__ == '__main__':
    main()
