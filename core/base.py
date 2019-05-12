import torch
import torch.nn as nn
import torch.nn.init
import math
import torch.nn.functional as F


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.bias.data.zero_()


def get_n_params(model):
    return sum([param.nelement() for param in list(model.parameters())])


def conv3x3(in_planes, out_planes, stride=1, bias=False, dilation=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class MultModel(nn.Module):

    def __init__(self, models):
        super(MultModel, self).__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outs = [m(x) for m in self.models]
        return outs


class MultOutModel(nn.Module):

    def __init__(self, model, nb_outs=2):
        super(MultOutModel, self).__init__()
        self.nb_outs = nb_outs
        self.model = model

    def forward(self, x):
        outs = [self.model(x) for _ in range(self.nb_outs)]
        return outs


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SymmetricOutModel(nn.Module):
    def __init__(self, model):
        super(SymmetricOutModel, self).__init__()
        self.model = model

    def forward(self, x):
        y = flip(x, 2)
        return [self.model(x), self.model(y)]


def add_arguments(parser):
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--b', '--batchsize', dest='batchsize', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')