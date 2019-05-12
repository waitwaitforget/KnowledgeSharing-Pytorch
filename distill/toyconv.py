from __future__ import absolute_import
import torch.nn as nn 
import torch.nn.functional as F 
import sys
sys.path.append('../')
from KnowledgeSharing.utils.netutils import ConvBnReLU, ConvBn, count_params


class ToyConvNet(nn.Module):
    def __init__(self, n_in, n_channels, n_out, depth):
        super(ToyConvNet, self).__init__()

        factor = 2
        self.base = ConvBnReLU(n_in, n_channels, 3)
        self.pool = nn.MaxPool2d(2)

        self.net = nn.ModuleList()
        bc = n_channels
        oc = bc * factor
        for i in range(3):
            self.net.append(ConvBnReLU(bc, oc, 3))
            bc = oc
            oc = oc * factor if depth <=3 else oc
        self.bottleneck = ConvBnReLU(bc, 128, 3, stride=1, padding=1)

        self.linear = nn.Linear(16*16*128, n_out)

    def forward(self, x):
        fx = self.pool(self.base(x))
        for net in self.net:
            fx = net(fx)
        fx = self.bottleneck(fx)
        return self.linear(fx.view(fx.size(0), -1))

    def loss_func(self, func):
        self.calc_loss = func
        return func
    
    def loss(self,pred, target):
        return self.calc_loss(pred, target)
   

def test():
    import torch 
    x= torch.rand(16, 3, 32, 32)
    m = ToyConvNet(3, 16, 10, 5)
    print(count_params(m))
    @m.loss_func
    def func(x,y):
        return nn.functional.mse_loss(x,y)
    z = torch.rand(16,10)
    y = (m(x))
    loss = m.loss(y,z)
    print(loss)

if __name__ == '__main__':
    test()