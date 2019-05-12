import torch.nn as nn

class Ensemble(nn.Module):
    def __init__(self, m, basenets, devices=None):
        super(Ensemble, self).__init__()
        self.m = m
        
        self.devices=devices
        self.net = nn.ModuleList(basenets)
    
    def to(self,devices):
        if self.devices is None:
            [self.net[i].to('cpu') for i in range(self.m)]
        if len(devices)<self.m:
            for i in range(self.m):
                self.net[i].to('cuda:%s'%self.devices[0])
        else:
            for i in range(self.m):
                self.net[i].to('cuda:%s'%self.devices[i])

    def forward(self, x):
        indlogits = [net(x.to(net.device)) for net in self.net]
        indlogits = [p.to(indlogits[0].device) for p in indlogits]
        emblogits = sum(indlogits)/self.m
        
        return (indlogits, emblogits)

    def loss_fn(self, fn):
        self._loss = fn
        return fn

    def loss(self, pred, targets):
        self._loss(pred, targets)