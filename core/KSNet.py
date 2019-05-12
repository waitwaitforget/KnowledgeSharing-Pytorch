import torch.nn as nn

class KSNet(nn.Module):
    def __init__(self, n_specialist, basenets, devices=None):
        super(KSNet, self).__init__()

        if len(basenets) != n_specialist+1:
            raise ValueError('num of basenet is not correct')
        self.n_specialist = n_specialist
    
        self.specialists = nn.ModuleList(basenets[:-1])
        self.generalist = basenets[-1]
        # if len(devices) < n_specialist+1:
        #    raise ValueError("Number of devices is smaller than number of specialists and generalist.")
        self.devices = devices

    def to(self, devices):
        if self.devices is None:
            [self.specialists[i].to('cpu') for i in range(self.n_specialist)]
            self.generalist.to('cpu')

        if len(self.devices) < self.n_specialist:
            [self.specialists[i].to('cuda:%s'%self.devices[0]) for i in range(self.n_specialist)]
            self.generalist.to('cuda:%s'%self.devices[0])
        else:
            for i in range(self.n_specialist):
                self.specialists[i].to('cuda:%s'%self.devices[i])
            self.generalist.to('cuda:%s'%self.devices[-1])

    def forward(self, x):
        spec_logits = [net(x.to(net.device)) for net in self.specialists]
        gene_logits = self.generalist(x.to(self.generalist.device))

        # transfer to devices[0]
        spec_logits = [p.to(spec_logits[0].device) for p in spec_logits]
        gene_logits = gene_logits.to(spec_logits[0].device)
        return spec_logits, gene_logits
            
    def loss_fn(self, fn):
        self._loss = fn
        return fn

    def loss(self, pred, targets):
        self._loss(pred, targets)