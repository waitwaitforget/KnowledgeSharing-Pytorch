from .resnet import build_bottleneck_resnet
import torch.nn as nn



class KSNet(nn.Module):
    """
    distill mcl prediction to average
    """
    def __init__(self, n_specialist, n_classes, devices=None):
        super(KSNet, self).__init__()

        self.n_specialist = n_specialist
        self.n_classes = n_classes

        self.specialists = nn.ModuleList([build_bottleneck_resnet(56,nb_classes=n_classes) for _ in range(n_specialist)])
        self.generalist = build_bottleneck_resnet(56,nb_classes=n_classes)

        if len(devices) < n_specialist+1:
            raise ValueError("Number of devices is smaller than number of specialists and generalist.")
        self.devices = devices

    def to(self, devices):
        for i in range(self.n_specialist):
            self.specialists[i].to('cuda:%s'%self.devices[i])
        self.generalist.to('cuda:%s'%self.devices[-1])

    def forward(self, xs):
        spec_logits = [net(x) for net,x in zip(self.specialists,xs[:-1])]
        gene_logits = self.generalist(xs[-1])

        # transfer to devices[0]
        spec_logits = [p.to('cuda:%s'%self.devices[0]) for p in spec_logits]
        gene_logits = gene_logits.to('cuda:%s'%self.devices[0])
        return spec_logits, gene_logits
            