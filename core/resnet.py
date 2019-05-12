from .base import *
from torchvision.models.resnet import resnet50, resnet152


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(CifarResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out
    @property
    def device(self):
        return list(self.parameters())[0].device

    def loss_fn(self, fn):
        self._loss = fn
        return fn

    def loss(self, pred, targets):
        self._loss(pred, targets)

def build_resnet56(nb_classes):
    return CifarResNet(BasicBlock, [9, 9, 9], nb_classes)


def build_bottleneck_resnet(depth, nb_classes, mode='cifar'):
    if mode == 'cifar':
        nb = (depth - 2) // 9
        return CifarResNet(Bottleneck, [nb, nb, nb], nb_classes)
    elif mode == 'imagenet':
        if depth == 50:
            return resnet50(pretrained=False)
        elif depth ==  152:
            return resnet152(pretrained=False)
        else:
            raise Exception("Only support resent50 and resnet 152 on Imagenet.")


def unit_test():
    net = build_bottleneck_resnet(56, nb_classes=10)
    #net = build_bottleneck_resnet56(152, 1000, 'imagenet')
    #from torch.autograd import Variable
    #y = net(Variable(torch.randn(1, 3, 224, 224)))
    #print y

    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params num: ', params)
    print('sss', get_n_params(net))
    #print net
    x=nn.Sequential(nn.BatchNorm2d(64), nn.BatchNorm2d(128), nn.BatchNorm2d(256))
    print(get_n_params(x))
    print(get_n_params(net) - get_n_params(x))
    # test()

if __name__ == "__main__":
    unit_test()
