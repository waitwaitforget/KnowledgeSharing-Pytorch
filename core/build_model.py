import torch 
from .resnet2 import build_bottleneck_resnet
from .smallnet import ToyConvNet

def build_model(model_type,depth, args):
    if model_type == 'ksnet':
        return None
    elif model_type=='resnet':
        return build_bottleneck_resnet(depth=depth, nb_classes=args.nb_classes)
    elif model_type=='small':
        return ToyConvNet(3, 16, 100, 4)
