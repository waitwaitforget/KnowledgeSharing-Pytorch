import argparse

def parse():
    parser = argparse.ArgumentParser('Common Argument Parser')
    parser.add_argument('--memo', default='exp', type=str, help='memo for the experiment')
    parser.add_argument('--exp-name', default='exp', type=str, help='memo for the experiment')
    # model definition 
    parser.add_argument('--model', default='resnet', type=str, help='type of model')
    parser.add_argument('--depth', default=32, type=int)
    parser.add_argument('--teach-depth', default=56, type=int)
    parser.add_argument('--nb-classes', default=100, type=int, help='number of classes')
    ## model sepcific
    parser.add_argument('--mult-mode', default='mult_model',type=str,help='multi model modes')
    parser.add_argument('--multi-arch', default='resnet32,resnet32', type=str, help='type of model')
    parser.add_argument('--nmodel', default=5, type=int, help='number of models')
    parser.add_argument('-T', default=2, type=float, help='temperature for KL-divergence')
    parser.add_argument('--loss', default='kl_div', type=str, help='option for loss function')
    # born agin network
    parser.add_argument('-c', default=1, type=float, help='coefficient of regularization')
    parser.add_argument('--born-time', default=3, type=int, help='number of models')
    parser.add_argument('--mode', default='common', type=str, help='type of born loss')

    # cb 
    parser.add_argument('--ratio', default=0.5, type=float, help='coefficient of regularization')
    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str, help='type of optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='learning rate')
    parser.add_argument('--milestones', default='150,225', type=str, help='learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma for lr scheduler rate')
    # training configuration
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--start_epoch', default=0, type=int, help='start point')
    parser.add_argument('--max-epoch', default=300, type=int, help='maxial training epoch')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--num-workers', default=4, type=int, help='type of optimizer')
    
    parser.add_argument('--accumulate',default=1, type=int, help='use accumulation for update gradients')
    parser.add_argument('--dataset', default='cifar100', type=str, help='type of optimizer')
    parser.add_argument('--data-dir', default='./data', type=str, help='type of optimizer')
    parser.add_argument('--ckpt-dir', default='./checkpoints', type=str, help='type of optimizer')
    parser.add_argument('--devices', default='cpu', type=str, help='need specify devices')
    
    parser.add_argument('--tensorboard', default=1,type=int, help='need specify devices')

    parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--print-interval', default=10, type=int)
    parser.add_argument('--ckpt-interval', default=10,type=int) 
    parser.add_argument('--verbose',default=1, type=int, help='display batch details')   
    args = parser.parse_args()
    return args
