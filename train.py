import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets

import torchvision.transforms as transforms

import os
import string
import random
import numpy as np
from core.KSNet import KSNet
from core.ensemble import Ensemble
from utils.trainutils import AverageMeter
from utils.metricutils import accuracy, oracle_accuracy
from utils.datautils import build_vision_dataloader
from utils.misc import random_string
from utils.trainutils import PerformanceReporter

import argparse
import time
# tensorboard
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser('Training Knowledge sharing network.')
parser.add_argument('--memo', default='exp', type=str, help='memo for the experiment')
parser.add_argument('--model', default='ks', type=str, help='type of model')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-T', default=2, type=float, help='temperature for KL-divergence')
parser.add_argument('--momentum', default=0.9, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='learning rate')
parser.add_argument('--max-epoch', default=300, type=int, help='maxial training epoch')
parser.add_argument('--batch-size', default=256, type=int, help='batch size')
parser.add_argument('--nmodel', default=5, type=int, help='number of models')
parser.add_argument('--devices', default=None, required=True, help='need specify devices')
parser.add_argument('--depth', default=32, type=int)
parser.add_argument('--print-interval', default=10, type=int)
args = parser.parse_args()

des = '-'.join([args.memo, args.model]+ list(map(str,[args.nmodel, args.depth,args.T, args.lr, args.batch_size])) + [random_string(6)])
devices = args.devices.split(',')
writer = SummaryWriter(os.path.join('runs',des))
# define models and optimizers
if args.model == 'ks':
    model = KSNet(5, args.depth,100, devices)
elif args.model == 'ensemble':
    model = Ensemble(5, args.depth,100, devices)
model.to(devices)
optimizers = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizers, [150, 225], gamma=0.1)
criterion = nn.CrossEntropyLoss(reduce=False)
# define datasets and dataloader
transform= transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]],
                                                     std=[n/255. for n in [68.2, 65.4, 70.4]])])
train_loader, test_loader = build_vision_dataloader('./data',dsname='CIFAR100',
                                                    transform=transform,batch_size=args.batch_size)
reporter = PerformanceReporter(None, acc=accuracy, oracle_acc=oracle_accuracy)

def compute_oracle_loss(preds, targets):

    losses = [criterion(pred, targets) for pred in preds]
    losses = [loss.unsqueeze(1) for loss in losses]
    losses = torch.cat(losses, 1)
    valid_loss, valid_index = torch.topk(losses, k=1, dim=1, largest=False)
    logits = torch.cat([pred.unsqueeze(2) for pred in preds],dim=2)

    valid_index = valid_index.repeat(1, logits.size(1))
    valid_logits = logits.gather(2, valid_index.unsqueeze(2))
    return valid_loss.mean(), valid_logits

def compute_kl_loss(pred, target, temperature=1.0):
    def softmax(x, T):
        y = torch.exp(x/T)
        y = y/(y.sum(1).unsqueeze(1)+1e-8)
        return y
    probs = softmax(pred.squeeze(),temperature)
    targp = softmax(target.squeeze(), temperature)
    loss = temperature ** 2 * F.kl_div(probs, targp)
    return loss

# register loss function before training
if args.model == 'ks':
    @model.loss_fn
    def ks_loss_fn(pred, targets):
        spec_logits, gene_logits = pred
        batch_losses, valid_logits = compute_oracle_loss(spec_logits, targets)
        gene_loss = criterion(gene_logits, targets).mean()
        kl_loss = compute_kl_loss(gene_logits, valid_logits.detach(), temperature=args.T)
        batch_loss = batch_losses  + kl_loss # + gene_loss
        return batch_loss
elif args.model == 'ensemble':
    @model.loss_fn
    def ie_loss_fn(pred, targets):
        batch_loss = sum([criterion(pred, targets).mean() for pred in pred[0]])
        return batch_loss

def train_epoch(model, optimizer, criterion, train_loader, epoch):
    model.train()
    loss = AverageMeter()

    for i,(data, targets) in enumerate(train_loader):
        data_time = time.time()
        batch_datas = [data.to('cuda:%s'% dev) for dev in devices]
        targets = targets.to('cuda:%s'%devices[0])
        data_time = time.time() - data_time
        # forward
        batch_time = time.time()
        pred = model(batch_datas)
        batch_time = time.time() - batch_time
        batch_loss = model.loss(pred, targets)

        measures = reporter.write(pred, targets)
        # update metrics
        optimizer.zero_grad()
        batch_loss.backward()
        loss.update(batch_loss.item())
        optimizer.step()

        if (i+1) % args.print_interval == 0:
            cmdstr = 'Epoch {}/{}: data_time: {:.4f}, batch_time: {:.4f}, batch_loss: {:.4f}, total_loss: {:.4f},'.format(i, epoch, data_time, batch_time, batch_loss.item(),
                    loss.avg)
            metricstr = ', '.join(['%s: %.4f' for k,v in reporter.metrics.items()])
            print(cmdstr + metricstr)
    reporter.reset()

    # write to summary
    writer.add_scalar('data/train_loss', loss.avg, epoch)
    writer.add_scalar('data/train_acc1', acc1.avg, epoch)
    writer.add_scalar('data/train_acc5', acc5.avg, epoch)
    writer.add_scalar('data/train_oracle', oracle_acc.avg, epoch)

def evaluate(model, criterion, val_loader, epoch):
    loss = AverageMeter()

    with torch.no_grad():
        for i,(data, targets) in enumerate(val_loader):
            data_time = time.time()
            batch_datas = [data.to('cuda:%s'% dev) for dev in devices]
            targets = targets.to('cuda:%s'%devices[0])
            data_time = time.time() - data_time
            # forward
            batch_time = time.time()
            spec_logits, gene_logits = model(batch_datas)
            batch_time = time.time() - batch_time
            batch_loss = model.loss(pred, targets)

            measures = reporter.write(pred, targets)
            # update metrics
            loss.update(batch_loss.item())

            if (i+1) % args.print_interval == 0:
                cmdstr = 'Eval epoch {}/{}: data_time: {:.4f}, batch_time: {:.4f}, batch_loss: {:.4f}, total_loss: {:.4f},'.format(i, epoch, data_time, batch_time, batch_loss.item(),
                        loss.avg)
                metricstr = ', '.join(['%s: %.4f' for k,v in reporter.metrics.items()])
                print(cmdstr + metricstr)
    # write to summary
    writer.add_scalar('data/val_loss', loss.avg, epoch)
    writer.add_scalar('data/val_acc1', acc1.avg, epoch)
    writer.add_scalar('data/val_acc5', acc5.avg, epoch)
    writer.add_scalar('data/val_oracle', oracle_acc.avg, epoch)

def main():
    for epoch in range(args.max_epoch):
        lr_scheduler.step()
        train_epoch(model, optimizers, criterion, train_loader, epoch)
        evaluate(model, criterion, test_loader, epoch)
def kidding():
    print('wtf')
if __name__=='__main__':
    main()
