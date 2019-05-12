
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F 

import torchvision.transforms as transforms
import sys
import os
sys.path.append('../')

import KnowledgeSharing.distill as distill
from KnowledgeSharing.utils.trainutils import AverageMeter
from KnowledgeSharing.utils.metricutils import accuracy
import KnowledgeSharing.utils as utils
import KnowledgeSharing.core as core
import KnowledgeSharing.common as common_args
import time
from tqdm import tqdm

from copy import deepcopy
from tensorboardX import SummaryWriter


args = common_args.parse()
des = '-'.join([args.memo, '_hinton_distll_'+args.model]+ list(map(str,[args.nmodel, args.depth,args.T, args.lr, args.batch_size])) + [utils.random_string(6)])
writer = SummaryWriter(os.path.join('runs',des))

teacher_model = core.build_bottleneck_resnet(args.teach_depth, args.nb_classes)
teacher_model.to(args.devices)
ckpt_name = os.path.join(args.ckpt_dir, '%s_%i_tea_best_checkpoint.pt'%(args.model, args.teach_depth))
print('try loading ckpt',ckpt_name)
if os.path.exists(ckpt_name):
    state_ = torch.load(ckpt_name)
    teacher_model.load_state_dict(state_['state_dict'])
else:
    # training from scratch
    raise Exception('checkpoint not found')

student_model = core.build_bottleneck_resnet(args.depth, args.nb_classes)
student_model.to(args.devices)
model = distill.DistillModel(teacher_model, student_model)
optimizer, lr_scheduler = utils.trainutils.build_optimizer_scheduler(model, args)


def compose_name(*args):
    return '-'.join(args)

@model.loss_fn
def loss(pred, target):
    s_pred, t_pred = pred
    return distill.loss.hinton_distill_loss(t_pred, s_pred, target)

""" training teacher model and distill it to student"""
def train_epoch(model, optimizer, train_loader, epoch, callback=None, **kwargs):
    global writer
    model.train()
    loss = AverageMeter()
    metrics = AverageMeter()
    data_time = time.time()
    train_bar = tqdm(train_loader)
    devices = 'cuda:0' if 'devices' not in kwargs.keys() else kwargs['devices']
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(devices)
        target = target.to(devices)
        data_time = time.time() - data_time
        # forward
        batch_time = time.time()
        pred = model(data)
        batch_time = time.time() - batch_time
        batch_loss = model.loss(pred, target)
        metrics.update(accuracy(pred[0], target), data.size(0))

        loss.update(batch_loss.item(), data.size(0))
        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # print(metrics.avg,loss.avg,type(loss.avg))
        train_bar.set_description('Epoch {}/{}, data_time: {:.4f}, batch_time: {:.4f}, loss: {:.4f}, accuracy: {:.4f}'.format(idx, epoch,
                                    data_time, batch_time, loss.avg, metrics.avg.item()))
        data_time = time.time()

    writer.add_scalar('data/teacher_train_loss', loss.avg, epoch)
    writer.add_scalar('data/teachertrain_acc', metrics.avg.item(),epoch)


def validate(model, val_loader, epoch, callback=None, **kwargs):
    global writer
    model.eval()
    loss = AverageMeter()
    metrics = AverageMeter()
    data_time = time.time()
    val_bar = tqdm(val_loader)
    devices = 'cuda:0' if 'devices' not in kwargs.keys() else kwargs['devices']
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data = data.to(devices)
            target = target.to(devices)
            data_time = time.time() - data_time
            # forward
            batch_time = time.time()
            pred = model(data)
            batch_time = time.time() - batch_time
            batch_loss = model.loss(pred, target)
            metrics.update(accuracy(pred[0], target), data.size(0))

            loss.update(batch_loss.item(), data.size(0))
            

            val_bar.set_description('Eval epoch {}/{}, data_time: {:.4f}, batch_time: {:.4f}, loss: {:.4f}, accuracy: {:.4f}'.format(idx, epoch,
                                        data_time, batch_time, loss.avg, metrics.avg.item()))
            data_time = time.time()
        writer.add_scalar('data/teacher_val_loss', loss.avg, epoch)
        writer.add_scalar('data/teacher_val_acc', metrics.avg.item(),epoch)
    return metrics.avg

#  just training the student model

print('training student model')
train_transform= transforms.Compose([transforms.RandomHorizontalFlip(), 
                            transforms.RandomCrop(32, padding=4),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], 
                                                    std=[n/255. for n in [68.2, 65.4, 70.4]])])

val_transform= transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], 
                                                    std=[n/255. for n in [68.2, 65.4, 70.4]])])                                         
train_loader, val_loader = utils.build_vision_dataloader(args.data_dir, args.dataset,
            batch_size=args.batch_size, num_workers=args.num_workers, train_transform=train_transform,val_transform=val_transform)
best_accuracy = 0.0
for epoch in range(args.max_epoch):
    lr_scheduler.step(epoch)
    train_epoch(model, optimizer, train_loader, epoch,devices=args.devices)
    val_acc = validate(model, val_loader, epoch,devices=args.devices)
    if val_acc > best_accuracy or (epoch+1)%args.ckpt_interval == 0:
        utils.trainutils.save_checkpoints(args.ckpt_dir, model, val_acc, best_accuracy, epoch, name=compose_name(args.memo, args.exp_name, args.model,'student'))
        best_accuracy = val_acc
