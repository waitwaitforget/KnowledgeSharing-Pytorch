'''
Born-Again Neural Network
The pipeline is similar to the knowledge distillation with more steps.
ref: https://arxiv.org/abs/1805.04770

Author: Kai Tian
Date: 13/12/2018
'''

from __future__ import absolute_import
import sys
import os
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import torchvision.transforms as transforms

import KnowledgeSharing.distill as distill
import KnowledgeSharing.utils as utils
import KnowledgeSharing.core as core
from KnowledgeSharing.utils.metricutils import accuracy
import KnowledgeSharing.common as common_args
from KnowledgeSharing.utils.trainutils import AverageMeter
import time
from tqdm import tqdm

from copy import deepcopy
from tensorboardX import SummaryWriter
from ban import BAN

args = common_args.parse()
des = '-'.join([args.memo, args.model]+ list(map(str,[args.born_time, args.depth, args.c, args.lr, args.batch_size])) + [utils.random_string(6)])
writer=SummaryWriter(des)


""" training teacher model and distill it to student"""
def train_epoch(model, optimizer, train_loader, epoch, callback=None, **kwargs):
    global writer
    
    loss = AverageMeter()
    metrics = AverageMeter()
    data_time = time.time()
    train_bar = tqdm(train_loader)
    devices = 'cuda:0' if 'devices' not in kwargs.keys() else 'cuda:'+kwargs['devices'][0]
    
    for idx, (data, target) in enumerate(train_bar):
        data = data.to(devices)
        target = target.to(devices)
        data_time = time.time() - data_time
        # forward
        batch_time = time.time()
        pred = model(data)
        batch_time = time.time() - batch_time
        batch_loss = model.loss(pred, target, mode=kwargs['mode'])
        metrics.update(accuracy(pred, target), data.size(0))

        loss.update(batch_loss.item(), data.size(0))
        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # print(metrics.avg,loss.avg,type(loss.avg))
        train_bar.set_description('Epoch {}/{}, data_time: {:.4f}, batch_time: {:.4f}, loss: {:.4f}, accuracy: {:.4f}'.format(idx, epoch,
                                    data_time, batch_time, loss.avg, metrics.avg.item()))
        data_time = time.time()

    
    return loss.avg, metrics.avg.item()

def validate(model, val_loader, epoch, callback=None, **kwargs):
    model.eval()
    global writer
    loss = AverageMeter()
    metrics = AverageMeter()
    data_time = time.time()
    val_bar = tqdm(val_loader)
    devices = 'cuda:0' if 'devices' not in kwargs.keys() else 'cuda:'+kwargs['devices'][0]
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_bar):
            data = data.to(devices)
            target = target.to(devices)
            data_time = time.time() - data_time
            # forward
            batch_time = time.time()
            pred = model(data)
            batch_time = time.time() - batch_time
            batch_loss = model.loss(pred, target,mode=kwargs['mode'])
            metrics.update(accuracy(pred, target), data.size(0))

            loss.update(batch_loss.item(), data.size(0))
            

            val_bar.set_description('Eval epoch {}/{}, data_time: {:.4f}, batch_time: {:.4f}, loss: {:.4f}, accuracy: {:.4f}'.format(idx, epoch,
                                        data_time, batch_time, loss.avg, metrics.avg.item()))
            data_time = time.time()
        
    return loss.avg, metrics.avg.item()


def born_agin_nn(model, student_model, iter, tea_acc=0.0):
    global writer
    if student_model is not None:
        model.expand(student_model, tea_acc)
        optimizer = optim.SGD(model.student.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    else:
        optimizer = optim.SGD(model.teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, list(map(int,args.milestones.split(','))), gamma=args.gamma)
    # move to gpu
    model.to('cuda:'+args.devices[0])
    transform= transforms.Compose([transforms.RandomHorizontalFlip(), 
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], 
                                                     std=[n/255. for n in [68.2, 65.4, 70.4]])])
    val_transform= transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]],
                                                     std=[n/255. for n in [68.2, 65.4, 70.4]])])
    train_loader, val_loader = utils.build_vision_dataloader(args.data_dir, args.dataset,
                batch_size=args.batch_size, num_workers=args.num_workers, train_transform=transform,val_transform=val_transform)
    best_accuracy = 0.0
    for epoch in range(args.max_epoch):
        scheduler.step(epoch)
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, epoch, devices=args.devices, mode=args.mode)
        val_loss, val_acc = validate(model, val_loader, epoch,devices=args.devices, mode=args.mode)

        writer.add_scalar('data/born_%i_train_loss'%(iter+1), train_loss, epoch)
        writer.add_scalar('data/born_%i_train_acc'%(iter+1), train_acc,epoch)

        writer.add_scalar('data/born_%i_val_loss'%(iter+1), val_loss, epoch)
        writer.add_scalar('data/born_%i_val_acc'%(iter+1), val_acc,epoch)
    return model, val_acc


teacher_model = core.build_model(args.model, args.depth, args)
student_model = None
# initialize model
model = BAN(teacher_model, c=args.c, ckpt_dir=os.path.join(args.ckpt_dir, 'ban', des))
tea_acc = 0.0
# born again 
# born time plus one, as the zero is the teacher
print('training born agagin neural network')
for i in range(args.born_time+1):
    print('born %i times'%i)
    if i==0:
        model, tea_acc = born_agin_nn(model, student_model, i, tea_acc)
    else:
        model.expand(student_model, tea_acc)
        model, tea_acc = born_agin_nn(model, student_model, i, tea_acc)
    student_model = core.build_model(args.model, args.depth, args)
# save all models
model.save(tea_acc)

