import sys
sys.path.append(".")
sys.path.append("..")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BAN(nn.Module):

    def __init__(self, teacher_model, student_model=None, basic_criterion=None,
                        c=1.0, ckpt_dir='./checkpoint/ban/'):
        super(BAN, self).__init__()

        self.teacher = teacher_model
        self.student = student_model

        self.iter = 0
        self.ckpt_dir = ckpt_dir
        self.teacher_acc = 0.0
        self.student_acc = 0.0
        self.c = c
        self.basic_criterion =  basic_criterion# is None else basic_criterion

    def forward(self, x):
        if self.student is None:
            return self.teacher(x)
        else:
            with torch.no_grad():
                tea_p = self.teacher(x)
            stu_p = self.student(x)
            return [tea_p, stu_p]

    @staticmethod
    def _save_checkpoint(ckpt_dir, model, acc, iter, name):
        state = {'state_dict': model.state_dict(),
                 'acc': acc,
                 'iter': iter}
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save(state, os.path.join(ckpt_dir, name))


    def expand(self, model, tea_acc):
        #if self.student is None:
        #    raise ValueError(
        #        'There is no well trained student available for expand!')
        # save current teacher to disk
        teacher_ckpt_path = 'teacher_%i.pt' % self.iter
        BAN._save_checkpoint(self.ckpt_dir, self.teacher,
                         self.teacher_acc, self.iter, teacher_ckpt_path)
        if self.student is not None:
            self.teacher = self.student
        self.student = model
        self.iter += 1
        self.teacher_acc = tea_acc

    
    def save(self, stu_acc):
        self.student_acc = stu_acc
        # save current teacher and student model to disk
        teacher_ckpt_path = 'teacher_%i.pt' % self.iter
        student_ckpt_path = 'student_%i.pt' % self.iter

        BAN._save_checkpoint(self.ckpt_dir, self.teacher,
                         self.teacher_acc, self.iter, teacher_ckpt_path)
        BAN._save_checkpoint(self.ckpt_dir, self.student,
                         self.student_acc, self.iter, student_ckpt_path)

    @staticmethod
    def _soft_cross_entropy(x, y):
        b = - torch.sum(F.softmax(y, dim=1) * F.log_softmax(x, dim=1), dim=1)
        
        return b.mean()

    def _common_fn(self, pred, target):
        if isinstance(pred, list):
            tea_logits, stu_logits = pred[0], pred[1]
            if self.basic_criterion is not None:
                basic_loss = self.basic_criterion(stu_logits, target)
            else:
                basic_loss = 0
            # distill loss
            distill_loss = BAN._soft_cross_entropy(stu_logits, tea_logits.detach())
            return basic_loss + self.c * distill_loss
        else:
            return F.cross_entropy(pred, target)

    def _cwtm_fn(self, pred, target):
        if not isinstance(pred, list):
            raise ValueError('cwtm mode need a student model')
        #
        tea_logits, stu_logits = pred[0], pred[1]
        tea_prob = F.softmax(tea_logits.detach())
        # reweight the samples
        mvalue, _ = tea_prob.max(1)
        weight = mvalue / mvalue.sum()
        # compute the weighted cross entropy
        b = tea_prob * F.log_softmax(stu_logits, dim=1)
        
        b = weight.unsqueeze(1) * b
        distill_loss = -1.0 * b.sum()
        basic_loss = self.basic_criterion(stu_logits, target)
        return basic_loss + self.c * distill_loss

    @staticmethod
    def _permutate_non_argmax(x):
        m, n = x.size(0), x.size(1)
        x_max, x_idx = x.max(1)
        idx = torch.stack([torch.randperm(n)
                           for _ in range(m)]).long().to(x.device)
        y = torch.zeros(x.size()).to(x.device)
        y.scatter_(1, idx, x)
        y_idx = idx.gather(1, x_idx.view(-1, 1))
        y_place = y.gather(1, x_idx.view(-1, 1))

        y.scatter_(1, y_idx.view(-1, 1), y_place.view(-1, 1))
        y.scatter_(1, x_idx.view(-1, 1), x_max.view(-1, 1))
        return y

    def _dkpp_fn(self, pred, target):
        if not isinstance(pred, list):
            raise ValueError('cwtm mode need a student model')
        tea_logits, stu_logits = pred[0], pred[1]
        tea_prob = F.softmax(tea_logits.detach(), dim=1)
        # permutate the non-max args
        permuted_tea_prob = BAN._permutate_non_argmax(tea_prob)

        # calculate loss
        basic_loss = self.basic_criterion(stu_logits, target)
        distill_loss = permuted_tea_prob * F.log_softmax(stu_logits, dim=1)
        distill_loss = -1.0 * distill_loss.sum()

        return basic_loss + self.c * distill_loss

    def loss(self, pred, target, mode='common'):
        '''
        compute loss with three mode:
        common: the same as hinton knowledge distillation without soften target
        cwtm: reweight the sample
        dkpp: permute the non-argmax targets
        '''
        if mode == 'common':
            return self._common_fn(pred, target)
        elif mode == 'cwtm':
            return self._cwtm_fn(pred, target)
        elif mode == 'dkpp':
            return self._dkpp_fn(pred, target)
        else:
            raise ValueError(
                'Not supported mode. Only "common", "cwtm","dkpp" for selection')
