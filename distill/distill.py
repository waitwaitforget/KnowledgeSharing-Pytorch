import torch.nn as nn
import torch

class DistillModel(nn.Module):

    def __init__(self, teacher, student):
        super(DistillModel, self).__init__()

        self.teacher = teacher
        self.student = student

    def forward(self, x):
        with torch.no_grad():
            ty = self.teacher(x)
        sy = self.student(x)

        return sy,ty

    def loss_fn(self, fn):
        self._loss = fn
        return fn

    def loss(self, pred, target):
        return self._loss(pred, target)
        
