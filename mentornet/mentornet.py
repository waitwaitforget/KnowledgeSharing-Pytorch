import torch.nn as nn
import torch.nn.functional as F 
import torch

class Mentor(nn.Module):
    def __init__(self, nb_classes, n_step=2):
        super(Mentor, self).__init__()

        #self.c1 = c1
        #self.c2 = c2 
        self.label_emb = nn.Embedding(nb_classes, 2)
        self.percent_emb = nn.Embedding(100, 5)

        self.lstm = nn.LSTM(2, 10, 1, bidirectional=True)
        # self.h0 = torch.rand(2,)
        self.fc1 = nn.Linear(27, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, data):
        label, pt, l = data
        x_label = self.label_emb(label)
        x_percent = self.percent_emb(pt)
        # print(x_label.size())
        h0 = torch.rand(2, label.size(0), 10)
        c0 = torch.rand(2, label.size(0), 10)

        output, (hn,cn) = self.lstm(l, (h0,c0))
        output = output.sum(0).squeeze()
        x = torch.cat((x_label, x_percent, output), dim=1)
        z = F.tanh(self.fc1(x))
        z = F.sigmoid(self.fc2(z))
        return z


class MentorNet(nn.Module):
    def __init__(self, student_model, basic_criterion=None,c1=1,c2=1,percentile=75, nb_classes=100, ckpt_dir='./checkpoint/mentor/'):
        super(MentorNet, self).__init__()

        self.student = student_model # student model
        self.mentor = Mentor(nb_classes, 2) # mentor net

        self.ckpt_dir = ckpt_dir
        self.c1 = c1
        self.c2 = c2
        self.nb_classes = nb_classes
        self.percentile = percentile
        self.basic_criterion = nn.CrossEntropyLoss() if basic_criterion is None else basic_criterion
        
    def forward(self, x):
        pred = self.student(x)
        return pred
    
    #def target_to_one_hot(self, y):
    #    return torch.randint(self.nb_classes, (y,)).long()

    def loss(self, pred, target, global_step_ratio):
        # compute common cross entropy loss
        ce_loss = F.cross_entropy(pred, target, reduce=False)
        student_loss = ce_loss.mean()
        # compute mentor loss
        if self.c2 == 0:
            mentor_target = ce_loss.le(self.c1).long()
        else:
            mentor_target = torch.min(torch.max(0, 1 - (ce_loss - self.c1) / self.c2), 1)
        percent = torch.ones(pred.size(0)).long() * global_step_ratio

        
import torch
mentor = Mentor(100, 2)
label = torch.randint(100, (10,)).long()
#label = torch.eye(100)[label]
#label = label.long()

percent = torch.randint(100, (10,)).long()
#percent = torch.eye(100)[percent]
#percent = percent.long()

l = torch.rand(2, 10, 2)

y = mentor((label, percent, l))
print(y)