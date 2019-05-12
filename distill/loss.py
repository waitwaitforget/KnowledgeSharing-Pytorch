import torch
from torch.autograd import grad
import torch.nn.functional as F

def scaled_softmax(logits, T=2):
    '''
    smooth the logits by divide a temperature
    '''
    logits = logits / T
    return F.softmax(logits)

def hinton_distill_loss(t_pred, s_pred, target, T=2, alpha=1):
    # compute ce loss
    celoss = F.cross_entropy(s_pred, target)
    # distll loss
    s_s_pred = scaled_softmax(s_pred)
    s_t_pred = scaled_softmax(t_pred)
    disllloss = F.mse_loss(s_s_pred, s_t_pred.detach())
    return celoss + alpha * T**2 * disllloss

# comppute derivative
def _calc_jacobian(x, pred, target):
    pred.gather(1, target.view(target.size(0),1)).sum().backward(retain_graph=True)
    return x.grad.detach()

def jacobian_matching_loss(*args):
    x, t_pred, s_pred, target = args
    alpha = 1.0
    # compute ce loss
    celoss = F.cross_entropy(s_pred, target)
    # compute match loss
    teacher_jacob = _calc_jacobian(x, t_pred, target)
    # zero grad 
    x.grad.zero_()
    # compute stdudnt jacob
    student_jacob = _calc_jacobian(x, s_pred, target)
    matchloss = torch.norm(student_jacob - teacher_jacob.detach(),2)
    # distll loss
    disllloss = F.mse_loss(s_pred, t_pred.detach())
    # note that by doing loss.baward(), all the gradients will multiply 2, so the lr should divide by 2
    loss = celoss + alpha * matchloss + disllloss
    return loss

def intermediate_matching_loss(pred, target):
    pass
    
