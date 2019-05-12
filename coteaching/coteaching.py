
import sys
import os
sys.path.append('..')

import KnowledgeSharing.core as core
import torch.nn as nn

class MutltiModel(nn.Module):
    def __init__(self, coherts, coeff=1.0, rate=50):
        super(MutualLearning, self).__init__()

        self.models = nn.ModuleList(coherts)
        self.n_model = len(coherts)
        self.coeff = 1.0
        self.criterion = nn.CrossEntropyLoss()
        self.percentile = rate

    def forward(self, x):
        return [model(x) for model in self.models]
    
    
