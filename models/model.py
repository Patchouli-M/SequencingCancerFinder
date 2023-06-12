import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np 

# a FcNet 
class FcNet(nn.Module):
    def __init__(self,in_size:int = 4096, mid_size:int = 512):
        super(FcNet, self).__init__()
        self.desen_layer = nn.Sequential(
            nn.Linear(in_size,in_size),
            nn.Dropout(),
            nn.Linear(in_size,mid_size)
        )
        self.in_features=mid_size
    def forward(self,x):
        x = self.desen_layer(x)
        return x
    
def get_fea(args):
    return FcNet()

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256):
        super(feat_classifier, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, class_num)
    def forward(self, x):
        x = self.fc(x)
        return x

class VREx(nn.Module):
    def __init__(self, args):
        super(VREx, self).__init__()
        self.featurizer = get_fea(args)
        self.classifier = feat_classifier(args.num_classes, self.featurizer.in_features)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.register_buffer('update_count', torch.tensor([0]))
        self.args = args

    def update(self, minibatches, opt, sch):
        if self.update_count >= self.args.anneal_iters:
            penalty_weight = self.args.lam
        else:
            penalty_weight = 1.0
        nll = 0.
        if self.args.gpu_id:
            all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        else:
            all_x = torch.cat([data[0].float() for data in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, data in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx +
                                data[0].shape[0]]
            all_logits_idx += data[0].shape[0]
            if self.args.gpu_id:
                nll = F.cross_entropy(logits, data[1].cuda().long())
            else:
                nll = F.cross_entropy(logits, data[1].long())
            losses[i] = nll
        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        self.update_count += 1
        return {'loss': loss.item(),'mean': mean.item(), 'penalty': penalty.item()}


    def predict(self, x):
        return self.network(x)
