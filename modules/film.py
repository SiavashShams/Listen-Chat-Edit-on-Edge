import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingFusor(nn.Module):

    def __init__(self, d_in=None, d_task=1, d_out=None, mode='cat',
        task2int={'extract': 1, 'remove': -1}
    ):
        super().__init__()
        assert mode in ['cat', 'linear_sum']
        if mode == 'linear_sum':
            self.fusor = nn.Sequential(
                nn.Linear(d_in+d_task, d_out),
                nn.ReLU(inplace=True)
            )
        self.d_in = d_in
        self.d_task = d_task
        self.d_out = d_out
        self.task2int = task2int
        self.mode = mode

    def forward(self, x, tasks):
        '''
        x: a list of (B, D) tensors
        tasks: a list of B strings
        '''
        S, B, D = len(x), x[0].shape[0], x[0].shape[1]

        y = torch.zeros((B, S, D+self.d_task), device=x[0].device)
        for s in range(S):
            for b in range(B):
                y[b][s][:-self.d_task] += x[s][b]
                y[b][s][-self.d_task] +=  self.task2int[tasks[s][b]]
            
        if self.mode == 'cat':
            y = y.view(B, -1)

        return y

class FiLM(nn.Module):

    def __init__(self, in_dim, out_dim, n_layer=1, scale=True):
        super(FiLM, self).__init__()
        self.scale = scale
        if n_layer == 1:
            self.scaler = nn.Linear(in_dim, out_dim) if scale else None
            self.shifter = nn.Linear(in_dim, out_dim)
        elif n_layer == 2:
            self.scaler = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            ) if scale else None
            self.shifter = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )
        elif n_layer == 2.4:
            self.scaler = nn.Sequential(
                nn.Linear(in_dim, in_dim//4),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim//4, out_dim)
            ) if scale else None
            self.shifter = nn.Sequential(
                nn.Linear(in_dim, in_dim//4),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim//4, out_dim)
            )
        elif n_layer == 3:
            self.scaler = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            ) if scale else None
            self.shifter = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )

    def _forward_convtasnet(self, feat, cond):
        '''
        feat: (B, T, out_dim)
        cond: (B, in_dim)
        '''
        if self.scale:
            gamma = self.scaler(cond).unsqueeze(dim=1) # (B, 1, out_dim)
            beta = self.shifter(cond).unsqueeze(dim=1) # (B, 1, out_dim)
            cond_feat = feat * gamma + beta # time invariant

        else:
            beta = self.shifter(cond).unsqueeze(dim=1) # (B, 1, out_dim)
            cond_feat = feat + beta # time invariant
        
        return cond_feat

    def _forward_sepformer(self, feat, cond):
        '''
        feat: (B, out_dim, K, S)
        cond: (B, in_dim)
        '''
        if self.scale:
            gamma = self.scaler(cond).unsqueeze(dim=-1).unsqueeze(dim=-1) # (B, out_dim, 1, 1)
            beta = self.shifter(cond).unsqueeze(dim=-1).unsqueeze(dim=-1)  # (B, out_dim, 1, 1)
            cond_feat = feat * gamma + beta # time invariant

        else:
            beta = self.shifter(cond).unsqueeze(dim=-1).unsqueeze(dim=-1) # (B, out_dim, 1, 1)
            cond_feat = feat + beta # time invariant
        
        return cond_feat

    def forward(self, feat, cond):
        '''
        feat: (B, T, out_dim) for convtaset or (B, out_dim, K, S) for sepformer
        cond: (B, in_dim)
        '''
        if feat.dim() == 3:
            return self._forward_convtasnet(feat, cond)
        elif feat.dim() == 4:
            return self._forward_sepformer(feat, cond)
        else:
            raise NotImplementedError
