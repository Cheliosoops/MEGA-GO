import torch
import torch.nn as nn
import random

class adaAF_1(torch.nn.Module):
    def __init__(self, in_ch, latent_dim,out_ch):
        super(adaAF_1, self).__init__()
        self.BN = nn.BatchNorm1d(in_ch)
        self.GAP = nn.AdaptiveAvgPool1d(in_ch)
        self.FC1 = nn.Linear(in_ch, latent_dim)
        self.ReLu = nn.ReLU()
        self.FC2 = nn.Linear(latent_dim, out_ch)
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, Finput):
        Fv = self.BN(Finput)
        Mv = Finput - Fv
        Mv_1 = self.GAP(Mv)
        Mv_2 = self.FC1(Mv_1)
        Mv_3 = self.ReLu(Mv_2)
        Mv_4 = self.FC2(Mv_3)
        Mv_5 = self.Sigmoid(Mv_4)

        return Mv_5

class adaAF_2(torch.nn.Module):
    def __init__(self,in_ch):
        super(adaAF_2,self).__init__()
        self.BN = nn.BatchNorm1d(in_ch)
        self.GAP2 = nn.AdaptiveAvgPool1d(512)
    def forward(self, Finput1):
        catBN = self.BN(Finput1)
        catBN = self.GAP2(catBN)
        return catBN
        
