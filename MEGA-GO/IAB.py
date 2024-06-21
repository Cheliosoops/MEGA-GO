import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class Stage1(nn.Module):
    def __init__(self, in_channel=512, out_channel=512, kernels=[1,3,5], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, in_channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(out_channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(out_channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, out_channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)

        U = sum(conv_outs)

        S = U.mean(-1).mean(-1)
        Z = self.fc(S)

        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, 128, 1, 1))
        
        attention_weughts = torch.stack(weights, 0)
        attention_weughts = self.softmax(attention_weughts)

        V = (attention_weughts * feats).sum(0)
        return V
class Stage2(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Stage2, self).__init__()
        self.se1 = Stage1(in_channel=in_ch,out_channel=out_ch, reduction=in_ch / 4)

    def forward(self, input1,sim):
        input1 = input1 * sim.unsqueeze(1)
        input1 = input1.view([input1.shape[0], input1.shape[1]])
        s1, c1 = input1.shape
        input1 = input1.view(s1, c1, 1, 1)
        input1_en = self.se1(input1)
        input1_en = input1_en.view(input1_en.shape[0], input1_en.shape[1])
        return input1_en
