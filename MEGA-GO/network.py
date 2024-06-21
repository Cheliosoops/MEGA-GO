import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,SAGEConv
from adaSAB import GraphMultisetTransformer
from torch_geometric.nn import global_max_pool as gmp
from IAB import Stage2
from adaAF import adaAF_1,adaAF_2

class DH_GNN(nn.Module):
    def __init__(self, channel_dims=[512, 512, 512], fc_dim=512, num_classes=256, pooling='MTP'):
        super(DH_GNN, self).__init__()

        self.sage1 = SAGEConv(512, 64)
        self.sage1_hot = SAGEConv(512, 64)
        self.sage1_cold = SAGEConv(512, 64)
        
        self.sage2 = SAGEConv(64, 128)
        self.sage2_hot = SAGEConv(64, 128)
        self.sage2_cold = SAGEConv(64, 128)
        
        self.sage3_1 = SAGEConv(448, 512)
        self.sage3_2hot = SAGEConv(320, 512)
        self.sage3_2cold = SAGEConv(320, 512)

        #IAB definition
        self.adapter_Mhot = Stage2(128,128)
        self.adapter_hotM = Stage2(128,128)
        self.adapter_Mcold = Stage2(128,128)
        self.adapter_coldM = Stage2(128,128)
        #IAB definition

        self.cos_sim = nn.CosineSimilarity(dim=1)
        
        self.pooling = pooling

        #adaSAB
        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(512, 256, 512, None, 10000, 0.25, ['GMPool_G', 'SelfAtt', 'GMPool_I'], num_heads=8, layer_norm=True)
        else:
            self.pool = gmp
        #adaSAB

        self.drop1 = nn.Dropout(p=0.2)

    def forward(self, x, data):
        
        x = self.drop1(x)
        Main = F.relu(self.sage1(x,data.edge_index.long()))# 64
        hot = F.relu(self.sage1_hot(x,data.edge_index.long()))# 64
        cold = F.relu(self.sage1_cold(x,data.edge_index.long()))# 64
        
        #############################################################################
        
        random_noise = torch.rand_like(hot).to(hot.device)
        hot = hot + torch.sign(hot) * F.normalize(random_noise, dim=-1) * 0.3
        random_noise = torch.rand_like(cold).to(cold.device)
        cold = cold - torch.sign(cold) * F.normalize(random_noise, dim=-1) * 0.3
        
        #############################################################################
        
        Main2 = F.relu(self.sage2(Main,data.edge_index.long())) # 128
        hot2 = F.relu(self.sage2_hot(hot,data.edge_index.long()))# 128
        cold2 = F.relu(self.sage2_cold(cold,data.edge_index.long()))# 128

        cos_m_h = self.cos_sim(Main2.float(),hot2.float())
        cos_m_c = self.cos_sim(Main2.float(),cold2.float())
        
        Main2_hot = self.adapter_hotM(Main2,cos_m_h)
        adahot2 = self.adapter_Mhot(hot2,cos_m_h)
        
        Main2_cold = self.adapter_Mcold(Main2,cos_m_c)
        adacold2 = self.adapter_coldM(cold2,cos_m_c)
        
        Main2 = torch.cat([Main2,Main,adahot2,adacold2],dim=1) #448
        
        random_noise = torch.rand_like(hot2).to(hot2.device)
        hot2 = hot2 + torch.sign(hot2) * F.normalize(random_noise, dim=-1) * 0.3
        random_noise = torch.rand_like(cold2).to(cold2.device)
        cold2 = cold2 - torch.sign(cold2) * F.normalize(random_noise, dim=-1) * 0.3


        hot2 = torch.cat([hot2,hot,Main2_hot],dim=1) # 320
        cold2 = torch.cat([cold2,cold,Main2_cold],dim=1)# 320
        
        #############################################################################
        
        Main3 = F.relu(self.sage3_1(Main2,data.edge_index.long()))
        hot3 = F.relu(self.sage3_2hot(hot2,data.edge_index.long()))
        cold3 = F.relu(self.sage3_2cold(cold2,data.edge_index.long()))

        #############################################################################
        
        random_noise = torch.rand_like(hot3).to(hot3.device)
        hot3 = hot3 + torch.sign(hot3) * F.normalize(random_noise, dim=-1) * 0.3
        random_noise = torch.rand_like(cold3).to(cold3.device)
        cold3 = cold3 - torch.sign(cold3) * F.normalize(random_noise, dim=-1) * 0.3
        
        #############################################################################
        if self.pooling == 'MTP':
            Main = self.pool(Main3, data.batch, data.edge_index.long(),Weather = None)
            alpha = self.pool(hot3, data.batch, data.edge_index.long(),Weather = "hot")
            beta = self.pool(cold3, data.batch, data.edge_index.long(),Weather = "cold")

        return Main,alpha,beta


class MEGA_GO(torch.nn.Module):
    def __init__(self, out_dim, pooling='MTP'):
        super(MEGA_GO,self).__init__()

        self.out_dim = out_dim
        self.one_hot_embed = nn.Embedding(21,96)
        self.proj_onehot = nn.Linear(96, 512) 
        self.proj_esm = nn.Linear(1280, 512) 

        self.adaAF_1_onehot = adaAF_1(96,128,512)
        self.adaAF_1_esm = adaAF_1(1280,768,512)
        self.adaAF_2_fuse = adaAF_2(512)

        self.pooling = pooling
        self.dhgnn = DH_GNN(pooling=pooling)

        self.readoutINT = nn.Sequential(
                        nn.Linear(1536,1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, out_dim),
                        nn.Sigmoid()
        )
        self.readout = nn.Sequential(
                        nn.Linear(512,1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, out_dim),
                        nn.Sigmoid()
        )
    def forward(self, data,idx_batch,ith_epoch):
        
        x_one_hot_origin = self.one_hot_embed(data.native_x.long())
        x_one_hot = self.proj_onehot(x_one_hot_origin)

        x_esm_origin = data.x.float()
        x_esm = self.proj_esm(x_esm_origin)

        x_fuse = x_one_hot + x_esm
        
        #adaAF
        x_ONE_HOT = self.adaAF_1_onehot(x_one_hot_origin)
        x_ESM = self.adaAF_1_esm(x_esm_origin)
        
        x_FUSE = x_ONE_HOT + x_ESM
        x_FUSE = self.adaAF_2_fuse(x_FUSE)
        
        x = F.relu(0.7*x_fuse+ 0.3*x_FUSE)
        #adaAF

        #Readout
        Main,alpha,beta = self.dhgnn(x, data)
        integration = torch.cat((Main,alpha,beta),dim=1)
        y_pred = self.readoutINT(integration)
        y_pred_alpha = self.readout(alpha)
        y_pred_beta = self.readout(beta)
        #Readout
        return y_pred,y_pred_alpha,y_pred_beta
