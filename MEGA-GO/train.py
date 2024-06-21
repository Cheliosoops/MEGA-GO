from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import MEGA_GO
from sklearn import metrics
from utils import log
import argparse
from config import get_config

def train(config, task, suffix):

    train_set = GoTermDataset("train", task, config.AF2model)
    valid_set = GoTermDataset("val", task, config.AF2model)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    
    output_dim = valid_set.y_true.shape[-1]
    model = MEGA_GO(output_dim, config.pooling).to(config.device)
    
    optimizer = torch.optim.Adam(
        params = model.parameters(), 
        **config.optimizer,
        )

    bce_loss = torch.nn.BCELoss(reduce=False)
    
    train_loss = []
    val_loss = []
    val_aupr = []
    val_Fmax = []

    y_true_all = valid_set.y_true.float().reshape(-1)

    for ith_epoch in range(config.max_epochs):
        torch.cuda.empty_cache()
        for idx_batch, batch in enumerate(train_loader):
            model.train()
            y_pred,y_pred_alpha,y_pred_beta= model(batch[0].to(config.device),idx_batch,ith_epoch)
            y_true = batch[1].to(config.device)
            _loss = bce_loss(y_pred, y_true)
            _loss = _loss.mean()
            hot_loss = bce_loss(y_pred_alpha,y_true)
            hot_loss = hot_loss.mean()
            cold_loss = bce_loss(y_pred_beta,y_true)
            cold_loss = cold_loss.mean()

            loss = (_loss + hot_loss + cold_loss)

    
            # if 35 < idx_epoch < 75:
            #     for param in base_optimizer.param_groups:
            #         param['lr'] = 1e-7
            # elif 75 < idx_epoch < 150:
            #    for param in base_optimizer.param_groups:
            #         param['lr'] = 1e-8
            log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)}")
            train_loss.append(loss.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_loss = 0
        model.eval()
        y_pred_all = []
        n_nce_all = []
        y_pred_hot_all = []
        y_pred_cold_all = []
        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                y_pred,y_pred_alpha,y_pred_beta = model(batch[0].to(config.device),idx_batch,ith_epoch)
                y_pred_all.append(y_pred)
                y_pred_alpha_all.append(y_pred_alpha)
                y_pred_beta_all.append(y_pred_beta)
            
            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)
            y_pred_alpha_all = torch.cat(y_pred_alpha_all, dim=0).cpu().reshape(-1)
            y_pred_beta_all = torch.cat(y_pred_beta_all, dim=0).cpu().reshape(-1)
            eval_loss = bce_loss(y_pred_all, y_true_all).mean() + bce_loss(y_pred_alpha_all, y_true_all).mean() + bce_loss(y_pred_beta_all, y_true_all).mean()
            
                
            aupr = metrics.average_precision_score(y_true_all.numpy(), y_pred_all.numpy(), average="samples")
            val_aupr.append(aupr)
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)} ||| aupr: {round(float(aupr),3)}")
            val_loss.append(eval_loss.numpy())

            torch.save(model.state_dict(), config.model_save_path + task + f"{suffix}.pt")
            torch.save(
                    {
                        "train_bce": train_loss,
                        "val_bce": val_loss,
                        "val_aupr": val_aupr,
                    }, config.loss_save_path + task + f"{suffix}.pt"
                )

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='bp', choices=['bp','mf','cc'], help='')
    p.add_argument('--suffix', type=str, default='Lee', help='')
    p.add_argument('--device', type=str, default="0" , help='')
    p.add_argument('--pooling', default='MTP', type=str, choices=['MTP','GMP'], help='Set your operation mode for pooling')
    p.add_argument('--AF2model', default=False, type=str2bool, help='whether to use AF2model for training')
    p.add_argument('--batch_size', type=int, default=32, help='')
    
    args = p.parse_args()
    config = get_config()
    config.optimizer['lr'] = 1e-6
    config.batch_size = args.batch_size
    config.max_epochs = 150
    if args.device != '':
        config.device = "cuda:" + args.device
    print(args)
    config.pooling = args.pooling
    config.AF2model = args.AF2model
    train(config, args.task, args.suffix)
