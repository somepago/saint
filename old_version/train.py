import torch
from torch import nn
from models import SAINT, SAINT_vision

from data import data_prep,DataSetCatCon

import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, imputations_acc_justy
from augmentations import embed_data_mask
from augmentations import add_noise

import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='1995_income', type=str, choices=['1995_income','bank_marketing','qsar_bio','online_shoppers','blastchar','htru2','shrutime','spambase','philippine','mnist','loan_data','arcene','volkert','creditcard','arrhythmia','forest','kdd99']) 
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--active_log', action = 'store_true')

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix','gauss_noise'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default= 0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])


opt = parser.parse_args()
torch.manual_seed(opt.set_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opt.attentiontype in ['colrow','row']:
    opt.ff_dropout = 0.8
    opt.transformer_depth = 1     
    if opt.dataset in ['arrhythmia','philippine','creditcard']:
        opt.embedding_size = 8
        opt.attention_heads = 4

if opt.dataset in ['arrhythmia']:
    opt.embedding_size = 8
    if opt.attentiontype in ['col']:
        opt.transformer_depth = 1

if opt.dataset in ['philippine']:
    opt.batchsize = 128
    if opt.attentiontype in ['col']:
        opt.embedding_size = 8

if opt.dataset in ['arcene']:
    opt.embedding_size = 4
    if opt.attentiontype in ['colrow','col']:
        opt.attention_heads = 1
        opt.transformer_depth = 4

if opt.dataset in ['mnist']:
    opt.batchsize = 32
    opt.attention_heads = 4
    if opt.attentiontype in ['col']:
        opt.embedding_size = 12
    else:
        opt.embedding_size = 8


print(f"Device is {device}.")

modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.dataset,opt.run_name)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb
    if opt.ssl_avail_y > 0 and opt.pretrain:
        wandb.init(project="saint_ssl", group=opt.run_name, name = opt.run_name + '_' + str(opt.attentiontype)+ '_' +str(opt.dataset))
    else:
        wandb.init(project="saint_all", group=opt.run_name, name = opt.run_name + '_' + str(opt.attentiontype)+ '_' +str(opt.dataset))
    wandb.config.update(opt)



# mask parameters are used to similate missing data scenrio. Set to default 0s otherwise. (pt_mask_params is for pretraining)
mask_params = {
    "mask_prob":opt.train_mask_prob,
    "avail_train_y": 0,
    "test_mask":opt.train_mask_prob
}

pt_mask_params = {
        "mask_prob":opt.mask_prob,
        "avail_train_y": 0,
        "test_mask": 0
    }

print('Downloading and processing the dataset, it might take some time.')
if opt.dataset not in ['mnist']:
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(opt.dataset, opt.set_seed, mask_params)
    continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 
    if opt.dataset == 'volkert':
        y_dim = 10
    else:
        y_dim = 2
else:
    from data import vision_data_prep
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, _, _ = vision_data_prep(opt.dataset, opt.set_seed, mask_params)
    continuous_mean_std = None 
    y_dim = 10

train_bsize = opt.batchsize    
if opt.ssl_avail_y>0:
    train_pts_touse = np.random.choice(X_train['data'].shape[0], opt.ssl_avail_y)
    X_train['data'] = X_train['data'][train_pts_touse,:]
    y_train['data'] = y_train['data'][train_pts_touse]
    
    X_train['mask'] = X_train['mask'][train_pts_touse,:]
    y_train['mask'] = y_train['mask'][train_pts_touse]
    train_bsize = min(opt.ssl_avail_y//4,opt.batchsize)
    
train_ds = DataSetCatCon(X_train, y_train, cat_idxs,continuous_mean_std, is_pretraining=True)
trainloader = DataLoader(train_ds, batch_size=train_bsize, shuffle=True,num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,continuous_mean_std, is_pretraining=True)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs,continuous_mean_std, is_pretraining=True)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

# Creating a different dataloader for the pretraining.
if opt.pretrain:
    if opt.dataset not in ['mnist']:
        _, cat_idxs, _, X_train_pt, y_train_pt, _, _, _, _, train_mean, train_std = data_prep(opt.dataset, opt.set_seed, pt_mask_params)
        ctd = np.array([train_mean,train_std]).astype(np.float32)
    else:
        _, cat_idxs, _, X_train_pt, y_train_pt, _, _, _, _, _, _ = vision_data_prep(opt.dataset, opt.set_seed, pt_mask_params)
        ctd = None
    pt_train_ds = DataSetCatCon(X_train_pt, y_train_pt, cat_idxs,ctd, is_pretraining=True)
    pt_trainloader = DataLoader(pt_train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)


if opt.dataset not in ['mnist','volkert']:
    cat_dims = np.append(np.array(cat_dims),np.array([2])).astype(int) # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification
    model = SAINT(
    categories = tuple(cat_dims), 
    num_continuous = len(con_idxs),                
    dim = opt.embedding_size,                           
    dim_out = 1,                       
    depth = opt.transformer_depth,                       
    heads = opt.attention_heads,                         
    attn_dropout = opt.attention_dropout,             
    ff_dropout = opt.ff_dropout,                  
    mlp_hidden_mults = (4, 2),       
    continuous_mean_std = continuous_mean_std, 
    cont_embeddings = opt.cont_embeddings,
    attentiontype = opt.attentiontype,
    final_mlp_style = opt.final_mlp_style,
    y_dim = y_dim
    )
    vision_dset = False
elif opt.dataset == 'volkert':
    cat_dims = np.append(np.array(cat_dims),np.array([10])).astype(int)
    model = SAINT(
    categories = tuple(cat_dims), 
    num_continuous = len(con_idxs),                
    dim = opt.embedding_size,                           
    dim_out = 1,                       
    depth = opt.transformer_depth,                       
    heads = opt.attention_heads,                         
    attn_dropout = opt.attention_dropout,             
    ff_dropout = opt.ff_dropout,                  
    mlp_hidden_mults = (4, 2),       # relative multiples of each hidden dimension of the last mlp to logits
    continuous_mean_std = continuous_mean_std,
    cont_embeddings = opt.cont_embeddings,
    attentiontype = opt.attentiontype,
    final_mlp_style = opt.final_mlp_style,
    y_dim = y_dim
    )
    vision_dset = False

elif opt.dataset == 'mnist':
    cat_dims = np.append(np.array(cat_dims),np.array([10])).astype(int)
    model = SAINT_vision(
    categories = tuple(cat_dims), 
    num_continuous = len(con_idxs),                
    dim = opt.embedding_size,                           
    dim_out = 1,                       
    depth = opt.transformer_depth,                       
    heads = opt.attention_heads,                         
    attn_dropout = opt.attention_dropout,             
    ff_dropout = opt.ff_dropout,                  
    mlp_hidden_mults = (4, 2),       # relative multiples of each hidden dimension of the last mlp to logits
    continuous_mean_std = continuous_mean_std,
    cont_embeddings = opt.cont_embeddings,
    attentiontype = opt.attentiontype,
    final_mlp_style = opt.final_mlp_style,
    y_dim = y_dim
    )
    vision_dset = True
else:
    print('This dataset is not valid')


criterion = nn.CrossEntropyLoss().to(device)
model.to(device)

if opt.pretrain:
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
    pt_aug_dict = {
        'noise_type' : opt.pt_aug,
        'lambda' : opt.pt_aug_lam
    }
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    print("Pretraining begins!")
    for epoch in range(opt.pretrain_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(pt_trainloader, 0):
            optimizer.zero_grad()
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            
            # embed_data_mask function is used to embed both categorical and continuous data.
            if 'cutmix' in opt.pt_aug:
                from augmentations import add_noise
                x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model,vision_dset)
            else:
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            
            if 'mixup' in opt.pt_aug:
                from augmentations import mixup_data
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=opt.mixup_lam)

            loss = 0
            if 'contrastive' in opt.pt_tasks:
                aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                if opt.pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif opt.pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')
                logits_per_aug1 = aug_features_1 @ aug_features_2.t()/opt.nce_temp
                logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/opt.nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion(logits_per_aug1, targets)
                loss_2 = criterion(logits_per_aug2, targets)
                loss   = opt.lam0*(loss_1 + loss_2)/2
            elif 'contrastive_sim' in opt.pt_tasks:
                aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_1 = model.pt_mlp(aug_features_1)
                aug_features_2 = model.pt_mlp2(aug_features_2)
                c1 = aug_features_1 @ aug_features_2.t()
                loss+= opt.lam1*torch.diagonal(-1*c1).add_(1).pow_(2).sum()
            if 'denoising' in opt.pt_tasks:
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                con_outs =  torch.cat(con_outs,dim=1)
                l2 = criterion2(con_outs, x_cont)
                l1 = 0
                for j in range(len(cat_dims)-1):
                    l1+= criterion1(cat_outs[j],x_categ[:,j])
                loss += opt.lam2*l1 + opt.lam3*l2    
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch: {epoch}, Running Loss: {running_loss}')
        if opt.active_log:
            wandb.log({'pt_epoch': epoch ,'pretrain_epoch_loss': running_loss
            })

optimizer = optim.AdamW(model.parameters(),lr=opt.lr)

best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0

print('Training begins now.')
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
        reps = model.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,len(cat_dims)-1,:]
        y_outs = model.mlpfory(y_reps)
        loss = criterion(y_outs,x_categ[:,len(cat_dims)-1]) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if opt.active_log:
        wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
        'loss': loss.item()
        })
    
    if epoch%5==0:
            model.eval()
            with torch.no_grad():
                if opt.dataset in ['mnist','volkert']:
                    from utils import multiclass_acc_justy
                    accuracy, auroc = multiclass_acc_justy(model, validloader, device)
                    test_accuracy, test_auroc = multiclass_acc_justy(model, testloader, device)
                    if accuracy > best_valid_accuracy:
                        best_valid_accuracy = accuracy
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                else:
                    accuracy, auroc = imputations_acc_justy(model, validloader, device)
                    test_accuracy, test_auroc = imputations_acc_justy(model, testloader, device)


                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                    (epoch + 1, accuracy,auroc ))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                    (epoch + 1, test_accuracy,test_auroc ))
                if opt.active_log:
                    wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })     
                    wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                if auroc > best_valid_auroc:
                    best_valid_auroc = auroc
                    best_test_auroc = test_auroc
                    best_test_accuracy = test_accuracy               
                    torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))         
            model.train()
                


total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
if opt.dataset not in ['mnist','volkert']:
    print('AUROC on best model:  %.3f' %(best_test_auroc))
else:
    print('Accuracy on best model:  %.3f' %(best_test_accuracy))
if opt.active_log:
    wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc , 
    'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })     

