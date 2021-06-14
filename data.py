import pandas as pd
import numpy as np

import os
import wget, zipfile
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# use this class for uci/ kaggle datasets.
def data_download(dataset):
    url= None
    if dataset=='1995_income':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        dataset_name = '1995_income'
        target = 14
    elif dataset=='bank_marketing':
        dataset_name = 'bank-full'
        target = 16
    elif dataset == 'qsar_bio':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        target = 41
        dataset_name = 'qsar_bio'
    elif dataset == 'online_shoppers':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'
        target = 17
        dataset_name = 'online_shoppers'
    elif dataset == 'blastchar':
        dataset_name = 'blastchar'
        target = 20
    elif dataset == 'htru2':
        dataset_name = 'htru2'
        target = 8
    elif dataset == 'shrutime':
        dataset_name = 'Churn_Modelling'
        target = 11
    elif dataset == 'spambase':
        dataset_name = 'spambase'
        target = 57
        
    elif dataset == 'loan_data':
        dataset_name = 'loan_data'
        target = 13
    elif dataset == 'arcene':
        dataset_name = 'arcene'
        target = 'target'
    elif dataset == 'creditcard':
        dataset_name = 'creditcard'
        target = 30
    elif dataset == 'arrhythmia':
        dataset_name = 'arrhythmia'
        target = 226
    elif dataset == 'forest':
        dataset_name = 'forest'
        target = 49
    elif dataset == 'kdd99':
        dataset_name = 'kdd99'
        target = 39
    else:
        print('TODO: HAVE TO DO THIS DATASET!')
        
    out = Path(os.getcwd()+'/data/'+dataset_name+'.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        if url is not None:
            print(f"Downloading {dataset} file...")
            wget.download(url, out.as_posix())
        else:
            raise 'Download the dataset from the link mentioned in github first'
        
    if dataset=='1995_income':
        train = pd.read_csv(out,header=None)
    elif dataset=='bank_marketing':
        train = pd.read_csv(out,sep=';',header=None, skiprows=1)
    elif dataset == 'qsar_bio':
        train = pd.read_csv(out,sep=';',header=None)
    elif dataset == 'online_shoppers':
        train = pd.read_csv(out,sep=',',header=None, skiprows=1)
    elif dataset == 'blastchar':
        train = pd.read_csv(out,sep=',',header=None, skiprows=1)
    elif dataset == 'htru2':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
    elif dataset=='shrutime':
        train = pd.read_csv(out,sep=',',header=None, skiprows=1)
        train = train.iloc[:,2:]
        train.columns = range(train.shape[1])
    elif dataset == 'spambase':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
        train = train.iloc[:,1:]
    elif dataset == 'loan_data':
        train = pd.read_csv(out,sep=',', skiprows=0)
        train.columns = range(train.shape[1])
#         cat_cols = [0,1,6,10,11,12]
        cat_cols = [1]
        return train, target, cat_cols
    elif dataset == 'arcene':
        train = pd.read_csv(out,sep=',',skiprows=0)
        tg_list = train['target'].tolist()
        # train = train.loc[:, train.std() > 7]
        train = train.loc[:, train.std() > 150]
        train['target'] = tg_list
    elif dataset == 'creditcard':
        train = pd.read_csv(out,header=None,skiprows=1)
        train = train.iloc[:,1:]
    elif dataset == 'arrhythmia':
        train = pd.read_csv(out,header=None,skiprows=1)
    elif dataset == 'forest':
        train = pd.read_csv(out,header=None,skiprows=1)
    elif dataset == 'kdd99':
        train = pd.read_csv(out,header=None,skiprows=1)
    return train, target


#use this class only for automl datasets since dataset format is different.
def data_prep_automl(dataset, datasplit=[.65, .15, .2]):
    if dataset == 'philippine':
        url = 'http://www.causality.inf.ethz.ch/AutoML/philippine.zip'
        dataset_name = 'philippine'
    elif dataset == 'volkert':
        url = 'http://www.causality.inf.ethz.ch/AutoML/volkert.zip'
        dataset_name = 'volkert'
    out = Path(os.getcwd()+'/data/'+dataset_name+'/'+dataset_name+'.zip')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        print("Downloading file...")
        wget.download(url, out.as_posix())
        with zipfile.ZipFile(out, 'r') as zip_ref:
            zip_ref.extractall(Path(os.getcwd()+'/data/'+dataset_name+'/'))
       
    p1 = Path(os.getcwd()+'/data/'+dataset_name+'/'+dataset_name+'_train.data') 
    p2 = Path(os.getcwd()+'/data/'+dataset_name+'/'+dataset_name+'_train.solution')    
    p3 = Path(os.getcwd()+'/data/'+dataset_name+'/'+dataset_name+'_feat.type')    

    train = pd.read_csv(p1,sep=' ',header=None, skiprows=0)
    train = train.iloc[:,:-1]
    if dataset == 'volkert':
        okindx = np.where(np.array(train.std()) !=0)[0]
        train = train.loc[:,okindx]
        train.columns = range(train.shape[1])
        y = pd.read_csv(p2,sep=' ',header=None, skiprows=0).drop(columns=[10]).to_numpy()
        y = list(np.argmax(y, axis=1))
    else:    
        y = pd.read_csv(p2,sep=' ',header=None, skiprows=0)[0].tolist()
    train['target'] = y
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(train.shape[0],))
        
    data_types = pd.read_csv(p3,sep=' ',header=None, skiprows=0)[0].tolist()
    
    if dataset == 'volkert':
        return train, np.array(data_types)[okindx]
    return train.loc[:,], np.array(data_types)

        
def data_mask_split(X,y,mask,y_mask,indices,mask_det,stage):
    try:
        x_d = {
            'data': X.values[indices],
            'mask': mask.values[indices]
        }
    except:
        x_d = {
            'data': X.values[indices],
            'mask': mask[indices]
        }
    y_d = {
        'data': y.values[indices].reshape(-1, 1),
        'mask': y_mask[indices].reshape(-1, 1)
    } 
    if mask_det is not None:
        if stage == 'train' and mask_det['avail_train_y'] > 0:
            avail_ys = np.random.choice(y_d['mask'].shape[0], mask_det['avail_train_y'], replace=False)
            y_d['mask'][avail_ys,:] = 1
        
        if stage != 'train' and mask_det['test_mask'] < 10e-3:
            x_d['mask'] = np.ones_like(x_d['mask'])

    return x_d, y_d
    
def data_prep(dataset,seed,mask_det=None, datasplit=[.65, .15, .2]):
    np.random.seed(seed) 
    if dataset in ['1995_income','bank_marketing','qsar_bio','online_shoppers','blastchar','htru2','shrutime','spambase','arcene','creditcard','arrhythmia','forest','kdd99']:
        train,target = data_download(dataset)
        temp = train.fillna("ThisisNan")
        if "Set" not in train.columns:
            train["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(train.shape[0],))
        unused_feat = ['Set']
        features = [ col for col in train.columns if col not in unused_feat+[target]] 
        train_indices = train[train.Set=="train"].index
        valid_indices = train[train.Set=="valid"].index
        test_indices = train[train.Set=="test"].index
        categorical_columns = []
        categorical_dims =  {}
        for col in train.columns[train.dtypes == object]:
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

        for col in train.columns[train.dtypes == 'float64']:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)
        for col in train.columns[train.dtypes == 'int64']:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)

        
        cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
        con_idxs = list(set(range(len(features))) - set(cat_idxs))
        cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
        train[target] = train[target].astype(int)  
    elif dataset in ['loan_data']:
        train,target,categorical_columns = data_download(dataset)
        cont_columns = set(train.columns.tolist()) - set(categorical_columns)
        temp = train.fillna("ThisisNan")
        if "Set" not in train.columns:
            train["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(train.shape[0],))
        unused_feat = ['Set']
        features = [ col for col in train.columns if col not in unused_feat+[target]] 
        train_indices = train[train.Set=="train"].index
        valid_indices = train[train.Set=="valid"].index
        test_indices = train[train.Set=="test"].index
        categorical_dims =  {}
        for col in categorical_columns:
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_dims[col] = len(l_enc.classes_)
        for col in cont_columns:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)

         
        cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
        con_idxs = list(set(range(len(features))) - set(cat_idxs))
        cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
        train[target] = train[target].astype(int) 
    elif dataset in ['philippine','volkert']:
        train, data_types = data_prep_automl(dataset)
        temp = train.fillna("ThisisNan")
        target = 'target'
        train_indices = train[train.Set=="train"].index
        valid_indices = train[train.Set=="valid"].index
        test_indices = train[train.Set=="test"].index
        unused_feat = ['Set']
        features = [ col for col in train.columns if col not in unused_feat+[target]]
        
        cat_idxs = np.where(np.isin(data_types, ['Categorical','Binary']))[0]
        con_idxs = np.where(data_types == 'Numerical')[0]
        for col in cat_idxs:
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = train[col].astype(str)
            train[col] = l_enc.fit_transform(train[col].values)
        for col in con_idxs:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)
        if len(cat_idxs)>0:
            cat_dims = train[cat_idxs].nunique().tolist()
        else:
            cat_dims = np.array([])

    if mask_det is not None:
        temp = temp[features]
        nan_mask = temp.ne("ThisisNan").astype(int)
        gen_mask = np.random.choice(2,(train[features].shape),p=[mask_det["mask_prob"],1-mask_det["mask_prob"]])
        mask = np.multiply(nan_mask,gen_mask)
        y_mask = np.zeros_like(train[target])
    else:
        mask = np.ones_like(train[features])
        y_mask = np.zeros_like(train[target])
    X = train[features]
    Y = train[target]
    X_train, y_train = data_mask_split(X,Y,mask,y_mask,train_indices,mask_det,'train')
    X_valid, y_valid = data_mask_split(X,Y,mask,y_mask,valid_indices,mask_det,'valid')
    X_test, y_test = data_mask_split(X,Y,mask,y_mask,test_indices,mask_det,'test')
    # print(X_train['data'].shape, y_train['data'].shape,X_valid['data'].shape,y_valid['data'].shape,X_test['data'].shape,y_test['data'].shape)
    
    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std


        

class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,continuous_mean_std=None, is_pretraining=False,tag=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.is_pretraining = is_pretraining
        self.tag = tag
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
            self.y = Y['data']
            self.y_mask = Y['mask']
            
        else:
            self.y = np.expand_dims(np.array(Y['data']),axis=-1)
            self.y_mask = np.expand_dims(np.array(Y['mask']),axis=-1)
        
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        
        

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        if self.is_pretraining:
            if self.tag is not None:
                return np.concatenate((self.X1[idx], self.y[idx])), self.X2[idx], np.concatenate((self.X1_mask[idx], self.y_mask[idx])), self.X2_mask[idx],self.tag[idx]
            else:
                return np.concatenate((self.X1[idx], self.y[idx])), self.X2[idx], np.concatenate((self.X1_mask[idx], self.y_mask[idx])), self.X2_mask[idx]
        else:
            return self.X1[idx], self.X2[idx], self.y[idx]


def vision_data_prep(dataset,seed,mask_det=None, datasplit=[0.8,0.2]):
    np.random.seed(seed) 
    import torch
    from torchvision import transforms, datasets
    data_dir = './data/vision/'

    X_train, y_train, X_valid, y_valid, X_test, y_test = {},{},{},{},{},{}
    if dataset == 'mnist':
        temp = datasets.MNIST(root=data_dir, train=True, 
                download=True)
        split_labels = np.random.choice(["train", "valid"], p = datasplit, size=(len(temp),))
        train_loc, valid_loc = np.where(split_labels == 'train')[0], np.where(split_labels == 'valid')[0] 
        X_train['data'], y_train['data'] = torch.flatten(temp.data[train_loc,:,:].long(), start_dim=1, end_dim=-1).numpy() , temp.targets[train_loc].numpy()
        X_valid['data'], y_valid['data'] = torch.flatten(temp.data[valid_loc,:,:].long(), start_dim=1, end_dim=-1).numpy(), temp.targets[valid_loc].numpy()
        test_set =  datasets.MNIST(root=data_dir, train=False, 
                download=True)
        X_test['data'], y_test['data'] = torch.flatten(test_set.data.long(), start_dim=1, end_dim=-1).numpy(), test_set.targets.numpy()
        
    else:
        raise 'Dataset not found'

    cat_dims, cat_idxs, con_idxs = np.repeat(256,X_train['data'].shape[-1]),np.arange(X_train['data'].shape[-1]),np.array([])
    if mask_det is not None:
        mp = mask_det["mask_prob"]
        X_train['mask'] = np.random.choice(2,(X_train['data'].shape),p=[mp,1-mp])
        X_valid['mask'] = np.random.choice(2,(X_valid['data'].shape),p=[mp,1-mp])
        X_test['mask'] = np.random.choice(2,(X_test['data'].shape),p=[mp,1-mp])
    else:
        X_train['mask'] = np.ones_like(X_train['data'])
        X_valid['mask'] = np.ones_like(X_valid['data'])
        X_test['mask'] = np.ones_like(X_test['data'])


    y_train['mask'] = np.zeros_like(y_train['data'])
    y_valid['mask'] = np.zeros_like(y_valid['data'])
    y_test['mask'] = np.zeros_like(y_test['data'])

    

    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, 0, 0



