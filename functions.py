# functions and utils
import pandas as pd
import h5py as h5

import numpy as np
from sklearn.model_selection import KFold
import torch


# select data that passed QC
# https://www.sciencedirect.com/science/article/pii/S1053811919304963#bib28
def get_selected_data(datafile):
    filename=r'/AbideI_sbjID_Khola_v2.csv'
    select_ID_I=pd.read_csv(filename, header=None)

    filename=r'/Abide_I_II_sbjID_Khola_v2.csv'
    select_ID_I_II=pd.read_csv(filename, header=None)

    select_ID=pd.concat([select_ID_I, select_ID_I_II]) 
    select_ID=select_ID.to_numpy()

    hfile = h5.File(datafile)
    cid=hfile['summaries'].attrs['SUB_ID']

    select_ID_i=[]
    missing_ID=[]
    for s in select_ID:
        itemindex=np.where(cid == s)
        if np.size(itemindex)==0:
            missing_ID.append(s)
        
        else: 
            select_ID_i.append(itemindex[0][0])
    return select_ID_i

# divide the data use for CV from data used for testing
def split_cv_test(datafile, summaryMeasure, select_ID_i):

    split_ratios=[0.9,0.1]
    hfile = h5.File(datafile)
    
    #select data
    data_all = hfile['summaries/'+summaryMeasure].value[select_ID_i] 
    label_all= hfile['summaries'].attrs['DX_GROUP'][select_ID_i]
    
    dataset_size = len(label_all)
    
    #RANDOMIZATION OF THE DATA WITH SEED
    np.random.seed(23)
    idx = np.arange(len(data_all))
    np.random.shuffle(idx)

    data_all = data_all[idx]
    label_all = label_all[idx]
    hfile.close()

    split_cross = int(split_ratios[0] * dataset_size) 
    data = data_all[range(0,split_cross)]
    label = label_all[range(0,split_cross)]
    return data, label


def get_train_val_splits(n_splits):
    kf = KFold(n_splits=n_splits,random_state=42, shuffle=True)

    n_subjects = 1045  #len(data) - test data HARD CODED SO CAN BE OUTSIDE OF THE LOOK
    X = np.zeros((n_subjects, 1))

    train_split = []
    val_split = []
    split_name = []

    for i_split, (train_list, val_list) in enumerate(kf.split(X)):

        train_split.append(train_list)
        val_split.append(val_list)
    return train_split, val_split

class load_data(Dataset):
    def __init__(self, 
                 split,
                 train_split, val_split, 
                 summaryMeasure, 
                 data, label
                 ):
                 
        trData=data[train_split]
        meanTrData = np.mean(trData, axis=0)
        stdTrData = np.std(trData, axis=0)
        
        # SPLIT DATA               
        if split=='train':
            self.data = (trData - meanTrData)/(stdTrData + 1.0)
            self.label = label[train_split]
            print('TRAINING n ads {}/{}'.format((self.label==1).sum(),len(self.label)))
        elif split =='val':
            valData= data[val_split]
            self.data = (valData - meanTrData)/(stdTrData + 1.0)
            self.label =label[val_split]
            print('VALIDATION n ads {}/{}'.format((self.label==1).sum(),len(self.label)))
        elif split =='test':
            testData = data[test_split]
            self.data = (testData - meanTrData)/(stdTrData + 1.0)
            self.label = label[test_split]
        else:
             raise ValueError('Error! the split name is not recognized')
                           
        # reshape for 3 d 
        nsbj= len(self.label)  
        self.data = self.data.reshape(nsbj, 1, 45, 54, 45)       
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, item):
        return torch.tensor(self.data[item]), torch.tensor(self.label[item])   


def split_get_test_data(datafile, summaryMeasure,select_ID_i):

    split_ratios=[0.9,0.1]

    hfile = h5.File(datafile)
    
    #select the good data only
    data_all = hfile['summaries/'+summaryMeasure].value[select_ID_i] 
    label_all= hfile['summaries'].attrs['DX_GROUP'][select_ID_i]

    dataset_size = len(label_all)
  #RANDOMIZATION OF THE DATA WITH SEED
    np.random.seed(23)
    idx = np.arange(len(data_all))
    np.random.shuffle(idx)
    
    data_all = data_all[idx]
    label_all = label_all[idx]
    hfile.close()

    split_cross = int(split_ratios[0] * dataset_size) 
    data = data_all[range(split_cross, dataset_size)]
    label = label_all[range(split_cross, dataset_size)]
    return data, label


def split_cv_test_MMmodel(datafile, select_ID_i,SummaryMeasures_List):

    split_ratios=[0.9,0.1]

    hfile = h5.File(datafile)
    
    #select the good data only
    data_all=[0]
    for summaryMeasure in SummaryMeasures_List:

        data_sm= hfile['summaries/'+summaryMeasure].value[select_ID_i]
        if len(data_all)==1:
            data_all = data_sm
        else:
            data_all = np.concatenate((data_all, data_sm), axis = 4)
            print('loading data...')
    print('data_loaded')  
    
    label_all= hfile['summaries'].attrs['DX_GROUP'][select_ID_i]
    
    dataset_size = len(label_all)
    
    #RANDOMIZATION OF THE DATA WITH SEED
    np.random.seed(23)
    idx = np.arange(len(data_all))
    np.random.shuffle(idx)

    data_all = data_all[idx]
    label_all = label_all[idx]
    hfile.close()

    split_cross = int(split_ratios[0] * dataset_size) 
    data = data_all[range(0,split_cross)]
    label = label_all[range(0,split_cross)]
    return data, label