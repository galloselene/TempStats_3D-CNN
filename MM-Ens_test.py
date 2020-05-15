# Multi Measures Ensabled Model

import numpy as np
import pandas as pd
from  sklearn.metrics  import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from functions import split_cv_test, split_get_test_data
from SM_train_val_kford_CV import SM_test

def MMens_test(SummaryMeasures_List, n_split, batch_size, kfold, datafile, select_ID_i):
    store_labels=np.empty([5*len(SummaryMeasures_List), 117], dtype=int)
    store_predictions=np.empty([5*len(SummaryMeasures_List), 117], dtype=int)

    for summaryMeasure in SummaryMeasures_List:
        data, label =  split_cv_test(datafile, summaryMeasure, select_ID_i)
        data_test, label_test = split_get_test_data(datafile, summaryMeasure, select_ID_i) #load test data

        labels, predictions = SM_test(summaryMeasure, n_split, data, label, data_test, label_test, batch_size, is_mmens=True)
        store_labels[n, :] = labels
        store_predictions[n, :] = store_predictions


    kfold=n_split
    cv_kfold_res = np.empty((2,5))
    for k in range(0, kfold): 
        
        i = np.arange(0, 45, 5) +k
        lab = store_labels[i[0],:]
        pred = store_predictions[i,:]
        pred = pred.sum(axis=0)>4 
        
        F1_Score =f1_score(lab, pred, average='weighted')#, zero_division=1) 
        bal_acc= balanced_accuracy_score(lab, pred)

        cv_kfold_res[0,k]=bal_acc
        cv_kfold_res[1,k]=F1_Score
    print('MM-Ensemble model cross-validates balanced accuracy: {}, f1-score: {}'.format(cv_kfold_res[0,:].mean(), cv_kfold_res[1,:].mean()))
