# Single Modality model train and validate
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from  sklearn.metrics  import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

from functions import get_train_val_splits, load_data
from architechtures import CNN_model

MODEL_STORE_PATH = [] # path to results
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_split, val_split = get_train_val_splits(n_split=5)

def SM_train_val_kfoldCV(summaryMeasure, data, label, batch_size, num_epochs,n_split):
    
    # CROSSVALIDATION LOOP
    for k in np.arange(n_split): 
        train_split_k=train_split[k]
        val_split_k=val_split[k]
        
        #reset the model before new cv
        model=CNN_model()
        model.to(device) 
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(),momentum=0.9, lr = 0.001, weight_decay=1e-3)

        # load data
        trainData = load_data(split='train', 
                                train_split=train_split_k, val_split=val_split_k, 
                                summaryMeasure=summaryMeasure, data = data, label= label)
        valData = load_data(split='val', 
                            train_split=train_split_k, val_split=val_split_k, 
                            summaryMeasure=summaryMeasure, data = data, label= label)
        

        train_loader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=valData, batch_size=batch_size, shuffle=True)

    # START TRAINING FOR THIS CV

        # collect loss and acc of each epoch
        epoch_loss_train = []
        epoch_acc_train = []

        epoch_loss_val  = []
        epoch_balacc_val= []
        epoch_F1_Score = []
        best_acc = None

        
        # TRAIN THE MODEL
        for epoch in range(num_epochs):
            total_ep = 0.0
            correct_ep = 0.0
            loss_ep = 0.0

            model.train()

            for i, (images, labels) in enumerate(train_loader):

                if device.type == "cuda":
                    images= images.to(device)     
                    labels= labels.to(device)
                labels=labels.reshape(len(labels),1).type(torch.cuda.FloatTensor)

                # Run the forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backprop and perform  optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                predicted= outputs.data>0.0 
                correct = (predicted == labels).sum().item()

                total_ep += total
                correct_ep += correct
                loss_ep += (loss.item()*len(labels))

            epoch_acc_train.append((correct_ep/ total_ep))  
            epoch_loss_train.append(loss_ep/ total_ep)

            # Test the model on validation data
            model.eval()

            total_ep = 0.0
            correct_ep = 0.0
            loss_ep = 0.0
                    
            # F1 calc
            F1_labels=[]
            F1_pred=[]

            for i, (images, labels) in enumerate(val_loader): 

                if device.type == "cuda":
                    images= images.to(device)     
                    labels= labels.to(device)
                labels=labels.reshape(len(labels),1).type(torch.cuda.FloatTensor)

                outputs_v = model(images) # the forward uses the entire batch together
                predicted= outputs_v.data>0.0 #define True if >0.5 and False if <0.5

                # collect true labels and predicted for metrix calc
                if i==0:
                    F1_labels=labels.int().cpu().numpy()
                    F1_pred=predicted.int().cpu().numpy()
                else:
                    F1_labels= np.concatenate((F1_labels, labels.int().cpu().numpy()))
                    F1_pred = np.concatenate((F1_pred, predicted.int().cpu().numpy()))
                
                loss_v = criterion(outputs_v, labels) 

                total_v = labels.size(0) 
                correct_v = (predicted == labels).sum().item() 

                total_ep += total_v 
                correct_ep += correct_v 
                loss_ep += (loss_v.item()*len(labels))
            
            # calculate metrices of this epoch
            current_ep_loss=loss_ep/total_ep
            
            current_ep_acc=(correct_ep/total_ep)
            acc = accuracy_score(F1_labels, F1_pred)
            balacc= balanced_accuracy_score(F1_labels, F1_pred)
            
            F1_Score = f1_score(F1_labels, F1_pred, average='weighted')
            tn, fp, fn, tp = confusion_matrix(F1_labels, F1_pred).ravel() 
            
            
            if not best_acc or best_acc < current_ep_acc:
                best_acc = current_ep_acc
                torch.save(model.state_dict(), MODEL_STORE_PATH + '/' + summaryMeasure + '_BEST_model'
                                + '_CV' + str(k+1)
                                + '.pt')  

            # collect matrixes of all the epochs
            epoch_loss_val.append((loss_ep/total_ep))           
            epoch_balacc_val.append(balacc)
            
        print('{}, CV_{}, best accuracy = {}'.format(summaryMeasure, k+1, acc))



def SM_test(summaryMeasure, k, data, label, data_test, label_test, batch_size, is_mmens=False):
    
        train_split_k=train_split[k]
        val_split_k=val_split[k]
    #reset the model before new cv
        model=CNN_model()
        model.to(device) 
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(),momentum=0.9, lr = 0.001, weight_decay=1e-3)
               
        # load data
        train_split_k=train_split[k]
        testData = load_data(split='test', 
                              train_split=train_split_k, val_split=val_split_k,
                              summaryMeasure=summaryMeasure, 
                              data = data, label= label, # need train data for mean and sd normalization
                              data_test = data_test, label_test= label_test)
        
        test_loader = DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

        # set and load best model
        model = CNN_model()
        model.load_state_dict(torch.load(
        MODEL_STORE_PATH + '/' + summaryMeasure + '_BEST_model'
                                       + '_CV' + str(k+1)
                                       + '.pt'))
        
        model.to(device)
    # START TRAINING FOR THIS CV
        # TEST EVALUATION 
        model.eval()
        
        total_ep = 0.0
        correct_ep = 0.0
        loss_ep = 0.0
        current_ep_acc=0.0
        current_ep_loss=0.0
        total_step_v = len(test_loader)
        
    # F1 calc
        F1_labels=[]
        F1_pred=[]

        for i, (images, labels) in enumerate(test_loader): #TEST THE MODEL ON THE VALIDATION

            if device.type == "cuda":
                images= images.to(device)     
                labels= labels.to(device)
            labels=labels.reshape(len(labels),1).type(torch.cuda.FloatTensor)

            outputs_v = model(images) # the forward uses the entire batch together
            predicted= outputs_v.data>0.0 #define True if >0.5 and False if <0.5

            # collect true labels and predicted for metrix calc

            if i==0:
                F1_labels=labels.int().cpu().numpy()
                F1_pred=predicted.int().cpu().numpy()
            else:
                F1_labels= np.concatenate((F1_labels, labels.int().cpu().numpy()))
                F1_pred = np.concatenate((F1_pred, predicted.int().cpu().numpy()))

            loss_v = criterion(outputs_v, labels) 

            total_v = labels.size(0) # get it in case the last batch has different number of sample
            correct_v = (predicted == labels).sum().item() # sum the correct answers for this batch
            #acc_list_v.append(correct_v / total_v) # n of correct / n of sample in this batch

            total_ep += total_v # sum all the samples. it must end up being the tot training set
            correct_ep += correct_v # sum together the right answers in entire training set
            loss_ep += (loss_v.item()*len(labels))
        
            current_ep_acc=(correct_ep/total_ep)*100
            current_ep_loss=loss_ep/total_ep
        
        acc = accuracy_score(F1_labels, F1_pred)
        bal_acc= balanced_accuracy_score(F1_labels, F1_pred)
        F1_Score =f1_score(F1_labels, F1_pred, average='weighted')#, zero_division=1) 
        tn, fp, fn, tp = confusion_matrix(F1_labels, F1_pred).ravel()
        

        print('CV {}, sumMeasure: {} Test Accuracy of the model is: {}, F1: {}%'
                          .format(k+1, summaryMeasure, current_ep_acc, F1_Score))

        if is_mmens == True:
            return F1_labels.reshape(-1), F1_pred.reshape(-1)
        else:
            return bal_acc, f1_score 

def SM_test_kfold_CV(summaryMeasure, n_split, data, label, data_test, label_test, batch_size, is_mmens=False):
    cv_kfold_res = np.empty((2,5))
    for k in np.arange(n_split): 
        print('Start Cross-validation k= {}'.format(k+1))       

        bal_acc, F1_score = SM_test(summaryMeasure, k, data, label, data_test, label_test, batch_size, is_mmens=False)
        
        cv_kfold_res[0,k]=bal_acc
        cv_kfold_res[1,k]=F1_score
    print('SM model cross-validates balanced accuracy: {}, f1-score: {}'.format(cv_kfold_res[0,:].mean(), cv_kfold_res[1,:].mean()))