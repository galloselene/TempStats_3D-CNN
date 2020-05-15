# MM-Model train validation and test

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

def MM_train_val_kfoldCV(SummaryMeasures_List, data, label, batch_size, num_epochs,n_split):
    n_channels = len(SummaryMeasures_List)
    # CROSSVALIDATION LOOP
    for k in np.arange(n_split): 
        train_split_k=train_split[k]
        val_split_k=val_split[k]
        #reset the model before new cv
        model=CNN_model(n_channels)
        model.to(device) 
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(),momentum=0.9, lr = 0.001, weight_decay=1e-3)

        # load data
        trainData = load_data(split='train', 
                            train_split=train_split_k, val_split=val_split_k, 
                            data = data, label= label)
        valData = load_data(split='val', 
                            train_split=train_split_k, val_split=val_split_k, 
                            data = data, label= label)


        train_loader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=valData, batch_size=batch_size, shuffle=True)

    # START TRAINING FOR THIS CV
        total_step = len(train_loader)
        total_step_v = len(val_loader)

        # collect loss and acc of each batch
        loss_list = []
        acc_list = []

        loss_v_list=[]
        acc_list_v=[]
        # collect loss and acc of each epoch
        epoch_loss_train = []
        epoch_acc_train = []

        epoch_loss_val  = []
        epoch_myacc_val = []
        epoch_acc_val   = []
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
                #print('outputs: {}'.format(outputs))
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # Backprop and perform  optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                predicted= outputs.data>0.0 #define True if >0.5 and False if <0.5
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)
                #print('predic == labab {}'.format(predicted == labels[:,0]))        

                total_ep += total
                correct_ep += correct
                loss_ep += (loss.item()*len(labels))
                #print('summing correct this batch {}'. format(correct_ep))

            if (i + 1)  == total_step: #print only the last one
                    print('CV {}, Epoch [{}/{}], Step [{}/{}], Training Accuracy of the model this epoch is: {} %'
                        .format(k+1, epoch + 1, num_epochs, i + 1, total_step, (correct_ep/ total_ep) * 100))

            epoch_acc_train.append((correct_ep/ total_ep))  
            epoch_loss_train.append(loss_ep/ total_ep)


            # Test the model 

            model.eval()

            total_ep = 0.0
            correct_ep = 0.0
            loss_ep = 0.0


            # F1 calc
            F1_labels=[]
            F1_pred=[]

            for i, (images, labels) in enumerate(val_loader): #TEST THE MODEL ON THE VALIDATION
            #for i, (images, labels) in enumerate(train_loader): #TEST THE MODEL ON THE TRAINING      

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
                # Although the loss is one value, it represents the *average* of losses over all examples in the batch
                loss_v_list.append(loss_v.item()) # one item at the time

                total_v = labels.size(0) # get it in case the last batch has different number of sample
                correct_v = (predicted == labels).sum().item() # sum the correct answers for this batch
                acc_list_v.append(correct_v / total_v) # n of correct / n of sample in this batch

                total_ep += total_v # sum all the samples. it must end up being the tot training set
                correct_ep += correct_v # sum together the right answers in entire training set
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
                torch.save(model.state_dict(), MODEL_STORE_PATH + '/BEST_model'
                            + '_CV' + str(k+1)
                            + '.pt')  

def MMmodel_test_CV(n_channels, data, label, data_test, label_test,batch_size):
    cv_kfold_res= np.empty(2,5)
    for k in np.arange(5): 
        print('Start Cross-validation k= {}'.format(k+1))

        #reset the model before new cv
        model=CNN_model(n_channels)
        model.to(device) 
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(),momentum=0.9, lr = 0.001, weight_decay=1e-3)

        # load data
        train_split_k=train_split[k]
        testData = load_data(split='test', 
                            train_split=train_split_k, val_split=[],
                            data = data, label= label, # need train data for mean and sd normalization
                            data_test = data_test, label_test= label_test)

        test_loader = DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

        # set and load best model
        model=CNN_model(n_channels)

        model.load_state_dict(torch.load(
        MODEL_STORE_PATH + '/BEST_model'
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

        print('CV {}, Test Accuracy of the model is: {}, F1: {}%'
                        .format(k+1, current_ep_acc, F1_Score))
        cv_kfold_res[0,k] = bal_acc
        cv_kfold_res[1,k] = F1_Score

    print('SM model cross-validates balanced accuracy: {}, f1-score: {}'.format(cv_kfold_res[0,:].mean(), cv_kfold_res[1,:].mean()))