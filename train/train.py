import torch
import time
import numpy as np
import h5py
import datetime
import os
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import torch.utils.data as Data
import scipy.io as sio   
from sklearn.model_selection import train_test_split
import argparse
import torch.optim as optim
import random
from tqdm import tqdm
import copy
from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.append('./')
from models.Model_A import CNN_A
from models.Model_B import CNN_B
from models.Model_C import CNN_C
from models.Model_D import CNN_D
from models.Model_E import CNN_E

import warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created")

class Dependent_TrainModel():
    def __init__(self):
        super(Dependent_TrainModel, self).__init__()
    
    def dependent_getdata(self, i):   
        torch.cuda.empty_cache()
        print("processing: ", short_name[i], "......")
        file_path = os.path.join(args.dataset_dir, 'DE_s'+short_name[i])
        file = sio.loadmat(file_path)
        data = file['data']
        y_v = file['valence_labels'][0]
        y_a = file['arousal_labels'][0]
       
        data_x, data_y = self.dataprocess(data, y_v, y_a)

        return data_x, data_y
    
    def dataprocess(self, data, y_v, y_a):

        y_v = torch.tensor(y_v).long()
        y_a = torch.tensor(y_a).long()
        
        y_v = torch.nn.functional.one_hot(y_v.clone().detach(), num_classes=args.num_classes)
        y_a = torch.nn.functional.one_hot(y_a.clone().detach(), num_classes=args.num_classes)
        
        one_falx = data.transpose([0, 2, 3, 1])
        one_y_v = np.empty([0, 2])
        one_y_a = np.empty([0, 2])
        
        for j in range(int(len(y_a))):
            one_y_v = np.vstack((one_y_v, y_v[j]))
            one_y_a = np.vstack((one_y_a, y_a[j]))

        if args.flag=='v':
            one_y = one_y_v
        else:
            one_y = one_y_a
            
        data_x = one_falx
        data_y = one_y

        data_x = torch.tensor(data_x).permute(0, 3, 1, 2) 
            
        return data_x, data_y       
    
    def subject_dependent_dataset_split(self, n, data_x, data_y):
        data_x = np.reshape(data_x, (40, 120, 4, 9, 9))
        data_y = np.reshape(data_y, (40, 120, 2))

        x_test, y_test = data_x[n*4 : (n+1)*4, :, :, :, :], data_y[n*4 : (n+1)*4,:,:]
        x_test = np.reshape(x_test, (-1, 4, 9, 9))
        y_test = np.reshape(y_test, (-1, 2))

        data_x_train = np.delete(data_x, slice(n*4,(n+1)*4), axis=0)
        data_x_train = np.reshape(data_x_train, (-1,4,9,9))

        data_y_train = np.delete(data_y, slice(n*4,(n+1)*4), axis=0)
        data_y_train = np.reshape(data_y_train, (-1,2))
          
        train_loader, test_loader = self.make_dataset(data_x_train, x_test, data_y_train, y_test)
        
        return train_loader, test_loader
    
    def make_dataset(self, x_train, x_test, y_train, y_test):
        
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()
        
        train_loader = Data.DataLoader(Data.TensorDataset(x_train.clone().detach().float(),
                                                    y_train.clone().detach().long()),
                                    batch_size=args.batch_size, shuffle=False) 
        
        
        test_loader = Data.DataLoader(Data.TensorDataset(x_test.clone().detach().float(),
                                                    y_test.clone().detach().long()),
                                    batch_size=args.batch_size, shuffle=False) 

        return train_loader, test_loader

    def train(self):
        print('Avaliable device:' + str(torch.cuda.get_device_name(torch.cuda.current_device())) + '\n')
        Acc_all = []
        # For each subject
        for i in range(len(short_name)):
            subject_acc = []
            std_sub = []
            
            print("Subject: {}".format(short_name[i]))
            
            data_x, data_y = self.dependent_getdata(i)
            indices = np.random.permutation(data_x.shape[0])
            data_x_rand = data_x[indices]
            data_y_rand = data_y[indices]
            
            
            # For each fold
            for n in range(10):
                # print("Fold: {}/{}\n".format(n+1,10))

                # choose the model
                if args.model_used == 'Model_A':
                    model = CNN_A()
                elif args.model_used == 'Model_B':
                    model = CNN_B()
                elif args.model_used == 'Model_C':
                    model = CNN_C()
                elif args.model_used == 'Model_D':
                    model = CNN_D()
                elif args.model_used == 'Model_E':
                    model = CNN_E()
                    
                    
                # model = nn.DataParallel(model)
                model.to(device)
    
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
                
                train_loader, test_loader = self.subject_dependent_dataset_split(n, data_x_rand, data_y_rand)
            
                # Train the model
                for epoch in tqdm(range(args.epoch_num)):
                    running_loss = 0.0
                    for m, (x_batch,y_batch) in enumerate(train_loader):
                        inputs, labels = x_batch, y_batch
                        inputs, labels = inputs.to(device), labels.to(device)
                        model.train()
                        outputs = model(inputs)

                        outputs = outputs.float()
                        labels = labels.float()
                        loss = criterion(outputs, labels)
                        loss.backward()
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        running_loss += loss.item() 
                
                save_path = os.path.join(args.model_save_path, 'model_' + short_name[i] + '_' + str(n) + '.pth')
                torch.save(model, save_path)
                    
                # Test the model
                correct = 0
                total = 0
                with torch.no_grad():
                    for m, (x_batch,y_batch) in enumerate(test_loader):
                        inputs, labels = x_batch, y_batch
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)

                        _, predicted = torch.max(outputs.data, 1)
                        true_labels = torch.argmax(labels, dim=1)
                        
                        total += labels.size(0)
                        correct += (predicted == true_labels).sum().item()
                   
                test_acc = 100 * correct / total
                print('Fold [{}/{}], Epoch [{}/{}], Loss: {}, Accuracy: {}'.format(n+1, 10, epoch+1, args.epoch_num, running_loss/len(train_loader), test_acc))
                subject_acc.append(test_acc)
                
            mean_acc = np.mean(subject_acc)
            sub_std = np.std(subject_acc)
            print('Subject [{}/{}], Mean Accuracy: {}, Std: {}\nAccuracy of Each Fold: {}'.format(short_name[i], len(short_name), mean_acc, sub_std, subject_acc))


            file = open(args.dependent_save_file, 'w')
            file.write("\n" + str(datetime.datetime.now()) + '\nSubject: ' + str(short_name[i]) +
                    '\nModel: ' + str(args.model_used) + ' flag: ' + str(args.flag) +
                    ' epoch_number: ' + str(args.epoch_num) +
                    '\nMean Acc:'+ str(mean_acc) + '\nStd:' + str(sub_std) +'\nAcc of Each Fold:'+ str(subject_acc))
        
            file.close()

            Acc_all.append(mean_acc)

            std_sub.append(sub_std)
            

        std_all = np.std(Acc_all)

        mean_acc = np.mean(Acc_all)

        print('Acc_all: {}'.format(Acc_all))
        print('Mean Acc: {}'.format(mean_acc))
        print('Mean Std: {}'.format(std_all))
        

        file = open(args.dependent_save_file, 'w')
        file.write("\n" + str(datetime.datetime.now()) + '\n The Final Result' + str(args.dataset_dir) +
                '\nModel: ' + str(args.model_used) + ' flag: ' + str(args.flag) +
                '\nlearning_rate: ' + str(args.learning_rate) + ' batch_size: '+ str(args.batch_size) +
                ' epoch_number: ' + str(args.epoch_num) +
                '\nACC_all:'+ str(Acc_all) + '\nSub_std:' + str(std_sub) +'\nMean ACC:'+ str(mean_acc) +
                '\nMean Std:'+ str(std_all) +'\n')
    
        file.close()
        
    
          
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
      
if __name__ == '__main__':
    # torch.backends.cuda.max_split_size_mb = 512
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch-num', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--dataset-dir', type=str, default='./datasets/DEAP_0p5_81_deblurred_1_map/')
    parser.add_argument('--model-save-path', type=str, default='./checkpoints/pth')
    parser.add_argument('--dependent-save-file', type=str, default='./checkpoints/results/ten_fold_cv_result_save.txt')
    parser.add_argument('--flag', type=str, default='a')
    parser.add_argument('--model-used', type=str, default='Model_A')
    
    # parser.add_argument('--save-feature', type=str, default='/data/gaoxuange/gxg/4D-CRNN-master/DEAP/features_save_a_72')
    args=parser.parse_args() 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    short_name = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24', '25', '26', '27',
                '28', '29', '30', '31', '32']
    
    seed = 22
    set_seed_everywhere(seed)    
    

    model_train = Dependent_TrainModel()
    model_train.train()
        
