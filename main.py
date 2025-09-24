import os
import pandas as pd
import numpy as np
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn.functional as F 
import pickle
import argparse
import sys, random
import json

from model import *

'''
mfcc_feature, 
spec,
spec_alex_feature_input.detach().cpu().numpy(),
spec_resnet50_feature_input,
spec_resnet101_feature_input,
hubert_feature.detach().cpu().numpy(),
label

{'disgust': 192, 'happy': 192, 'fear': 192, 'sad': 192, 'angry': 192, 'neutral': 96}
'''

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], \
               'train_precision': [], 'train_recall': [], 'val_precision': [], 'val_recall': [], \
               'train_conf': [], 'val_conf': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_labels = []
        train_predicted = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[1], labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs[-1], 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_labels.extend(labels.tolist())
            train_predicted.extend(predicted.tolist())
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        train_precision = precision_score(train_labels, train_predicted, average='macro')
        train_recall = recall_score(train_labels, train_predicted, average='macro')
        train_conf_matrix = json.dumps(confusion_matrix(train_labels, train_predicted).tolist())
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_labels = []
        val_predicted = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
                outputs = model(inputs)
                loss = criterion(outputs[1], labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs[-1], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_labels.extend(labels.tolist())
                val_predicted.extend(predicted.tolist())
        
        val_loss /= len(test_loader.dataset)
        val_acc = val_correct / val_total

        val_precision = precision_score(val_labels, val_predicted, average='macro')
        val_recall = recall_score(val_labels, val_predicted, average='macro')
        val_conf_matrix = json.dumps(confusion_matrix(val_labels, val_predicted).tolist())
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        scheduler.step(val_loss)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['train_conf'].append(train_conf_matrix)
        history['val_conf'].append(val_conf_matrix)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), 'best_model.pth')

    with open('training_history.csv', 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,train_precision,train_recall,val_precision,val_recall,train_conf,val_conf\n')
        for i in range(num_epochs):
            f.write(f"{history['epoch'][i]},{history['train_loss'][i]:.4f},{history['train_acc'][i]:.4f},{history['val_loss'][i]:.4f},{history['val_acc'][i]:.4f},{history['train_precision'][i]:.4f},{history['train_recall'][i]:.4f},{history['val_precision'][i]:.4f},{history['val_recall'][i]:.4f}, {history['train_conf'][i]}, {history['val_conf'][i]}\n")

    with open('training_train_conf.csv', 'w') as f:
        for i in range(num_epochs):
            f.write(f"{history['train_conf'][i]}\n")

    with open('training_val_conf.csv', 'w') as f:
        for i in range(num_epochs):
            f.write(f"{history['val_conf'][i]}\n")
    
    return model

def train_my_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], \
               'train_precision': [], 'train_recall': [], 'val_precision': [], 'val_recall': [], \
               'train_conf': [], 'val_conf': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_labels = []
        train_predicted = []
        

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs1, inputs2, labels = inputs[-2].to('cuda:0'), inputs[3].to('cuda:0'), labels.to('cuda:0')
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs[1], labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs[0].size(0)
            _, predicted = torch.max(outputs[-1], 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_labels.extend(labels.tolist())
            train_predicted.extend(predicted.tolist())
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        train_precision = precision_score(train_labels, train_predicted, average='macro')
        train_recall = recall_score(train_labels, train_predicted, average='macro')
        train_conf_matrix = json.dumps(confusion_matrix(train_labels, train_predicted).tolist())
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_labels = []
        val_predicted = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs1, inputs2, labels = inputs[-2].to('cuda:0'), inputs[3].to('cuda:0'), labels.to('cuda:0')
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs[1], labels)
                
                val_loss += loss.item() * inputs[0].size(0)
                _, predicted = torch.max(outputs[-1], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_labels.extend(labels.tolist())
                val_predicted.extend(predicted.tolist())
        
        val_loss /= len(test_loader.dataset)
        val_acc = val_correct / val_total

        val_precision = precision_score(val_labels, val_predicted, average='macro')
        val_recall = recall_score(val_labels, val_predicted, average='macro')
        val_conf_matrix = json.dumps(confusion_matrix(val_labels, val_predicted).tolist())
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        scheduler.step(val_loss)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['train_conf'].append(train_conf_matrix)
        history['val_conf'].append(val_conf_matrix)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), 'best_model.pth')

    with open('training_history.csv', 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,train_precision,train_recall,val_precision,val_recall,train_conf,val_conf\n')
        for i in range(num_epochs):
            f.write(f"{history['epoch'][i]},{history['train_loss'][i]:.4f},{history['train_acc'][i]:.4f},{history['val_loss'][i]:.4f},{history['val_acc'][i]:.4f},{history['train_precision'][i]:.4f},{history['train_recall'][i]:.4f},{history['val_precision'][i]:.4f},{history['val_recall'][i]:.4f}, {history['train_conf'][i]}, {history['val_conf'][i]}\n")

    with open('training_train_conf.csv', 'w') as f:
        for i in range(num_epochs):
            f.write(f"{history['train_conf'][i]}\n")

    with open('training_val_conf.csv', 'w') as f:
        for i in range(num_epochs):
            f.write(f"{history['val_conf'][i]}\n")
    
    return model
    
if __name__ == '__main__':

    seed_everything()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train SER model")

    #Features
    parser.add_argument('--model', type=str,
        help='choose which model')
    
    parser = parser.parse_args()
    print(parser)

    with open('arrays.pkl', 'rb') as f:
        arrays = pickle.load(f)

    arrays = train_test_split(arrays, [i[-1] for i in arrays], test_size=0.25, random_state=42)

    if parser.model == 'AlexNet':
        train_dataset = AudioEmotionDatasetAlexNet(arrays[0])
        test_dataset = AudioEmotionDatasetAlexNet(arrays[1])

        model = SER_AlexNet().to('cuda:0')

        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
    elif parser.model == 'ResNet50':
        train_dataset = AudioEmotionDatasetResNet50(arrays[0])
        test_dataset = AudioEmotionDatasetResNet50(arrays[1])

        model = SER_ResNet50().to('cuda:0')

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    elif parser.model == 'ResNet101':
        train_dataset = AudioEmotionDatasetResNet101(arrays[0])
        test_dataset = AudioEmotionDatasetResNet101(arrays[1])

        model = SER_ResNet101().to('cuda:0')

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    elif parser.model == 'LSTM':
        train_dataset = AudioEmotionDatasetLSTM(arrays[0])
        test_dataset = AudioEmotionDatasetLSTM(arrays[1])

        model = LSTM().to('cuda:0')

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    elif parser.model == 'LSTMHUBERT':
        train_dataset = AudioEmotionDatasetLSTMHUBERT(arrays[0])
        test_dataset = AudioEmotionDatasetLSTMHUBERT(arrays[1])

        model = LSTM_HUBERT().to('cuda:0')

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    elif parser.model == 'my_model':
        train_dataset = AudioEmotionDatasetMYMODEL(arrays[0])
        test_dataset = AudioEmotionDatasetMYMODEL(arrays[1])

        model = my_model().to('cuda:0')

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    elif parser.model == 'my_model_no_resnet50':
        train_dataset = AudioEmotionDatasetMYMODEL(arrays[0])
        test_dataset = AudioEmotionDatasetMYMODEL(arrays[1])

        model = my_model_no_resnet50().to('cuda:0')

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    elif parser.model == 'my_model_no_transformer':
        train_dataset = AudioEmotionDatasetMYMODEL(arrays[0])
        test_dataset = AudioEmotionDatasetMYMODEL(arrays[1])

        model = my_model_no_transformer().to('cuda:0')

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    else:
        print('Choose a valid model! ---- AlexNet, ResNet50, ResNet101, LSTM, LSTMHUBERT, my_model')
        sys.exit(1)
    
    num_epochs = 30
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=0)

    print(model)
    criterion = nn.CrossEntropyLoss()
    
    
    if parser.model == 'my_model' or parser.model == 'my_model_no_resnet50' or parser.model == 'my_model_no_transformer':
        model = train_my_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)
    else:
        model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)