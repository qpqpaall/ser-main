import os
import pandas as pd
import numpy as np
import torchaudio
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
# import matplotlib.pyplot as plt
# import seaborn as sns
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn.functional as F 
    
label_dict = {
    'disgust': 0, 
    'happy': 1, 
    'fear': 2, 
    'sad': 3, 
    'angry': 4, 
    'neutral': 5
}

class AudioEmotionDatasetAlexNet(Dataset):
    def __init__(self, dataframe):  # 3 seconds at 16000 Hz
        self.arrays = dataframe
    
    def __len__(self):
        return len(self.arrays)
    
    def __getitem__(self, idx):
        return self.arrays[idx][2], label_dict[self.arrays[idx][-1]]
    
class AudioEmotionDatasetResNet50(Dataset):
    def __init__(self, dataframe):  # 3 seconds at 16000 Hz
        self.arrays = dataframe
    
    def __len__(self):
        return len(self.arrays)
    
    def __getitem__(self, idx):
        return self.arrays[idx][3], label_dict[self.arrays[idx][-1]]
    
class AudioEmotionDatasetResNet101(Dataset):
    def __init__(self, dataframe):  # 3 seconds at 16000 Hz
        self.arrays = dataframe
    
    def __len__(self):
        return len(self.arrays)
    
    def __getitem__(self, idx):
        return self.arrays[idx][4], label_dict[self.arrays[idx][-1]]
    
class AudioEmotionDatasetLSTM(Dataset):
    def __init__(self, dataframe):  # 3 seconds at 16000 Hz
        self.arrays = dataframe
    
    def __len__(self):
        return len(self.arrays)
    
    def __getitem__(self, idx):
        return self.arrays[idx][0], label_dict[self.arrays[idx][-1]]
    

class AudioEmotionDatasetLSTMHUBERT(Dataset):
    def __init__(self, dataframe):  # 3 seconds at 16000 Hz
        self.arrays = dataframe
    
    def __len__(self):
        return len(self.arrays)
    
    def __getitem__(self, idx):
        return self.arrays[idx][-2], label_dict[self.arrays[idx][-1]]
    
class AudioEmotionDatasetMYMODEL(Dataset):
    def __init__(self, dataframe):  # 3 seconds at 16000 Hz
        self.arrays = dataframe
    
    def __len__(self):
        return len(self.arrays)
    
    def __getitem__(self, idx):
        return self.arrays[idx], label_dict[self.arrays[idx][-1]]
    
    
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
class SER_AlexNet(nn.Module):

    def __init__(self,num_classes=6, channel=3, pretrained=True):
        super(SER_AlexNet, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        self.classifier = model.classifier

        self.classifier[6] = nn.Linear(4096, num_classes)

        self._init_weights(pretrained=pretrained)
        
        print(' >>>>>>>>>>>>>>>>>>>>>>>>> initialized SER AlexNet ')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x_ = torch.flatten(x, 1)
        out = self.classifier(x_)

        return x, out

    def _init_weights(self, pretrained=True):

        init_layer(self.classifier[6])

        if pretrained == False:
            init_layer(self.features[0])
            init_layer(self.features[3])
            init_layer(self.features[6])
            init_layer(self.features[8])
            init_layer(self.features[10])
            init_layer(self.classifier[1])
            init_layer(self.classifier[4])


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class SER_ResNet50(nn.Module):
    def __init__(self,num_classes=6, channel=3, pretrained=True):
        super(SER_ResNet50, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        # self.resnet = model.load_state_dict(torch.load('./resnet-50/pytorch_model.bin'))

        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
                                    nn.Dropout(0.5),
                                    layer4
                                    )
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(2048, num_classes)
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        print(' >>>>>>>>>>>>>>>>>>>>>>>>> initialized SER ResNet50 ')

    def forward(self, x):
        out = self.resnet(x)
        return x, out


class SER_ResNet101(nn.Module):
    def __init__(self,num_classes=6, channel=3, pretrained=True):
        super(SER_ResNet101, self).__init__()

        self.resnet = torchvision.models.resnet101(pretrained=pretrained)
        num_features = self.resnet.fc.in_features

        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
                                    nn.Dropout(0.5),
                                    layer4
                                    )
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        print(' >>>>>>>>>>>>>>>>>>>>>>>>> initialized SER ResNet101 ')

    def forward(self, x):
        out = self.resnet(x)
        return x, out

class LSTM(nn.Module):
    def __init__(self,num_classes=6, channel=3, pretrained=True):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=1880, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True)

        self.fc = nn.Linear(256*2, 6)

        print(' >>>>>>>>>>>>>>>>>>>>>>>>> initialized SER LSTM ')

    def forward(self, x):
        lstm_out, (ho, co) = self.lstm(x)
        out = self.fc(lstm_out)
        
        return x, out


class LSTM_HUBERT(nn.Module):
    def __init__(self,num_classes=6, channel=3, pretrained=True):
        super(LSTM_HUBERT, self).__init__()

        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True)

        self.fc = nn.Linear(256*2*149, 6)

        print(' >>>>>>>>>>>>>>>>>>>>>>>>> initialized SER LSTM_HUBERT ')

    def forward(self, x):
        batch_size, channel, H, W = x.size()
        x = x.view(batch_size, H, W)
        lstm_out, (ho, co) = self.lstm(x)
        lstm_out = lstm_out.reshape(batch_size, -1)
        out = self.fc(lstm_out)

        return x, out
    
class my_model(nn.Module):
    def __init__(self, num_classes=6, channel=3):
        super(my_model, self).__init__()

        # resnet50
        self.resnet = torchvision.models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features

        # self.drop = nn.Dropout(p=0.2)

        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
                                    nn.Dropout(0.5),
                                    layer4
                                    )
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(num_features, int(num_features / 8))

        # lstm-hubert
        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True)

        self.fc = nn.Linear(256*2*149, int(num_features / 8))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x1, x2):
        # lstm_out
        batch_size, channel, H, W = x1.size()
        x1 = x1.view(batch_size, H, W)
        lstm_out, (ho, co) = self.lstm(x1)
        lstm_out = lstm_out.reshape(batch_size, -1)
        lstm_out = self.fc(lstm_out)

        # lstm_out = self.drop(lstm_out)

        # resnet50_out
        resnet_out = self.resnet(x2)

        # transformer_out
        transformer_in = torch.cat((lstm_out, resnet_out), dim=1)
        transformer_out = self.transformer_encoder(transformer_in)
        transformer_out = self.fc2(transformer_out)
        return (x1,x2), transformer_out

class my_model_no_resnet50(nn.Module):
    def __init__(self, num_classes=6, channel=3):
        super(my_model_no_resnet50, self).__init__()

        # lstm-hubert
        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True)

        self.fc = nn.Linear(256*2*149, 512)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x1, x2):
        # lstm_out
        batch_size, channel, H, W = x1.size()
        x1 = x1.view(batch_size, H, W)
        lstm_out, (ho, co) = self.lstm(x1)
        lstm_out = lstm_out.reshape(batch_size, -1)
        lstm_out = self.fc(lstm_out)

        # transformer_out
        # transformer_in = torch.cat((lstm_out, resnet_out), dim=1)
        transformer_in = lstm_out
        transformer_out = self.transformer_encoder(transformer_in)
        transformer_out = self.fc2(transformer_out)
        return (x1,x2), transformer_out


class my_model_no_transformer(nn.Module):
    def __init__(self, num_classes=6, channel=3):
        super(my_model_no_transformer, self).__init__()

        # resnet50
        self.resnet = torchvision.models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features

        # self.drop = nn.Dropout(p=0.2)

        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
                                    nn.Dropout(0.5),
                                    layer4
                                    )
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(num_features, int(num_features / 8))

        # lstm-hubert
        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True)

        self.fc = nn.Linear(256*2*149, int(num_features / 8))

        # Transformer
        # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x1, x2):
        # lstm_out
        batch_size, channel, H, W = x1.size()
        x1 = x1.view(batch_size, H, W)
        lstm_out, (ho, co) = self.lstm(x1)
        lstm_out = lstm_out.reshape(batch_size, -1)
        lstm_out = self.fc(lstm_out)

        # lstm_out = self.drop(lstm_out)

        # resnet50_out
        resnet_out = self.resnet(x2)

        # transformer_out
        transformer_in = torch.cat((lstm_out, resnet_out), dim=1)
        transformer_out = self.fc2(transformer_in)
        return (x1,x2), transformer_out
