import librosa
import pandas as pd
import numpy as np
import os
os.sched_setaffinity(0, {i for i in range(32)})
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageOps
from torchaudio.models import hubert_pretrain_base
from torchaudio.transforms import Resample
import torchaudio
from transformers import Wav2Vec2Processor, HubertForCTC, AutoImageProcessor, ResNetForImageClassification

resampler = Resample(orig_freq=48000, new_freq=16000)
processor = Wav2Vec2Processor.from_pretrained("./hubert-large-ls960-ft")
hubert_model = HubertForCTC.from_pretrained("./hubert-large-ls960-ft", output_hidden_states=True).cuda()

RAVDESS = "./audio_speech_actors_01-24/"

ravdess_dir_lis = os.listdir(RAVDESS)
path_list = []
gender_list = []
emotion_list = []

alexnet_preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]) 

resnet50_preprocess = AutoImageProcessor.from_pretrained("./resnet-50")
resnet101_preprocess = AutoImageProcessor.from_pretrained("./resnet-101")

emotion_dic = {
    '03' : 'happy',
    '01' : 'neutral',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fear',
    '07' : 'disgust',
}

for directory in ravdess_dir_lis:
    actor_files = os.listdir(os.path.join(RAVDESS, directory))
    for audio_file in actor_files: 
        part = audio_file.split('.')[0]
        key = part.split('-')[2]
        if key in emotion_dic:
            gender_code = int(part.split('-')[6])
            path_list.append(f"{RAVDESS}{directory}/{audio_file}")
            gender_list.append('female' if gender_code & 1 == 0 else 'male')
            emotion_list.append(emotion_dic[key])
            
df = pd.concat([
    pd.DataFrame(path_list, columns=['path']),
    pd.DataFrame(gender_list, columns=['sex']),
    pd.DataFrame(emotion_list, columns=['emotion'])
], axis=1)

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data, frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=16000,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

# DATA AUGMENTATION
# NOISE
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def get_spec_alex(data):
    data = np.clip(data, 0.0, 1.0)
    data = (data*255.0).astype(np.uint8)
    data = np.stack([data] * 3, axis=-1)

    return alexnet_preprocess(Image.fromarray(data, mode='RGB'))

def get_spec_resnet50(data):
    data = np.clip(data, 0.0, 1.0)
    data = (data*255.0).astype(np.uint8)
    data = np.stack([data] * 3, axis=-1)

    return resnet50_preprocess(data)['pixel_values'][0]

def get_spec_resnet101(data):
    data = np.clip(data, 0.0, 1.0)
    data = (data*255.0).astype(np.uint8)
    data = np.stack([data] * 3, axis=-1)

    return resnet101_preprocess(data)['pixel_values'][0]

def get_hubert_feature(data):
    feat = processor(data[0][:16000*3], sampling_rate=16000 , return_tensors="pt").input_values
    hubert_feat = hubert_model(feat.cuda())['hidden_states'][-1]
    return hubert_feat

def get_features(path, duration=3, offset=0.6):
    # data,sr=librosa.load(path, sr=16000, duration=duration, offset=offset)
    data,sr=librosa.load(path, sr=16000, duration=duration)

    print(data.shape)

    # aud=extract_features(data)
    # audio=np.array(aud)
    
    # noised_audio=noise(data)
    # aud2=extract_features(noised_audio)
    # audio=np.vstack((audio,aud2))
    
    # pitched_audio=pitch(data,sr)
    # aud3=extract_features(pitched_audio)
    # audio=np.vstack((audio,aud3))
    
    # pitched_audio1=pitch(data,sr)
    # pitched_noised_audio=noise(pitched_audio1)
    # aud4=extract_features(pitched_noised_audio)
    # audio=np.vstack((audio,aud4))

    # feature alexnet-spec
    mfcc_feature = mfcc(data, sr, frame_length=2048,hop_length=512)
    spec = librosa.feature.melspectrogram(y=data, sr=16000)
    spec_alex_feature_input = get_spec_alex(spec)
    spec_resnet50_feature_input = get_spec_resnet50(spec)
    spec_resnet101_feature_input = get_spec_resnet101(spec)

    _data, _sr = torchaudio.load(path)
    _data = resampler(_data)
    hubert_feature = get_hubert_feature(_data)

    return mfcc_feature, spec, spec_alex_feature_input, spec_resnet50_feature_input, spec_resnet101_feature_input, hubert_feature

row, col = df.shape

mfcc_feature_list = []
spec_feature_list = []
spec_alex_feature_list = []
spec_resnet50_feature_list = []
spec_resnet101_feature_list = []
hubert_feature_list = []
label_list = []

feature_label_list = []

# path, sex, emotion
for i in tqdm(range(row)):
    mfcc_feature, spec, spec_alex_feature_input, spec_resnet50_feature_input, spec_resnet101_feature_input, hubert_feature = get_features(df.iloc[i]['path'])
    label = df.iloc[i]['emotion']

    mfcc_feature_list.append(mfcc_feature) # 
    spec_feature_list.append(spec) # 128 * 94
    spec_alex_feature_list.append(spec_alex_feature_input.detach().cpu().numpy()) # channel =3 
    spec_resnet50_feature_list.append(spec_resnet50_feature_input) # channel =3 
    spec_resnet101_feature_list.append(spec_resnet101_feature_input) # channel =3 
    hubert_feature_list.append(hubert_feature.detach().cpu().numpy()) # channel = 1
    label_list.append(label)

    feature_label_list.append([mfcc_feature, 
                               spec,
                               spec_alex_feature_input.detach().cpu().numpy(),
                               spec_resnet50_feature_input,
                               spec_resnet101_feature_input,
                               hubert_feature.detach().cpu().numpy(),
                               label])


import pickle
with open('arrays.pkl', 'wb') as f: 
    pickle.dump(feature_label_list, f)