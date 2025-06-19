# dataset.py
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import random

def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
       res= np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
       return res
    else:
       return feat

    
class XDDataset_Whole_test_train(data.Dataset):
    def __init__(self, file_path: str, test_mode: bool,attnwindow):
        self.df = pd.read_csv(file_path)
        self.test_mode = test_mode        
        self.attnwindow = attnwindow
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        
        clip_feature = torch.tensor(clip_feature)
        clip_length = clip_feature.shape[0]
        if clip_length < self.attnwindow:
            clip_feature=pad(clip_feature, self.attnwindow)
        elif clip_length% self.attnwindow != 0:
            clip_feature = pad(clip_feature, (clip_length//self.attnwindow+1)*self.attnwindow)
        video_name = index
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length, video_name


class UCFDataset_Whole_test_train(data.Dataset):
    def __init__(self,  file_path: str, test_mode: bool,normal: bool ,attnwindow:int):
        self.df = pd.read_csv(file_path)
        self.test_mode = test_mode
        self.normal = normal
        self.attnwindow=attnwindow
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        clip_feature = torch.tensor(clip_feature)
        clip_length = clip_feature.shape[0]
        if clip_length < self.attnwindow:
            clip_feature = pad(clip_feature, self.attnwindow)
        elif clip_length % self.attnwindow != 0:
            clip_feature = pad(clip_feature, (clip_length // self.attnwindow + 1) * self.attnwindow)
        video_name = index
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length, video_name


class UCFDataset_robust(data.Dataset):
    def __init__(self,  file_path: str, test_mode: bool,  normal: bool ,attnwindow:int,frac):
        self.df = pd.read_csv(file_path)
        self.test_mode = test_mode 
        self.normal = normal
        self.attnwindow=attnwindow
        self.frac=frac
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        clip_feature = torch.tensor(clip_feature)
        clip_length = clip_feature.shape[0]
        if clip_length < self.attnwindow:
            clip_feature = pad(clip_feature, self.attnwindow)
        elif clip_length % self.attnwindow != 0:
            clip_feature = pad(clip_feature, (clip_length // self.attnwindow + 1) * self.attnwindow)
        video_name = index
        clip_label = self.df.loc[index]['label']
        length_to_zero = int(clip_feature.shape[0] * (self.frac))
        indices_to_zero = random.sample(range(clip_feature.shape[0]), length_to_zero)
        clip_feature[indices_to_zero, :] = 0

        return clip_feature, clip_label, clip_length, video_name

    
class XDDataset_robust(data.Dataset):
    def __init__(self, file_path: str, test_mode: bool,attnwindow,frac):
        self.df = pd.read_csv(file_path)
        self.test_mode = test_mode
        self.attnwindow = attnwindow
        self.frac=frac

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        clip_feature = torch.tensor(clip_feature)
        clip_length = clip_feature.shape[0]
        if clip_length < self.attnwindow:
            clip_feature=pad(clip_feature, self.attnwindow)
        elif clip_length% self.attnwindow != 0:
            clip_feature = pad(clip_feature, (clip_length//self.attnwindow+1)*self.attnwindow)
        video_name = index
        clip_label = self.df.loc[index]['label']
        length_to_zero = int(clip_feature.shape[0] * (self.frac))
        indices_to_zero = random.sample(range(clip_feature.shape[0]), length_to_zero)
        clip_feature[indices_to_zero, :] = 0
        return clip_feature, clip_label, clip_length, video_name
    


