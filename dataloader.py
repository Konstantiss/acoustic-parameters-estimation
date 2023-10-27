import torch as pt
from torch import nn
import torchaudio as ta
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import numpy as numpy
import os
import subprocess
import tqdm as tqdm
import pandas as pd

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)

class Dataset(Dataset):

    def __init__(self, annotations_file, transformation, target_sample_rate, num_samples, device):
        self.datalist = pd.read_csv(annotations_file, usecols=['files'])
        self.drr = pd.read_csv(annotations_file, usecols=['FBDRRMean(Ch)'])
        self.rt60 = pd.read_csv(annotations_file, usecols=['FBT60Mean(Ch)'])
        self.device = device
        self.transformation = transformation.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        audio_file_path = self.datalist[idx]
        drr = self.drr[idx]
        rt60 = self.rt60[idx]
        waveform, sample_rate = ta.load(audio_file_path)  # (num_channels,samples) -> (1,samples) makes the waveform mono
        waveform = waveform.to(self.device)
        waveform = self._resample(waveform, sample_rate)
        waveform = self._mix_down(waveform)
        waveform = self._cut(waveform)
        waveform = self._right_pad(waveform)
        waveform = self.transformation(waveform)
        return waveform, float(drr), float(rt60)

    def _resample(self, waveform, sample_rate):
        # used to handle sample rate
        resampler = ta.transforms.Resample(sample_rate, self.target_sample_rate)
        return resampler(waveform)

    def _mix_down(self, waveform):
        # used to handle channels
        waveform = pt.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _cut(self, waveform):
        # cuts the waveform if it has more than certain samples
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        return waveform

    def _right_pad(self, waveform):
        # pads the waveform if it has less than certain samples
        signal_length = waveform.shape[1]
        if signal_length < self.num_samples:
            num_padding = self.num_samples - signal_length
            last_dim_padding = (
                0, num_padding)  # first arg for left second for right padding. Make a list of tuples for multi dim
            waveform = pt.nn.functional.pad(waveform, last_dim_padding)
        return waveform
