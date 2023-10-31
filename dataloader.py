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

class ACEDataset(Dataset):

    def __init__(self, annotations_file, transformation, target_sample_rate, num_samples, device):
        data = pd.read_csv(annotations_file)
        self.path_list = data['file'].tolist()
        self.drr_list = data['FBDRRMean(Ch)'].tolist()
        self.rt60_list = data['FBT60Mean(Ch)'].tolist()
        self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        audio_file_path = self.path_list[idx]
        drr = self.drr_list[idx]
        rt60 = self.rt60_list[idx]
        waveform, sample_rate = ta.backend.soundfile_backend.load(audio_file_path)  # (num_channels,samples) -> (1,samples) makes the waveform mono
        #waveform = waveform.to(self.device)
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
