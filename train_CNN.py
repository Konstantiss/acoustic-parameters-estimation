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
from dataloader import *
from CNN import *

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)


def train_single_epoch(model, dataloader, loss_fn, optimizer, device):
    for waveform, drr, rt60 in tqdm.tqdm(dataloader):
        waveform = waveform.to(device)
        drr = drr.to(device)
        rt60 = rt60.to(device)
        # calculate loss and preds
        drr_estimate, rt60_estimate = model(waveform)
        loss = loss_fn([drr_estimate.float(), rt60_estimate.float()], [drr.float(), rt60.float()])
        # backpropogate the loss and update the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss:{loss.item()}")


def train(model, dataloader, loss_fn, optimizer, device, epochs):
    for i in tqdm.tqdm(range(epochs)):
        print(f"epoch:{i + 1}")
        train_single_epoch(model, dataloader, loss_fn, optimizer, device)
        print('-------------------------------------------')
    print('Finished Training')


EVAL = True

if EVAL:
    DATA_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
    annotations_file_path = DATA_PATH + 'features_and_ground_truth_eval.csv'
else:
    DATA_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Dev/Speech/'
    annotations_file_path = DATA_PATH + 'features_and_ground_truth_dev.csv'

    
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 128
EPOCHS = 1

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
dataset = ACEDataset(annotations_file_path, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = CNNNetwork().cuda()
loss_fn = pt.nn.MSELoss()
optimizer = pt.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS)
