import torch
import torch as pt
from torch import nn
import torchaudio as ta
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import numpy as numpy
import os
import subprocess
from torcheval.metrics import R2Score
from tqdm import tqdm
import pandas as pd
from dataloader import *
from CNN import *
import time
import matplotlib.pyplot as plt

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)


def train(model, dataloader, loss_fn, optimizer, device, epochs):
    for epoch in range(EPOCHS):
        losses_per_epoch_drr = []
        losses_per_epoch_rt60 = []
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        with tqdm.tqdm(train_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
            for waveform, drrs_true, rt60s_true in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                waveform = waveform.to(device)
                drrs_true = drrs_true.to(device)
                rt60s_true = rt60s_true.to(device)
                # calculate loss and preds
                drr_estimates, rt60_estimates = model(waveform)
                loss_drr = loss_fn(drr_estimates.float(), drrs_true.float())
                loss_rt60 = loss_fn(rt60_estimates.float(), rt60s_true.float())
                losses_per_epoch_drr.append(loss_drr)
                losses_per_epoch_rt60.append(loss_rt60)
                # backpropogate the losses and update the gradients
                optimizer.zero_grad()
                loss_drr.backward(retain_graph=True)
                loss_rt60.backward()
                optimizer.step()
                tepoch.set_postfix(loss_drr=loss_drr.item(), loss_rt60=loss_rt60.item())
                # print(f"Loss:{total_loss.item()}")
        mean_loss_per_epoch_drr.append(sum(losses_per_epoch_drr) / len(losses_per_epoch_drr))
        mean_loss_per_epoch_rt60.append(sum(losses_per_epoch_rt60) / len(losses_per_epoch_rt60))
    print("Mean loss per epoch DRR:", mean_loss_per_epoch_drr)
    print("Mean loss per epoch RT60:", mean_loss_per_epoch_rt60)


EVAL = True

if EVAL:
    DATA_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
    annotations_file_path = DATA_PATH + 'features_and_ground_truth_eval.csv'
else:
    DATA_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Dev/Speech/'
    annotations_file_path = DATA_PATH + 'features_and_ground_truth_dev.csv'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 512
EPOCHS = 10

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
dataset = ACEDataset(annotations_file_path, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = CNNNetwork().cuda()
loss_fn = pt.nn.MSELoss()
optimizer = pt.optim.SGD(model.parameters(), lr=10e-7, momentum=0.9)

start_time = time.time()
mean_loss_per_epoch_drr = []
mean_loss_per_epoch_rt60 = []
train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS)
print('Total execution time: {:.4f} minutes', format((time.time() - start_time) / 60))

plt.figure(figsize=(10, 5))
plt.title("DRR and RT60 estimation loss per epoch")
plt.plot(mean_loss_per_epoch_drr, label="drr")
plt.plot(mean_loss_per_epoch_rt60, label="rt60")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()