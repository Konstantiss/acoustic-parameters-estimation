import torchvision.models as models
from torchvision.transforms import transforms
from torchsummary import summary
from torch import nn
import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
import numpy as numpy
import os
import subprocess
from torcheval.metrics import R2Score
import datetime
import pickle
from tqdm import tqdm
import pandas as pd
from train_and_evaluate_model import *
from evaluate_model import *
from dataloader import *
from CNN import *
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)

RESULTS_DIR = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Results/'

DATA_PATH_TRAIN = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Train/Speech/'

DATA_PATH_EVAL = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'

annotation_file_names = ['single_0dB_eval.csv', 'single_10dB_eval.csv', 'single_20dB_eval.csv',
                         'chromebook_0dB_eval.csv', 'chromebook_10dB_eval.csv', 'chromebook_20dB_eval.csv',
                         'mobile_0dB_eval.csv', 'mobile_10dB_eval.csv', 'mobile_20dB_eval.csv',
                         'crucif_0dB_eval.csv', 'crucif_10dB_eval.csv', 'crucif_20dB_eval.csv',
                         'lin8ch_0dB_eval.csv', 'lin8ch_10dB_eval.csv', 'lin8ch_20dB_eval.csv',
                         'em32_0dB_eval.csv', 'em32_10dB_eval.csv', 'em32_20dB_eval.csv']

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 64
EPOCHS = 30

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)

model = CNNNetwork().to(device)

model.load_state_dict(torch.load(RESULTS_DIR + 'cnn-save-2024-03-02 024311.642104-15.bin'))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
start_time = time.time()

losses_drr = []
losses_rt60 = []

for i, file in enumerate(annotation_file_names):
    annotation_file_path = DATA_PATH_EVAL + file

    eval_dataset = ACEDataset(annotation_file_path, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_drr, loss_rt60 = evaluate(model=model, eval_dataloader=eval_dataloader,
                                   loss_fn=loss_fn, device=device)
    print(file.split('_')[0] + " mic SNR: " + file.split('_')[1].split('_')[0] + " loss DRR:", loss_drr)
    print(file.split('_')[0] + " mic SNR: " + file.split('_')[1].split('_')[0] + " loss RT60:", loss_rt60)
    losses_drr.append(loss_drr)
    losses_rt60.append(loss_rt60)

execution_time = (time.time() - start_time) / 60
date_time = str(datetime.datetime.now())

print('Total execution time: {:.4f} minutes', format(execution_time))
print("Evaluation loss DRR:", losses_drr)
print("Evaluation loss RT60:", losses_rt60)
