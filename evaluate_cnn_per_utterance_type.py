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

annotation_file_names = ['s1_utterances_train.csv', 's2_utterances_train.csv', 's3_utterances_train.csv',
                         's4_utterances_train.csv', 's5_utterances_train.csv']

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 64
EPOCHS = 30

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)

model = CNNNetwork().to(device)

model.load_state_dict(torch.load(RESULTS_DIR + 'cnn-save-2023-11-23 223852.841903-30.bin'))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
start_time = time.time()

losses_drr = []
losses_rt60 = []

for i, file in enumerate(annotation_file_names):
    annotation_file_path = DATA_PATH_TRAIN + file

    eval_dataset = ACEDataset(annotation_file_path, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_drr, loss_rt60 = evaluate(model=model, eval_dataloader=eval_dataloader,
                                   loss_fn=loss_fn, device=device)
    print("S" + str(i + 1) + " utterance loss DRR:", loss_drr)
    print("S" + str(i + 1) + " utterance loss RT60:", loss_rt60)
    losses_drr.append(loss_drr)
    losses_rt60.append(loss_rt60)

execution_time = (time.time() - start_time) / 60
date_time = str(datetime.datetime.now())

s1_utterance_loss_drr = losses_drr[0]
s1_utterance_loss_rt60 = losses_rt60[0]
s2_utterance_loss_drr = losses_drr[1]
s2_utterance_loss_rt60 = losses_rt60[1]
s3_utterance_loss_drr = losses_drr[2]
s3_utterance_loss_rt60 = losses_rt60[2]
s4_utterance_loss_drr = losses_drr[3]
s4_utterance_loss_rt60 = losses_rt60[3]
s5_utterance_loss_drr = losses_drr[4]
s5_utterance_loss_rt60 = losses_rt60[4]

results = {
    "model": model.__class__.__name__,
    "s1_utterance_loss_drr": s1_utterance_loss_drr,
    "s1_utterance_loss_rt60": s1_utterance_loss_rt60,
    "s2_utterance_loss_drr": s2_utterance_loss_drr,
    "s2_utterance_loss_rt60": s2_utterance_loss_rt60,
    "s3_utterance_loss_drr": s3_utterance_loss_drr,
    "s3_utterance_loss_rt60": s3_utterance_loss_rt60,
    "s4_utterance_loss_drr": s4_utterance_loss_drr,
    "s4_utterance_loss_rt60": s4_utterance_loss_rt60,
    "s5_utterance_loss_drr": s5_utterance_loss_drr,
    "s5_utterance_loss_rt60": s5_utterance_loss_rt60,
    "datetime": datetime.datetime.now(),
    "execution_time": execution_time
}

print('Total execution time: {:.4f} minutes', format(execution_time))
print("Evaluation loss DRR:", losses_drr)
print("Evaluation loss RT60:", losses_rt60)

results_filename = RESULTS_DIR + 'results-cnn-per-utterance-' + date_time + '.pkl'
with open(results_filename, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
