import torch
from torch import nn
import torchaudio as ta
from torchsummary import summary
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
from dataloader import *
from CNN import *
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)

DATA_PATH_TRAIN = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Train/Speech/'
annotations_file_path_train = DATA_PATH_TRAIN + 'features_and_ground_truth_train.csv'

DATA_PATH_EVAL = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
annotations_file_path_eval = DATA_PATH_EVAL + 'features_and_ground_truth_eval.csv'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 256
EPOCHS = 5

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
train_dataset = ACEDataset(annotations_file_path_train, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataset = ACEDataset(annotations_file_path_eval, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = CNNNetwork().cuda()
# model.load_state_dict(torch.load('cnn-save2023-11-03 17:48:40.414314.bin'))

loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=10e-6, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
start_time = time.time()
mean_loss_per_epoch_train_drr, mean_loss_per_epoch_train_rt60, \
mean_loss_per_epoch_eval_drr, mean_loss_per_epoch_eval_rt60, \
mean_r2_per_epoch_train_drr, mean_r2_per_epoch_train_rt60, \
mean_r2_per_epoch_eval_drr, mean_r2_per_epoch_eval_rt60 = train_evaluate(
    model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, loss_fn=loss_fn,
    optimizer=optimizer,
    device=device, epochs=EPOCHS)

date_time = str(datetime.datetime.now())
model_save_filename = 'cnn-save' + date_time + '-' + str(EPOCHS) + '.bin'

torch.save(model.state_dict(), model_save_filename)

results = {
    "model": model.__class__.__name__,
    "train_loss_drr": mean_loss_per_epoch_train_drr,
    "train_loss_rt60": mean_loss_per_epoch_train_rt60,
    "train_r2_drr": mean_r2_per_epoch_train_drr,
    "train_r2_rt60": mean_r2_per_epoch_train_rt60,
    "eval_loss_drr": mean_loss_per_epoch_eval_drr,
    "eval_loss_rt60": mean_loss_per_epoch_eval_rt60,
    "eval_r2_drr": mean_r2_per_epoch_eval_drr,
    "eval_r2_rt60": mean_r2_per_epoch_eval_rt60,
    "datetime": datetime.datetime.now()
}

print('Total execution time: {:.4f} minutes', format((time.time() - start_time) / 60))
print("Mean training loss per epoch DRR:", mean_loss_per_epoch_train_drr)
print("Mean training loss per epoch RT60:", mean_loss_per_epoch_train_rt60)
print("Mean training R2 score per epoch DRR:", mean_r2_per_epoch_train_drr)
print("Mean training R2 score per epoch RT60:", mean_r2_per_epoch_train_rt60)
print("Mean evaluation loss per epoch DRR:", mean_loss_per_epoch_eval_drr)
print("Mean evaluation loss per epoch RT60:", mean_loss_per_epoch_eval_rt60)
print("Mean evaluation R2 score per epoch DRR:", mean_r2_per_epoch_eval_drr)
print("Mean evaluation R2 score per epoch RT60:", mean_r2_per_epoch_eval_rt60)

results_filename = 'results-' + date_time + '-' + str(EPOCHS) + '.pkl'
with open(results_filename, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

plot_filename = 'figs/cnn-loss-plot-train-' + date_time + '-' + str(EPOCHS) + '.png'
plt.figure(figsize=(10, 5))
plt.title("CNN DRR and RT60 training loss per epoch")
plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_train_drr, linestyle='solid', marker='o', label="drr")
plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_train_rt60, linestyle='solid', marker='o', label="rt60")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.legend()
plt.savefig(plot_filename)
plt.show()

plot_filename = 'figs/cnn-r2-plot-train-' + date_time + '-' + str(EPOCHS) + '.png'
plt.figure(figsize=(10, 5))
plt.title("CNN DRR and RT60 training R2 score per epoch")
plt.plot(range(1, EPOCHS + 1), mean_r2_per_epoch_train_drr, linestyle='solid', marker='o', label="drr")
plt.plot(range(1, EPOCHS + 1), mean_r2_per_epoch_train_rt60, linestyle='solid', marker='o', label="rt60")
plt.xlabel("Epoch")
plt.ylabel("R2 score")
plt.ylim(0, 1)
plt.legend()
plt.savefig(plot_filename)
plt.show()

plot_filename = 'figs/cnn-loss-plot-eval-' + date_time + '-' + str(EPOCHS) + '.png'
plt.figure(figsize=(10, 5))
plt.title("CNN DRR and RT60 evaluation loss per epoch")
plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_eval_drr, linestyle='solid', marker='o', label="drr")
plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_eval_rt60, linestyle='solid', marker='o', label="rt60")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.legend()
plt.savefig(plot_filename)
plt.show()

plot_filename = 'figs/cnn-r2-plot-eval-' + date_time + '-' + str(EPOCHS) + '.png'
plt.figure(figsize=(10, 5))
plt.title("CNN DRR and RT60 evaluation R2 score per epoch")
plt.plot(range(1, EPOCHS + 1), mean_r2_per_epoch_eval_drr, linestyle='solid', marker='o', label="drr")
plt.plot(range(1, EPOCHS + 1), mean_r2_per_epoch_eval_rt60, linestyle='solid', marker='o', label="rt60")
plt.xlabel("Epoch")
plt.ylabel("R2 score")
plt.ylim(0, 1)
plt.legend()
plt.savefig(plot_filename)
plt.show()
