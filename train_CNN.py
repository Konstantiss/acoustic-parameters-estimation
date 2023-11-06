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
from dataloader import *
from CNN import *
import time
import matplotlib.pyplot as plt

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)


def train(model, train_dataloader, eval_dataloader, loss_fn, optimizer, device, epochs):
    model = model.train()
    for epoch in range(EPOCHS):
        losses_per_epoch_train_drr = []
        losses_per_epoch_train_rt60 = []
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
                losses_per_epoch_train_drr.append(loss_drr.item())
                losses_per_epoch_train_rt60.append(loss_rt60.item())
                # backpropogate the losses and update the gradients
                optimizer.zero_grad()
                loss_drr.backward(retain_graph=True)
                loss_rt60.backward()
                optimizer.step()
                tepoch.set_postfix(loss_drr=loss_drr.item(), loss_rt60=loss_rt60.item())
        print(f"Mean DRR training loss for epoch {epoch + 1}:",
              sum(losses_per_epoch_train_drr) / len(losses_per_epoch_train_drr))
        print(f"Mean RT60 training loss for epoch {epoch + 1}:",
              sum(losses_per_epoch_train_rt60) / len(losses_per_epoch_train_rt60))
        mean_loss_per_epoch_train_drr.append(sum(losses_per_epoch_train_drr) / len(losses_per_epoch_train_drr))
        mean_loss_per_epoch_train_rt60.append(sum(losses_per_epoch_train_rt60) / len(losses_per_epoch_train_rt60))
    model = model.train()
    for epoch in range(EPOCHS):
        losses_per_epoch_train_drr = []
        losses_per_epoch_train_rt60 = []
        with tqdm.tqdm(eval_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
            for waveform, drrs_true, rt60s_true in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                waveform = waveform.to(device)
                drrs_true = drrs_true.to(device)
                rt60s_true = rt60s_true.to(device)
                # calculate loss and preds
                drr_estimates, rt60_estimates = model(waveform)
                loss_drr = loss_fn(drr_estimates.float(), drrs_true.float())
                loss_rt60 = loss_fn(rt60_estimates.float(), rt60s_true.float())
                losses_per_epoch_train_drr.append(loss_drr.item())
                losses_per_epoch_train_rt60.append(loss_rt60.item())
                # backpropogate the losses and update the gradients
                optimizer.zero_grad()
                loss_drr.backward(retain_graph=True)
                loss_rt60.backward()
                optimizer.step()
                tepoch.set_postfix(loss_drr=loss_drr.item(), loss_rt60=loss_rt60.item())
        print(f"Mean DRR training loss for epoch {epoch + 1}:",
              sum(losses_per_epoch_train_drr) / len(losses_per_epoch_train_drr))
        print(f"Mean RT60 training loss for epoch {epoch + 1}:",
              sum(losses_per_epoch_train_rt60) / len(losses_per_epoch_train_rt60))
        mean_loss_per_epoch_train_drr.append(sum(losses_per_epoch_train_drr) / len(losses_per_epoch_train_drr))
        mean_loss_per_epoch_train_rt60.append(sum(losses_per_epoch_train_rt60) / len(losses_per_epoch_train_rt60))

        print("---- Evaluation ---\n")
        if EVAL:
            model = model.eval()
            losses_per_epoch_eval_drr = []
            losses_per_epoch_eval_rt60 = []
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
                    losses_per_epoch_eval_drr.append(loss_drr.item())
                    losses_per_epoch_eval_rt60.append(loss_rt60.item())
                    # backpropogate the losses and update the gradients
                    optimizer.zero_grad()
                    loss_drr.backward(retain_graph=True)
                    loss_rt60.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss_drr=loss_drr.item(), loss_rt60=loss_rt60.item())
            print(f"Mean DRR evaluation loss for epoch {epoch + 1}:",
                  sum(losses_per_epoch_eval_drr) / len(losses_per_epoch_eval_drr))
            print(f"Mean RT60 evaluation loss for epoch {epoch + 1}:",
                  sum(losses_per_epoch_eval_rt60) / len(losses_per_epoch_eval_rt60))
            mean_loss_per_epoch_eval_drr.append(sum(losses_per_epoch_eval_drr) / len(losses_per_epoch_eval_drr))
            mean_loss_per_epoch_eval_rt60.append(sum(losses_per_epoch_eval_rt60) / len(losses_per_epoch_eval_rt60))


EVAL = True

DATA_PATH_TRAIN = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Train/Speech/'
annotations_file_path_train = DATA_PATH_TRAIN + 'features_and_ground_truth_train.csv'

DATA_PATH_EVAL = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
annotations_file_path_eval = DATA_PATH_EVAL + 'features_and_ground_truth_eval.csv'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 256
EPOCHS = 1

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
train_dataset = ACEDataset(annotations_file_path_train, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataset = ACEDataset(annotations_file_path_eval, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = CNNNetwork().cuda()
model.load_state_dict(torch.load('cnn-save2023-11-03 17:48:40.414314.bin'))

loss_fn = pt.nn.MSELoss()
# optimizer = pt.optim.SGD(model.parameters(), lr=10e-6, momentum=0.9)
optimizer = pt.optim.Adam(model.parameters(), lr=10e-4)
start_time = time.time()
mean_loss_per_epoch_train_drr = []
mean_loss_per_epoch_train_rt60 = []
mean_loss_per_epoch_eval_drr = []
mean_loss_per_epoch_eval_rt60 = []
train(model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, loss_fn=loss_fn,
      optimizer=optimizer,
      device=device, epochs=EPOCHS)
date_time = str(datetime.datetime.now())
model_save_filename = 'cnn-save' + date_time + '.bin'

torch.save(model.state_dict(), model_save_filename)

results = {
    "model": model.__class__.__name__,
    "train_loss_drr": mean_loss_per_epoch_train_drr,
    "train_loss_rt60": mean_loss_per_epoch_train_rt60,
    "eval_loss_drr": mean_loss_per_epoch_eval_drr,
    "eval_loss_rt60": mean_loss_per_epoch_eval_rt60,
    "datetime": datetime.datetime.now()
}

print('Total execution time: {:.4f} minutes', format((time.time() - start_time) / 60))
print("Mean training loss per epoch DRR:", mean_loss_per_epoch_train_drr)
print("Mean training loss per epoch RT60:", mean_loss_per_epoch_train_rt60)
print("Mean evaluation loss per epoch DRR:", mean_loss_per_epoch_eval_drr)
print("Mean training loss per epoch RT60:", mean_loss_per_epoch_eval_rt60)

results_filename = 'results-' + date_time + '.pkl'
with open(results_filename, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

plot_filename = 'figs/loss-plot-train-' + date_time + '.png'
plt.figure(figsize=(10, 5))
plt.title("DRR and RT60 training loss per epoch")
plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_train_drr, linestyle='solid', marker='o', label="drr")
plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_train_rt60, linestyle='solid', marker='o', label="rt60")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.legend()
plt.savefig(plot_filename)
plt.show()

if EVAL:
    plot_filename = 'figs/loss-plot-eval-' + date_time + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("DRR and RT60 evalutaion loss per epoch")
    plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_eval_drr, linestyle='solid', marker='o', label="drr")
    plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_eval_rt60, linestyle='solid', marker='o', label="rt60")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()