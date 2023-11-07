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


def train_evaluate(model, train_dataloader, eval_dataloader, loss_fn, optimizer, device, epochs):
    mean_loss_per_epoch_train_drr = []
    mean_loss_per_epoch_train_rt60 = []
    mean_loss_per_epoch_eval_drr = []
    mean_loss_per_epoch_eval_rt60 = []
    model = model.train()
    for epoch in range(epochs):
        print("---- Training ---\n")
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

        print("---- Evaluation ---\n")

        model = model.eval()
        losses_per_epoch_eval_drr = []
        losses_per_epoch_eval_rt60 = []
        with tqdm.tqdm(eval_dataloader, unit="batch", total=len(eval_dataloader)) as tepoch:
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
                tepoch.set_postfix(loss_drr=loss_drr.item(), loss_rt60=loss_rt60.item())
        print(f"Mean DRR evaluation loss for epoch {epoch + 1}:",
              sum(losses_per_epoch_eval_drr) / len(losses_per_epoch_eval_drr))
        print(f"Mean RT60 evaluation loss for epoch {epoch + 1}:",
              sum(losses_per_epoch_eval_rt60) / len(losses_per_epoch_eval_rt60))
        mean_loss_per_epoch_eval_drr.append(sum(losses_per_epoch_eval_drr) / len(losses_per_epoch_eval_drr))
        mean_loss_per_epoch_eval_rt60.append(sum(losses_per_epoch_eval_rt60) / len(losses_per_epoch_eval_rt60))

    return mean_loss_per_epoch_train_drr, mean_loss_per_epoch_train_rt60, mean_loss_per_epoch_eval_drr, mean_loss_per_epoch_eval_rt60
