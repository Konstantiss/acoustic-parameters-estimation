import torch
from torch import nn
import torchaudio as ta
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import numpy as numpy
import os
import subprocess
from torcheval.metrics.functional import r2_score
from torchmetrics.regression import PearsonCorrCoef
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from tqdm import tqdm
import pandas as pd
from dataloader import *
from CNN import *
import time
import matplotlib.pyplot as plt


def evaluate(model, eval_dataloader, loss_fn, device):
    mean_loss_per_epoch_eval_drr = []
    mean_loss_per_epoch_eval_rt60 = []

    print("---- Evaluation ---\n")
    model = model.to(device)
    model = model.eval()
    losses_per_epoch_eval_drr = []
    losses_per_epoch_eval_rt60 = []
    error_per_epoch_eval_drr = []
    error_per_epoch_eval_rt60 = []
    with tqdm.tqdm(eval_dataloader, unit="batch", total=len(eval_dataloader)) as tepoch:
        for waveform, drrs_true, rt60s_true in tepoch:
            tepoch.set_description(f"Evaluation")
            waveform = waveform.to(device)
            drrs_true = drrs_true.to(device)
            rt60s_true = rt60s_true.to(device)
            # calculate loss and preds
            if model.__class__.__name__ == 'CNNNetwork':
                drr_estimates, rt60_estimates = model(waveform)
            else:
                estimates = model(waveform)
                drr_estimates = estimates[:, 0]
                rt60_estimates = estimates[:, 1]
            loss_drr = loss_fn(drr_estimates.float(), drrs_true.float())
            loss_rt60 = loss_fn(rt60_estimates.float(), rt60s_true.float())
            #pearson = PearsonCorrCoef().to(device)
            error_rt60 = sum(rt60_estimates.float() - rt60s_true.float()) / len(rt60s_true)
            error_drr = sum(drr_estimates.float() - drrs_true.float()) / len(drrs_true)
            # pearson_drr = pearson(drrs_true.float(), drr_estimates.float())
            # pearson_rt60 = pearson( rt60s_true.float(), rt60_estimates.float())
            losses_per_epoch_eval_drr.append(loss_drr.item())
            losses_per_epoch_eval_rt60.append(loss_rt60.item())
            error_per_epoch_eval_drr.append(error_drr.item())
            error_per_epoch_eval_rt60.append(error_rt60.item())
            tepoch.set_postfix(loss_drr=loss_drr.item(), loss_rt60=loss_rt60.item())
    current_epoch_loss_eval_drr = sum(losses_per_epoch_eval_drr) / len(losses_per_epoch_eval_drr)
    current_epoch_loss_eval_rt60 = sum(losses_per_epoch_eval_rt60) / len(losses_per_epoch_eval_rt60)
    current_epoch_error_eval_drr = sum(error_per_epoch_eval_drr) / len(error_per_epoch_eval_drr)
    current_epoch_error_eval_rt60 = sum(error_per_epoch_eval_rt60) / len(error_per_epoch_eval_rt60)
    print(f"DRR evaluation loss:",
          current_epoch_loss_eval_drr)
    print(f"RT60 evaluation loss:",
          current_epoch_loss_eval_rt60)
    print(f"DRR mean error:", current_epoch_error_eval_drr)
    print(f"RT60 mean error :", current_epoch_error_eval_rt60)

    mean_loss_per_epoch_eval_drr.append(current_epoch_loss_eval_drr)
    mean_loss_per_epoch_eval_rt60.append(current_epoch_loss_eval_rt60)

    return mean_loss_per_epoch_eval_drr, mean_loss_per_epoch_eval_rt60
