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

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] < 3 else x),  # convert images to 3 channels
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

model = models.vgg11(pretrained=True).to(device)

# Freeze all the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=2, device=device)
)

model.load_state_dict(torch.load(RESULTS_DIR + 'vgg-save-2024-03-02 164800.562394-15.bin'))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
start_time = time.time()

losses_drr = []
losses_rt60 = []

for i, file in enumerate(annotation_file_names):
    annotation_file_path = DATA_PATH_EVAL + file
    eval_dataset = ACEDataset(annotation_file_path, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device, resnet=True,
                              image_transformation=transform)
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
