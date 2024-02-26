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
annotations_file_path_train = DATA_PATH_TRAIN + 'features_and_ground_truth_train.csv'

DATA_PATH_EVAL = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
annotations_file_path_eval = DATA_PATH_EVAL + 'features_and_ground_truth_eval.csv'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 256
EPOCHS = 30

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
train_dataset = ACEDataset(annotations_file_path_train, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device, resnet=True,
                           image_transformation=transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataset = ACEDataset(annotations_file_path_eval, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device, resnet=True,
                          image_transformation=transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = models.resnet18(pretrained=True).to(device)

# Freeze all the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2, device=device)

model.load_state_dict(torch.load(RESULTS_DIR + 'resnet-save-2023-11-23 155931.877396-30.bin'))

loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=10e-6, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
start_time = time.time()

mean_loss_per_epoch_eval_drr, mean_loss_per_epoch_eval_rt60 = evaluate(model=model, eval_dataloader=eval_dataloader, loss_fn=loss_fn, device=device)

execution_time = (time.time() - start_time) / 60
date_time = str(datetime.datetime.now())

results = {
    "model": model.__class__.__name__,
    "eval_loss_drr": mean_loss_per_epoch_eval_drr,
    "eval_loss_rt60": mean_loss_per_epoch_eval_rt60,
    "datetime": datetime.datetime.now(),
    "execution_time": execution_time
}

print('Total execution time: {:.4f} minutes', format(execution_time))
print("Evaluation loss DRR:", mean_loss_per_epoch_eval_drr)
print("Evaluation loss RT60:", mean_loss_per_epoch_eval_rt60)

results_filename = RESULTS_DIR + 'results-resnet-' + date_time + '-' + str(EPOCHS) + '.pkl'
# with open(results_filename, 'wb') as handle:
#     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
