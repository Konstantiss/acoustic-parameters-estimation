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

annotation_file_names = ['single_eval.csv', 'chromebook_eval.csv', 'mobile_eval.csv',
                         'crucif_eval.csv', 'lin8ch_eval.csv', 'em32_eval.csv']

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 64
EPOCHS = 30

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)


model = models.densenet121(pretrained=True).to(device)

# Freeze all the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(1024, 2, device=device)
)

model.load_state_dict(torch.load(RESULTS_DIR + 'densenet-save-2024-03-02 110634.272891-15.bin'))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
start_time = time.time()

losses_drr = []
losses_rt60 = []

for i, file in enumerate(annotation_file_names):
    annotation_file_path = DATA_PATH_EVAL + file
    print(file)
    eval_dataset = ACEDataset(annotation_file_path, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device, resnet=True,
                              image_transformation=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_drr, loss_rt60 = evaluate(model=model, eval_dataloader=eval_dataloader,
                                   loss_fn=loss_fn, device=device)
    print(file.split('_')[0] + " mic loss DRR:", loss_drr)
    print(file.split('_')[0] + " mic loss RT60:", loss_rt60)
    losses_drr.append(loss_drr)
    losses_rt60.append(loss_rt60)

execution_time = (time.time() - start_time) / 60
date_time = str(datetime.datetime.now())

single_loss_drr = losses_drr[0]
single_loss_rt60 = losses_rt60[0]
chromebook_loss_drr = losses_drr[1]
chrombook_loss_rt60 = losses_rt60[1]
mobile_loss_drr = losses_drr[2]
mobile_loss_rt60 = losses_rt60[2]
crucif_loss_drr = losses_drr[3]
crucif_loss_rt60 = losses_rt60[3]
lin8ch_loss_drr = losses_drr[4]
lin8ch_loss_rt60 = losses_rt60[4]
em32_loss_drr = losses_drr[5]
em32_loss_rt60 = losses_rt60[5]

results = {
    "model": model.__class__.__name__,
    "single_loss_drr": single_loss_drr,
    "single_loss_rt60": single_loss_rt60,
    "chromebook_loss_drr": chromebook_loss_drr,
    "chrombook_loss_rt60": chrombook_loss_rt60,
    "mobile_loss_drr": mobile_loss_drr,
    "mobile_loss_rt60": mobile_loss_rt60,
    "crucif_loss_drr": crucif_loss_drr,
    "crucif_loss_rt60": crucif_loss_rt60,
    "lin8ch_loss_drr": lin8ch_loss_drr,
    "lin8ch_loss_rt60": lin8ch_loss_rt60,
    "em32_loss_drr": em32_loss_drr,
    "em32_loss_rt60": em32_loss_rt60,
    "datetime": datetime.datetime.now(),
    "execution_time": execution_time
}

print('Total execution time: {:.4f} minutes', format(execution_time))
print("Evaluation loss DRR:", losses_drr)
print("Evaluation loss RT60:", losses_rt60)

results_filename = RESULTS_DIR + 'results-densenet-per-mic-' + date_time + '.pkl'
results_filename = results_filename.replace(":", "")
with open(results_filename, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
