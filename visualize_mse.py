import torchvision.models as models
from torchvision.transforms import transforms
from train_and_evaluate_model import *
from dataloader import *
from CNN import *
from get_estimates_and_true_values import *
import time
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)

print("--- CNN ---")

RESULTS_DIR = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Results/'

DATA_PATH_TRAIN = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Train/Speech/'
annotations_file_path_train = DATA_PATH_TRAIN + 'features_and_ground_truth_train.csv'

DATA_PATH_EVAL = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
annotations_file_path_eval = DATA_PATH_EVAL + 'features_and_ground_truth_eval.csv'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 32

melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
eval_dataset = ACEDataset(annotations_file_path_eval, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = CNNNetwork().cuda()
melspectogram = ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=10e-4, momentum=0.9)

model.load_state_dict(torch.load(RESULTS_DIR + 'cnn-save-2024-03-02 024311.642104-15.bin'))

drr_true, drr_estimates, rt60_true, rt60_estimates = get_estimates_and_true_values(model=model,
                                                                                   eval_dataloader=eval_dataloader,
                                                                                   loss_fn=loss_fn,
                                                                                   optimizer=optimizer,
                                                                                   device=device)
x = len(drr_true) + 1

lines_drr = []
for i, j, k in zip(range(1, x), drr_true, drr_estimates):
    pair = [(i, j), (i, k)]
    lines_drr.append(pair)

lines_rt60 = []
for i, j, k in zip(range(1, x), rt60_true, rt60_estimates):
    pair = [(i, j), (i, k)]
    lines_rt60.append(pair)

linecoll = matcoll.LineCollection(lines_drr, colors='k', label="Estimation Error")

plot_filename = RESULTS_DIR + 'figs/cnn-drr-true-vs-estimates.png'
fig, ax = plt.subplots(figsize=(11, 5))
plt.title("CNN DRR Estimations VS True Values")
plt.plot(range(1, x), drr_true, 'o', color='k', label="True values")
plt.plot(range(1, x), drr_estimates, 'o', color='r', label="Estimates")
ax.add_collection(linecoll)
plt.ylabel("Direct to Reverberant Ratio")
plt.legend()
plt.savefig(plot_filename)
plt.show()

linecoll = matcoll.LineCollection(lines_rt60, colors='k', label="Estimation Error")

plot_filename = RESULTS_DIR + 'figs/cnn-rt60-true-vs-estimates.png'
fig, ax = plt.subplots(figsize=(11, 5))
plt.title("CNN RT60 Estimations VS True Values")
plt.plot(range(1, len(rt60_true) + 1), rt60_true, 'o', color='k', label="True values")
plt.plot(range(1, len(rt60_estimates) + 1), rt60_estimates, 'o', color='r', label="Estimates")
ax.add_collection(linecoll)
plt.ylabel("Reverberation Time")
plt.legend()
plt.savefig(plot_filename)
plt.show()

print("--- ResNet ---")

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


eval_dataset = ACEDataset(annotations_file_path_eval, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device, resnet=True,
                          image_transformation=transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = models.resnet18(pretrained=True).to(device)

# Freeze all the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(RESULTS_DIR + 'resnet-save-2024-03-02 064213.893337-15.bin'))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=10e-4, momentum=0.9)

drr_true, drr_estimates, rt60_true, rt60_estimates = get_estimates_and_true_values(model=model,
                                                                                   eval_dataloader=eval_dataloader,
                                                                                   loss_fn=loss_fn,
                                                                                   optimizer=optimizer,
                                                                                   device=device)
lines_drr = []
for i, j, k in zip(range(1, x), drr_true, drr_estimates):
    pair = [(i, j), (i, k)]
    lines_drr.append(pair)

lines_rt60 = []
for i, j, k in zip(range(1, x), rt60_true, rt60_estimates):
    pair = [(i, j), (i, k)]
    lines_rt60.append(pair)

linecoll = matcoll.LineCollection(lines_drr, colors='k', label="Estimation Error")

plot_filename = RESULTS_DIR + 'figs/resnet-drr-true-vs-estimates.png'
fig, ax = plt.subplots(figsize=(11, 5))
plt.title("ResNet DRR Estimations VS True Values")
plt.plot(range(1, x), drr_true, 'o', color='k', label="True values")
plt.plot(range(1, x), drr_estimates, 'o', color='r', label="Estimates")
ax.add_collection(linecoll)
plt.ylabel("Direct to Reverberant Ratio")
plt.legend()
plt.savefig(plot_filename)
plt.show()

linecoll = matcoll.LineCollection(lines_rt60, colors='k', label="Estimation Error")

plot_filename = RESULTS_DIR + 'figs/resnet-rt60-true-vs-estimates.png'
fig, ax = plt.subplots(figsize=(11, 5))
plt.title("ResNet RT60 Estimations VS True Values")
plt.plot(range(1, len(rt60_true) + 1), rt60_true, 'o', color='k', label="True values")
plt.plot(range(1, len(rt60_estimates) + 1), rt60_estimates, 'o', color='r', label="Estimates")
ax.add_collection(linecoll)
plt.ylabel("Reverberation Time")
plt.legend()
plt.savefig(plot_filename)
plt.show()

print("--- VGG ---")


eval_dataset = ACEDataset(annotations_file_path_eval, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device, resnet=True,
                          image_transformation=transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

drr_true, drr_estimates, rt60_true, rt60_estimates = get_estimates_and_true_values(model=model,
                                                                                   eval_dataloader=eval_dataloader,
                                                                                   loss_fn=loss_fn,
                                                                                   optimizer=optimizer,
                                                                                   device=device)
lines_drr = []
for i, j, k in zip(range(1, x), drr_true, drr_estimates):
    pair = [(i, j), (i, k)]
    lines_drr.append(pair)

lines_rt60 = []
for i, j, k in zip(range(1, x), rt60_true, rt60_estimates):
    pair = [(i, j), (i, k)]
    lines_rt60.append(pair)

linecoll = matcoll.LineCollection(lines_drr, colors='k', label="Estimation Error")

plot_filename = RESULTS_DIR + 'figs/vgg-drr-true-vs-estimates.png'
fig, ax = plt.subplots(figsize=(11, 5))
plt.title("VGG DRR Estimations VS True Values")
plt.plot(range(1, x), drr_true, 'o', color='k', label="True values")
plt.plot(range(1, x), drr_estimates, 'o', color='r', label="Estimates")
ax.add_collection(linecoll)
plt.ylabel("Direct to Reverberant Ratio")
plt.legend()
plt.savefig(plot_filename)
plt.show()

linecoll = matcoll.LineCollection(lines_rt60, colors='k', label="Estimation Error")

plot_filename = RESULTS_DIR + 'figs/vgg-rt60-true-vs-estimates.png'
fig, ax = plt.subplots(figsize=(11, 5))
plt.title("VGG RT60 Estimations VS True Values")
plt.plot(range(1, len(rt60_true) + 1), rt60_true, 'o', color='k', label="True values")
plt.plot(range(1, len(rt60_estimates) + 1), rt60_estimates, 'o', color='r', label="Estimates")
ax.add_collection(linecoll)
plt.ylabel("Reverberation Time")
plt.legend()
plt.savefig(plot_filename)
plt.show()

print("--- DenseNet ---")

eval_dataset = ACEDataset(annotations_file_path_eval, melspectogram, SAMPLE_RATE, NUM_SAMPLES, device, resnet=True,
                          image_transformation=transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = models.densenet121(pretrained=True).to(device)

# Freeze all the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(1024, 2, device=device)
)

model.load_state_dict(torch.load(RESULTS_DIR + 'densenet-save-2024-03-02 110634.272891-15.bin'))

optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
start_time = time.time()

drr_true, drr_estimates, rt60_true, rt60_estimates = get_estimates_and_true_values(model=model,
                                                                                   eval_dataloader=eval_dataloader,
                                                                                   loss_fn=loss_fn,
                                                                                   optimizer=optimizer,
                                                                                   device=device)
lines_drr = []
for i, j, k in zip(range(1, x), drr_true, drr_estimates):
    pair = [(i, j), (i, k)]
    lines_drr.append(pair)

lines_rt60 = []
for i, j, k in zip(range(1, x), rt60_true, rt60_estimates):
    pair = [(i, j), (i, k)]
    lines_rt60.append(pair)

linecoll = matcoll.LineCollection(lines_drr, colors='k', label="Estimation Error")

plot_filename = RESULTS_DIR + 'figs/densenet-drr-true-vs-estimates.png'
fig, ax = plt.subplots(figsize=(11, 5))
plt.title("DenseNet DRR Estimations VS True Values")
plt.plot(range(1, x), drr_true, 'o', color='k', label="True values")
plt.plot(range(1, x), drr_estimates, 'o', color='r', label="Estimates")
ax.add_collection(linecoll)
plt.ylabel("Direct to Reverberant Ratio")
plt.legend()
plt.savefig(plot_filename)
plt.show()

linecoll = matcoll.LineCollection(lines_rt60, colors='k', label="Estimation Error")

plot_filename = RESULTS_DIR + 'figs/densenet-rt60-true-vs-estimates.png'
fig, ax = plt.subplots(figsize=(11, 5))
plt.title("DenseNet RT60 Estimations VS True Values")
plt.plot(range(1, len(rt60_true) + 1), rt60_true, 'o', color='k', label="True values")
plt.plot(range(1, len(rt60_estimates) + 1), rt60_estimates, 'o', color='r', label="Estimates")
ax.add_collection(linecoll)
plt.ylabel("Reverberation Time")
plt.legend()
plt.savefig(plot_filename)
plt.show()


