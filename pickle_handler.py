import pickle
import matplotlib.pyplot as plt

pickle_file_path = 'results-2023-11-03 17:48:40.442050.pkl'

objects = []
with (open(pickle_file_path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
            # objects.append(torch.load(save_path,map_location=torch.device('cpu')))
        except EOFError:
            break

mean_loss_per_epoch_drr = objects[0]['loss_drr']
mean_loss_per_epoch_rt60 = objects[0]['loss_rt60']

print('Model:', objects[0]['model'])
print('Number of epochs:', len(mean_loss_per_epoch_drr))
print('DRR loss per epoch:', objects[0]['loss_drr'])
print('RT60 loss per epoch:', objects[0]['loss_rt60'])
print('Date and time:', objects[0]['datetime'])

PLOT = True

if PLOT:
    plot_filename = 'figs/loss-plot-' + str(objects[0]['datetime']) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("DRR and RT60 estimation loss per epoch")
    plt.plot(mean_loss_per_epoch_drr, linestyle='solid', marker='o', label="drr")
    plt.plot(mean_loss_per_epoch_rt60, linestyle='solid', marker='o', label="rt60")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()

