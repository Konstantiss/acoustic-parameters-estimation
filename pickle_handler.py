import pickle
import matplotlib.pyplot as plt

pickle_file_path = 'results-2023-11-06 13:14:44.401792.pkl'

pkl_contents = []
with (open(pickle_file_path, "rb")) as openfile:
    while True:
        try:
            pkl_contents.append(pickle.load(openfile))
            # pkl_contents.append(torch.load(save_path,map_location=torch.device('cpu')))
        except EOFError:
            break

mean_loss_per_epoch_train_drr = pkl_contents[0]['train_loss_drr']
mean_loss_per_epoch_train_rt60 = pkl_contents[0]['train_loss_rt60']
mean_loss_per_epoch_eval_drr = pkl_contents[0]['eval_loss_drr']
mean_loss_per_epoch_eval_rt60 = pkl_contents[0]['eval_loss_rt60']
num_epochs = len(mean_loss_per_epoch_train_drr)

print('Model:', pkl_contents[0]['model'])
print('Number of epochs:', num_epochs)
print('DRR train loss per epoch:', mean_loss_per_epoch_train_drr)
print('RT60 train loss per epoch:', mean_loss_per_epoch_train_rt60)
print('DRR evaluation loss per epoch:', mean_loss_per_epoch_eval_drr)
print('RT60 evaluation loss per epoch:', mean_loss_per_epoch_eval_rt60)
print('Date and time:', pkl_contents[0]['datetime'])

PLOT = True

if PLOT:
    plot_filename = 'figs/loss-plot-train-' + str(pkl_contents[0]['datetime']) + '-' + str(num_epochs) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("DRR and RT60 training loss per epoch")
    plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_train_drr, linestyle='solid', marker='o', label="drr")
    plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_train_rt60, linestyle='solid', marker='o', label="rt60")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(1, )
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()

    plot_filename = 'figs/loss-plot-eval-' + str(pkl_contents[0]['datetime']) + '-' + str(num_epochs) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("DRR and RT60 evaluation loss per epoch")
    plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_eval_drr, linestyle='solid', marker='o', label="drr")
    plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_eval_rt60, linestyle='solid', marker='o', label="rt60")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(1, )
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()

