import pickle
import matplotlib.pyplot as plt

RESULTS_DIR = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Results/'

pickle_file_path = RESULTS_DIR + 'results-resnet-per-utterance-2024-02-26 16:59:39.587180-30.pkl'

pkl_contents = []
with (open(pickle_file_path, "rb")) as openfile:
    while True:
        try:
            pkl_contents.append(pickle.load(openfile))
        except EOFError:
            break

if 'per-utterance' in pickle_file_path:
    print("---- Results per utterance ----")
    print('Model:', pkl_contents[0]['model'])
    print('S1 utterance loss DRR:', pkl_contents[0]['s1_utterance_loss_drr'])
    print('S1 utterance loss RT60:', pkl_contents[0]['s1_utterance_loss_rt60'])
    print('S2 utterance loss DRR:', pkl_contents[0]['s2_utterance_loss_drr'])
    print('S2 utterance loss RT60:', pkl_contents[0]['s2_utterance_loss_rt60'])
    print('S3 utterance loss DRR:', pkl_contents[0]['s3_utterance_loss_drr'])
    print('S3 utterance loss RT60:', pkl_contents[0]['s3_utterance_loss_rt60'])
    print('S4 utterance loss DRR:', pkl_contents[0]['s4_utterance_loss_drr'])
    print('S4 utterance loss RT60:', pkl_contents[0]['s4_utterance_loss_rt60'])
    print('S5 utterance loss DRR:', pkl_contents[0]['s5_utterance_loss_drr'])
    print('S5 utterance loss RT60:', pkl_contents[0]['s5_utterance_loss_rt60'])
    print('Date and time:', pkl_contents[0]['datetime'])
    print('Execution time:', pkl_contents[0]['execution_time'])
    model_name = pkl_contents[0]['model']
else:
    mean_loss_per_epoch_train_drr = pkl_contents[0]['train_loss_drr']
    mean_loss_per_epoch_train_rt60 = pkl_contents[0]['train_loss_rt60']
    mean_loss_per_epoch_eval_drr = pkl_contents[0]['eval_loss_drr']
    mean_loss_per_epoch_eval_rt60 = pkl_contents[0]['eval_loss_rt60']

    num_epochs = len(mean_loss_per_epoch_train_drr)
    #model_name = 'cnn' if pkl_contents[0]['model'] == 'CNNNetwork' else 'resnet'
    model_name = pkl_contents[0]['model']

    print('Model:', pkl_contents[0]['model'])
    print('Number of epochs:', num_epochs)
    print('DRR train loss per epoch:', mean_loss_per_epoch_train_drr)
    print('RT60 train loss per epoch:', mean_loss_per_epoch_train_rt60)
    print('DRR evaluation loss per epoch:', mean_loss_per_epoch_eval_drr)
    print('RT60 evaluation loss per epoch:', mean_loss_per_epoch_eval_rt60)
    print('Date and time:', pkl_contents[0]['datetime'])
    print('Execution time:', pkl_contents[0]['execution_time'])

    PLOT = True

    if PLOT:
        plot_filename = RESULTS_DIR + 'figs/' + model_name + '-loss-plot-train-' + str(pkl_contents[0]['datetime']) + '-' + str(num_epochs) + '.png'
        plt.figure(figsize=(10, 5))
        plt.title(model_name + "DRR and RT60 training loss per epoch")
        plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_train_drr, linestyle='solid', marker='o', label="drr")
        plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_train_rt60, linestyle='solid', marker='o', label="rt60")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(1, )
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(plot_filename)
        plt.show()

        # plot_filename = RESULTS_DIR + 'figs/' + model_name + 'loss-plot-eval-' + str(pkl_contents[0]['datetime']) + '-' + str(num_epochs) + '.png'
        # plt.figure(figsize=(10, 5))
        # plt.title(model_name + " DRR and RT60 evaluation loss per epoch")
        # plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_eval_drr, linestyle='solid', marker='o', label="drr")
        # plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_eval_rt60, linestyle='solid', marker='o', label="rt60")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.xlim(1, )
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.savefig(plot_filename)
        # plt.show()


