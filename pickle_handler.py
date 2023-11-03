import pickle

pickle_file_path = 'results-2023-11-03 13:30:25.959295.pkl'

objects = []
with (open(pickle_file_path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
            # objects.append(torch.load(save_path,map_location=torch.device('cpu')))
        except EOFError:
            break

print('Model:', objects[0]['model'])
print('DRR loss per epoch:', objects[0]['loss_drr'])
print('RT60 loss per epoch:', objects[0]['loss_rt60'])
print('Date and time:', objects[0]['datetime'])