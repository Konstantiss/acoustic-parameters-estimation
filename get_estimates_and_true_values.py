from tqdm import tqdm
from dataloader import *

EARLY_STOPPING_PATIENCE = 4


def get_estimates_and_true_values(model, eval_dataloader, loss_fn, optimizer, device):
    print("Model: ", model.__class__.__name__)
    model = model.to(device)
    for epoch in range(1):

        print("---- Evaluation ---\n")

        model = model.eval()

        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        with tqdm.tqdm(eval_dataloader, unit="batch", total=len(eval_dataloader)) as tepoch:
            for waveform, drrs_true, rt60s_true in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
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
                # backpropogate the losses and update the gradients
                optimizer.zero_grad()
                loss_drr.backward(retain_graph=True)
                loss_rt60.backward()
                optimizer.step()
                tepoch.set_postfix(loss_drr=loss_drr.item(), loss_rt60=loss_rt60.item())
                return drrs_true.tolist(), drr_estimates.tolist(), rt60s_true.tolist(), rt60_estimates.tolist()
