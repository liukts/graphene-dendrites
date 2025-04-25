import torch
from datasets import get_dataloader
from models_ann import getANN
from optimizers import adam,sgd
from tt_loops import train,test_ann,save
from tqdm import trange
import os
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

batch_sz = 1
date = '241114'
dataset = 'iris'
net_name = 'PercepNet'
epochs = 20
seeds = 1
lr = 1e-2

target_dir = f'{date}_{dataset}_{net_name}_lr{lr}'# _noise{noise_std}'

if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)

train_l,test_l = get_dataloader(dataset,batch_sz)

training_losses = []
mean_losses = []
test_losses = []
accuracies = []
highest = 0
for i in range(seeds):
    pbar = trange(epochs, ncols=100, unit="epoch")
    model = getANN(net_name).to(DEVICE)
    optimizer = adam(model.parameters(),lr=lr)

    for ep in pbar:
        training_loss,mean_loss = train(model, DEVICE, train_l, optimizer)
        test_loss,accuracy = test_ann(model, DEVICE, test_l)
        if accuracy > highest:
            model_path = "./outputs/" + target_dir + f"/{dataset}_{net_name}.pt"
            save(
                model_path,
                epoch=ep,
                model=model,
                optimizer=optimizer,
            )
            highest = accuracy
        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)       
        pbar.set_postfix(accuracy=accuracies[-3:])
    print(f"accuracies: {accuracies[-epochs:]}")

np.save("./outputs/" + target_dir + "/training_losses.npy", np.array(training_losses))
np.save("./outputs/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
np.save("./outputs/" + target_dir + "/test_losses.npy", np.array(test_losses).reshape(seeds,epochs).T)
np.save("./outputs/" + target_dir + "/accuracies.npy", np.array(accuracies).reshape(seeds,epochs).T)