import torch
from datasets import get_dataloader
from encoders import get_encoder,decode
from norse.torch import LIFParameters
from models import SeqNet,DendSeqNet,ConvNet,Model,getSNN
from optimizers import adam,sgd
from tt_loops import train,test,save
from tqdm import trange
import os
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

batch_sz = 100
date = '230219'
dataset = 'fmnist'
enc_name = 'poisson'
net_name = 'DendSeqNet2'
seq_len = 60
p_fmax = 100
dt = 1e-3
epochs = 20
seeds = 1
hidden = 200
lr = 1e-4
hchannels = 2
ochannels = 2
lif_params = LIFParameters(method='super',alpha=100,tau_mem_inv=torch.as_tensor(1/0.01))
# lif_params = LIFParameters(method='super',alpha=100,tau_mem_inv=torch.as_tensor(1.0/2e-2),tau_syn_inv=torch.as_tensor(1.0/1e-2))

target_dir = f'{date}_{dataset}_{net_name}_{enc_name}{p_fmax}_lr{lr}_tlen{seq_len}_hch{hchannels}_och{ochannels}_config1'

if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)

train_l,test_l = get_dataloader(dataset,batch_sz)
encode = get_encoder(enc_name,seq_len,lif_params,p_fmax,dt)

training_losses = []
mean_losses = []
test_losses = []
accuracies = []
spk_act = []
highest = 0
for i in range(seeds):
    pbar = trange(epochs, ncols=100, unit="epoch")
    spk_net = getSNN(net_name,hidden,lif_params,hchannels,ochannels,dt)
    # print(spk_net)
    model = Model(
        encoder=encode,
        snn=spk_net,
        decoder=decode
    ).to(DEVICE)
    optimizer = adam(model.parameters(),lr=lr)

    for ep in pbar:
        training_loss,mean_loss = train(model, DEVICE, train_l, optimizer)
        test_loss,accuracy,spks = test(model, DEVICE, test_l)
        # if accuracy > highest:
        #     model_path = "./outputs/" + target_dir + f"/{dataset}_{net_name}.pt"
        #     save(
        #         model_path,
        #         epoch=ep,
        #         model=model,
        #         optimizer=optimizer,
        #     )
        #     highest = accuracy
        training_losses += training_loss
        spk_act.append(spks)
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)       
        pbar.set_postfix(accuracy=accuracies[-3:],spks=spk_act[-3:])
    print(f"accuracies: {accuracies[-epochs:]}")
    print(f"spks: {spk_act[-epochs:]}")

np.save("./outputs/" + target_dir + "/training_losses.npy", np.array(training_losses))
np.save("./outputs/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
np.save("./outputs/" + target_dir + "/test_losses.npy", np.array(test_losses).reshape(seeds,epochs).T)
np.save("./outputs/" + target_dir + "/accuracies.npy", np.array(accuracies).reshape(seeds,epochs).T)
np.save("./outputs/" + target_dir + "/spk_act.npy", np.array(spk_act).reshape(seeds,epochs).T)