import torch
from datasets import get_dataloader
from encoders import get_encoder,decode
from norse.torch import LIFParameters
from models import SeqNet,ConvNet,Model
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
dataset = 'mnist'
enc_name = 'lif'
net_name = 'SeqNet'
seq_len = 50
p_fmax = 1000
dt = 0.001
epochs = 10
seeds = 1
hidden = 200
lif_params = LIFParameters(method='super',alpha=100)

target_dir = 'SeqNet_test_mnist'

train_l,test_l = get_dataloader(dataset,batch_sz)
encode = get_encoder(enc_name,seq_len,lif_params,p_fmax,dt)
if net_name == 'ConvNet':
    spk_net = ConvNet(h1=hidden,lifparams=lif_params)
elif net_name == 'SeqNet':
    spk_net = SeqNet(h1=hidden,lifparams=lif_params)
model = Model(
    encoder=encode,
    snn=spk_net,
    decoder=decode
).to(DEVICE)
optimizer = adam(model.parameters())

training_losses = []
mean_losses = []
test_losses = []
accuracies = []
for i in range(seeds):
    pbar = trange(epochs, ncols=100, unit="epoch")
    for ep in pbar:
        training_loss,mean_loss = train(model, DEVICE, train_l, optimizer)
        test_loss,accuracy = test(model, DEVICE, test_l)
        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)       
        pbar.set_postfix(accuracy=accuracies[-3:])
    print(f"accuracies: {accuracies[-epochs:]}")

if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)
np.save("./outputs/" + target_dir + "/training_losses.npy", np.array(training_losses))
np.save("./outputs/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
np.save("./outputs/" + target_dir + "/test_losses.npy", np.array(test_losses).reshape(seeds,epochs).T)
np.save("./outputs/" + target_dir + "/accuracies.npy", np.array(accuracies).reshape(seeds,epochs).T)