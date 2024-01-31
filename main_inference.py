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

from tqdm import tqdm, trange

# folder to save results
target_dir = "230112_inf_neurmw"
if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)
load_dir = "230103_s05_fashion_noise0.0_seqnet0500_neurmw_ep20_lr0.001_T80_alpha100_beta5_pf1e3_bn_initxunif"

batch_sz = 100
date = '230927'
dataset = 'fmnist'
enc_name = 'poisson'
net_name = 'SeqNet'
seq_len = 6
p_fmax = 500
dt = 1e-3
epochs = 20
seeds = 1
hidden = 200
lr = 1e-3
hchannels = 1
ochannels = 1

lif_params = LIFParameters(method='super',alpha=100,tau_mem_inv=torch.as_tensor(1.0/0.01),tau_syn_inv=torch.as_tensor(1.0/0.001))
config = 1

target_dir = f'{date}_{dataset}_{net_name}_{enc_name}{p_fmax}_lr{lr}_tlen{seq_len}_hch{hchannels}_och{ochannels}_config{config}'

# dataiter = iter(test_loader)
# images,labels = dataiter.next()
# figim,axim = plt.subplots(1,1,figsize=(2,1.4))
# axim.imshow((images[0].reshape(28,28)),cmap='gray')
# axim.get_xaxis().set_visible(False)
# axim.get_yaxis().set_visible(False)
# figim.savefig(f'noiseimage_{noise_std}.svg')

class Model(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y

def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='test', unit='batch', ncols=80, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

test_losses = []
accuracies = []
highest = 0

noise_std_range = np.linspace(0,3,16)

snn = SeqNet(h1=hidden,h2=hidden,beta=beta,alpha=alpha)
model = Model(
    # encoder=encode.ConstantCurrentLIFEncoder(T),
    encoder=encode.PoissonEncoder(seq_length=T,f_max=f_poisson),
    # snn=ConvNet(alpha=alpha,k=k,beta=beta),
    snn=snn,
    decoder=decode
).to(DEVICE)

model.load_state_dict(torch.load("./outputs/"+load_dir+"/fmnist_dwmtj.pt")["model_state_dict"])
pbar = tqdm(noise_std_range,ncols=80)
for noise in pbar:
    for s in range(seeds):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                AddGaussianNoise(mean=0,std=noise),
            ]
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(
                root=".",
                train=False,
                transform=transform,
            ),
            batch_size=BATCH_SIZE
        )
        test_loss, accuracy = test(model, DEVICE, test_loader)
        test_losses.append(test_loss)
        accuracies.append(accuracy)  
    pbar.set_postfix(noise=noise,acc=np.mean(np.array(accuracies)[-5:]))
accs_np = np.array(accuracies).reshape(len(noise_std_range),seeds)
print(accs_np)

if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)
np.save("./outputs/" + target_dir + "/accs_np.npy",accs_np)
np.save("./outputs/" + target_dir + "/noise_std_range.npy",noise_std_range)
# np.save("./outputs/" + target_dir + "/training_losses.npy", np.array(training_losses))
# np.save("./outputs/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
# np.save("./outputs/" + target_dir + "/test_losses.npy", np.array(test_losses).reshape(seeds,EPOCHS).T)
# np.save("./outputs/" + target_dir + "/accuracies.npy", np.array(accuracies).reshape(seeds,EPOCHS).T)