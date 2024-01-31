import matplotlib.pyplot as plt
from norse.torch import LIFParameters
import torch
import norse
import numpy as np

plt.style.use(['science','nature'])

lif_params = LIFParameters(method='super',alpha=100,tau_mem_inv=torch.as_tensor(1/0.02),tau_syn_inv=torch.as_tensor(1/0.02))

activation = norse.torch.LI(p=lif_params)
data = torch.zeros(500,1)
data[0] = 1.0
data[100] = 1.0
data[200] = 1.0
data[300] = 1.0
data[400] = 1.0

voltage_trace, _ = activation(data)
fig1,ax1 = plt.subplots(1,1,figsize=(1.5,1.15))
ax1.plot(voltage_trace.detach())

lif_params = LIFParameters(method='super',alpha=100,tau_mem_inv=torch.as_tensor(1/0.02),tau_syn_inv=torch.as_tensor(1/0.001))

activation = norse.torch.LI(p=lif_params)

data[0] = 8.0
data[100] = 8.0
data[200] = 8.0
data[300] = 8.0
data[400] = 8.0

voltage_trace, _ = activation(data)
ax1.plot(voltage_trace.detach())

ax1.set_xlim([0,500])
fig1.savefig('./outputs/spike.svg')