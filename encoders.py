from norse.torch import ConstantCurrentLIFEncoder,PoissonEncoder
import torch

def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

def get_encoder(name,seq_length,lif_params,p_fmax,dt):
    if name == 'lif':
        return ConstantCurrentLIFEncoder(seq_length=seq_length,p=lif_params,dt=dt)
    elif name == 'poisson':
        return PoissonEncoder(seq_length=seq_length,f_max=p_fmax,dt=dt)