import torch
from norse.torch import LIFCell
from norse.torch.module.leaky_integrator import LInoweight
from module_dendr import DENDCell
from norse.torch.functional.threshold import threshold

class DLIFWrapper(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,lifparams,dt):
        super(DLIFWrapper, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = torch.nn.ModuleList([torch.nn.Linear(in_channel,self.out_channels, bias=False) 
                                            for in_channel in self.in_channels])
        self.lif = LIFCell(p=lifparams,dt=dt)
        self.dendrite = DENDCell(p=lifparams,dt=dt)
        self.relu = torch.nn.ReLU()
    
    def forward(self, groups, sd, sh):
        sd_o = [None]*len(groups)
        x0,sd_o[0] = self.dendrite(self.weights[0]((groups[0])),sd[0])
        for i in range(1,len(groups)):
            temp,sd_o[i] = self.dendrite(self.weights[i]((groups[i])),sd[i])
            x0 += temp
        z,sh_o = self.lif(x0,sh)
        # for i in range(len(groups)):
        #     v = (1-z)*sd_o[i].v
        #     sd_o[i] = sd_o[i]._replace(v=v)
        return z,sd_o,sh_o

class DOutWrapper(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,lifparams,dt):
        super(DOutWrapper, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = torch.nn.ModuleList([torch.nn.Linear(in_channel,self.out_channels, bias=False) 
                                            for in_channel in self.in_channels])
        self.li = LInoweight(p=lifparams,dt=dt)
        self.dendrite = DENDCell(p=lifparams,dt=dt)
    
    def forward(self, groups, sd, so):
        sd_o = [None]*len(groups)
        x0,sd_o[0] = self.dendrite(self.weights[0]((groups[0])),sd[0])
        for i in range(1,len(groups)):
            temp,sd_o[i] = self.dendrite(self.weights[i]((groups[i])),sd[i])
            x0 += temp
        z,so_o = self.li(x0,so)
        return z,sd_o,so_o
    
class DLIFWrapper2(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,lifparams,dt):
        super(DLIFWrapper2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = torch.nn.ModuleList([torch.nn.Linear(in_channel,self.out_channels, bias=False) 
                                            for in_channel in self.in_channels])
        self.dendrite = DENDCell(p=lifparams,dt=dt)
        self.p = lifparams
    
    def forward(self, groups, sd):
        sd_o = [None]*len(groups)
        x0,sd_o[0] = self.dendrite(self.weights[0]((groups[0])),sd[0])
        for i in range(1,len(groups)):
            temp,sd_o[i] = self.dendrite(self.weights[i]((groups[i])),sd[i])
            x0 += temp
        z = threshold(x0 - self.p.v_th, self.p.method, self.p.alpha)
        print(z)
        for i in range(len(groups)):
            v = (1-z)*sd_o[i].v
            sd_o[i] = sd_o[i]._replace(v=v)
        return z,sd_o

class DOutWrapper2(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,lifparams,dt):
        super(DOutWrapper2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = torch.nn.ModuleList([torch.nn.Linear(in_channel,self.out_channels, bias=False) 
                                            for in_channel in self.in_channels])
        self.dendrite = DENDCell(p=lifparams,dt=dt)
    
    def forward(self, groups, sd):
        sd_o = [None]*len(groups)
        x0,sd_o[0] = self.dendrite(self.weights[0]((groups[0])),sd[0])
        for i in range(1,len(groups)):
            temp,sd_o[i] = self.dendrite(self.weights[i]((groups[i])),sd[i])
            x0 += temp
        z = x0
        return z,

class DLIFWrapper3(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,lifparams,dt):
        super(DLIFWrapper3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = torch.nn.ModuleList([torch.nn.Linear(in_channel,self.out_channels, bias=False) 
                                            for in_channel in self.in_channels])
        self.dendrite = DENDCell(p=lifparams,dt=dt)
        self.p = lifparams
    
    def forward(self, groups, sd):
        sd_o = [None]*len(groups)
        x0,sd_o[0] = self.dendrite(self.weights[0]((groups[0])),sd[0])
        for i in range(1,len(groups)):
            temp,sd_o[i] = self.dendrite(self.weights[i]((groups[i])),sd[i])
            x0 += temp
        z = threshold(x0 - self.p.v_th, self.p.method, self.p.alpha)
        # print(z)
        # for i in range(len(groups)):
        #     v = (1-z)*sd_o[i].v
        #     sd_o[i] = sd_o[i]._replace(v=v)
        return z,sd_o

class DOutWrapper3(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,lifparams,dt):
        super(DOutWrapper3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = torch.nn.ModuleList([torch.nn.Linear(in_channel,self.out_channels, bias=False) 
                                            for in_channel in self.in_channels])
        self.dendrite = DENDCell(p=lifparams,dt=dt)
    
    def forward(self, groups, sd):
        sd_o = [None]*len(groups)
        x0,sd_o[0] = self.dendrite(self.weights[0]((groups[0])),sd[0])
        for i in range(1,len(groups)):
            temp,sd_o[i] = self.dendrite(self.weights[i]((groups[i])),sd[i])
            x0 += temp
        z = x0
        return z,sd_o