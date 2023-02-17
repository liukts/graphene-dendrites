import torch
from norse.torch import LIFCell,LILinearCell,LIFParameters
from module_neur import DLIFWrapper,DOutWrapper,DLIFWrapper2,DOutWrapper2,DLIFWrapper3,DOutWrapper3

def getSNN(net_name,hidden,lif_params,hchannels,ochannels,dt):
    if net_name == 'ConvNet':
        spk_net = ConvNet(h1=hidden,lifparams=lif_params,dt=dt)
    elif net_name == 'SeqNet':
        spk_net = SeqNet(h1=hidden,lifparams=lif_params,dt=dt)
    elif net_name == 'DendSeqNet':
        spk_net = DendSeqNet(h1=hidden,lifparams=lif_params,hchannels=hchannels,ochannels=ochannels,dt=dt)
    elif net_name == 'DendSeqNet2':
        spk_net = DendSeqNet2(h1=hidden,lifparams=lif_params,hchannels=hchannels,ochannels=ochannels,dt=dt)
    elif net_name == 'DendSeqNet3':
        spk_net = DendSeqNet3(h1=hidden,lifparams=lif_params,hchannels=hchannels,ochannels=ochannels,dt=dt)
    elif net_name == 'DendSeqNetSVHN':
        spk_net = DendSeqNetSVHN(h1=hidden,lifparams=lif_params,hchannels=hchannels,ochannels=ochannels,dt=dt)
    elif net_name == 'DendSeqNetSVHN2':
        spk_net = DendSeqNetSVHN2(h1=hidden,lifparams=lif_params,hchannels=hchannels,ochannels=ochannels,dt=dt)
    elif net_name == 'DendSeqNetSVHN3':
        spk_net = DendSeqNetSVHN3(h1=hidden,lifparams=lif_params,hchannels=hchannels,ochannels=ochannels,dt=dt)
    elif net_name == 'SeqNetSVHN':
        spk_net = SeqNetSVHN(h1=hidden,lifparams=lif_params,dt=dt)
    if net_name == 'ConvNetSVHN':
        spk_net = ConvNetSVHN(h1=hidden,lifparams=lif_params,dt=dt)
    return spk_net

# wrapper for encoder, snn, and decoder
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

# MLP containing one hidden layer
class SeqNet(torch.nn.Module):

    def __init__(
        self,  h1=200, feature_size=28, lifparams=LIFParameters(method='super', alpha=100), dt=0.001
    ):

        super(SeqNet, self).__init__()

        self.fc0 = torch.nn.Linear(feature_size*feature_size, h1)
        self.lif = LIFCell(p=lifparams,dt=dt)
        self.out = LILinearCell(h1, 10, p=lifparams,dt=dt)

    def forward(self,x):

        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # initial states
        s0 = None
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = x[ts, :].view(-1, 28*28)
            z = self.fc0(z)
            z, s0 = self.lif(z, s0)
            v, so = self.out(z, so)
            voltages[ts, :, :] = v
        return voltages
    
# Dendritic MLP containing one hidden layer
class DendSeqNet(torch.nn.Module):

    def __init__(
        self,  h1=200, feature_size=28, lifparams=LIFParameters(method='super', alpha=100), hchannels=2, ochannels=4, dt=0.001
    ):

        super(DendSeqNet, self).__init__()

        self.hidden = DLIFWrapper(in_channels=[feature_size**2//hchannels]*hchannels,out_channels=h1,lifparams=lifparams,dt=dt)
        self.out = DOutWrapper(in_channels=[h1//ochannels]*ochannels,out_channels=10,lifparams=lifparams,dt=dt)
        self.hchannels = hchannels
        self.ochannels = ochannels
        self.hl = h1
        self.feature_size = feature_size
        # for mod in self.modules():
        #     if isinstance(mod, torch.nn.Linear):
        #         torch.nn.init.normal_(mod.weight, std=1.)

    def forward(self,x):

        # dendrite splits
        spl_1 = self.feature_size**2//self.hchannels
        spl_2 = self.hl//self.ochannels

        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # initial states
        sdh = [None]*self.hchannels
        sh = None
        sdo = [None]*self.ochannels
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = x[ts, :].view(-1, self.feature_size**2)
            z_splits = []
            for i in range(self.hchannels):
                z_splits.append(z[:,spl_1*i:spl_1*(i+1)])
            z,sdh,sh = self.hidden(z_splits,sdh,sh)
            z_splits = []
            for i in range(self.ochannels):
                z_splits.append(z[:,spl_2*i:spl_2*(i+1)])
            v,sdo,so = self.out(z_splits,sdo,so)
            voltages[ts, :, :] = v
            # print(v)
        return voltages

# Dendritic MLP containing one hidden layer without expressive neurons
class DendSeqNet2(torch.nn.Module):

    def __init__(
        self,  h1=200, feature_size=28, lifparams=LIFParameters(method='super', alpha=100), hchannels=2, ochannels=4, dt = 0.001
    ):

        super(DendSeqNet2, self).__init__()

        self.hidden = DLIFWrapper2(in_channels=[feature_size**2//hchannels]*hchannels,out_channels=h1,lifparams=lifparams,dt=dt)
        self.out = DOutWrapper2(in_channels=[h1//ochannels]*ochannels,out_channels=10,lifparams=lifparams,dt=dt)
        self.hchannels = hchannels
        self.ochannels = ochannels
        self.hl = h1
        self.feature_size = feature_size
        # for mod in self.modules():
        #     if isinstance(mod, torch.nn.Linear):
        #         torch.nn.init.normal_(mod.weight, std=1.)

    def forward(self,x):

        # dendrite splits
        spl_1 = self.feature_size**2//self.hchannels
        spl_2 = self.hl//self.ochannels

        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # initial states
        sdh = [None]*self.hchannels
        sdo = [None]*self.ochannels

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = x[ts, :].view(-1, self.feature_size**2)
            z = z
            z_splits = []
            for i in range(self.hchannels):
                z_splits.append(z[:,spl_1*i:spl_1*(i+1)])
            z,sdh = self.hidden(z_splits,sdh)
            z_splits = []
            for i in range(self.ochannels):
                z_splits.append(z[:,spl_2*i:spl_2*(i+1)])
            v,sdo = self.out(z_splits,sdo)
            voltages[ts, :, :] = v
            # print(v)
        return voltages

# Dendritic MLP containing one hidden layer without expressive neurons
class DendSeqNet3(torch.nn.Module):

    def __init__(
        self,  h1=200, feature_size=28, lifparams=LIFParameters(method='super', alpha=100), hchannels=2, ochannels=4, dt = 0.001
    ):

        super(DendSeqNet3, self).__init__()

        self.hidden = DLIFWrapper3(in_channels=[feature_size**2//hchannels]*hchannels,out_channels=h1,lifparams=lifparams,dt=dt)
        self.out = DOutWrapper3(in_channels=[h1//ochannels]*ochannels,out_channels=10,lifparams=lifparams,dt=dt)
        self.hchannels = hchannels
        self.ochannels = ochannels
        self.hl = h1
        self.feature_size = feature_size
        # for mod in self.modules():
        #     if isinstance(mod, torch.nn.Linear):
        #         torch.nn.init.normal_(mod.weight, std=1.)

    def forward(self,x):

        # dendrite splits
        spl_1 = self.feature_size**2//self.hchannels
        spl_2 = self.hl//self.ochannels

        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # initial states
        sdh = [None]*self.hchannels
        sdo = [None]*self.ochannels

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = x[ts, :].view(-1, self.feature_size**2)
            z = z
            z_splits = []
            for i in range(self.hchannels):
                z_splits.append(z[:,spl_1*i:spl_1*(i+1)])
            z,sdh = self.hidden(z_splits,sdh)
            z_splits = []
            for i in range(self.ochannels):
                z_splits.append(z[:,spl_2*i:spl_2*(i+1)])
            v,sdo = self.out(z_splits,sdo)
            voltages[ts, :, :] = v
            # print(v)
        return voltages

# MLP containing one hidden layer
class SeqNetSVHN(torch.nn.Module):

    def __init__(
        self,  h1=200, feature_size=32, lifparams=LIFParameters(method='super', alpha=100), dt=0.001
    ):

        super(SeqNetSVHN, self).__init__()

        self.fc0 = torch.nn.Linear(feature_size*feature_size*3, h1)
        self.lif0 = LIFCell(p=lifparams,dt=dt)
        self.fc1 = torch.nn.Linear(h1, h1)
        self.lif1 = LIFCell(p=lifparams,dt=dt)
        self.out = LILinearCell(h1, 10, p=lifparams)

    def forward(self,x):

        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # initial states
        s0 = None
        s1 = None
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = x[ts, :].flatten(1)
            z = self.fc0(z)
            z, s0 = self.lif0(z, s0)
            # z = self.fc1(z)
            # z, s1 = self.lif1(z, s1)
            v, so = self.out(z, so)
            voltages[ts, :, :] = v
        return voltages
    

# Dendritic MLP containing one hidden layer without expressive neurons, for SVHN
class DendSeqNetSVHN(torch.nn.Module):

    def __init__(
        self,  h1=200, feature_size=32, lifparams=LIFParameters(method='super', alpha=100), hchannels=9, ochannels=4, dt=0.001
    ):

        super(DendSeqNetSVHN, self).__init__()

        self.L = 10
        self.M = 12
        self.R = 10
        self.hidden = DLIFWrapper2(in_channels=[feature_size*self.L]*3+[feature_size*self.M]*3+[feature_size*self.R]*3,out_channels=h1,lifparams=lifparams,dt=dt)
        self.out = DOutWrapper2(in_channels=[h1//ochannels]*ochannels,out_channels=10,lifparams=lifparams,dt=dt)
        self.hchannels = hchannels
        self.ochannels = ochannels
        self.hl = h1
        self.feature_size = feature_size
        # for mod in self.modules():
        #     if isinstance(mod, torch.nn.Linear):
        #         torch.nn.init.normal_(mod.weight, std=1.)

    def forward(self,x):

        # dendrite splits
        spl_1_1 = self.L
        spl_1_2 = self.L+self.M
        spl_2 = self.hl//self.ochannels

        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # initial states
        sdh = [None]*self.hchannels
        sdo = [None]*self.ochannels

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z_splits = []
            z_splits.append(x[ts,:,0,:,:spl_1_1].flatten(1))
            z_splits.append(x[ts,:,1,:,:spl_1_1].flatten(1))
            z_splits.append(x[ts,:,2,:,:spl_1_1].flatten(1))
            z_splits.append(x[ts,:,0,:,spl_1_1:spl_1_2].flatten(1))
            z_splits.append(x[ts,:,1,:,spl_1_1:spl_1_2].flatten(1))
            z_splits.append(x[ts,:,2,:,spl_1_1:spl_1_2].flatten(1))
            z_splits.append(x[ts,:,0,:,spl_1_2:].flatten(1))
            z_splits.append(x[ts,:,1,:,spl_1_2:].flatten(1))
            z_splits.append(x[ts,:,2,:,spl_1_2:].flatten(1))

            z,sdh = self.hidden(z_splits,sdh)
            z_splits = []
            for i in range(self.ochannels):
                z_splits.append(z[:,spl_2*i:spl_2*(i+1)])
            v,sdo = self.out(z_splits,sdo)
            voltages[ts, :, :] = v
            # print(v)
        return voltages

# Dendritic MLP containing one hidden layer without expressive neurons, for SVHN
class DendSeqNetSVHN2(torch.nn.Module):

    def __init__(
        self,  h1=200, feature_size=32, lifparams=LIFParameters(method='super', alpha=100), hchannels=9, ochannels=4, dt=0.001
    ):

        super(DendSeqNetSVHN2, self).__init__()

        self.L = 10
        self.M = 12
        self.R = 10
        self.hidden = DLIFWrapper(in_channels=[feature_size*self.L]*3+[feature_size*self.M]*3+[feature_size*self.R]*3,out_channels=h1,lifparams=lifparams,dt=dt)
        self.out = DOutWrapper(in_channels=[h1//ochannels]*ochannels,out_channels=10,lifparams=lifparams,dt=dt)
        self.hchannels = hchannels
        self.ochannels = ochannels
        self.hl = h1
        self.feature_size = feature_size
        # for mod in self.modules():
        #     if isinstance(mod, torch.nn.Linear):
        #         torch.nn.init.normal_(mod.weight, std=0.7)

    def forward(self,x):

        # dendrite splits
        spl_1_1 = self.L
        spl_1_2 = self.L+self.M
        spl_2 = self.hl//self.ochannels

        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # initial states
        sdh = [None]*self.hchannels
        sh = None
        sdo = [None]*self.ochannels
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z_splits = []
            z_splits.append(x[ts,:,0,:,:spl_1_1].flatten(1))
            z_splits.append(x[ts,:,1,:,:spl_1_1].flatten(1))
            z_splits.append(x[ts,:,2,:,:spl_1_1].flatten(1))
            z_splits.append(x[ts,:,0,:,spl_1_1:spl_1_2].flatten(1))
            z_splits.append(x[ts,:,1,:,spl_1_1:spl_1_2].flatten(1))
            z_splits.append(x[ts,:,2,:,spl_1_1:spl_1_2].flatten(1))
            z_splits.append(x[ts,:,0,:,spl_1_2:].flatten(1))
            z_splits.append(x[ts,:,1,:,spl_1_2:].flatten(1))
            z_splits.append(x[ts,:,2,:,spl_1_2:].flatten(1))

            z,sdh,sh = self.hidden(z_splits,sdh,sh)
            z_splits = []
            for i in range(self.ochannels):
                z_splits.append(z[:,spl_2*i:spl_2*(i+1)])
            v,sdo,so = self.out(z_splits,sdo,so)
            voltages[ts, :, :] = v
            # print(v)
        return voltages

# Dendritic MLP containing one hidden layer without expressive neurons, for SVHN
class DendSeqNetSVHN3(torch.nn.Module):

    def __init__(
        self,  h1=200, feature_size=32, lifparams=LIFParameters(method='super', alpha=100), hchannels=9, ochannels=4, dt=0.001
    ):

        super(DendSeqNetSVHN3, self).__init__()

        self.hidden = DLIFWrapper3(in_channels=[feature_size**2]*3,out_channels=h1,lifparams=lifparams,dt=dt)
        self.out = DOutWrapper3(in_channels=[h1//ochannels]*ochannels,out_channels=10,lifparams=lifparams,dt=dt)
        self.hchannels = hchannels
        self.ochannels = ochannels
        self.hl = h1
        self.feature_size = feature_size
        # for mod in self.modules():
        #     if isinstance(mod, torch.nn.Linear):
        #         torch.nn.init.normal_(mod.weight, std=1.)

    def forward(self,x):

        # dendrite splits
        spl_2 = self.hl//self.ochannels

        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # initial states
        sdh = [None]*self.hchannels
        sdo = [None]*self.ochannels

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z_splits = []
            z_splits.append(x[ts,:,0,:,:].flatten(1))
            z_splits.append(x[ts,:,1,:,:].flatten(1))
            z_splits.append(x[ts,:,2,:,:].flatten(1))

            z,sdh = self.hidden(z_splits,sdh)
            z_splits = []
            for i in range(self.ochannels):
                z_splits.append(z[:,spl_2*i:spl_2*(i+1)])
            v,sdo = self.out(z_splits,sdo)
            voltages[ts, :, :] = v
            # print(v)
        return voltages

# CNN with 2 convolutional filters and 1 hidden layer
class ConvNet(torch.nn.Module):

    def __init__(
        self, h1 = 200, num_channels=1, feature_size=28, lifparams=LIFParameters(method='super', alpha=100), dt=0.001
    ):

        super(ConvNet, self).__init__()

        self.features = int(((feature_size - 4)/2-4)/2)
        self.conv1 = torch.nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.fc1= torch.nn.Linear(self.features*self.features*50, h1)
        self.out = LILinearCell(h1, 10,dt=dt)
        self.lif0 = LIFCell(p=lifparams,dt=dt)
        self.lif1 = LIFCell(p=lifparams,dt=dt)
        self.lif2 = LIFCell(p=lifparams,dt=dt)

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = None
        s1 = None
        s2 = None
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif0(z, s0)
            z = self.maxpool(z)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = self.maxpool(z)
            z = z.view(-1, 4**2 * 50)
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages
    
# CNN with 2 convolutional filters and 1 hidden layer
class ConvNetSVHN(torch.nn.Module):

    def __init__(
        self, h1 = 200, num_channels=3, feature_size=32, lifparams=LIFParameters(method='super', alpha=100), dt=0.001
    ):

        super(ConvNetSVHN, self).__init__()

        self.features = int(((feature_size - 4)/2-4)/2)
        self.conv1 = torch.nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.fc1= torch.nn.Linear(self.features*self.features*50, h1)
        self.out = LILinearCell(h1,10,dt=dt)
        self.lif0 = LIFCell(p=lifparams,dt=dt)
        self.lif1 = LIFCell(p=lifparams,dt=dt)
        self.lif2 = LIFCell(p=lifparams,dt=dt)

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = None
        s1 = None
        s2 = None
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif0(z, s0)
            z = self.maxpool(z)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = self.maxpool(z)
            z = z.view(-1, self.features**2 * 50)
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages
    