import torch
from norse.torch import LIFCell,LILinearCell,LIFParameters

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
        self,  h1=200, feature_size=28, lifparams=LIFParameters(method='super', alpha=100), dendrites=False
    ):

        super(SeqNet, self).__init__()

        self.fc0 = torch.nn.Linear(feature_size*feature_size, h1)
        self.lif = LIFCell(p=lifparams)
        self.out = LILinearCell(h1, 10, p=lifparams)

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
            z = self.fc0(x[ts, :].view(-1, 28*28))
            z, s0 = self.lif(z, s0)
            v, so = self.out(z, so)
            voltages[ts, :, :] = v
        return voltages

# CNN with 2 convolutional filters and 1 hidden layer
class ConvNet(torch.nn.Module):

    def __init__(
        self, h1 = 200, num_channels=1, feature_size=28, lifparams=LIFParameters(method='super', alpha=100), dendrites=False
    ):

        super(ConvNet, self).__init__()

        self.features = int(((feature_size - 4)/2-4)/2)
        self.conv1 = torch.nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.fc1= torch.nn.Linear(self.features*self.features*50, h1)
        self.out = LILinearCell(h1, 10)
        self.lif0 = LIFCell(p=lifparams)
        self.lif1 = LIFCell(p=lifparams)
        self.lif2 = LIFCell(p=lifparams)

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