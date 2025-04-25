import torch
from norse.torch import LIFCell,LILinearCell,LIFParameters
from module_neur import DLIFWrapper,DOutWrapper,DLIFWrapper2,DOutWrapper2,DLIFWrapper3,DOutWrapper3
import torch.nn.functional as F

def getANN(net_name):
    if net_name == 'PercepNet':
        spk_net = PercepNet()
    else:
        print(f'{net_name} is not a valid network')
    return spk_net

# MLP containing one hidden layer
class PercepNet(torch.nn.Module):

    def __init__(
        self,  feature_size=4, out_size=3
    ):

        super(PercepNet, self).__init__()
        self.fc0 = torch.nn.Linear(feature_size,out_size)

    def forward(self,x):
        x = self.fc0(x)
        return F.log_softmax(x)
