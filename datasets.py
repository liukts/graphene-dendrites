import torch.utils.data as td
import torchvision
from noise_transform import AddGaussianNoise
import os
import sklearn.datasets
from sklearn.model_selection import train_test_split
import torch

transformMNIST = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

transformFMNIST = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.2859,), (0.3530,)),
    ]
)

transformSVHN = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

def get_dataloader(name='fmnist',batch_size=100,noise_std=0):

    if not os.path.isdir('./dataset_repo'):
        os.mkdir('./dataset_repo')

    if name == 'mnist':

        train_data = torchvision.datasets.MNIST(
            root='./dataset_repo',
            train=True,
            download=True,
            transform=transformMNIST
            )
        train_loader = td.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
            )
        test_loader = td.DataLoader(
            torchvision.datasets.MNIST(
                root='./dataset_repo',
                train=False,
                transform=transformMNIST
            ),
            batch_size=batch_size
        )

    elif name == 'fmnist':

        train_data = torchvision.datasets.FashionMNIST(
            root='./dataset_repo',
            train=True,
            download=True,
            transform=transformFMNIST
            )
        train_loader = td.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
            )
        test_loader = td.DataLoader(
            torchvision.datasets.FashionMNIST(
                root='./dataset_repo',
                train=False,
                transform=transformFMNIST
            ),
            batch_size=batch_size
        )

    elif name == 'fmnistinf':

        transformFMNISTinf = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.2859,), (0.3530,)),
                AddGaussianNoise(mean=0,std=noise_std),
            ]
        )

        train_data = torchvision.datasets.FashionMNIST(
            root='./dataset_repo',
            train=True,
            download=True,
            transform=transformFMNIST
            )
        train_loader = td.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
            )
        test_loader = td.DataLoader(
            torchvision.datasets.FashionMNIST(
                root='./dataset_repo',
                train=False,
                transform=transformFMNIST
            ),
            batch_size=batch_size
        )

    elif name == 'svhn':

        train_data = torchvision.datasets.SVHN(
            root='./dataset_repo',
            split='train',
            download=True,
            transform=transformSVHN
            )
        test_data = torchvision.datasets.SVHN(
            root='./dataset_repo',
            split='test',
            download=True,
            transform=transformSVHN
            )
        train_loader = td.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
            )
        test_loader = td.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False
        )
    elif name == 'iris':
        iris = sklearn.datasets.load_iris()
        X = iris.data
        y = iris.target
        x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.2, train_size=0.8)
        x_train_to_tensor = torch.from_numpy(x_train).to(torch.float32)
        y_train_to_tensor = torch.from_numpy(y_train).to(torch.long) 
        x_test_to_tensor = torch.from_numpy(x_test).to(torch.float32)
        y_test_to_tensor = torch.from_numpy(y_test).to(torch.long)
        train_set = td.TensorDataset(x_train_to_tensor,y_train_to_tensor)
        test_set = td.TensorDataset(x_test_to_tensor,y_test_to_tensor)
        train_loader = td.DataLoader(train_set,batch_size=batch_size)
        test_loader = td.DataLoader(test_set,batch_size=batch_size)
    return train_loader,test_loader