import torch.utils.data as td
import torchvision
from noise_transform import AddGaussianNoise
import os

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

    return train_loader,test_loader