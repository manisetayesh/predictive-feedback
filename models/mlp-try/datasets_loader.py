from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import ToTensor, Compose

def load_mnist():
    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = ToTensor(),
        download = True)

    test_data = datasets.MNIST(
        root = 'data',
        train = False,
        transform = ToTensor(),
        download = True)
    return train_data, test_data


def load_extended_mnist():
    transform = Compose([  
        ToTensor(),
        lambda x: x.transpose(1, 2).flip(1)  
    ])
    train_dataset = torchvision.datasets.EMNIST(
        root='./data',
        split='byclass',
        train=True,
        download=True,
        transform=transform
    ) #Change this once data is downloaded in the data folder.

    test_dataset = torchvision.datasets.EMNIST(
        root='./data',
        split='byclass', #currenlty contains upper case letters, lower case letters and digits
        train=False,
        download=True,
        transform=transform
    )
    return train_dataset, test_dataset

def load_omniglot():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        lambda x: 1 - x  
    ])

    omniglot_train = datasets.Omniglot(root="./data", background=True, download=True, transform=transform)
    return omniglot_train