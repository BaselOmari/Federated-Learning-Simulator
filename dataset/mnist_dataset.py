import os

import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


def load_dataset(isTrainDataset=True) -> Dataset:
    mnistDataset = datasets.MNIST(
        os.path.dirname(os.path.realpath(__file__)) + "/data",
        train=isTrainDataset,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    return mnistDataset


def split_client_datasets(dataset, clientNum, roundNum):
    countPerSet = len(dataset) // (clientNum * roundNum)
    clientDatasets = [[] for _ in range(clientNum)]
    for client in range(clientNum):
        for round in range(roundNum):
            low = countPerSet * (round + client * roundNum)
            high = low + countPerSet
            subsetIndices = [i for i in range(low, high)]
            clientDatasets[client].append(Subset(dataset, subsetIndices))
    return clientDatasets


def get_dataloader(dataset, batchSize=64):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batchSize, shuffle=True, drop_last=True
    )
    return dataloader
