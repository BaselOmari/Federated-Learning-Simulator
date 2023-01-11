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


def get_dataloader(dataset, batchSize=64):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batchSize, shuffle=True, drop_last=True
    )
    return dataloader


def create_data_subset(count, index, fullDataset=load_dataset) -> Subset:
    countPerSet = len(fullDataset) // count
    low = countPerSet * index
    high = low + countPerSet
    subsetIndices = [i for i in range(low, high)]
    return Subset(fullDataset, subsetIndices)


def get_round_dataloader(roundNumber, roundCount, batchSize, userDataset) -> DataLoader:
    subset = create_data_subset(roundCount, roundNumber, userDataset)
    dataloader = DataLoader(
        subset,
        batch_size=batchSize,
        shuffle=True,
    )
    return dataloader
