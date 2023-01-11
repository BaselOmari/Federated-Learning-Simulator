import torch

from copy import deepcopy

from dataset import mnist_dataset

from model.layers import CNN
from model.train import train, test

def fedAvg(clientModels):
    averagedModel = deepcopy(clientModels[0])
    with torch.no_grad():
        for model in clientModels[1:]:
            for param1, param2 in zip(averagedModel.parameters(), model.parameters()):
                param1.data += param2.data
        for param in averagedModel.parameters():
            param.data /= len(clientModels)
    return averagedModel


class federatedConfig:
    clientNum = 4
    trainingRounds = 2


def federated():
    config = federatedConfig()

    trainSet = mnist_dataset.load_dataset(isTrainDataset=True)
    trainLoader = mnist_dataset.get_dataloader(trainSet)

    testSet = mnist_dataset.load_dataset(isTrainDataset=False)
    testLoader = mnist_dataset.get_dataloader(testSet)

    serverModel = CNN()

    for round in range(config.trainingRounds):
        print(f"Round {round} started")
        clientModels = []
        for client in range(config.clientNum):
            clientModel = deepcopy(serverModel)
            train(clientModel, trainLoader)
            clientModels.append(clientModel)
            print(f"Client {client} done")
        serverModel = fedAvg(clientModels)
        testAcc = test(serverModel, testLoader)
        print(f"Round {round} done\tAccuracy = {testAcc}\n\n")


if __name__ == "__main__":
    federated()
