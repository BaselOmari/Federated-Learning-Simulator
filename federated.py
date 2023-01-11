from copy import deepcopy
from threading import Thread, Lock

import torch

from dataset import mnist_dataset
from model.layers import CNN
from model.train import test, train

clientModels = []
clientModelsLock = Lock()


def clientTraining(serverModel, clientDatasets, client, round):
    global clientModels
    clientTrainingSet = clientDatasets[client][round]
    trainLoader = mnist_dataset.get_dataloader(clientTrainingSet)
    clientModel = deepcopy(serverModel)
    trainedClientModel = train(clientModel, trainLoader)

    clientModelsLock.acquire()
    clientModels.append(trainedClientModel)
    clientModelsLock.release()

    print(f"Client {client+1} done")


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
    clientNum = 3
    trainingRounds = 2


def federated():
    global clientModels
    config = federatedConfig()

    trainSet = mnist_dataset.load_dataset(isTrainDataset=True)
    clientDatasets = mnist_dataset.split_client_datasets(
        trainSet, config.clientNum, config.trainingRounds
    )

    testSet = mnist_dataset.load_dataset(isTrainDataset=False)
    testLoader = mnist_dataset.get_dataloader(testSet)

    serverModel = CNN()

    for round in range(config.trainingRounds):
        print(f"Round {round+1} started")

        clientModels.clear()
        clientThreads = []
        for client in range(config.clientNum):
            t = Thread(
                target=clientTraining, args=(serverModel, clientDatasets, client, round)
            )
            t.start()
            clientThreads.append(t)

        for t in clientThreads:
            t.join()

        serverModel = fedAvg(clientModels)

        testAcc = test(serverModel, testLoader)
        print(f"Round {round+1} done\tAccuracy = {testAcc}")


if __name__ == "__main__":
    federated()
