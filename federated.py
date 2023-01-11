import torch

from copy import deepcopy

from model.layers import CNN
from model.train import train

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
    pass



if __name__ == "__main__":
    federated()
