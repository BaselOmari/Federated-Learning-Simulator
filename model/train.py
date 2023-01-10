import torch.nn as nn
from torch import optim

from torch.utils.data import random_split


def train(model, dataset):

    ##### TRAINING HYPERPARAMETERS #####
    epochs = 100
    learningRate = 0.001
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    criterion = nn.NLLLoss()
    ####################################

    for epoch in range(epochs):

        epochLoss = 0
        for input, target in dataset:

            # Reset such that only data that pertains
            # to the current input is used
            optimizer.zero_grad()

            # Forward
            output = model(input)

            # No need to shape target to one-hot encoding
            loss = criterion(output, torch.tensor([target]))
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        if epoch % 2 == 0:
            print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model

def test(model, test_set):
    criterion = nn.MSELoss()
    loss = 0
    for input, target in test_set:
        output = model(input)
        loss += criterion(output, target).item()
    return loss / len(test_set)

def train_test_split(dset, trainPercentage):
    assert trainPercentage > 0 and trainPercentage < 1
    
    trainCount = int(len(dset)*trainPercentage)
    testCount = len(dset) - trainCount
    train, test = random_split(
        dset, [trainCount, testCount])
    return train, test
