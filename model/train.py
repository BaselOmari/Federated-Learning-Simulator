import torch
import torch.nn as nn

from torch import optim

from torch.utils.data import random_split

from tqdm import tqdm


def train(model, dataset):

    ##### TRAINING HYPERPARAMETERS #####
    epochs = 3
    learningRate = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    criterion = nn.NLLLoss()
    ####################################

    for epoch in range(epochs):

        epochLoss = 0
        for input, target in tqdm(dataset):

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

        print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model

def test(model, testSet):
    criterion = nn.NLLLoss()
    loss = 0
    for input, target in tqdm(testSet):
        output = model(input)
        loss += criterion(output, torch.tensor([target])).item()
    return loss / len(testSet)
