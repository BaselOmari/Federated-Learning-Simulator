import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from math import ceil, floor


def train(model, dataset):

    ##### TRAINING HYPERPARAMETERS #####
    epochs = 24
    learningRate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()
    ####################################

    for epoch in range(epochs):

        epochLoss = 0
        for input, target in dataset:

            # Reset such that only data that pertains
            # to the current input is used
            optimizer.zero_grad()

            # Forward
            output, _ = model(input)

            # No need to shape target to one-hot encoding
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        if epoch % 2 == 0:
            print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model
