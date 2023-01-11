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

            # Reset such that only gradients that pertain
            # to the current input are used
            optimizer.zero_grad()

            # Forward
            output = model(input)

            # No need to shape target to one-hot encoding
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model

def test(model, testSet):
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for input, target in testSet:
            output = model(input)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    return correct / total
