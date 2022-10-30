import torch.nn as nn
from torch import optim


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
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        if epoch % 2 == 0:
            print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model
