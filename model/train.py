from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.utils.data import random_split


def train(model, dataset, reverseVocab, vocab):

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
            print(generate(model, "And she shall than", 100, reverseVocab, vocab))

    return model


def sample(output):
    output = F.softmax(torch.squeeze(output), dim=0)
    dist = Categorical(output)
    index = dist.sample()
    return index.item()


def generate(model, startStr, length, reverseVocab, vocab):
    encodedStr = torch.tensor([reverseVocab[c] for c in startStr])
    finalStr = startStr

    # take in starting string
    _, hidden = model(encodedStr)

    # generate next characters
    prevChar = encodedStr[-1]
    for i in range(length):
        prevChar = prevChar.reshape([1])
        output, hidden = model(prevChar, hidden)
        generatedCharNum = sample(output)
        generatedChar = vocab[generatedCharNum]
        finalStr += generatedChar

        prevChar = torch.tensor(generatedCharNum)

    return finalStr


import sys

sys.path.append(r"/Users/alomarb/non-adsk/FL Research/FL-Sim")


from layers import CharRNN

from client import clientDataset

rounds = 4

roundSplitDataset = clientDataset.get_user_data(5, 0, rounds, 20202)
vocab, reverseVocab = clientDataset.get_vocab()

for i in range(rounds):
    currRoundText = roundSplitDataset[i]
    currRoundDataset = clientDataset.TextDataset(currRoundText, reverseVocab, 100)

    trainSet, testSet = random_split(
        currRoundDataset,
        [ceil(len(currRoundDataset) * 0.8), floor(len(currRoundDataset) * 0.2)],
    )

    freshModel = CharRNN(
        len(reverseVocab),
        16,
        len(reverseVocab),
        nLayers=2,
    )
    trainedModel = train(freshModel, trainSet, reverseVocab, vocab)
