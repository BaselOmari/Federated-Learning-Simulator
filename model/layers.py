import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, nLayers=2):
        super(CharRNN, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.nLayers = nLayers

        self.encoder = nn.Embedding(inputSize, hiddenSize)
        self.rnn = nn.LSTM(hiddenSize, hiddenSize, nLayers)
        self.decoder = nn.Linear(hiddenSize, outputSize)

    def forward(self, input, hidden=None):
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded, hidden)
        output = self.decoder(output)
        return output, hidden
