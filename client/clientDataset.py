import os
import random

from torch.utils.data import DataLoader, Dataset, random_split

DATASET = "tiny-shakespeare.txt"


class TextDataset(Dataset):
    def __init__(self, text, reverseVocab, chunkLength):
        self.tokenizedDataset = [reverseVocab[c] for c in text]
        self.chunkLength = chunkLength

    def __getitem__(self, index):
        low = index * self.chunkLength
        high = (index + 1) * self.chunkLength
        x = self.tokenizedDataset[low:high]
        y = None
        try:
            y = self.tokenizedDataset[low + 1 : high + 1]
        except IndexError:
            y = x
            x = self.tokenizedDataset[low - 1 : high - 1]
        return x, y

    # Number of batches
    def __len__(self):
        return len(self.tokenizedDataset) // self.chunkLength


def get_abs_path(relativePath):
    scriptDirectory = os.path.dirname(__file__)
    return os.path.join(scriptDirectory, relativePath)


def get_user_data(userCount, userID, rounds, seed):
    random.seed(seed)

    text = ""
    with open(get_abs_path(f"data/{DATASET}"), "r") as f:
        text = f.read()

    splitParagraphs = text.split("\n\n")
    random.shuffle(splitParagraphs)
    numParagraphsPerUser = len(splitParagraphs) / userCount
    numParagraphsPerRound = numParagraphsPerUser / rounds

    start = int(userID * numParagraphsPerUser)

    round_split_data = []
    for i in range(rounds):
        low = int(start + (numParagraphsPerRound * i))
        high = int(start + (numParagraphsPerRound * (i + 1)))
        round_split_data.append("\n\n".join(splitParagraphs[low:high]))

    return round_split_data


def get_vocab():
    text = ""
    with open(get_abs_path(f"data/{DATASET}"), "r") as f:
        text = f.read()

    chars = tuple(set(text))
    vocab = dict(enumerate(chars))
    reverseVocab = {c: i for i, c in vocab.items()}

    return vocab, reverseVocab
