import collections
import random
import torch


def get_user_data(userCount, userID, rounds, seed):
    random.seed(seed)

    text = ""
    with open("data/tiny-shakespeare.txt", "r") as f:
        text = f.read()

    splitParagraphs = text.split("\n\n")
    print(len(splitParagraphs))
    random.shuffle(splitParagraphs)
    numParagraphsPerUser = len(splitParagraphs) / userCount
    numParagraphsPerRound = numParagraphsPerUser / rounds

    start = int(userID * numParagraphsPerUser)

    round_split_data = []
    for i in range(rounds):
        low = int(start + (numParagraphsPerRound * i))
        high = int(start + (numParagraphsPerRound * (i + 1)))
        print(low, high)
        round_split_data.append("\n\n".join(splitParagraphs[low:high]))

    return round_split_data


def get_vocab():
    text = ""
    with open("data/tiny-shakespeare.txt", "r") as f:
        text = f.read()

    chars = tuple(set(text))
    vocab = dict(enumerate(chars))
    reverseVocab = {c: i for i, c in vocab.items()}

    return vocab, reverseVocab
