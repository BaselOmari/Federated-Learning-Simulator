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

    freq = collections.Counter(text).most_common()

    vocab = dict()
    reverseVocab = dict()
    for c, i in enumerate(freq):
        vocab[c] = i[0]
        reverseVocab[i[0]] = c

    return vocab, reverseVocab

    return vocab, reverse_vocab


text = ""
with open("data/tiny-shakespeare.txt", "r") as f:
    text = f.read()

vocab, reverse_vocab = vocab(text)

USER_COUNT = 25
USER_I = 0

rawDataset = get_user_data(text, USER_COUNT, USER_I)
