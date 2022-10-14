import collections


def get_user_data(text, user_count, user_idx):
    split_paragraphs = text.split("\n\n")
    num_paragraphs_per_user = len(split_paragraphs) / user_count

    low = int(user_idx * num_paragraphs_per_user)
    high = int((user_idx + 1) * num_paragraphs_per_user)

    return "\n\n".join(split_paragraphs[low:high])


def vocab(text):
    freq = collections.Counter(text).most_common()

    vocab = dict()
    reverse_vocab = dict()
    for c, i in enumerate(freq):
        vocab[c] = i[0]
        reverse_vocab[i[0]] = c

    return vocab, reverse_vocab


text = ""
with open("data/tiny-shakespeare.txt", "r") as f:
    text = f.read()

vocab, reverse_vocab = vocab(text)

USER_COUNT = 25
USER_I = 0

rawDataset = get_user_data(text, USER_COUNT, USER_I)
