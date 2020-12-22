from os.path import abspath
import numpy as np


def init():
    with open(
        abspath("./training_data/messages.txt"), "r", encoding="utf-8"
    ) as messages_file:
        text = messages_file.read()
        vocab = sorted(set(text))
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        int_repr = np.array([char2idx[c] for c in text])
    return (int_repr, len(vocab), char2idx, idx2char)


CHECKPOINT_DIR = abspath("./training_checkpoints")
int_repr, vocab_size, char2idx, idx2char = init()
