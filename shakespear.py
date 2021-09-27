"""
Assignment fun from http://www.cs.toronto.edu/~guerzhoy/tmp/understand-rnn/handout/index.html
Trained weights are used to generate Shakespeare-like text with edits to min-char-rnn script by Andrej Karpathy
"""

import numpy as np

# data = open('shakespeare_train.txt', 'r').read()
data = 'thou shall not see thy enemy nor thee'
# chars = list(set(data))
# data_size, vocab_size = len(data), len(chars)
# char2ix = {ch:idx for idx, ch in enumerate(chars)}
# idx2char = {idx:ch for idx, ch in enumerate(chars)}

# RNN hyperparams
hidden_size = 250   # hidden neurons
seq_length = 25     # Sequence of text unrolled into RNN
learning_rate = 1e-1

# Model parameters : Loaded from provided np file
a = np.load('char-rnn-snapshot.npz', allow_pickle=True)  # load from provided weights
Wxh = a["Wxh"]  # input to hidden
Whh = a["Whh"]  # hidden to hidden
Why = a["Why"]  # hidden to output
bh = a["bh"]    # hidden bias
by = a["by"]    # output bias
mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
mbh, mby = a["mbh"], a["mby"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), \
                                                       a["vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()

inputs = char_to_ix[data[0]]
hprev = np.zeros((hidden_size,1))   # RNN memory is null
temperature = 0.7
alpha = 1/temperature

def charpred(h, inputs):
    """
    :param h: hidden state, initially its hprev
    :param input: integer value of input character
    :return: integer value of predicted character
    """
    x = np.zeros((vocab_size,1))
    x[inputs] = 1    # Input char is one hot encoded
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(alpha*y)/np.sum(np.exp(alpha*y))
    idx = np.random.choice(range(vocab_size), p=p.ravel())
    return h, idx


def updatehidden(string, hprev):
    """
    :param string: input string
    :return: hidden state at the end of the string fed into RNN
    """
    length = len(string)
    count = 0
    h_state = hprev
    while count < length:
        charin = char_to_ix[string[count]]
        h_state, _ = charpred(h_state, charin)
        count += 1

    return h_state


seq_length = 1500
i = 0
text = []
h = updatehidden(data, hprev)

while i <= seq_length:
    h, inputs = charpred(h, inputs)
    text.append(ix_to_char[inputs])
    i += 1

txtout = ''.join(text)
print("Here thy new Shakespear\n ", txtout, "\n----------")