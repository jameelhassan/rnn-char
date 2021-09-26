"""
Assignment fun from http://www.cs.toronto.edu/~guerzhoy/tmp/understand-rnn/handout/index.html
Trained weights are used to generate Shakespeare-like text with edits to min-char-rnn script by Andrej Karpathy
"""

import numpy as np

# data = open('shakespeare_train.txt', 'r').read()
data = 'Thou shall not see thy enemy nor thee'
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

inputs = char_to_ix[data[-1]]
hprev = np.zeros((hidden_size,1))   # RNN memory is null
temperature = 1
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


seq_length = 500
i = 0
text = []

while i <= seq_length:
    hprev, inputs = charpred(hprev, inputs)
    text.append(ix_to_char[inputs])
    i += 1

txtout = ''.join(text)
print("Here thy new Shakespear\n ", txtout, "\n----------")

