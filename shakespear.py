"""
Assignment fun from http://www.cs.toronto.edu/~guerzhoy/tmp/understand-rnn/handout/index.html
Trained weights are used to generate Shakespeare-like text with edits to min-char-rnn script by Andrej Karpathy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as figure

# data = open('shakespeare_train.txt', 'r').read()
# data = 'thou shall not see thy enemy nor thee'
data = "First Citizen:\nLet us kill him, and we'll have corn at our own price.\nIs't a verdict?"
# chars = list(set(data))
# data_size, vocab_size = len(data), len(chars)
# char2ix = {ch:idx for idx, ch in enumerate(chars)}
# idx2char = {idx:ch for idx, ch in enumerate(chars)}

# RNN hyperparams
hidden_size = 250   # hidden neurons
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
temperature = 0.8
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

def neuronvisual(text, hiddenstates, neuronid):
    """
    :param text: list of input texts to the RNN model
    :param hiddenstates: Corresponding hidden states for the character
    :param neuronid: chosen neuron for visualization
    :return: plot of chosen neuron firing for each character
    """
    hiddenarr = np.array(hiddenstates)
    hiddenarr = hiddenarr[:,neuronid]
    fig, ax = plt.subplots(1,1,figsize=(20,2))
    ax.matshow(hiddenarr.T, cmap = plt.get_cmap("RdBu"), aspect="auto")

    for i in range(hiddenarr.shape[0]):
        ax.text(i, 0, text[i], va='center', ha='center')
    # ax.set_yticks(np.arange(len(text)))
    # ax.set_yticklabels(text)
    # plt.yticks(fontsize = 7, rotation = -30)
    plt.plot()


NEWTEXT = False     # If you want to generate a new text, or load a text file
seq_length = 100
i = 0
text = []
hiddenlist = []
h = updatehidden(data, hprev)
inputs = char_to_ix[data[-1]]   # Setting input as last char of input string

if NEWTEXT:
    while i <= seq_length:
        # Perform RNN prediction and append to list containing text
        h, inputs = charpred(h, inputs)
        text.append(ix_to_char[inputs])
        hiddenlist.append(h)
        i += 1

    txtout = ''.join(text)
    print("Here thy new Shakespear\n ", txtout, "\n----------")
else:
    hiddennp = np.load('hiddenlist.npy')
    hiddenlist = hiddennp.tolist()

    with open('file.txt') as f:
        contents = f.read()
        text.append(contents)

    text = [ch for ch in text[0]]


neuronvisual(text, hiddenlist, 159)
plt.show()

# saving hidden neuron and text lists
np.save("hiddenlist", np.array(hiddenlist))
with open('file.txt', 'w') as f:
    for char in text:
        f.write("%s" %char)

