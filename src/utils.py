"""
This module provides functions for corpus data management.

Functions
----------
batch_stochastic_replacement
    Reinitializes and modifies copy_tensors in-place.
save_dict
    Saves dictionary in a file.
load_dict
    Loads dictionary from a file
sentence_to_tensors
    Converts a sentence (list of tokens) into a tensor.
"""

import torch

def batch_stochastic_replacement(device, tensors, copy_tensors, words2i, pword=0.3):
    """
    Reinitializes and modify in place copy_tensors
    Args:
        device, chars2i, words2i: device object, mappers character/index to index
        tensors: original long tensors for tokens and chars of train corpus
        copy_tensors: slightly differ from original and used as input for training
        pchar: replaces each character by "<UNK>" with probability pchar
        pword: replaces each low frequency token by "<UNK>" with probability pword
                low frequency = 2/3 least frequent tokens
    """
    # indexes for unknown char and unknown word
    #cunk = chars2i["<UNK>"]
    wunk = words2i["<UNK>"]
    # only replace the 2/3 less frequent words
    repi = len(words2i) // 3
    
    #longest = max([len(w) for w in words2i]) + 3
    #cmaskr = torch.rand(longest, device=device)
    # uint8: boolean selection of indices (torch.long -> index selection)
    #cmaski = torch.tensor(list(range(longest)), dtype=torch.uint8, device=device) 

    # 150: hard constraint on length of sentence
    wmaskr = torch.rand(150, device=device)
    wmaski = torch.tensor(list(range(150)), dtype=torch.uint8, device=device)

    for s1, s2 in zip(tensors, copy_tensors):
#        for c1, c2 in zip(s1[0], s2[0]):
#            c2.copy_(c1)
#            cmaskr.uniform_(0, 1)
#            cmaski.copy_(cmaskr > (1-pchar))
#            cmaski[0] = 0 # do not replace <START> and <STOP> symbols
#            cmaski[-1] = 0
#            c2[cmaski[:len(c2)]] = cunk

        w1, w2 = s1[1], s2[1]
        w2.copy_(w1)
        wmaskr.uniform_(0, 1)
        wmaski.copy_(wmaskr > (1-pword))

        wmaski[:len(w2)] *= w2 > repi
        w2[wmaski[:len(w2)]] = wunk

def save_dict(d, filename):
    # Saves dictionary to filename
    with open(filename, "w") as f:
        for k in d:
            f.write("{}\n".format(k))

def load_dict(filename):
    # Loads dictionary from filename
    i2l = []
    with open(filename, "r") as f:
        for k in f:
            i2l.append(k.strip())
    return i2l, {k: i for i, k in enumerate(i2l)}

def sentence_to_tensors(sentence, words2i, device):
#    tensors = []
#    start, stop = [char2i["<START>"]], [char2i["<STOP>"]]
#    chars = []
#    for tok in sentence:
#        if tok in {"-LRB-", "-RRB-", "#RRB#", "#LRB#"}:
#            chars.append([start[0], char2i[tok], stop[0]])
#        else:
#            chars.append(start + [char2i[c] if c in char2i else char2i["<UNK>"] for c in tok] + stop)
    words = [words2i[w] if w in words2i else words2i["<UNK>"] for w in sentence]
    return (list(sentence), torch.tensor(words, dtype=torch.long, device=device))
    #return ([torch.tensor(cs, dtype=torch.long, device=device) for cs in chars],
    #         torch.tensor(words, dtype=torch.long, device=device))