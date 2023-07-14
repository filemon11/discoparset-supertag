import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dropout import Dropout

class Words2Tensors():
    """Stores each token as a long tensor"""
    def __init__(self, device, char2i, word2i, pchar=None):
        self.w2tensor = {}
        self.cunk = char2i["<UNK>"]
        self.char2i = char2i
        self.pchar = pchar

        longest = max([len(w) for w in word2i]) + 3
        self.cmaskr = torch.rand(longest, device=device)
        # uint8: boolean selection of indices (torch.long -> index selection)
        self.cmaski = torch.tensor(list(range(longest)), dtype=torch.uint8, device=device)
        self.initialize(word2i)

    def initialize(self, word2i):
        for tok in word2i:
            self.add(tok)

    def add(self, tok):
        char2i = self.char2i
        start, stop = [char2i["<START>"]], [char2i["<STOP>"]]
        if tok in {"-LRB-", "-RRB-", "#RRB#", "#LRB#"}:
            chars_idx = [start[0], char2i[tok], stop[0]]
        else:
            chars_idx = start + [char2i[c] if c in char2i else char2i["<UNK>"] for c in tok] + stop

        self.w2tensor[tok] = (
            torch.tensor(chars_idx, dtype=torch.long, device=self.cmaskr.device),
            torch.tensor(chars_idx, dtype=torch.long, device=self.cmaskr.device)
        )

    def get(self, words, training=False):
        for w in words :
            if w not in self.w2tensor:
                self.add(w)
        if not training or self.pchar is None:
            return [self.w2tensor[w][0] for w in words]
        tensors = [self.w2tensor[w] for w in words]
        for c1, c2 in tensors:
            c2.copy_(c1)
            self.cmaskr.uniform_(0, 1)
            self.cmaski.copy_(self.cmaskr > (1-self.pchar))
            # do not replace <START> and <STOP> symbols
            self.cmaski[0] = 0
            self.cmaski[-1] = 0
            c2[self.cmaski[:len(c2)]] = self.cunk
        return [t for _, t in tensors]

    def to(self, device):
        if device != self.cmaskr.device:
            self.cmaskr = self.cmaskr.to(device)
            self.cmaski = self.cmaski.to(device)
            self.w2tensor = {k : (v[0].to(device), v[1].to(device)) for k, v in self.w2tensor.items()}

class CharacterLstmLayer(nn.Module):
    
    def __init__(self, emb_dim, voc_size, out_dim, words2tensors=None, dropout=0.2, embed_init=0.1):
        """
        Args:
            emb_dim: dimension of input embeddings
            voc_size: size of vocabulary (0 = padding)
            out_dim: dimension of bi-lstm output (each direction is out_dim // 2)

        """
        super(CharacterLstmLayer, self).__init__()

        self.words2tensors = words2tensors
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.voc_size = voc_size

        self.embeddings = nn.Embedding(voc_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, out_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        if dropout > 0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
        self.initialize_parameters(embed_init)

    def initialize_parameters(self, embed_init):
        self.embeddings.weight.data.uniform_(-embed_init, embed_init)
        self.embeddings.weight.data[0].fill_(0)
        for p in self.lstm.parameters():
            m = (6 / sum(p.data.shape))**0.5
            p.data.uniform_(-m, m)

    def forward(self, input):
        """
        Args:
            input: list of torch.long tensors OR
                   list of tokens (str) if self.words2tensors is not None

        Returns:
            res: tensor of size (batch, out_dim)
        """
        
        if self.words2tensors is not None:
            self.words2tensors.to(next(self.parameters()).device)
            input = self.words2tensors.get(input, training=self.training)

        # Pytorch rnn batches need to be sorted by decreasing lengths:
        order, sorted_by_length = zip(* sorted(enumerate(input), key = lambda x: len(x[1]), reverse=True))

        lengths = [len(i) for i in sorted_by_length]
    
        padded_char_seqs =  torch.nn.utils.rnn.pad_sequence(sorted_by_length, batch_first=True)

        padded_char_seqs_embeddings = self.embeddings(padded_char_seqs)
        if self.dropout is not None:
            padded_char_seqs_embeddings = self.dropout(padded_char_seqs_embeddings)
        
        packed_padded_char_seqs = torch.nn.utils.rnn.pack_padded_sequence(
                                    padded_char_seqs_embeddings, lengths, batch_first=True)

        _, (hn_xdir_bat_xdim, _) = self.lstm(packed_padded_char_seqs)

        # hn is (num dir, batch, outdim)
        lstm_output = torch.cat([hn_xdir_bat_xdim[0], hn_xdir_bat_xdim[1]], 1)

        # reorder idxes
        rev, _ = zip(*sorted(enumerate(order), key = lambda x : x[1]))
        
        res = torch.embedding(lstm_output, torch.tensor(rev, dtype=torch.long).to(lstm_output.device))
        return res



class CharacterConvolutionLayer(nn.Module):

    def __init__(self, emb_dim, voc_size, out_dim, kernel_sizes=[3, 3], activations="relu", pooling=None, feed_forward=None):
        """
        Customizable character-based encoder layer

        Params:
            emb_dim: dimension of character embeddings
            voc_size: size of vocabulary
            out_dim: dimension of encoder (size of output)
            kernel_sizes: list of kernel size for each layer, len(kernel_sizes) determines the depth of the encoder
            pooling: if not None, add pooling layer between each convolution with specified kernel_size
            feed_forward: add <arg>:int feed forward layers
        """
        super(CharacterConvolutionLayer, self).__init__()

        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.voc_size = voc_size

        self.embeddings = nn.Embedding(voc_size, emb_dim, padding_idx=0)

        depth = len(kernel_sizes)

        input_sizes = [emb_dim] + [out_dim for i in range(depth-1)]

        act_map = {"relu": nn.ReLU, "tanh": nn.Tanh}
        act = act_map[activations]

    
        convolutions = [(nn.Conv1d(in_dim, out_dim, kernel_size=k, padding=1), act()) for in_dim, k in zip(input_sizes, kernel_sizes)]
        if pooling is not None:
            convolutions = [ (a, b, nn.MaxPool1d(pooling)) for a, b in convolutions ]
        convolutions = [l for layer in convolutions for l in layer]

        convolutions.append(nn.AdaptiveMaxPool1d(1))

        if feed_forward is not None:
            if type(feed_forward) != int:
                feed_forward = 1
            for i in range(feed_forward):
                convolutions.append(nn.Linear(out_dim, out_dim))
                convolutions.append(act())

        self.convs = nn.Sequential(*convolutions)

    def forward(self, input, replace_char=None):
        """
        Args:
            input: list of torch.long tensors

        Returns:
            res: tensor of size (batch, out_dim)
        """

        padded_input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)

        embeds = self.embeddings(padded_input).transpose(1, 2)

        return self.convs(embeds).squeeze(2)


