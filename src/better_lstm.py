import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from typing import *
from torch.autograd import Variable
import torch.nn.functional as F


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: bool = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor | PackedSequence) -> torch.Tensor | PackedSequence:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            assert(isinstance(x, PackedSequence))
            x, seq_lens = pad_packed_sequence(x, batch_first=self.batch_first)
            max_batch_size = x.size(0 if self.batch_first else 1)
        else:
            assert(isinstance(x, torch.Tensor))
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return pack_padded_sequence(x, seq_lens, batch_first=self.batch_first)
        else:
            return x

class LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)

        self.dropoutw : float | None
        self.input_drop : VariationalDropout | None
        self.output_drop : VariationalDropout | None

        self.unit_forget_bias = unit_forget_bias

        self.dropoutw = dropoutw if dropoutw > 0 else None
        if dropouti > 0:
            self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        else:
            self.input_drop = None
        
        if dropouto > 0:
            self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        else:
            self.output_drop = None

        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    # fix resetting weights
    def _drop_weights(self):
        if self.training:
            left_out = {}
            for name, param in self.named_parameters():
                if "weight_hh" in name:
                    w_mask = Variable(torch.ones(param.data.shape, device=param.data.device), requires_grad=False)
                    w_mask = F.dropout(w_mask, p=self.dropoutw, training=self.training)
    
                    left_out[name] = param.data * (1-w_mask)
                    getattr(self, name).data = param.data * w_mask
            return left_out

    def _reset_w(self, left_out):
        if self.training:
            for name, param in self.named_parameters():
                if "weight_hh" in name:
                    getattr(self, name).data = param.data + left_out[name]

    def forward(self, input, hx=None):
        self.flatten_parameters()

        if self.dropoutw is not None:
            left_out = self._drop_weights()
        
        if self.input_drop is not None:
            input = self.input_drop(input)

        seq, state = super().forward(input, hx=hx)

        if self.dropoutw is not None:
            self._reset_w(left_out)

        if self.output_drop is not None:
            seq = self.output_drop(seq)
        return seq, state