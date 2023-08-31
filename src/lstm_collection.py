"""
Collection of LSTM implementations.

Classes
----------
Residual
    Gated residual connection for LSTM.
LSTMModel
    LSTM supporting variational dropout
    and residual connection.
BiLSTM
    BiLSTM based on lstm_collection.LSTMModel.
SampleDrop
    Module for applying the same dropout mask
    on sequential input.
IntegratedBiLSTMModel
    BiLSTM with the same dropout variational dropout
    mask on both forward and backward direction.
FastVariationalDropout
    Module for applying the same dropout mask
    on sequential input supporting packed sequences.
FastLSTM
    Fast LSTM with variational dropout
    supporting packed sequences.

"""


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from typing import *
from torch.autograd import Variable
import torch.nn.functional as F


class Residual(nn.Module):
    """
    Gated residual connection for LSTM.
    TODO
    """
    def __init__(self, i_size, h_size, r_size):
        super(Residual, self).__init__()
        self.W = nn.Parameter(torch.Tensor(i_size + h_size, h_size), requires_grad = True)
        self.T = nn.Parameter(torch.Tensor(r_size, h_size), requires_grad = True) if h_size != r_size else nn.Identity()
                              
    def forward(self, hl, ht, h_o):
        g = torch.matmul(torch.cat((hl, ht), dim = -1), self.W)
        g = torch.sigmoid(g)
        return torch.matmul(h_o, self.T) * g


class LSTMModel(nn.Module):
    """
    LSTM supporting variational dropout
    and residual connection.

    Taken from https://github.com/josephdviviano/lstm-variational-dropout/blob/master/model.py
    """
    def __init__(self, input_size, n_layers, hidden_size,
                 dropout_i=0, dropout_h=0, return_states=True, reverse = False, residual = 0):
        """

        An LSTM model with Variational Dropout applied to the inputs and
        model activations. For details see Eq. 7 of

        A Theoretically Grounded Application of Dropout in Recurrent
        Neural Networks. Gal & Ghahramani, 2016.

        Note that this is equivalent to the weight-dropping scheme they
        propose in Eq. 5 (but not Eq. 6).

        Returns the hidden states for the final layer. Optionally also returns
        the hidden and cell states for all layers.

        Args:
            input_size (int): input feature size.
            n_layers (int): number of LSTM layers.
            hidden_size (int): hidden layer size of all layers.
            dropout_i (float): dropout rate of the inputs (t).
            dropout_h (float): dropout rate of the state (t-1).
            return_states (bool): If true, TODO
        """
        super(LSTMModel, self).__init__()

        assert all([0 <= x < 1 for x in [dropout_i, dropout_h]])
        assert all([0 < x for x in [input_size, n_layers, hidden_size]])
        assert isinstance(return_states, bool)

        self._input_size = input_size
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._dropout_i = dropout_i
        self._dropout_h = dropout_h
        self._dropout_res = 0   # TODO
        self._return_states = return_states

        cells = []
        for i in range(n_layers):
            cells.append(nn.LSTMCell(input_size if i == 0 else hidden_size,
                                     hidden_size,
                                     bias=True))

        self._cells = nn.ModuleList(cells)
        self._input_drop = SampleDrop(dropout=self._dropout_i)
        self._state_drop = SampleDrop(dropout=self._dropout_h)
        self._res_drop = SampleDrop(dropout=self._dropout_res)

        self._reverse = reverse

        if residual > 0:
            self.residual = Residual(input_size, hidden_size, residual)
        else:
            self.residual = None

    @property
    def input_size(self):
        return self._input_size

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def dropout_i(self):
        return self._dropout_i

    @property
    def dropout_h(self):
        return self._dropout_h

    def _new_state(self, batch_size):
        """Initalizes states."""
        h = Variable(torch.zeros(batch_size, self._hidden_size))
        c = Variable(torch.zeros(batch_size, self._hidden_size))

        return (h, c)

    def forward(self, X, residual_X = None):
        """Forward pass through the LSTM.

        Args:
            X (tensor): input with dimensions batch_size, seq_len, input_size

        Returns: Output ht from the final LSTM cell, and optionally all
            intermediate states.
        """
        states = [] if self._return_states else None
        
        X = X.permute(1, 0, 2)
        if residual_X != None:
            residual_X = residual_X.permute(1, 0, 2)

        seq_len, batch_size, input_size = X.shape

        for cell in self._cells:
            ht, ct = [], []

            # Initialize new state.
            h, c = self._new_state(batch_size)
            h = h.to(X.device)
            c = c.to(X.device)

            # Fix dropout weights for this cell.
            self._input_drop.set_weights(X[0, ...])  # Removes time dimension.
            self._state_drop.set_weights(h)
            if residual_X != None:
                self._res_drop.set_weights(residual_X[0, ...])

            idx_range = range(X.shape[0])
            if self._reverse:
                idx_range = reversed(idx_range)

            for i in idx_range:
                h_drop = self._state_drop(h)
                i_drop = self._input_drop(X[i])
                h, c = cell(i_drop, (h_drop, c))

                if self.residual is not None:
                    h += self.residual(i_drop, h_drop, self._res_drop(residual_X[i]))
                
                ht.append(h)
                ct.append(c)

            if self._reverse:
                ht = ht[::-1]
                ct = ct[::-1]

            # Output is again [batch, seq_len, n_feat].
            ht = torch.stack(ht, dim=0).permute(1, 0, 2)
            ct = torch.stack(ct, dim=0).permute(1, 0, 2)

            if self._return_states:
                states.append((h, c))

            X = ht.clone().permute(1, 0, 2)  # Input for next cell.
        #print(states)
        return (ht, (torch.stack([s[0] for s in states]), torch.stack([s[1] for s in states])))

class BiLSTM(nn.Module):
    """
    BiLSTM based on lstm_collection.LSTMModel.
    """
    def __init__(self, input_size, hidden_size,
                 dropout_i=0, dropout_h=0, gated_residual = 0):
        super().__init__()

        self.input_size = input_size
        self.lstm_forward = LSTMModel(input_size = input_size, n_layers = 1, 
                                      hidden_size = hidden_size, dropout_i = dropout_i, 
                                      dropout_h = dropout_h, residual = gated_residual)
        self.lstm_backward = LSTMModel(input_size = input_size, n_layers = 1, 
                                       hidden_size = hidden_size, dropout_i = dropout_i, 
                                       dropout_h = dropout_h, reverse = True, residual = gated_residual)
    
    def forward(self, x, residual_x = None):
        """Assumes x is of shape (batch, sequence, feature)"""
        if residual_x == None:
            hidden_sequence_f, (h_t_f, c_t_f) = self.lstm_forward(x)

            hidden_sequence_b, (h_t_b, c_t_b) = self.lstm_backward(x)
        else:
            
            hidden_sequence_f, (h_t_f, c_t_f) = self.lstm_forward(x, residual_x)

            hidden_sequence_b, (h_t_b, c_t_b) = self.lstm_backward(x, residual_x)


        hidden_sequence = torch.cat((hidden_sequence_f, hidden_sequence_b), dim = 2)
        h_t = torch.cat((h_t_f, h_t_b), dim=0)
        c_t = torch.cat((c_t_f, c_t_b), dim=0)
        
        return hidden_sequence, (h_t, c_t)
    

class SampleDrop(nn.Module):
    """
    Module for applying the same dropout mask
    on sequential input.

    Based on https://github.com/josephdviviano/lstm-variational-dropout/blob/master/model.py
    Applies dropout to input samples with a fixed mask."""
    def __init__(self, dropout=0):
        super().__init__()

        assert 0 <= dropout < 1
        self._mask = None
        self._dropout = dropout

    def set_weights(self, X):
        """Calculates a new dropout mask."""
        assert len(X.shape) == 2

        mask = Variable(torch.ones(X.size(0), X.size(1)), requires_grad=False)

        if X.is_cuda:
            mask = mask.cuda()

        self._mask = F.dropout(mask, p=self._dropout, training=self.training)

    def forward(self, X):
        """Applies dropout to the input X."""
        if not self.training or not self._dropout:
            return X
        else:
            return X * self._mask

class IntegratedBiLSTMModel(nn.Module):
    """
    BiLSTM with the same dropout variational dropout
    mask on both forward and backward direction.

    Based on https://github.com/josephdviviano/lstm-variational-dropout/blob/master/model.py
    """
    def __init__(self, input_size, n_layers, hidden_size,
                 dropout_i=0, dropout_h=0, residual = 0, return_states = True):
        """
        TODO.
        """
        super(IntegratedBiLSTMModel, self).__init__()

        assert all([0 <= x < 1 for x in [dropout_i, dropout_h]])
        assert all([0 < x for x in [input_size, n_layers, hidden_size]])

        self._input_size = input_size
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._dropout_i = dropout_i
        self._dropout_h = dropout_h
        self._dropout_res = 0   # TODO

        cells_f = []
        cells_b = []
        for i in range(n_layers):
            cells_f.append(nn.LSTMCell(input_size if i == 0 else hidden_size,
                                     hidden_size,
                                     bias=True))
            cells_b.append(nn.LSTMCell(input_size if i == 0 else hidden_size,
                                     hidden_size,
                                     bias=True))
            
        self._cells_f = nn.ModuleList(cells_f)
        self._cells_b = nn.ModuleList(cells_b)

        self._input_drop = SampleDrop(dropout=self._dropout_i)
        self._state_drop = SampleDrop(dropout=self._dropout_h)
        self._res_drop = SampleDrop(dropout=self._dropout_res)

        if residual > 0:
            self.residual_f = Residual(input_size, hidden_size, residual)
            self.residual_b = Residual(input_size, hidden_size, residual)
        else:
            self.residual_f = None
            self.residual_b = None
            
        self._return_states = return_states

    @property
    def input_size(self):
        return self._input_size

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def dropout_i(self):
        return self._dropout_i

    @property
    def dropout_h(self):
        return self._dropout_h

    def _new_state(self, batch_size):
        """Initalizes states."""
        h = Variable(torch.zeros(batch_size, self._hidden_size))
        c = Variable(torch.zeros(batch_size, self._hidden_size))

        return (h, c)

    def forward(self, X, residual_x = None):
        """Forward pass through the LSTM.

        Args:
            X (tensor): input with dimensions batch_size, seq_len, input_size

        Returns: Output ht from the final LSTM cell, and optionally all
            intermediate states.
        """

        states_f = [] if self._return_states else None
        states_b = [] if self._return_states else None
        
        X = X.permute(1, 0, 2)
        if residual_x != None:
            residual_x = residual_x.permute(1, 0, 2)

        _, batch_size, _ = X.shape

        for cell_f, cell_b in zip(self._cells_f, self._cells_b):
            ht_f, ct_f = [], []
            ht_b, ct_b = [], []

            # Initialize new state.
            h_f, c_f = self._new_state(batch_size)
            h_f = h_f.to(X.device)
            c_f = c_f.to(X.device)

            # Fix dropout weights.
            self._input_drop.set_weights(X[0, ...])  # Removes time dimension.
            self._state_drop.set_weights(h_f)
            if residual_x != None:
                self._res_drop.set_weights(residual_x[0, ...])

            idx_range = range(X.shape[0])

            for i in idx_range:
                h_drop = self._state_drop(h_f)
                i_drop = self._input_drop(X[i])
                h_f, c_f = cell_f(i_drop, (h_drop, c_f))

                if self.residual_f is not None:
                    h_f += self.residual_f(i_drop, h_drop, self._res_drop(residual_x[i]))
                
                ht_f.append(h_f)
                ct_f.append(c_f)
            
            idx_range = reversed(idx_range)

            h_b, c_b = self._new_state(batch_size)
            h_b = h_b.to(X.device)
            c_b = c_b.to(X.device)

            for i in idx_range:
                h_drop = self._state_drop(h_b)
                i_drop = self._input_drop(X[i])
                h_b, c_b = cell_b(i_drop, (h_drop, c_b))

                if self.residual_b is not None:
                    h_b += self.residual_b(i_drop, h_drop, self._res_drop(residual_x[i]))
                
                ht_b.append(h_b)
                ct_b.append(c_b)

            ht_b = ht_b[::-1]
            ct_b = ct_b[::-1]

            ht_f = torch.stack(ht_f, dim=0)
            ht_b = torch.stack(ht_b, dim=0)
            ht = torch.cat((ht_f, ht_b), dim=2).permute(1, 0, 2)

            if self._return_states:
                states_f.append((h_f, c_f))
                states_b.append((h_b, c_b))

            X = ht.clone().permute(1, 0, 2)  # Input for next cell.
        
        stacked_h = torch.stack([s[0] for s in states_f + states_b])
        stacked_c = torch.stack([s[1] for s in states_f + states_b])
         
        return (ht, (stacked_h, stacked_c))


class FastVariationalDropout(nn.Module):
    """
    Module for applying the same dropout mask
    on sequential input supporting packed sequences.
    
    From https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py

    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: bool = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: Union[torch.Tensor, PackedSequence]) -> Union[torch.Tensor, PackedSequence]:
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

class FastLSTM(nn.LSTM):
    """
    Fast LSTM with variational dropout
    supporting packed sequences.
    
    From https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py"""
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)

        self.dropoutw : Optional[float]
        self.input_drop : Optional[FastVariationalDropout]
        self.output_drop : Optional[FastVariationalDropout]

        self.unit_forget_bias = unit_forget_bias

        self.dropoutw = dropoutw if dropoutw > 0 else None
        if dropouti > 0:
            self.input_drop = FastVariationalDropout(dropouti,
                                             batch_first=batch_first)
        else:
            self.input_drop = None
        
        if dropouto > 0:
            self.output_drop = FastVariationalDropout(dropouto,
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
