"""
This module provides vertically
stacked LSTM models.

Classes
----------
LSTMStack
    Simple stacked biLSTM with residual connections.

VardropLSTMStack
    Variable dropout stacked biLSTM with residual connections.

Functions
----------
unpad
    Retrieves list of tensors from batch padded tensor.

"""


import torch
import torch.nn as nn
from vardroplstm import BiLSTM

torch.split
def unpad(tensor, lengths):
    """
    Returns a list of tensors from batch padded tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Batch padded tensor to retrieve individual
        tensors from. Shape (Batch x Sequence x ...)
    lengths : List[int]
        List of lengths of the sequences
        in ``tensor``. The length should
        equal the first dimension of ``tensor``.

    Returns
    -------
    List[torch.Tensor]
        A list with length ``lengths``
        of tensors with dimension (l x ...)
        for each l in ``lengths``.
    """
    #output = [t.squeeze(0) for t in tensor.split([1 for _ in lengths], dim=0)]
    output = [t[:l] for t, l in zip(tensor, lengths)]
    return output

class LSTMStack(nn.Module):
    """
    Simple stacked biLSTM with residual connections.

    Attributes
    ----------
    lstm_list : nn.ModuleList
        List of ``nn.LSTM`` biLSTM modules.
    residual : bool
        Whether to add the input vector
        to the output vector for each
        biLSTM application.

    Functions
    -------
    initialize_parameters
        Initialisation of weights.
    forward
        Applies the module.
    """
    def __init__(self, depth, dim_lstm_in, dim_lstm_stack, residual = True):
        """
        Initialising method for the ``LSTMStack`` module.
        Includes a call to ``initialize_parameters`` to
        initialise weights.
    
        Parameters
        ----------
        depth : int
            Number of stacked biLSTM modules.
        dim_lstm_in :
            Input dimension of the first
            biLSTM module.
        dim_lstm_stack :
            Input, hidden and output dimension
            of all biLSTMs except for the first
            input dimension.
        """
        super(LSTMStack, self).__init__()

        self.lstm_list = nn.ModuleList([nn.LSTM(input_size              = dim_lstm_in if n == 0 else dim_lstm_stack,
                                                        hidden_size     = dim_lstm_stack //2,
                                                        num_layers      = 1,
                                                        batch_first     = True,
                                                        bidirectional   = True) 
                                                    for n in range(depth)])

        self.residual = residual
        
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialises the weights of the biLSTM
        models with Xavier initalisation.
    
        Returns
        -------
        None
        """
        # Xavier initialization for every layer
        for mod in self.lstm_list:
            for p in mod.parameters():
                n_ids = sum(p.data.shape)
                m = (6 / n_ids)**0.5
                p.data.uniform_(-m, m)

    def forward(self, X, depth = -1, batch=False, lengths=None):
        """
        Forward method for applying the model. Supports
        batch input of a list of variable list sequence 
        tensors. The tensors are padded and packed to allow
        for fast processing and unpacked and unpadded before
        returning. If the ``residual`` attribute is true,
        each LSTM's input is added to its output except
        for the first input (residual connections).
    
        Parameters
        ----------
        X : torch.Tensor | List[torch.Tensor]
            Tensor of two dimensions or list of
            tensors with variable first dimension
            (i.e. sequence length) and uniform
            second dimension.
        depth : int, default = -1, meaning maximum depth    
            Up to which stack level to compute
            outputs. n means computing the output 
            up to (and including) the nth LSTM.

        Returns
        -------
        List[torch.Tensor] | List[List[torch.Tensor]]
            The list of all biLSTM outputs. The first element
            is the first LSTM's input, i.e. the ``X``
            parameter. For batches, the output is
            given as a double list where the first list
            corresponds to the LSTM stack level and the
            second to the batch sequences output at that
            level.
        """
        output_list = [X]

        if depth == -1:
            depth = len(self.lstm_list)

        if not batch:
            current_input = X.unsqueeze(0)

            for lstm, n in zip(self.lstm_list, range(depth)):
                output, _ = lstm(current_input)

                if n != 0 and self.residual:
                    current_input = current_input + output
                else: 
                    current_input = output

                output_list.append(current_input.squeeze(0))

            return output_list
        
        else:
            lengths = [len(s) for s in X]
            
            current_input = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)

            for lstm, n in zip(self.lstm_list, range(depth)):
                output, _ = lstm(torch.nn.utils.rnn.pack_padded_sequence(
                                    current_input, lengths, batch_first=True))

                output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                
                if n != 0:
                    current_input = current_input + output
                else: 
                    current_input = output

                output_list.append(unpad(current_input, lengths))

            return output_list
        

# Batch padded LSTM stack
class VardropLSTMStack(nn.Module):
    """multi-task LSTM stack
    TODO"""
    def __init__(self, num_tasks, dim_lstm_in, dim_lstm_stack, emb_init, residual = True):
        super(VardropLSTMStack, self).__init__()

        self.lstm_list = nn.ModuleList([BiLSTM(         input_size      = dim_lstm_in if n == 0 else dim_lstm_stack,
                                                        hidden_size     = dim_lstm_stack //2,
                                                        dropout_i       = 0.3,
                                                        dropout_h       = 0.3,
                                                        residual        = False) 
                                                    for n in range(num_tasks)])

        self.residual = residual

        self.initialize_parameters(emb_init)

    def initialize_parameters(self, emb_init):
        # Xavier initialization for every layer
        # uniform initialization for embeddings
        for mod in self.lstm_list:
            for p in mod.parameters():
                n_ids = sum(p.data.shape)
                m = (6 / n_ids)**0.5
                p.data.uniform_(-m, m)

    def forward(self, X, depth = -1, batch=False, lengths=None):
        """
        depth: level of stack to return; 0 is LSTM input
        batch: true if sentence is a list of sentences
        """
        output_list = []

        if depth == -1:
            depth = len(self.lstm_list)

        output_list.append(X)

        if batch:
            current_input = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        else:
            current_input = X.unsqueeze(0)


        for lstm, n in zip(self.lstm_list, range(depth)):
            output, _ = lstm(current_input)

            if n != 0 and self.residual:
                current_input = current_input + output
            else: 
                current_input = output

            if batch:
                output_list.append(unpad(current_input, lengths))
            else:
                output_list.append(current_input.squeeze(0))

        return output_list