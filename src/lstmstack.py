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
from vardroplstm import IntegratedBiLSTMModel

import better_lstm as bl


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
    output = [t[:l] for t, l in zip(tensor, lengths)]
    return output

class LSTMStack(nn.Module):
    """
    Simple stacked biLSTM with residual connections and
    optional fast variational dropout.

    Attributes
    ----------
    lstm_list : nn.ModuleList[better_lstm.LSTM]
        List of biLSTM modules.
    residual : bool
        Whether to add the input vector
        to the output vector for each
        biLSTM application.
    initial : nn.Linear | nn.Identity
        Linear transformation at
        input level.
    layernorms : nn.ModuleList[nn.Identity] | nn.ModuleList[nn.LayerNorm]
        List of layer normalisations used for LSTM
        input and at the last output.
    gates : nn.ModuleList[GatedResidual] | List[None]
        List of gates used for residual
        connections to LSTM output.
    residual : int
        Depth of simple residual addition
        connection. 0 means none, 1 means from the
        preceding layer, 2 from the layer before that 
        and so forth.
    residual_add_gated : int
        Depth of residual gated connection
        at LSTM output. 0 means none, 1 means from the
        preceding layer, 2 from the layer before that 
        and so forth.

    Functions
    -------
    initialize_parameters
        Initialisation of weights.
    forward
        Applies the module.
    """
    def __init__(self, depth, dim_lstm_in, dim_lstm_stack, vardrop_i = 0, vardrop_h = 0, residual = 1, 
                    residual_add_gated = 0, layernorm = False, initial_transform = True):
        """
        Initialising method for the ``LSTMStack`` module.
        Includes a call to ``initialize_parameters`` to
        initialise weights.
    
        Parameters
        ----------
        depth : int
            Number of stacked biLSTM modules.
        dim_lstm_in : int
            Input dimension of the first
            biLSTM module.
        dim_lstm_stack : int
            Input, hidden and output dimension
            of all biLSTMs except for the first
            input dimension.
        vardrop_i : float, default = 0, meaning no dropout
            Value of variational LSTM cell input dropout. 
        vardrop_h : float, default = 0, meaning no dropout
            Value of variational LSTM cell hidden dropout.
        residual : int, default = 1
            Which preceding LSTM stack layer to add as a residual
            connection to each LSTM output. 0 means no residual
            connection, 1 means the previous layer, 2 means layer n-2
            and so forth. If ``initial_transform`` is False,
            no residual connection is added from the input dimension.
        residual_add_gated : int, default = 0
            Which preceding LSTm stack layer to add as a simple gated 
            residual connection to each LSTM output. 0 means no residual
            connection, 1 means the previous layer, 2 means layer n-2
            and so forth. If ``initial_transform`` is False,
            no residual connection is added from the input dimension.
        layernorm : bool, default = False
            Whether to use layer normalization between biLSTM layers
            and as a final operation after the last biLSTM in the stack.
        initial_transform : bool, default = False
            Whether to perform a simple linear transformation
            at input level to assign all LSTMs uniform input
            and hidden dimensionality. If residual connections
            are used, this adds a residual connection from
            the input level.
        """
        super(LSTMStack, self).__init__()

        if initial_transform:
            self.initial = nn.Linear(dim_lstm_in, dim_lstm_stack, bias = False)
            dim_lstm_in = dim_lstm_stack
        else:
            self.initial = nn.Identity()

        self.lstm_list = nn.ModuleList([bl.LSTM(input_size      = dim_lstm_in if n == 0 else dim_lstm_stack,
                                                hidden_size     = dim_lstm_stack // 2,
                                                num_layers      = 1,
                                                batch_first     = True,
                                                bidirectional   = True,
                                                dropouti        = vardrop_i,
                                                dropoutw        = vardrop_h,
                                                dropouto        = 0) 
                                                    for n in range(depth)])

        if layernorm:
            self.layernorms = nn.ModuleList([nn.LayerNorm(dim_lstm_in if n == 0 else dim_lstm_stack)
                                                for n in range(depth + 1)])
        else:
            self.layernorms = nn.ModuleList([nn.Identity() for _ in range(depth + 1)])
        
        if residual_add_gated == 0:
            self.gates = [None for _ in range(depth)]
        else:
            self.gates = nn.ModuleList([GatedResidual(dim_lstm_stack) for _ in range(depth)])
        
        self.residual = residual
        self.residual_add_gated = residual_add_gated
        
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

    def forward(self, X, depth = -1, batch = False):
        """
        Forward method for applying the model. Supports
        batch input of a list of variable list sequence 
        tensors. The tensors are padded and packed to allow
        for fast processing and unpacked and unpadded before
        returning.
    
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
        batch : bool
            Whether the input is a batch of tensors.

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
        output_list = []

        if depth == -1:
            depth = len(self.lstm_list)

        if not batch:
            X = X.unsqueeze(0)

            if hasattr(self, "initial"):
                X = self.initial(X)

            output_list.append(X)

            for lstm, norm, n in zip(self.lstm_list, self.layernorms, range(depth)):
                output, _ = lstm(norm(X))

                if n > n - self.residual >= -(-1 if (hasattr(self, "same_initial")) and isinstance(self.initial, nn.Linear) else 0):
                    X = output + output_list[-self.residual]
                else: 
                    X = output
                
                if(hasattr(self, "residual_add_gated")): 
                    if n > n - self.residual_add_gated >= (-1 if isinstance(self.initial, nn.Linear) else 0):
                        X = self.gates[n](output, output_list[-self.residual_add_gated])
                    else: 
                        X = output

                output_list.append(X)

            # final layer norm
            if depth == len(self.lstm_list) > 0:
                output_list[-1] = self.layernorms[-1](output_list[-1])

            return [layer.squeeze(0) for layer in output_list]
        
        else:
            lengths = [len(s) for s in X]
            
            X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)

            if hasattr(self, "initial"):
                X = self.initial(X)

            output_list.append(X)

            for lstm, norm, n in zip(self.lstm_list, self.layernorms, range(depth)):
                output, _ = lstm(torch.nn.utils.rnn.pack_padded_sequence(
                                    norm(X), lengths, batch_first=True))

                output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                
                if n > n - self.residual >= (-1 if (hasattr(self, "same_initial")) and isinstance(self.initial, nn.Linear) else 0):
                    X = output + output_list[-self.residual]
                else: 
                    X = output

                if(hasattr(self, "residual_add_gated")): 
                    if n > n - self.residual_add_gated >= (-1 if isinstance(self.initial, nn.Linear) else 0):
                        X = self.gates[n](output, output_list[-self.residual_add_gated])
                    else: 
                        X = output

                output_list.append(X)
            
            # final layer norm
            if depth == len(self.lstm_list) > 0:
                output_list[-1] = self.layernorms[-1](output_list[-1])

            return [unpad(layer, lengths) for layer in output_list]
        

# Change name to InternalResLSTMStack
class VardropLSTMStack(nn.Module):
    """
    Stacked biLSTM with residual connections and
    optional variational dropout. Slow due to manual
    LSTM implementation necessary for inclusion of
    residual connections to LSTM internals.

    Attributes
    ----------
    lstm_list : nn.ModuleList[better_lstm.LSTM] | nn.ModuleList[vardroplstm.IntegratedBiLSTMModel]
        List of biLSTM modules. Uses fast implementation 
        ``better_lstm.LSTM`` based on the native pytorch
        if no residual connection to the LSTM cell is used
        (i.e. ``residual_gated`` is 0). Else uses 
        ``vardroplstm.IntegratedBiLSTMModel``.
    residual : bool
        Whether to add the input vector
        to the output vector for each
        biLSTM application.
    initial : nn.Linear | nn.Identity
        Linear transformation at
        input level.
    layernorms : nn.ModuleList[nn.Identity] | nn.ModuleList[nn.LayerNorm]
        List of layer normalisations used for LSTM
        input and at the last output.
    gates : nn.ModuleList[GatedResidual] | List[None]
        List of gates used for residual
        connections to LSTM output.
    residual : int
        Depth of simple residual addition
        connection. 0 means none, 1 means from the
        preceding layer, 2 from the layer before that 
        and so forth.
    residual_gated : int
        Depth of simple residual gated connection
        to LSTM internals. 0 means none, 1 means from the
        preceding layer, 2 from the layer before that 
        and so forth.
    residual_add_gated : int
        Depth of residual gated connection
        at LSTM output. 0 means none, 1 means from the
        preceding layer, 2 from the layer before that 
        and so forth.

    Functions
    -------
    initialize_parameters
        Initialisation of weights.
    forward
        Applies the module.

    """
    def __init__(self, depth, dim_lstm_in, dim_lstm_stack, vardrop_i, vardrop_h, residual_add = 1, 
                    residual_gated = 0, residual_add_gated = 0, layernorm = False, initial_transform = True):
        """
        Initialising method for the ``VardropLSTMStack`` module.
        Includes a call to ``initialize_parameters`` to
        initialise weights.
    
        Parameters
        ----------
        depth : int
            Number of stacked biLSTM modules.
        dim_lstm_in : int
            Input dimension of the first
            biLSTM module.
        dim_lstm_stack : int
            Input, hidden and output dimension
            of all biLSTMs except for the first
            input dimension.
        vardrop_i : float, default = 0, meaning no dropout
            Value of variational LSTM cell input dropout. 
        vardrop_h : float, default = 0, meaning no dropout
            Value of variational LSTM cell hidden dropout.
        residual_add : int, default = 1
            Which preceding LSTM stack layer to add as a residual
            connection to each LSTM output. 0 means no residual
            connection, 1 means the previous layer, 2 means layer n-2
            and so forth. If ``initial_transform`` is False,
            no residual connection is added from the input dimension.
        residual_gated : int, default = 1
            Which preceding LSTM stack layer to add as a residual
            connection to each LSTM internals. 0 means no residual
            connection, 1 means the previous layer, 2 means layer n-2
            and so forth. If dimensions do not match, a linear
            transformation is performed at the residual connection.
        residual_add_gated : int, default = 0
            Which preceding LSTm stack layer to add as a simple gated 
            residual connection to each LSTM output. 0 means no residual
            connection, 1 means the previous layer, 2 means layer n-2
            and so forth. If ``initial_transform`` is False,
            no residual connection is added from the input dimension.
        layernorm : bool, default = False
            Whether to use layer normalization between biLSTM layers
            and as a final operation after the last biLSTM in the stack.
        initial_transform : bool, default = False
            Whether to perform a simple linear transformation
            at input level to assign all LSTMs uniform input
            and hidden dimensionality. If residual connections
            are used, this adds a residual connection from
            the input level.
        """
        super(VardropLSTMStack, self).__init__()

        if initial_transform:
            self.initial = nn.Linear(dim_lstm_in, dim_lstm_stack, bias = False)
            dim_lstm_in = dim_lstm_stack
        else:
            self.initial = nn.Identity()

        if residual_gated == 0:
            self.lstm_list = nn.ModuleList([bl.LSTM(
                                                input_size      = dim_lstm_in if n == 0 else dim_lstm_stack,
                                                hidden_size     = dim_lstm_stack //2,
                                                num_layers      = 1,
                                                batch_first     = True,
                                                bidirectional   = True,
                                                dropouti        = vardrop_i,
                                                dropoutw        = vardrop_h,
                                                dropouto        = 0,)
                                                                                for n in range(depth)])
        else:
            self.lstm_list = nn.ModuleList([IntegratedBiLSTMModel(
                                                input_size      = dim_lstm_in if n == 0 else dim_lstm_stack,
                                                n_layers        = 1,
                                                hidden_size     = dim_lstm_stack //2,
                                                dropout_i       = vardrop_i,
                                                dropout_h       = vardrop_h,
                                                residual        = (dim_lstm_in if n - residual_gated == -1 else dim_lstm_stack) 
                                                                            if residual_gated > 0 and n - residual_gated >= -1 else 0)
                                                    for n in range(depth)])
            
        if layernorm:
            self.layernorms = nn.ModuleList([nn.LayerNorm(dim_lstm_in if n == 0 else dim_lstm_stack)
                                                for n in range(depth + 1)])
        else:
            self.layernorms = nn.ModuleList([nn.Identity() for _ in range(depth + 1)])

        if residual_add_gated == 0:
            self.gates = [None for _ in range(depth)]
        else:
            self.gates = nn.ModuleList([GatedResidual(dim_lstm_stack) for _ in range(depth)])
        

        self.residual_add       = residual_add
        self.residual_gated     = residual_gated
        self.residual_add_gated = residual_add_gated

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialises the weights of the biLSTM
        models with Xavier initalisation.
    
        Returns
        -------
        None
        """
        for mod in self.lstm_list:
            for p in mod.parameters():
                n_ids = sum(p.data.shape)
                m = (6 / n_ids)**0.5
                p.data.uniform_(-m, m)

    def forward(self, X, depth = -1, batch = False):
        """
        Forward method for applying the model. Supports
        batch input of a list of variable list sequence 
        tensors. The tensors are padded for processing
        (but not packed) and unpadded before returning.
    
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
        batch : bool
            Whether the input is a batch of tensors.

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
        output_list = []

        if depth == -1:
            depth = len(self.lstm_list)

        if batch:
            lengths = [len(s) for s in X]
            X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        else:
            X = X.unsqueeze(0)
        
        if hasattr(self, "initial"):
                X = self.initial(X)

        output_list.append(X)

        for lstm, norm, n in zip(self.lstm_list, self.layernorms, range(depth)):
            if n > n - self.residual_gated >= -1:
                output, _ = lstm(norm(X), residual_x = output_list[-self.residual_gated])
            else:
                output, _ = lstm(norm(X))

            if n > n - self.residual_add >= (-1 if (hasattr(self, "initial")) and isinstance(self.initial, nn.Linear) else 0):        # cannot add input layer if different
                X = output + output_list[-self.residual_add]
            else:
                X = output

            if(hasattr(self, "residual_add_gated")):    # permits loading saved versions of this 
                                                        # class that did not have this attribute
                if n > n - self.residual_add_gated >= (-1 if isinstance(self.initial, nn.Linear) else 0):        # cannot add input layer if different
                    X = self.gates[n](output, output_list[-self.residual_add_gated])
                else:
                    X = output

            output_list.append(X)

        # final layer norm
        if depth == len(self.lstm_list) > 0:
            output_list[-1] = self.layernorms[-1](output_list[-1])
        
        if batch:
            return [unpad(layer, lengths) for layer in output_list]
        else:
            return [layer.squeeze(0) for layer in output_list]
        

class GatedResidual(nn.Module):
    """
    Gate module for the residual gated
    additive connections. Recieves the
    current layer input and a previous layer
    input, applies a gate to the previous layer
    output, adds the result to the current layer
    and returns the result.

    Attributes
    ----------
    linear : nn.Linear
        Linear transformation from concatenated
        current layer and previous layer to
        the current layer dimensionality.
    sigmoid : nn.Sigmoid
        Sigmoid function to apply to the
        gate.

    Functions
    -------
    initialize_parameters
        Initialisation of linear weights.
    forward
        Applies the module.

    """
    def __init__(self, dim_in, bias_init = 0.1):
        super(GatedResidual, self).__init__()

        self.linear = nn.Linear(2 * dim_in, dim_in, bias = True)
        self.sigmoid = nn.Sigmoid()

        self.initialize_parameters(bias_init)

    def initialize_parameters(self, bias_init):
        """
        Initialises the weights of ``self.linear``
        with Xavier initialisation and the bias 
        vector uniformly with the provided value.
    
        Parameters
        ----------
        bias_init : float
            Value for initialisation of
            linar transformation bias vector.

        Returns
        -------
        None
        """
        torch.nn.init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.fill_(bias_init)


    def forward(self, X, X_res):
        """
        Forward method for applying the gate.
        Returns X plus gated residual connetion.
        Input must have form (B x S x F) with
        B being the batch number and S the sequence
        length.

        Parameters
        ----------
        X : torch.Tensor
            Current layer tensor.
        X_res : torch.Tensor
            Previous layer tensor.

        Returns
        -------
        torch.Tensor
            Result of adding the gated residual
            connection to X.
        """
        gate = self.linear(torch.cat((X, X_res), dim = 2))
        gate = self.sigmoid(gate)
        return X + gate * X_res