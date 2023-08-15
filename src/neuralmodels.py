"""
Collection of neural network models.

Classes
----------
FeedForward

"""

import torch.nn as nn
from dropout import Dropout

from layernorm import LayerNormalization

class FeedForward(nn.Module):
    """
    Handy feed forward network
    model with variable hidden
    layer number and dropout or 
    layer normalisation

    Attributes
    ----------
    ff : nn.Sequential
        Sequence of nn.Linear modules
        and dropout or layer normalization.

    Functions
    -------
    forward
        Applies the module.

    """
    def __init__(self, d_in, d_hid, d_out, drop_in, final_bias = False, activation = nn.Tanh, 
                    layer_norm = False, n_hid = 1, drop_hid = 0):
        """
        Initialising method for the ``FeedForward`` module.
    
        Parameters
        ----------
        d_in : int
            Input dimension.
        d_hid : int
            Hidden dimension for the
            feed forward network.
        d_out : int
            Output dimension
        final_bias : bool
            Whether to include a bias
            in the output layer.
        activation : nn.Module, default = nn.Tanh
            Activation function to use between layers
        layer_norm : bool
            Whether to use layer normalisation
            after hidden linear layers and before
            the activation function.
        n_hid : int, default = 1
            Number of hidden layers. 0 means that
            the network consists of only a single 
            linear transformation.
        drop_hid : float, default = 0, meaning no dropout
            Dropout to use between hidden layers
            after the activation function.
        """
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
                        Dropout(drop_in),
                        *[nn.Sequential(
                            nn.Linear(d_in if i == 0 else d_hid, d_hid),
                            LayerNormalization(d_hid) if layer_norm else nn.Identity(),
                            activation(),
                            Dropout(drop_hid))
                                for i in range(n_hid)],
                        nn.Linear(d_in if n_hid == 0 else d_hid, d_out, bias = final_bias)
                        )
        
    def forward(self, X):
        """
        Applies the feed forward network
        to the input.
    
        Parameters
        ----------
        X : torch.Tensor
            The input tensor.
        """
        return self.ff(X)