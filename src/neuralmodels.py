"""TODO"""

import torch.nn as nn
from dropout import Dropout

from layernorm import LayerNormalization

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop_in, final_bias = False, activation = nn.Tanh, 
                    layer_norm = False, n_hid = 1, drop_hid = 0):
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
        return self.ff(X)