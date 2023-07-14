import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



class Attention(nn.Module):
    
    def __init__(self, dim_in, dim_k=64):
        super(Attention, self).__init__()
        self.Q = nn.Linear(dim_in, dim_k)
        self.K = nn.Linear(dim_in, dim_k)
        self.V = nn.Linear(dim_in, dim_k)
        self.O = nn.Linear(dim_k, dim_in)

        self.dim_in = dim_in
        self.dim_k = dim_k

        self.softmax = nn.Softmax(dim=2)

        self.layer_norm = nn.LayerNorm(dim_in)

    def forward(self, X):
        queries = self.Q(X)
        keys = self.K(X)
        values = self.V(X)
        attention = self.softmax(torch.bmm(queries, keys.transpose(1, 2)) / self.dim_k**0.5)
        attended_values = torch.bmm(attention, values)
        return self.layer_norm(X + self.O(attended_values))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_in, dim_k=64):
        super(TransformerEncoderLayer, self).__init__()
        self.atn = Attention(dim_in, dim_k)
        self.ff = nn.Sequential(nn.Linear(dim_in, dim_in),
                                nn.ReLU(),
                                nn.Linear(dim_in, dim_in))
        self.layer_norm = nn.LayerNorm(dim_in)

    def forward(self, X):
        Xp = self.atn(X)
        return self.layer_norm(Xp + self.ff(Xp))

    

if __name__ == "__main__":
    atn = TransformerEncoderLayer(800, 64)
    input = torch.randn(4, 10, 800)

    for i in range(1000):
        res = atn(input)
    print(res.shape)




