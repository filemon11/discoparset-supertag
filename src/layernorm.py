"""
This module contains the LayerNormalisation model
used in: 

@inproceedings{coavoux-2021-bert,
    title = "{BERT}-Proof Syntactic Structures: Investigating Errors in Discontinuous Constituency Parsing",
    author = "Coavoux, Maximin",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.288",
    doi = "10.18653/v1/2021.findings-acl.288",
    pages = "3259--3272",
}

The implementation can be found here: https://aclanthology.org/2021.findings-acl.288.pdf

Classes
----------
LayerNormalization

"""

import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out