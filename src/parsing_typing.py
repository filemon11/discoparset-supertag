"""
This module includes type aliases
for easy typing of methods in the
context of the discontinuous constituent
parsing project.

Type Aliases
----------
AnySentence

Sentence

AnyCorpus

Corpus

Device

"""

import torch
from typing import Sequence, List

AnySentence = Sequence[str]
"""Sequence[str]"""
Sentence    = List[str]
"""List[str]"""

AnyCorpus   = Sequence[AnySentence]
"""Sequence[AnySentence]"""
Corpus      = List[Sentence]
"""List[Sentence]"""

Device      = int | torch.device
"""int | torch.device"""