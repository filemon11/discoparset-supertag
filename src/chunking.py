"""
Module for importing chunking data.
Requires the huggingsface ``dataset`` package
to download the corpus.

The data was originally made available as part
of the CoNLL-2000 shared chunking task.

@inproceedings{tksbuchholz2000conll,
   author     = "Tjong Kim Sang, Erik F. and Sabine Buchholz",
   title      = "Introduction to the CoNLL-2000 Shared Task: Chunking",
   editor     = "Claire Cardie and Walter Daelemans and Claire
                 Nedellec and Tjong Kim Sang, Erik",
   booktitle  = "Proceedings of CoNLL-2000 and LLL-2000",
   publisher  = "Lisbon, Portugal",
   pages      = "127--132",
   year       = "2000"
}


Functions
----------
import_chunking_data
    Imports the chunking dataset.

"""


from datasets import load_dataset, Dataset, DatasetDict

from typing import List, Tuple, Literal, cast
from parsing_typing import Corpus


def import_chunking_data(split : Literal["test"] | Literal["train"] | Literal["dev"]) -> Tuple[Corpus, List[List[int]]]:
    """
    Imports the conll2000 chunking dataset with the help of the
    huggingface ``datasets`` module. Based on Wall Street Journal corpus (WSJ)
    sections 15-18 (train, 211.727 tokens) and section 20 (test, 47.377 tokens).
    Since the dataset has no development split, the last 400 sentences 
    of the train split are provided for development. They are removed
    for "train".

    Parameters
    ----------
    category : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : List[List[str]]
        The chunking values as numerical features.
    """

    def _remove_empty_sentences(tokens : Corpus, labels : List[List[int]]) -> Tuple[Corpus, List[List[int]]]:
        tokens_updated : Corpus             = []
        labels_updated : List[List[int]]    = []

        for t, l in zip(tokens, labels):
            if not len(t) == len(l) == 0:
                tokens_updated.append(t)
                labels_updated.append(l)
        
        return tokens_updated, labels_updated

    dataset : DatasetDict = cast(DatasetDict, load_dataset("conll2000"))

    tokens : Corpus
    labels : List[List[int]]
    
    if split == "dev":
        tokens = dataset["train"]["tokens"][-400:]
        labels = dataset["train"]["chunk_tags"][-400:]

    elif split == "train":
        tokens = dataset["train"]["tokens"][:-400]
        labels = dataset["train"]["chunk_tags"][:-400]

    else:
        tokens = dataset["test"]["tokens"]
        labels = dataset["test"]["chunk_tags"]

    return _remove_empty_sentences(tokens, labels)