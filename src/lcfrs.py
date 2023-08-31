"""
Module for importing LCFRS supertag data.
The data must be generated first using the following tool:
https://github.com/truprecht/lcfrs-supertagger

After generating, the data is placed in a folder
name ``.corpus_cache``.

More information on the extracted supertags:

@inproceedings{ruprecht-morbitz-2021-supertagging,
    title = "Supertagging-based Parsing with Linear Context-free Rewriting Systems",
    author = {Ruprecht, Thomas  and
      M{\"o}rbitz, Richard},
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.232",
    doi = "10.18653/v1/2021.naacl-main.232",
    pages = "2923--2935",
    abstract = "We present the first supertagging-based parser for linear context-free rewriting systems (LCFRS). It utilizes neural classifiers and outperforms previous LCFRS-based parsers in both accuracy and parsing speed by a wide margin. Our results keep up with the best (general) discontinuous parsers, particularly the scores for discontinuous constituents establish a new state of the art. The heart of our approach is an efficient lexicalization procedure which induces a lexical LCFRS from any discontinuous treebank. We describe a modification to usual chart-based LCFRS parsing that accounts for supertagging and introduce a procedure that transforms lexical LCFRS derivations into equivalent parse trees of the original treebank. Our approach is evaluated on the English Discontinuous Penn Treebank and the German treebanks Negra and Tiger.",
}

Functions
----------
import_lcfrs
    Imports the LCFRS supertagged corpus.
"""


import os.path

from typing import List, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


from parsing_typing import Corpus, Sentence, Split


def import_lcfrs(split : Split, path : str) -> Tuple[Corpus, Corpus]:
    """
    Imports a corpus annotated with LCFRS supertags using the LCFRS
    supertagger. The train, dev and test splits should already be in separate
    files named "train-tags" and accordingly.

    Parameters
    ----------
    category : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    path : str
        Directory of the annotated corpus files.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : List[List[str]]
        The LCFRS supertags.
    """

    tokens      : Corpus = []
    supertags   : Corpus = []

    with open(os.path.join(path, f"{split}-tags"), 'r') as file:

        tokens_sent     : Sentence = []
        supertags_sent  : Sentence = []

        for line in file:
            if len(line.split()) == 0:              # empty lines denote the start of a new sentence
                tokens.append(tokens_sent)
                supertags.append(supertags_sent)

                tokens_sent = []
                supertags_sent = []
            
            else:
                components : List[str] = line.split()

                tokens_sent.append(components[0])
                supertags_sent.append(components[1])
            
        return tokens, supertags