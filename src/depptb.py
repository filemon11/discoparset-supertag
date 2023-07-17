"""
Module for importing a dependency relation annotated 
version of the Penn Treebank corpus and to convert
dependency graphs to tag features. The provided corpus
must be in .conllu format.

Functions
----------
sentence_parse
    Extracts dependency features from a retrieved DepPTB sentence.
corpus_parse
    Retrieves tokens and dependency features of a split of the corpus.

CONSTANTS
----------
SPLITS
    Standard splits of the Penn Treebank by sentence number.
"""

from types import MappingProxyType

from conllu import parse
from conllu.models import TokenList
from io import open

from typing import List, Tuple, Dict, Literal

SPLITS : MappingProxyType[str, Tuple[int, int]] = MappingProxyType({"train"   : (3915, 43746 + 1),
                                                                    "dev"     : (43747, 45446 + 1),
                                                                    "test"    : (45447, 47862 + 1)})
"""
Standard splits of the Penn Treebank by sentence number.
Section 2-22 train, 22-23 dev, 24 test.
"""

def sentence_parse(sentence : TokenList, max_diff_in : int = 5, max_diff_out : int = 2) \
                                                            -> Tuple[List[str], List[str]]:
    """
    Converts a sentence and a dependency structure
    provided as a ``conllu.models.TokenList`` into
    a list of tokens and a list of dependency features.
    A feature consists of the dependency relation type of 
    the incoming arc as well as the relative position 
    of the maximum left-side outgoing arc and the minimum 
    right-side outgoing arc.

    Parameters
    ----------
    sentence : TokenList
        Input dependency graph.
    max_diff_in : int, default = 5
        Currently obsolete.
    max_diff_out : int, default = 2
        Maximum absolute relative output distance. 
        Larger distances are capped at this value.
        The default 2 maintains a small number of 
        tags while still capturing the notion of
        adjacent or long-range relationships.

    Returns
    -------
    tokens : List[str]
        The retrieved tokens.
    features : List[str]
        The retrieved dependency features.
    """
    tokens  : List[str] = []
    pos     : List[str] = []

    in_dep      : List[str] = []
    in_deptype  : List[str] = []

    out_dep     : List[Tuple[List[int], List[int]]] = [([],[]) for _ in range(len(sentence))]
    out_deptypes    : List[List[str]] = [[] for _ in range(len(sentence))]

    total_dep   : List[Tuple[List[int], List[int]]] = [([],[]) for _ in range(len(sentence))]

    # TODO: Explain what is happening here
    for token in sentence:
        tokens.append(token['form'])
        pos.append(token["upos"])
        
        diff : int = token['id']-token["head"]
        if diff > max_diff_out:
            diff = max_diff_out
        elif diff < -max_diff_out:
            diff = -max_diff_out

        if token["head"] > 0:
            out_dep[token["head"] - 1][0 if diff < 0 else 1].append(diff)
            total_dep[token["head"] -1][0 if diff < 0 else 1].append(token['id'] -1)
            out_deptypes[token["head"] -1].append(str(token["deprel"]).split(":")[0])

        in_diff : int = token["head"] - token["id"]
        if in_diff > max_diff_in:
            in_diff = max_diff_in
        elif in_diff < -max_diff_in:
            in_diff = -max_diff_in

        in_dep.append("" if token["deprel"] == "punct" and False else token["deprel"].split(":")[0] + "/" + str(in_diff))
        in_deptype.append(token["deprel"].split(":")[0])

    return tokens, [f"{i}_{str(max(set(o[0]))) if len(o[0]) > 0 else '0'}_{str(min(set(o[1]))) if len(o[1]) > 0 else '0'}" for i, o in zip(in_deptype, out_dep)]

def corpus_parse(filename : str, split : Literal["test"] | Literal["train"] | Literal["dev"], 
                 splits_dict : MappingProxyType[str, Tuple[int, int]] | Dict[str, Tuple[int, int]] = SPLITS) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Retrieves tokens and lexicalised dependency features
    of a split of a dependency annotated corpus.
    Each token in a sequence receives as feature the
    dependency relation type of its incoming arc as well as
    the relative position of the maximum left-side outgoing
    arc and the minimum right-side outgoing arc.

    Parameters
    ----------
    filename : str
        path to depPTB .conll file (not split)
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    splits_dict : MappingProxyType[str, Tuple[int, int]] | Dict[str, Tuple[int, int]]
        Mapping from split name to corresponding
        sentence number slice in the corpus.
        Standard splits are set as default.

    Returns
    -------
    tokens : List[List[str]]
        The retrieved tokens.
    features : List[List[str]]
        The retrieved dependency features.

    See Also
    -------
    depptb.SPLITS
    """
    
    data_file = open(filename, "r", encoding="utf-8").read()
    parselist = parse(data_file)[slice(*splits_dict[split])]
    tokens      : List[List[str]] = []
    features    : List[List[str]] = []

    for sentence in parselist:
        sen_tokens      : List[str]
        sen_features    : List[str]

        sen_tokens, sen_features = sentence_parse(sentence)

        tokens.append(sen_tokens)
        features.append(sen_features)

    return tokens, features