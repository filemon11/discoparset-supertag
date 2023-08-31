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

from typing import List, Tuple, Dict, Union, Set, FrozenSet

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


from parsing_typing import Sentence, Corpus, Split

SPLITS : MappingProxyType[str, Tuple[int, int]] = MappingProxyType({"train"   : (3915, 43746 + 1),
                                                                    "dev"     : (43747, 45446 + 1),
                                                                    "test"    : (45447, 47862 + 1)})
"""
Standard splits of the Penn Treebank by sentence number.
Section 2-22 train, 22-23 dev, 24 test.
"""

def sentence_parse_deprel(sentence : TokenList, max_diff_in : int = 5) -> Sentence:
    """
    Converts a dependency structure
    provided as a ``conllu.models.TokenList`` into
    a list of dependency labels.
    A feature consists of the label of the incoming arc
    and the position of the head of the incoming arc
    relative to the current position.

    Parameters
    ----------
    sentence : TokenList
        Input dependency graph.
    max_diff_in : int, default = 5

    Returns
    -------
    features : List[str]
        The retrieved dependency features.
    """
    in_dep      : Sentence = []

    for token in sentence:

        in_diff : int = token["head"] - token["id"]
        if in_diff > max_diff_in:
            in_diff = max_diff_in
        elif in_diff < -max_diff_in:
            in_diff = -max_diff_in

        in_dep.append(token["deprel"].split(":")[0] + "_" + str(in_diff))

    return in_dep

def sentence_parse_deprelPOS(sentence : TokenList, max_diff_in : int = 3) -> Sentence:
    """
    Converts a dependency structure
    provided as a ``conllu.models.TokenList`` into
    a list of tokens and a list of dependency features.
    A feature consists of the label of 
    the incoming arc as well as the POS tag p of the arcs
    head and the relative position relative to the current
    word only considering the words with POS tag p.

    Parameters
    ----------
    sentence : TokenList
        Input dependency graph.
    max_diff_in : int, default = 3

    Returns
    -------
    features : List[str]
        The retrieved dependency features.
    """

    in_dep      : Sentence = []

    #print(sentence)
    for token in sentence:
        
        if token["head"] == 0:
            in_dep.append(f"root_-1_{token['deprel']}")

        else:
            POS_count : int = 0
            diff : int = token["head"] - token['id']

            head_pos : str = sentence[token["head"] - 1]["upos"]
            while diff != 0:
                #print(diff, token['id'], token['head'])
                if sentence[token['id'] + diff - 1]["upos"] == head_pos:
                    POS_count += 1 if diff > 0 else -1
                diff += 1 if diff < 0 else -1    # reduce by one or add 1 if negative

            if POS_count > max_diff_in:
                POS_count = max_diff_in
            elif POS_count < -max_diff_in:
                POS_count = -max_diff_in

            in_dep.append(f"{head_pos}_{POS_count}_{token['deprel'].split(':')[0]}")

    return in_dep

def corpus_parse(filename : str, split : Split, 
                 splits_dict : Union[MappingProxyType[str, Tuple[int, int]], Dict[str, Tuple[int, int]]] = SPLITS,
                 encoding : FrozenSet[Literal["deprel", "deprelPos"]] = frozenset({"deprel"})) \
                                        -> Tuple[Corpus, Dict[str, Corpus]]:
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
    splits_dict : MappingProxyType[str, Tuple[int, int]] | Dict[str, Tuple[int, int]], default = depptb.SPLITS
        Mapping from split name to corresponding
        sentence number slice in the corpus.
        Standard splits are set as default.
    encoding : Literal["deprel"] | Literal["deprelPos"], default = {"deprel"}
        Type of encoding of dependency tree.
        cf. https://aclanthology.org/N19-1077.pdf
    
    Returns
    -------
    tokens : List[List[str]]
        The retrieved tokens.
    features : Dict[Union[Literal["deprel"], Literal["deprelPOS"]], Corpus]
        Dictionary of requested tasks.

    See Also
    -------
    depptb.SPLITS
    """
    
    data_file = open(filename, "r", encoding="utf-8").read()
    parselist = parse(data_file)[slice(*splits_dict[split])]
    tokens      : Corpus = []
    features    : Dict[str, Corpus] = {}
    
    # retrieve tokens
    for sentence in parselist:
        sentence_tokens : Sentence = []
        for token in sentence:
            sentence_tokens.append(token['form'])
        tokens.append(sentence_tokens)

    # construct tasks
    if "deprel" in encoding:
        deprel_features : Corpus = []
        for sentence in parselist:
            deprel_features.append(sentence_parse_deprel(sentence))

        features["deprel"] = deprel_features

    if "deprelPOS" in encoding:
        deprelPOS_features : Corpus = []
        for sentence in parselist:
            deprelPOS_features.append(sentence_parse_deprelPOS(sentence))
            
        features["deprelPOS"] = deprelPOS_features

    return tokens, features