"""
Module for importing the LTAG-spinal treebank
available at https://www.cis.upenn.edu/~xtag/spinal/

Functions
----------
extract_from_file
    Extracts tokens and supertags from LTAG-spinal file.
extract
    Extracts tokens and supertags for split of choice.

CONSTANTS
----------
SPLITS
    Standard splits of the Penn Treebank by sentence number.
"""

from types import MappingProxyType
from io import open
import os

from typing import List, Tuple, Dict, Union, Set, FrozenSet, Mapping

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


from parsing_typing import Sentence, Corpus, Split

SPLITS : Mapping[str, str] = MappingProxyType({ "train"   : "derivation.train",
                                                "dev"     : "derivation.sec22",
                                                "test"    : "derivation.test"})
"""
Standard splits of the Penn Treebank by sentence number.
Section 2-21 train, 22 dev, 23 test.
"""

def extract_from_file(filename : str) -> Tuple[Corpus, Corpus]:
    """
    Retrieves tokens and supertags of a file
    of the LTAG-spinal treebank.

    Parameters
    ----------
    filename : str
        Name of the file.
    
    Returns
    -------
    tokens : List[List[str]]
        The retrieved tokens.
    supertags : List[List[str]]
        The treieved supertags.
    """

    tokens      : Corpus    = []
    supertags   : Corpus    = []

    sen_num : int = -1

    with open(filename, "r", encoding="utf-8") as file:
        line = next(file)
        while True:
            
            if len(r := line.split()) > 1 and r[0] == "root":
                sen_num += 1
                tokens.append([])
                supertags.append([])
                

            elif line[0] == "#":
                tokens[sen_num].append(line.split()[1])
                
                line = next(file)
                supertags[sen_num].append(line[8:-1])
            
            try:
                line = next(file)

            except StopIteration:
                break

    return tokens, supertags
            

def extract(directory : str, split : Split, 
                 splits_dict : Mapping[str, str] = SPLITS,
                 version : Literal[1, 2] = 1) \
                                        -> Tuple[Corpus, Corpus]:
    """
    Retrieves tokens and supertags of a split
    of the LTAG-spinal treebank.

    Parameters
    ----------
    directory : str
        path to LTAG-spinal directoy
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    splits_dict : Mapping[str, str], default = LTAGspinal.SPLITS
        Mapping from split name to corresponding
        file name. Treebank must be provided already split.
    version : Literal[1, 2], default = 1
        Version of LTAG-spinal treebank to use.
        From LTAG-spinal README: "in v02, spinal templates and the 
        representation of predicate coordination are normalized"
    
    Returns
    -------
    tokens : List[List[str]]
        The retrieved tokens.
    supertags : List[List[str]]
        The treieved supertags.

    See Also
    -------
    LTAGspinal.SPLITS
    """
    
    filename : str = os.path.join(directory, f"{splits_dict[split]}.v0{version}")

    tokens      : Corpus
    supertags   : Corpus

    tokens, supertags = extract_from_file(filename)
    
    return tokens, supertags