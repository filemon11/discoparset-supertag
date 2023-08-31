"""
This module provides functions to import
supertag annotations and supertag distributions
from the depCCG model:

@inproceedings{yoshikawa:2017acl,
      author={Yoshikawa, Masashi and Noji, Hiroshi and Matsumoto, Yuji},
      title={A* CCG Parsing with a Supertag and Dependency Factored Model},
      booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      publisher={Association for Computational Linguistics},
      year={2017},
      pages={277--287},
      location={Vancouver, Canada},
      doi={10.18653/v1/P17-1026},
      url={http://aclweb.org/anthology/P17-1026}
    }


Functions
----------
load_model
    Load the standard depCCG model.
supertag
    Supertags a corpus with the 1-best supertag.
supertag_distribution
    Computes a supertag assignment probability for a corpus.
all_supertags
    Gives the list of possible lexical category assignments a depCCG can predict.

Constants
----------
CCG_CATS
    Supertag prediction vocabulary size depCCG works with.

"""

from depccg.instance_models import load_model as depccg_load_model
from depccg.types import ScoringResult
from depccg.chainer import lstm_parser_bi_fast

from types import MappingProxyType

from typing import List, Optional, Union, Union, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from parsing_typing import Corpus, AnyCorpus, Sentence, AnySentence

import torch

Device = Union[torch.device, int]
"""torch.device | int"""

Variants = Union[None, Literal["elmo", "rebank", "elmo_rebank"]]
"""Union[None, Literal["elmo", "rebank", "elmo_rebank"]]"""

CCG_CATS : MappingProxyType[Variants, int] = MappingProxyType({ None : 425,
                                                                "elmo" : 425,
                                                                "rebank" : 511,
                                                                "elmo_rebank" : 511 })
                             
"Total number of CCG lexical category assignments depCCG uses."

def load_model(device : int = -1, variant : Variants = None) -> lstm_parser_bi_fast.FastBiaffineLSTMParser:
    """
    Loads depCCG basic english model for efficient
    supertagging.
    Parameters
    ----------
    device : int, default = -1
        Device to run on. -1 is CPU.
        Larger integers denote GPUs of your
        system. Does not accept torch.device
        objects.
    variant : Union[None, Literal["elmo"], Literal["rebank"], Literal["elmo_rebank"]]
        Variant of depccg model. None is the
        standard model.
    
    Returns
    -------
    lstm_parser_bi_fast.FastBiaffineLSTMParser
    """
    return depccg_load_model(device = device, variant = variant)[0]

def supertag(corpus : AnyCorpus, model : Optional[lstm_parser_bi_fast.FastBiaffineLSTMParser] = None, 
            device : int = -1, variant : Variants = None) -> Corpus:
    """
    Supertags a corpus using the
    1-best predictin of a depCCG
    model.

    Parameters
    ----------

    corpus : List[List[str]]
        A list of sentences where 
        a sentences is a list of token strings.
    model : Optional[lstm_parser_bi_fast.FastBiaffineLSTMParser], default = None
        Pre-loaded depCCG model. If None,
        the standard model is loaded.
    device : int, default = -1
        Device to initialize the model on if
        no model is provided.
    variant : Union[None, Literal["elmo"], Literal["rebank"], Literal["elmo_rebank"]]
        Variant of depccg model to choose
        if no model is provided. None is the
        standard model.

    Returns
    -------
    List[List[str]]
        The supertagged corpus with
        each supertag represented as
        one string.
    """

    if model is None:
        model = load_model(device, variant = variant)

    score_result    : List[ScoringResult]
    categories      : List[str]
    
    score_result, categories = model.predict_doc(corpus)

    supertags : Corpus = [[categories[row.argmax()] for row in sentence.tag_scores] for sentence in score_result]
    
    return supertags

def supertag_distribution(corpus : AnyCorpus, tensor_device : Device = torch.device("cpu"),
                            model : Optional[lstm_parser_bi_fast.FastBiaffineLSTMParser] = None, 
                            model_device : int = -1, variant : Variants = None) -> List[torch.Tensor]:
    """
    Computes supertag probability
    distributions for a corpus.

    Parameters
    ----------
    corpus : List[List[str]]
        A list of sentences where 
        a sentences is a list of token strings.
    tensor_device : int | torch.device, default = torch.device("cpu")
        Device to locate output tensors on.
    model : Optional[lstm_parser_bi_fast.FastBiaffineLSTMParser], default = None
        Pre-loaded depCCG model. If None,
        it is loaded for this task separately.
    model_device : int, default = -1
        Device to initialize the model on if
        no model is provided.
    variant : Union[None, Literal["elmo"], Literal["rebank"], Literal["elmo_rebank"]]
        Variant of depccg model to choose
        if no model is provided. None is the
        standard model.

    Returns
    -------
    List[torch.Tensor]
        One tensor per input sentence with
        dimension ``(S, CCG_CATS)``, where S
        denotes the sequence length and CCG_CATS
        the number of possible supertags depCCG
        predicts.

    See Also
    -------
    depccg_util.CCG_CATS
    """
    
    # load model if not provided
    if model is None:
        model = load_model(model_device, variant = variant)

    score_result : List[ScoringResult]
    
    score_result, _ = model.predict_doc(corpus)

    # depCCG outputs log probabilities.
    probabilities : List[torch.Tensor] = [torch.exp(torch.tensor(sentence.tag_scores, requires_grad = False, device = tensor_device)) for sentence in score_result]

    return probabilities

def all_supertags(model : Optional[lstm_parser_bi_fast.FastBiaffineLSTMParser] = None, device : int = -1,
                    variant : Variants = None) -> List[str]:
    """
    Retrieves the list of all possible
    lexical category assignments depCCG
    can predict from given model. If no
    model is provided, it loads the standard
    model with ``load_model``.

    Parameters
    ----------

    model : Optional[lstm_parser_bi_fast.FastBiaffineLSTMParser], default = None
        Pre-loaded depCCG model. If None,
        the standard model is loaded.
    device : int, default = -1
        Device to initialize the model on if
        no model is provided.
    variant : Union[None, Literal["elmo"], Literal["rebank"], Literal["elmo_rebank"]]
        Variant of depccg model to choose
        if no model is provided. None is the
        standard model.
        
    Returns
    -------
    List[str]
        The list of all possible
        lexical category assignments.

    """
    if model is None:
        model, _ = load_model(device, variant = variant)

    categories : List[str] = model.predict_doc([[]])[1]
    
    return categories