from re import I
from depccg.instance_models import load_model as depccg_load_model
from depccg.types import ScoringResult
from depccg.chainer import lstm_parser_bi_fast
#from depccg.allennlp import supertagger

from multiprocessing import Pool

from collections import defaultdict

from typing import DefaultDict, List, Tuple, Optional, Dict, Set, Any, Hashable

import copy

import torch

import tree as T

def load_model(device : int = -1):
    return depccg_load_model(None)[0]

def supertag(corpus : List[List[str]], device : int = -1, model = None) -> List[List[str]]:

    if not model:
        model = load_model(device)

    score_result    : List[ScoringResult]
    categories_     : List[str]
    
    score_result, categories_ = model.predict_doc(corpus)

    supertags : List[List[str]] = [[categories_[row.argmax()] for row in sentence.tag_scores] for sentence in score_result]
    
    return supertags

def supertag_onehot(corpus : List[List[str]], device : int = -1, model = None):

    if not model:
        model = load_model(device)

    score_result    : List[ScoringResult]
    categories_     : List[str]
    
    score_result, categories_ = model.predict_doc(corpus)

    results = [torch.zeros(len(sen), len(categories_), requires_grad = False) for sen in corpus]

    for s_i, sentence in enumerate(score_result):
        for w_i, row in enumerate(sentence.tag_scores):
            
            results[s_i][w_i] = torch.exp(torch.tensor(row, requires_grad = False))
    
    return results

def supertag_features(corpus : List[List[str]], device : int = -1, model = None) -> List[List[str]]:

    if not model:
        model = load_model(device)

    r,_ = model.predict_doc(corpus, gpu = -1)

    return r

def all_supertags(device : int = -1, model = None) -> List[str]:

    if not model:
        model, _ = load_model(device)

    categories_ : List[str]
    
    _, categories_ = model.predict_doc([[]])
    
    return categories_


def prepare_ccg(corpus, model = None, tokens_input = False):
    if not tokens_input:
        raw_corpus = [[tok.token for tok in T.get_yield(corpus[i])] for i in range(len(corpus))]
    else:
        raw_corpus = corpus
    scores = supertag_onehot(raw_corpus, model)
    return scores