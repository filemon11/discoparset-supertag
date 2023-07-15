import torch
import torch.nn as nn

import CCGbank_import as ccgbank
import CCG_helper as ccg
import chunking as chunk

import depptb

import tree as T
from utils import batch_stochastic_replacement, save_dict, load_dict

from collections import defaultdict

import random
import copy

from typing import Dict, Tuple, List, Optional, Set, Literal, Any, TypeVar, cast


# standard splits for ccgrebank import
ccg_splits : Dict[str, Tuple[int, int]] = {"train" : (200,2200), "dev" : (2201,2300), "test" : (2301,2400)}

# specifying the annotated corpus the available auxiliary features are based on
tasks : Dict[str, Tuple[str, ...]] = {"ccg" : ("supertag", "scope", "leftaction", "rightaction", "head", "arg", "sketch", "argstruct", "near", "functor"),
                                      "depptb" : ("dep",),
                                      "conll2000" : ("chunking",)}

task_to_corpus : Dict[str, str] = {task : corpus for corpus, task_list in tasks.items() for task in task_list}

V = TypeVar("V")

def shuffle_and_limit(data1 : List[V], *data : List[V], limit : Optional[int]) -> Tuple[List[V], ...]:
    """
    Shuffle one or more lists in parallel
    and optionally limit the lists to
    a certain index.

    Parameters
    ----------
    data1 : List[V]
        List to shuffle
    data : Tuple[List[V], ...]
        Additional lists to shuffle
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None
        If not None: Number of sentences to sample.

    Returns
    -------
    data_return : Tuple[List[V], ...]
        Shuffled and optionally limited list(s).
    """

    data_temp = cast(List[Tuple[V, ...]], list(zip(data1, *data)))

    random.shuffle(data_temp)

    if limit is not None:
        data_temp = data_temp[:limit]

    data_return : List[List[V]] = [list(l) for l in list(zip(*data_temp))]
    
    return tuple(data_return)

def import_ccg_basic(ccg_dir : str, split : Literal["test"] | Literal["train"] | Literal["dev"], 
                            limit : Optional[int] = None) -> \
                                Tuple[List[List[str]], Dict[str, List[List[str]]]]:
    """
    Imports the CCGrebank (CCG annotated Penn Treebank) corpus 
    with the help of the the CCGbank_import module.
    The standard split is used (section 2-22 train, 22-23 dev, 24 test).
    The following features are extracted:
        supertags : lexical category assignments
        scopes : TODO 
        leftaction : TODO,
        rightaction : TODO

    Parameters
    ----------
    ccg_dir : str
        CCGrebank directory
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None
        If not None: Number of sentences to sample.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : Dict[str, List[List[str]]]
        Dictionary with the features and
        their values.
    """

    data : Dict[str, List[List[str]]] = {}

    tokens          : List[List[str]]
    supertags       : List[List[str]]
    scopes          : List[List[str]]
    leftactions     : List[List[str]]
    rightactions    : List[List[str]]
    
    supertags, tokens, scopes, leftactions, rightactions = \
                shuffle_and_limit(*ccgbank.import_complex_multi(range(*ccg_splits[split]), ccg_dir)[:5], limit = limit)

    data["supertag"]    = supertags
    data["scope"]       = scopes
    data["leftaction"]  = leftactions
    data["rightaction"] = rightactions

    return tokens, data

def import_dep(dep_dir, split : Literal["test"] | Literal["train"] | Literal["dev"], limit : Optional[int] = None) \
                    -> Tuple[List[List[str]], List[List[str]]]:
    """
    Imports dependency information from the
    depPTB (dependency annotated Penn Treebank) corpus
    with the help of the the depptb module.
    The standard split is used (section 2-22 train, 22-23 dev, 24 test).

    Parameters
    ----------
    dep_dir : str
        path to depPTB .conll file (not split)
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None
        If not None: Number of sentences to sample.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : List[List[str]]
        Dependency features.
    """

    dep_sentences   : List[List[str]]
    dep             : List[List[str]]

    dep_sentences, dep = shuffle_and_limit(*depptb.corpus_parse(dep_dir, split), limit=limit)

    return dep_sentences, dep

def import_chunking(split : Literal["test"] | Literal["train"] | Literal["dev"], limit : Optional[int] = None) \
                            -> Tuple[List[List[str]], List[List[str]]]:
    """
    Imports the conll2000 chunking dataset with the help of the
    huggingface dataset module. Based on Wall Street Journal corpus (WSJ)
    sections 15-18 (train, 211.727 tokens) and section 20 (test, 47.377 tokens).
    Since the dataset has no development split, the last 400 sentences 
    of the train split are provided for development. They are removed
    for "train".

    Parameters
    ----------
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None
        If not None: Number of sentences to sample.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : List[List[str]]
        Chunking features.
    """

    chunking_sentences   : List[List[str]]
    chunking_temp        : List[List[int]]

    chunking_sentences, chunking_temp = chunk.import_chunking_data(split)

    # convert chunking features into strings to match the formatting standard
    chunking : List[List[str]] = [[str(value) for value in sentence] for sentence in chunking_temp]

    chunking_sentences, chunking = shuffle_and_limit(chunking_sentences, chunking, limit=limit)

    return chunking_sentences, chunking

def import_data(tasks : List[List[str]], corpus_dirs : Dict[str, str], split : Literal["test"] | Literal["train"] | Literal["dev"],
                    limit : Optional[int] = None) -> Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]]:
    """
    Retrieves the datasets (tokens and annotation) for the
    tasks specified in the tasks parameter. In cases where
    several tasks are extracted from the same corpus, it is
    only imported once. The split is dependent on the
    the individual import functions.

    Parameters
    ----------
    tasks : List[List[str]]
        Hierarchical list of tasks.
    corpus_dirs : Dict[str, str]
        Mapping from corpus names to their paths.
    splits : str
        The split of the datasets to return.
    
    Returns
    -------
    data : Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]]
        Mapping from corpus names to tuple (tokens, tasks), where:
            tokens -- A list of sentences containing tokens.
            tasks -- A mapping from tasks based on the corpus to the features.
    """

    data_dict : Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]] = {}

    goal_tasks_by_corpus : Dict[str, List[str]] = defaultdict(list)
    for level in tasks:
        for task in level:
            if not task in ("parsing", "tag"):
                goal_tasks_by_corpus[task_to_corpus[task]].append(task)

    
    for corpus in goal_tasks_by_corpus.keys():

        if corpus == "conll2000":
            chunking_sentences, chunking = import_chunking(split, limit)

            data_dict[corpus] = (chunking_sentences, {"chunking" : chunking})

        elif corpus == "depptb":
            
            dep_sentences, dep = import_dep(corpus_dirs["depptb"], split, limit)

            data_dict[corpus] = (dep_sentences, {"dep" : dep})

        elif corpus == "ccg":
            tokens, ccg_basic_task_data = import_ccg_basic(corpus_dirs["ccg"], split, limit)

            data_dict[corpus] = (tokens, {})

            for key, data in ccg_basic_task_data.items():
                if key in goal_tasks_by_corpus[corpus]:

                    data_dict[corpus][1][key] = data

            # Additional tasks
            
            arg_list : Optional[List[List[Tuple[str, List[Tuple[str, str, str]]]]]] = None
            if any([task in ("head", "arg", "functor") for task in goal_tasks_by_corpus[corpus]]):
                arg_list = [[ccg.supertag_to_arg_list(supertag) for supertag in sentence] for sentence in ccg_basic_task_data["supertag"]]

            for task in goal_tasks_by_corpus[corpus]:
                
                # total head
                if task == "head":
                    assert(arg_list is not None)
                    data_dict[corpus][1]["head"] = [[arg_list[0] for arg_list in sentence] for sentence in arg_list] 
                
                # first argument and direction
                elif task == "arg":
                    assert(arg_list is not None)
                    data_dict[corpus][1]["arg"] = [[(arg_list[1][0][1]+arg_list[1][0][2] if len(arg_list[1]) > 0 else "") for arg_list in sentence] for sentence in arg_list] 

                elif task == "functor":
                    assert(arg_list is not None)
                    data_dict[corpus][1]["functor"] = [[(arg_list[1][0][0] if len(arg_list[1]) > 0 else "") for arg_list in sentence] for sentence in arg_list] 

                elif task == "sketch":
                    data_dict[corpus][1]["sketch"] = [[ccg.get_sketch(supertag) for supertag in sentence] for sentence in ccg_basic_task_data["supertag"]]

                elif task == "argstruct":
                    data_dict[corpus][1]["argstruct"] = [[ccg.supertag_to_arg_struct(supertag) for supertag in sentence] for sentence in ccg_basic_task_data["supertag"]]

                elif task == "near":
                    data_dict[corpus][1]["near"] = [ccg.create_near_action(sentence) for sentence in ccg_basic_task_data["supertag"]]


    return data_dict

def data_to_tensor(data_dict : Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]], words2i, device) -> Dict[str, List[Tuple[List[str], torch.Tensor]]]:

    def create_action_tensors(sentences : List[List[str]], words2i) -> List[Tuple[List[str], torch.Tensor]]:
        return [(sentence, torch.tensor([words2i[w] if w in words2i else \
                                words2i["<UNK>"] for w in sentence], dtype=torch.long, device=device)) \
                                                         for sentence in sentences]
    
    task_tensor_dict : Dict[str, List[Tuple[List[str], torch.Tensor]]] = {}
    for _, data in data_dict.items():
        tensors = create_action_tensors(data[0], words2i)
        
        for task in data[1]:
            task_tensor_dict[task] = tensors

    return task_tensor_dict

def data_to_vocab(data_dict : Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]]) -> Dict[str, Tuple[List[str], Dict[str, int]]]:

    vocab_dict : Dict[str, Tuple[List[str], Dict[str, int]]] = {}
    for data in data_dict.values():
        for task, tags in data[1].items():
            task_set : Set[str] = {*[tag for sentence in tags for tag in sentence]}
    
            print(f"number of {task} labels:", len(task_set))

            i2task : List[str]      = list(task_set)
            task2i : Dict[str, int] = {tag : i for i, tag in enumerate(i2task)}
        
            vocab_dict[task] = (i2task, task2i)

    return vocab_dict

def data_to_features(data_dict : Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]], vocab : Dict[str, Tuple[List[str], Dict[str, int]]], device) -> Dict[str, List[torch.Tensor]]:

    features_dict : Dict[str, List[torch.Tensor]] = {}
    for data in data_dict.values():
        for task, tags in data[1].items():
            features_dict[task] = [torch.tensor([vocab[task][1][tag] for tag in sentence], dtype=torch.long, device=device) for sentence in tags]

    return features_dict


class DataLoader():
    def __init__(self, tasks : Optional[List[List[str]]], corpus_dirs : Optional[Dict[str, str]], split = "train", limit : Optional[int] = None, words2i = None, device = "cpu", 
                    vocab : Optional[Dict[str, Tuple[List[str], Dict[str, int]]]] = None, data : Optional[Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]]] = None):
        """split -> 'train', 'dev', 'test'
        """
        self.data       : Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]]
        self.str_features   : Dict[str, List[List[str]]]

        self.vocab      : Optional[Dict[str, Tuple[List[str], Dict[str, int]]]]         = None
        self.features   : Optional[Dict[str, List[torch.Tensor]]]                       = None
        self.tensors    : Optional[Dict[str, List[Tuple[List[str], torch.Tensor]]]]     = None
        self.tensors_copy   : Optional[Dict[str, List[Tuple[List[str], torch.Tensor]]]]     = None

        self.words2i = None
        #TODO self.words2i : TYPE = None
        
        if data is None:
            assert(tasks is not None)
            assert(corpus_dirs is not None)
            self.data = import_data(tasks, corpus_dirs, split, limit)
        else:
            self.data = data

        self.str_features = {task : features for _, tasks in self.data.values() for task, features in tasks.items()}

        self.num_sentences  : Dict[str, int] = {task : len(corpus) for corpus, task_dict in self.data.values() for task in task_dict.keys()}
        self.num_tokens     : Dict[str, int] = {task : sum([len(sentence) for sentence in corpus]) for corpus, task_dict in self.data.values() for task in task_dict.keys()}

        if words2i is not None:
            self.init_tensors(words2i, device)

        if split == "train":
            self.vocab = data_to_vocab(self.data)
            self.features = data_to_features(self.data, self.vocab, device)

        elif split == "dev":
            self.vocab = vocab

    def init_tensors(self, words2i, device = "cpu"):
        self.words2i = words2i
        self.tensors = data_to_tensor(self.data, words2i, device)
        self.tensors_copy = copy.deepcopy(self.tensors)

    @property
    def token_lists(self) -> List[List[List[str]]]:
        return [d[0] for d in self.data.values()]

    @property
    def num_labels(self) -> Dict[str, int]:
        assert(self.vocab is not None)
        return {task : len(data[1]) for task, data in self.vocab.items()}

    def batch_stochastic_replacement(self, pword, device = "cpu"):
        assert(self.words2i is not None)

        for t, t_c in zip(self.tensors.values(), self.tensors_copy.values()):
            batch_stochastic_replacement(device, t, t_c, self.words2i, pword=pword)
    
    def get_sample(self, size : int, device):
        return DataLoader(None, None, "dev", None, None, device, self.vocab, {corpus : (tokens[:size], {task : tags[:size] for task, tags in tasks.items()}) for corpus, (tokens, tasks) in self.data.items()})
    
    def save_vocab(self, directory : str):
        assert(self.vocab is not None)
        for task, (i2t, _) in self.vocab.items():
            save_dict(i2t, "{}/i2{}".format(directory, task))

    def load_vocab(self, directory : str):
        self.vocab = {}
        for task in self.str_features.keys():
            i2t, t2i = load_dict("{}/i2{}".format(directory, task))
            self.vocab[task] = (i2t, t2i)