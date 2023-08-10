"""
Module for the management of auxiliary task corpora.
Brings separate modules for several auxiliary tasks
into one uniformly usable framework.

Classes
----------
DataLoader
    Helper class for data management of auxiliary task corpora.

Functions
----------
shuffle_and_limit
    Shuffle one or more list(s) in parallel and optionally sample a certain amount of entries.
import_ccg_basic
    Import CCG corpus data.
import_dep
    Import depPTB corpus data.
import_chunking
    Import data from the ConNLL-2000 chunking task.
import_lcfrs
    Import data from LCFRS supertag annotated corpus.
import_data
    Import all necessary data for a given list of task names. 
data_to_tensor
    Compute tensors for tokens of a multi-task dataset into.
features_to_vocab
    Compute vocabulary (index to feature, feature to index) mappings for a multi-task dataset.
features_to_tensor_features
    Compute tensors for feature values of a multi-task dataset.
"""

import torch
import torch.nn as nn

import CCGbank_import as ccgbank
import CCG_helper as ccg
import chunking as chunk
import lcfrs

import depptb

from utils import batch_stochastic_replacement, save_dict, load_dict, sentence_to_tensors
from helpers import corpus_apply

from collections import defaultdict

import random
import copy

from types import MappingProxyType

from typing import Dict, Tuple, List, Optional, Set, Literal, TypeVar, cast, Sequence, Mapping
from parsing_typing import Corpus, AnyCorpus, Device, Sentence

# specifying the annotated corpus the available auxiliary features are based on
TASKS : MappingProxyType[str, Tuple[str, ...]] = MappingProxyType({ "ccg" : ("supertag", "scope", "leftaction", 
                                                                            "rightaction", "head", "arg", "sketch", 
                                                                            "argstruct", "near", "functor"),
                                                                    "depptb" : ("dep",),
                                                                    "conll2000" : ("chunking",),
                                                                    "lcfrsptb" : ("lcfrs",)})

TASK2CORPUS : MappingProxyType[str, str] = MappingProxyType({task : corpus for corpus, task_list in TASKS.items() for task in task_list})

V = TypeVar("V")
Y = TypeVar("Y")

def shuffle_and_limit(data1 : Sequence[V], *data : Sequence[V], limit : Optional[int]) -> Tuple[List[V], ...]:
    """
    Shuffle one or more lists in parallel
    and optionally limit the lists to
    a certain index.

    Parameters
    ----------
    data1 : Sequence[V]
        List to shuffle
    data : Tuple[Sequence[V], ...]
        Additional lists to shuffle
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None, default = None, meaning no limit
        Number of sentences to sample.

    Returns
    -------
    Tuple[List[V], ...]
        Shuffled and optionally limited list(s).
    """

    # Construct one tuple per list(s) index. Enables shuffling content
    # from several lists in parallel.
    data_temp = cast(List[Tuple[V, ...]], list(zip(data1, *data)))

    random.shuffle(data_temp)

    if limit is not None:
        data_temp = data_temp[:limit]

    data_return : List[List[V]] = [list(l) for l in list(zip(*data_temp))]
    
    return tuple(data_return)

def import_ccg_basic(ccg_dir : str, split : Literal["test"] | Literal["train"] | Literal["dev"], 
                            limit : Optional[int] = None) -> \
                                Tuple[Corpus, Dict[str, Corpus]]:
    """
    Imports the CCGrebank (CCG annotated Penn Treebank) corpus 
    with the help of the the ``CCGbank_import`` module.
    The standard split is used (section 2-22 train, 22-23 dev, 24 test).
    The following features are extracted:
        supertags : lexical category assignments
        scopes : range boundaries of the supertag's predecessors in the derivation 
        leftaction : relative position of a supertag's left arguments,
        rightaction : relative position of a supertag's right arguments

    Parameters
    ----------
    ccg_dir : str
        CCGrebank directory
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None, default = None, meaning no limit
        Number of sentences to sample.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : Dict[str, List[List[str]]]
        Dictionary with the features and
        their values.
    """

    data : Dict[str, Corpus] = {}

    tokens          : Corpus
    supertags       : Corpus
    scopes          : Corpus
    leftactions     : Corpus
    rightactions    : Corpus
    
    tokens, supertags, scopes, leftactions, rightactions = \
                shuffle_and_limit(*ccgbank.import_complex_multi(range(*ccgbank.SPLITS[split]), ccg_dir)[:5], limit = limit)

    data["supertag"]    = supertags
    data["scope"]       = scopes
    data["leftaction"]  = leftactions
    data["rightaction"] = rightactions

    return tokens, data

def import_dep(dep_dir : str, split : Literal["test"] | Literal["train"] | Literal["dev"], limit : Optional[int] = None) \
                    -> Tuple[Corpus, Corpus]:
    """
    Imports dependency information from the
    depPTB (dependency annotated Penn Treebank) corpus
    with the help of the the ``depptb`` module.
    The standard split is used (section 2-22 train, 22-23 dev, 24 test).

    Parameters
    ----------
    dep_dir : str
        path to depPTB .conll file (not split)
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None, default = None, meaning no limit
        Number of sentences to sample.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : List[List[str]]
        Dependency features.
    """

    dep_sentences   : Corpus
    dep             : Corpus

    dep_sentences, dep = shuffle_and_limit(*depptb.corpus_parse(dep_dir, split), limit=limit)

    return dep_sentences, dep

def import_chunking(split : Literal["test"] | Literal["train"] | Literal["dev"], limit : Optional[int] = None) \
                            -> Tuple[Corpus, Corpus]:
    """
    Imports the conll2000 chunking dataset with the help of the
    huggingface ``datasets`` module. Based on Wall Street Journal corpus (WSJ)
    sections 15-18 (train, 211.727 tokens) and section 20 (test, 47.377 tokens).
    Since the dataset has no development split, the last 400 sentences 
    of the train split are provided for development. They are removed
    for "train".

    Parameters
    ----------
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None, default = None, meaning no limit
        Number of sentences to sample.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : List[List[str]]
        Chunking features.
    """

    chunking_sentences   : Corpus
    chunking_temp        : List[List[int]]

    chunking_sentences, chunking_temp = chunk.import_chunking_data(split)

    # convert chunking features into strings to match the formatting standard
    chunking : List[List[str]] = [[str(value) for value in sentence] for sentence in chunking_temp]

    chunking_sentences, chunking = shuffle_and_limit(chunking_sentences, chunking, limit = limit)

    return chunking_sentences, chunking

def import_lcfrs(lcfrs_dir : str, split : Literal["test"] | Literal["train"] | Literal["dev"], limit : Optional[int] = None) \
                            -> Tuple[Corpus, Corpus]:
    """
    Imports a LCFRS supertagged dataset. The splits are pre-determined
    by the conversion tool. See module ``lcfrs`` for more information.

    Parameters
    ----------
    split : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the dataset to return.
    limit : int | None, default = None, meaning no limit
        Number of sentences to sample.

    Returns
    -------
    tokens : List[List[str]]
        The tokens.
    features : List[List[str]]
        Supertags.
    """
    lcfrs_sentences : Corpus
    supertags       : Corpus

    lcfrs_sentences, supertags = shuffle_and_limit(*lcfrs.import_lcfrs(split, lcfrs_dir), limit = limit)
    
    return lcfrs_sentences, supertags



def import_data(tasks : Sequence[str], corpus_dirs : Mapping[str, str], split : Literal["test"] | Literal["train"] | Literal["dev"],
                    limit : Optional[int] = None) -> Dict[str, Tuple[Corpus, Dict[str, Corpus]]]:
    """
    Retrieves the datasets (tokens and annotation) for the
    tasks specified in the ``tasks`` parameter. In cases where
    several tasks are extracted from the same corpus, it is
    only imported once. What parts are included on the split
    is dependent on the individual import functions. However,
    sections 22-23 and 24 of of the Wall Street Journal 
    corpus (WSJ), usually used as dev and test, are never
    part of the train split.

    Parameters
    ----------
    tasks : Sequence[str]
        Sequence of tasks. "parsing" and "tag" are ignored.
    corpus_dirs : Mapping[str, str]
        Mapping from corpus names to their paths.
    splits : Literal["test"] | Literal["train"] | Literal["dev"]
        The split of the datasets to return.
    limit : int | None, default = None, meaning no limit
        Number of sentences to sample.
    
    Returns
    -------
    Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]]
        Mapping from corpus names to tuple (tokens, tasks), where:
            tokens -- A list of sentences containing tokens.
            tasks -- A mapping from tasks based on the corpus to the features.
    """

    data_dict : Dict[str, Tuple[Corpus, Dict[str, Corpus]]] = {}

    # sort tasks by corpus
    goal_tasks_by_corpus : Dict[str, List[str]] = defaultdict(list)
    for task in tasks:
        # ignore "parsing" and "tag" tasks
        if not task in ("parsing", "tag"):
            goal_tasks_by_corpus[TASK2CORPUS[task]].append(task)

    # import the necessary corpora
    for corpus in goal_tasks_by_corpus.keys():

        match corpus:
            case "conll2000":
                chunking_sentences, chunking = import_chunking(split, limit)

                data_dict[corpus] = (chunking_sentences, {"chunking" : chunking})

            case "depptb":
            
                dep_sentences, dep = import_dep(corpus_dirs["depptb"], split, limit)

                data_dict[corpus] = (dep_sentences, {"dep" : dep})
        
            case "lcfrsptb":
                
                lcfrs_sentences, supertags = import_lcfrs(corpus_dirs["lcfrsptb"], split, limit)

                data_dict[corpus] = (lcfrs_sentences, {"lcfrs" : supertags})

            case "ccg":

                # for CCG, four basic tasks are available through the import_ccg_basic method.
                # The other tasks need to be constructed using the supertags provided by the
                # import_ccg_basic method. 
                tokens, ccg_basic_task_data = import_ccg_basic(corpus_dirs["ccg"], split, limit)

                data_dict[corpus] = (tokens, {})

                # only add those basic tasks are actually requested
                for key, data in ccg_basic_task_data.items():
                    if key in goal_tasks_by_corpus[corpus]:
                        data_dict[corpus][1][key] = data

                # Additional CCG tasks

                # head, arg and functor tasks are constructed using one common method
                if any([task in ("head", "arg", "functor") for task in goal_tasks_by_corpus[corpus]]):
                    head    : Corpus      # total head
                    arg     : Corpus       # first argument and direction
                    functor : Corpus   
                    head, arg, functor = ccg.head_arg_functor(ccg_basic_task_data["supertag"])

                for task in goal_tasks_by_corpus[corpus]:

                    match task:
                        case "head":
                            data_dict[corpus][1]["head"] = head
                        case "arg":
                            data_dict[corpus][1]["arg"] = arg
                        case "functor":
                           data_dict[corpus][1]["functor"] = functor
                        case "sketch":
                            data_dict[corpus][1]["sketch"] = corpus_apply(ccg_basic_task_data["supertag"], ccg.get_sketch)
                        case "argstruct":
                            data_dict[corpus][1]["argstruct"] = corpus_apply(ccg_basic_task_data["supertag"], ccg.supertag_to_arg_struct)
                        case "near":
                            data_dict[corpus][1]["near"] = [ccg.create_near_action(sentence) for sentence in ccg_basic_task_data["supertag"]]

    return data_dict

def data_to_tensor(data_dict : Mapping[str, Tuple[AnyCorpus, Mapping[str, AnyCorpus]]], words2i : Dict[str, int], device : Device = torch.device("cpu")) \
                                    -> Dict[str, List[Tuple[Sentence, torch.Tensor]]]:
    """
    Creates tensor representations for corpora separated
    by tokens and sentences. Each word is substituted with
    the index provided by ``words2i``. For corpora with several
    tasks, each task in the output mapping points to the same
    tensor in the memory.

    Parameters
    ----------
    data_dict : Mapping[str, Tuple[Sequence[Sequence[str]], Mapping[str, Sequence[Sequence[int]]]]]
        Mapping from corpus names to tuples (tokens, tasks), where
        tokens are the tokens strings of the corpus and tasks is
        a mapping from tasks based on the corpus to the features.
    words2i : Dict[str, int]
        Mapping from words to indices.
    device : Device, default = torch.device("cpu")
        Device to locate the tensors on.
    
    Returns
    -------
    Dict[str, List[Tuple[List[str], torch.Tensor]]]
        Mapping from task names to their corpus tokens
        in tensor representation.
    """
    
    task_tensor_dict : Dict[str, List[Tuple[Sentence, torch.Tensor]]] = {}

    for _, data in data_dict.items():
        tensors = [sentence_to_tensors(sentence, words2i, device) for sentence in data[0]]
        
        for task in data[1]:
            task_tensor_dict[task] = tensors

    return task_tensor_dict

def features_to_vocab(features : Mapping[str, AnyCorpus], verbose : bool = False) -> Dict[str, Tuple[List[str], Dict[str, int]]]:
    """
    Computes vocabulary mappings (from features
    to indices, from indices to features) for all
    tasks in the provided ``features`` mapping.

    Parameters
    ----------
    features : Mapping[str, Sequence[Sequence[str]]]
        Mapping from task names to 
        their feature annotated corpus.
    verbose : bool, default = False
        If True, the number of task labels
        is printed.
    
    Returns
    -------
    Dict[str, Tuple[List[str], Dict[str, int]]]
        Mapping from task to tuple (i2task, task2i), where:
            i2task  --  A list containing all the individual
                        features values.
            task2i  --  A mapping from feature values to
                        indices corresponding to the list
                        indice in i2task.
    """


    vocab_dict : Dict[str, Tuple[List[str], Dict[str, int]]] = {}
    for task, tags in features.items():
        task_set : Set[str] = {*[tag for sentence in tags for tag in sentence]}

        if verbose:
            print(f"number of {task} labels:", len(task_set))
        i2task : List[str]      = list(task_set)
        task2i : Dict[str, int] = {tag : i for i, tag in enumerate(i2task)}
    
        vocab_dict[task] = (i2task, task2i)

    return vocab_dict

def features_to_tensor_features(features : Mapping[str, AnyCorpus], vocab : Mapping[str, Tuple[Sequence[str], Mapping[str, int]]], 
                                device : Device = torch.device("cpu")) -> Dict[str, List[torch.Tensor]]:
    """
    Converts feature annotated corpora to
    tensors for each task. The ``vocab`` parameter
    specifies the substitution from feature values
    to integers.

    Parameters
    ----------
    features : Mapping[str, Sequence[Sequence[str]]]
        Mapping from task names to 
        their str annotated corpus.
    vocab : Mapping[str, Tuple[Sequence[str], Mapping[str, int]]]
        Mapping from task names to their i2task
        and task2i mappings.
    device : Device, default = torch.device("cpu")
        Device to locate tensors on.

    Returns
    -------
    Dict[str, List[torch.Tensor]]
        Mapping from task to feature annotated corpus
        in index form as tensors.

    See Also
    --------
    features_to_vocab : Create vocab mappings.
    
    """

    features_dict : Dict[str, List[torch.Tensor]] = {}
    for task, tags in features.items():

        features_dict[task] = [torch.tensor([vocab[task][1][tag] for tag in sentence], dtype=torch.long, device=device) for sentence in tags]   #type: ignore

    return features_dict

class DataLoader():
    """
    Helper class for data management of auxiliary task
    corpora. Provides functions to load a variety of tasks 
    (tokens and annotations) and to convert into formats
    required by ``sfparser``. The class behaviour depends
    on the ``mode`` parameter when initialising:

    "train" --  Computes ``vocab`` and ``tensor_features``
        attributes from the retrieved data when initialising.
    "eval"  --  ``vocab`` must be provided separately either
        when initalising or later on from saved data on the disk
        via the ``load_vocab`` method.

    Attributes
    ----------
    data : Dict[str, Tuple[List[List[str]], Dict[str, List[List[str]]]]
        Mapping from corpus names to tuples (tokens, tasks), where
        tokens are the tokens strings of the corpus and tasks is
        a mapping from tasks based on the corpus to the features.
    vocab : None | Dict[str, Tuple[List[str], Dict[str, int]]]
        Mapping from task names to their i2task
        and task2i mappings.
    tensor_features : None | Optional[Dict[str, List[torch.Tensor]]]
        Mapping from task names to their features
        as distinc indices in ``torch.Tensor`` format.
    tensors : None | Optional[Dict[str, List[Tuple[List[str], torch.Tensor]]]]
        Mapping from task names to their corpus tokens
        in tensor representation.
    tensors_copy : None | Optional[Dict[str, List[Tuple[List[str], torch.Tensor]]]]
        Copy of ``tensors`` for ``batch_stochastic_replacement``.
    words2i : None | Dict[str, int]
        Mapping from words to unique indices.
    num_sentences : Dict[str, int]
        Mapping from tasks to the number of
        sentences that are annotated for the
        task.
    num_tokens : Dict[str, int]
        Mapping from tasks to the total number
        of tokens annotated for the task.
    features : Dict[str, List[List[str]]
        @property, mapping from task names to their feature
        values saved in ``data``.
    token_lists : List[List[List[str]]]
        @property, the string token lists.
    num_labels : Dict[str, int]
        @property, mapping from task name to the number of
        individual feature labels that occurr for it.

    Methods
    -------
    init_tensors(words2i, device = "cpu")
        Initialises the ``tensors`` and
        ``tensors_copy`` attributes.
    batch_stochastic_replacement(pword, device = "cpu")
        Performs batch stochastic replacement
        on the ``tensors_copy`` atrribute.
    get_sample(self, size, device = "cpu")
        Retrieves a new DataLoader object from
        the current DataLoader with the data limited 
        at index ``size``. Attention: The underlying
        objects in the memory remain the same.
    save_vocab(directory)
        Saves ``vocab`` attribute to ``directory``.
    load_vocab(directorx)
        Loads ``vocab`` attribute from ``directory``.
    """


    def __init__(self, tasks : Optional[Sequence[str]], corpus_dirs : Optional[Mapping[str, str]], 
                    split : Optional[Literal["test"] | Literal["train"] | Literal["dev"]] = "train", 
                    mode : Literal["train"] | Literal["eval"] = "train", limit : Optional[int] = None, 
                    words2i : Optional[Dict[str, int]] = None, device : Optional[Device] = torch.device("cpu"), 
                    vocab : Optional[Dict[str, Tuple[List[str], Dict[str, int]]]] = None, 
                    data : Optional[Dict[str, Tuple[Corpus, Dict[str, Corpus]]]] = None,
                    verbose : bool = False):
        """
        Initialising method for the ``DataLoader`` class.
        Automatically retrieves corpora specified in the
        ``tasks`` parameter. The retrieved corpora are shuffled. 
        If ``limit`` is specified, a certain number of sentences 
        from the result is retrieved resulting in random sampling.

        Parameters
        ----------
        tasks : None | Sequence[str]
            A sequence of tasks to retrieve tokens
            and features for. Must be None if data
            is provided through the ``data`` parameter.
        corpus_dirs : None | Mapping[str, str]
            Mapping from corpus names to their
            directories. Must be None if data
            is provided through the ``data`` parameter.
        split : Optional[Literal["test"] | Literal["train"] | Literal["dev"], default = "train]
            Which split to retrieve. Must be None if data
            is provided through the ``data`` parameter.
        mode : Literal["train"] | Literal["eval"]
            If "train", then ``vocab`` and
            ``tensor_features`` are computed from the retrieved
            data. If "eval", then ``vocab`` can be provided per
            parameter.
        limit : None | int, default = None, meaning no limit
            Number of sentences to sample per tasks.
        words2i : None | Dict[str, int], default = None
            Mapping from words to unique indices. Should
            be computed over all train corpora (including the 
            parsing task) and is therefore an optional argument.
        device : None | Device, default = torch.device("cpu")
            Device to use for tensors.
        vocab : None | Dict[str, Tuple[Corpus, Dict[str, Corpus]]], default = None
            Mapping from task names to their i2task
            and task2i mappings. Must be None if
            ``mode`` is "train".
        data : None | Dict[str, Tuple[Corpus, Dict[str, Corpus]]], default = None
            Data mapping, can be provided if extracted in
            before initialising the class. Then, the corpora
            are not retrieved a second time. Must be None,
            if ``task`` and ``corpus_dirs`` are specified.
        verbose : bool, default = False,
            If True, information about corpus extraction
            is written.

        """
        
        self.data : Dict[str, Tuple[Corpus, Dict[str, Corpus]]]
        """Mapping from corpus names to tuples (tokens, tasks), where
        tokens are the tokens strings of the corpus and tasks is
        a mapping from tasks based on the corpus to the features."""

        self.vocab : Optional[Dict[str, Tuple[List[str], Dict[str, int]]]] = None
        """Mapping from task names to their i2task
        and task2i mappings."""

        self.tensor_features : Optional[Dict[str, List[torch.Tensor]]] = None
        """Mapping from task names to their features
        as distinc indices in ``torch.Tensor`` format."""

        self.tensors : Optional[Dict[str, List[Tuple[Sentence, torch.Tensor]]]] = None
        """Mapping from task names to their corpus tokens
        in tensor representation."""

        self.tensors_copy : Optional[Dict[str, List[Tuple[Sentence, torch.Tensor]]]] = None
        """Copy of ``tensors`` for ``batch_stochastic_replacement``. One separate tensor
        for each task"""

        self.words2i : Optional[Dict[str, int]] = None
        """Mapping from words to unique indices."""
        
        self.num_sentences : Dict[str, int]
        """Mapping from tasks to the number of
        sentences that are annotated for the
        task."""

        self.num_tokens : Dict[str, int]
        """Mapping from tasks to the total number
        of tokens annotated for the task."""

        # generate corpus data if not provided
        if data is None:
            assert(tasks is not None)
            assert(corpus_dirs is not None)
            assert(split is not None)
            self.data = import_data(tasks, corpus_dirs, split, limit)
        else:
            assert(tasks is None)
            assert(corpus_dirs is None)
            assert(split is None)
            self.data = data

        self.num_sentences  = {task : len(corpus) for corpus, task_dict in self.data.values() for task in task_dict.keys()}
        self.num_tokens     = {task : sum([len(sentence) for sentence in corpus]) for corpus, task_dict in self.data.values() for task in task_dict.keys()}
        
        # initialise tensors if words2i mapping is provided
        if words2i is not None:
            assert(device is not None)
            self.init_tensors(words2i, device)
        
        if mode == "train":
            assert(device is not None)
            assert(vocab is None)
            features    = self.features
            self.vocab  = features_to_vocab(features, verbose)
            self.tensor_features = features_to_tensor_features(features, self.vocab, device)

        else:
            if vocab is not None:
                self.vocab = vocab

    @property
    def features(self) -> Dict[str, Corpus]:
        """
        Mapping from task names to their feature
        values saved in ``data``.
        """
        return {task : features for _, tasks in self.data.values() for task, features in tasks.items()}

    @property
    def token_lists(self) -> List[Corpus]:
        """
        Contains the string token lists (one per corpus)
        """
        return [d[0] for d in self.data.values()]

    @property
    def num_labels(self) -> Dict[str, int]:
        """
        Mapping from task name to the number of
        individual feature labels that occurr for it.
        """
        assert(self.vocab is not None)
        return {task : len(data[1]) for task, data in self.vocab.items()}
    
    def init_tensors(self, words2i : Dict[str, int], device : Device = torch.device("cpu")) -> None:
        """
        Initialises the ``tensors`` and the ``tensors_copy``
        attributes. For each corpus, the tokens are converted
        into indices in tensor format using the provided
        mapping. Then, each task is associated with a pointer
        to the corresponding tensor. ``tensors_copy`` is
        a deepcopy of ``tensors``.

        Parameters
        ----------
        words2i : Dict[str, int]
            Mapping from words to unique indices.
        device : Device, default = torch.device("cpu")
            Device to locate tensors on.

        Returns
        -------
        None
        """
        self.words2i = words2i
        self.tensors = data_to_tensor(self.data, words2i, device)
        self.tensors_copy = {task : copy.deepcopy(tensor) for task, tensor in self.tensors.items()}

    def batch_stochastic_replacement(self, pword : float, device : Device = torch.device("cpu")) -> None:
        """
        Performs in-place batch stochastic replacement for
        each task separately. In the ``tensors_copy``
        attribute each low frequency token (i.e. belonging 
        to the 2/3 least frequent tokens) is replaced
        with "<UNK>" with probability ``pword``.

        Parameters
        ----------
        pword : float
            Replacement probability
        device : Device, default = torch.device("cpu")
            Device to locate tensors on.

        Returns
        -------
        None
        """

        assert(self.words2i is not None)
        assert(self.tensors is not None)
        assert(self.tensors_copy is not None)

        for t, t_c in zip(self.tensors.values(), self.tensors_copy.values()):
            batch_stochastic_replacement(device, t, t_c, self.words2i, pword=pword)
    
    def get_sample(self, size : int, device) -> "DataLoader":
        """
        Retrieves a new DataLoader object from
        the current DataLoader with the data up until 
        index ``size``. The underlying objects in the memory 
        remain the same. ``words2i``, ``tensors`` and ``copy_tensors``
        are not transferred and must be re-initialised.
        Can be used to evaluate on only part of a corpus.

        Parameters
        ----------
        size : int
            Index to limit the corpus at.

        Returns
        -------
        DataLoader
            New DataLoader object with vocab and data
            specified.
        """
        return DataLoader(None, None, None, "eval", None, None, device, self.vocab, 
                            {corpus : (tokens[:size], {task : tags[:size] for task, tags in tasks.items()}) for corpus,
                            (tokens, tasks) in self.data.items()})
    
    def save_vocab(self, directory : str):
        """
        Saves the vocabulary information of each
        task from ``vocab`` in the specified directory.
        For each task, one separate file is created.

        Parameters
        ----------
        directory : str
            Directory to store the vocabulary information in.
            Suggestion: The model directory.

        Returns
        -------
        None
        """
        assert(self.vocab is not None)

        for task, (i2t, _) in self.vocab.items():
            save_dict(i2t, "{}/i2{}".format(directory, task))

    def load_vocab(self, directory : str):
        """
        Loades the vocabulary information of each
        task given in the ``features`` attribute
        from the speficied directory. For each task,
        one file with name "i2<task>" is expected.
        One feature per line. The line number represents
        the index. The reconstructed mappings
        from feature to index and from index to feature are
        saves as ``vocab`` attribute.

        Parameters
        ----------
        directory : str
            Directory to retrieve the vocabulary information from.

        Returns
        -------
        None
        """
        self.vocab = {}
        for task in self.features.keys():
            i2t, t2i = load_dict("{}/i2{}".format(directory, task))
            self.vocab[task] = (i2t, t2i)

