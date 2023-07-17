"""TODO


corpus_apply
    Apply a function to a list of lists of elements.

"""
from typing import Iterable, Callable, List, TypeVar, Tuple, overload

V = TypeVar("V")
Y = TypeVar("Y")
Z = TypeVar("Z")

C = TypeVar("C", bound = Iterable)
B = TypeVar("B", bound = Iterable)

def corpus_apply(corpus : Iterable[Iterable[V]], f : Callable[[V],Y], *args, **kwargs) -> List[List[Y]]:
    """
    Applies a function to each value of an iterable
    of an iterable.

    Parameters
    ----------
    corpus : Iterable[Iterable[V]]
        Data to apply the function on.
    f : Callable[[V],Y]
        Function to apply.
    *args
        Optional arguments are forwarded
        as parameters of ``f``.
    **kwargs
        Optional keyword arguments are
        forwarded as parameters of ``f``.

    Returns
    -------
    List[List[Y]]
        The resulting list of lists.
    """
    return [[f(entry, *args, **kwargs) for entry in sentence] for sentence in corpus]


@overload
def intuple(corpora : Tuple[Iterable[Iterable[V]]]) -> Tuple[Tuple[Tuple[V], ...], ...]:
    ...

@overload
def intuple(corpora : Tuple[Iterable[Iterable[V]], Iterable[Iterable[V]]]) -> Tuple[Tuple[Tuple[V, V], ...], ...]:
    ...

@overload
def intuple(corpora : Tuple[Iterable[Iterable[V]], Iterable[Iterable[V]], Iterable[Iterable[V]]]) -> Tuple[Tuple[Tuple[V, V, V], ...], ...]:
    ...

@overload
def intuple(corpora : Tuple[Iterable[Iterable[V]], ...]) -> Tuple[Tuple[Tuple[V, ...], ...], ...]:
    ...

def intuple(corpora):
    """TODO"""
    return [list(zip(*sentence)) for sentence in zip(*corpora)]

@overload
def outtuple(corpora : Iterable[Iterable[Tuple[V]]]) -> Tuple[Tuple[Tuple[V], ...]]:
    ...

@overload
def outtuple(corpora : Iterable[Iterable[Tuple[V, V]]]) -> Tuple[Tuple[Tuple[V, ...], ...], Tuple[Tuple[V, ...], ...]]:
    ...

@overload
def outtuple(corpora : Iterable[Iterable[Tuple[V, V, V]]]) -> Tuple[Tuple[Tuple[V, ...], ...], Tuple[Tuple[V, ...], ...], Tuple[Tuple[V, ...], ...]]:
    ...

@overload
def outtuple(corpora : Iterable[Iterable[Tuple[V, ...]]]) -> Tuple[Tuple[Tuple[V, ...], ...], ...]:
    ...

def outtuple(tupled_corpus):
    """TODO"""
    return tuple([list(element) for element in list(zip(*[[list(l) for l in list(zip(*sentence))] for sentence in tupled_corpus]))])

def _listify(t : Tuple[Tuple[V, ...], ...]) -> List[List[V]]:
    return [[j for j in i] for i in t]

@overload
def listify(t1 : Tuple[Tuple[V, ...], ...], t2 : None = None, t3 : None = None) -> List[List[V]]:
    ...

@overload
def listify(t1 : Tuple[Tuple[V, ...], ...], t2 : Tuple[Tuple[V, ...], ...], t3 : None = None) -> Tuple[List[List[V]], List[List[V]]]:
    ...

@overload
def listify(t1 : Tuple[Tuple[V, ...], ...], t2 : Tuple[Tuple[V, ...], ...], t3 : Tuple[Tuple[V, ...], ...]) -> Tuple[List[List[V]], List[List[V]], List[List[V]]]:
    ...

@overload
def listify(t1 : Tuple[Tuple[V, ...], ...], t2 : Tuple[Tuple[V, ...], ...], t3 : Tuple[Tuple[V, ...], ...], *args : Tuple[Tuple[V, ...], ...]) -> Tuple[List[List[V]], ...]:
    ...

def listify(t1, t2 = None, t3 = None, *args):
    if t2 is None and t3 is None:
        return _listify(t1)

    elif t3 is None:
        return (_listify(t1), _listify(t2))

    elif len(args) == 0:
        return (_listify(t1), _listify(t2), _listify(t3))

    else:
        return (_listify(t1), _listify(t2), _listify(t3)) + tuple(_listify(t) for t in args)
    

@overload
def mlistify(ts : Tuple[Tuple[Tuple[V, ...], ...]]) -> Tuple[List[List[V]]]:
    ...

@overload
def mlistify(ts : Tuple[Tuple[Tuple[V, ...], ...], Tuple[Tuple[V, ...], ...]]) -> Tuple[List[List[V]], List[List[V]]]:
    ...

@overload
def mlistify(ts : Tuple[Tuple[Tuple[V, ...], ...], Tuple[Tuple[V, ...], ...], Tuple[Tuple[V, ...], ...]]) -> Tuple[List[List[V]], List[List[V]], List[List[V]]]:
    ...

def mlistify(ts):
    return tuple(listify(t) for t in ts)