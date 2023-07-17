"""
Little helper functions for easier handling
of nested lists.

Functions
----------
corpus_apply
    Apply a function to a list of lists of elements.
intuple
    Move elements in separate lists of lists into
    one common structure where they are saved in an
    inner tuple. 
outtuple
    Reverse intuple.
listify
    Convert tuples of tuples of elements to a list of
    lists of elements.
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
def intuple(corpora : Tuple[Iterable[Iterable[V]], Iterable[Iterable[Y]]]) -> Tuple[Tuple[Tuple[V, Y], ...], ...]:
    ...

@overload
def intuple(corpora : Tuple[Iterable[Iterable[V]], Iterable[Iterable[Y]], Iterable[Iterable[Z]]]) -> Tuple[Tuple[Tuple[V, Y, Z], ...], ...]:
    ...

@overload
def intuple(corpora : Tuple[Iterable[Iterable[V]], ...]) -> Tuple[Tuple[Tuple[V, ...], ...], ...]:
    ...

def intuple(corpora):
    """"
    Converts a variable number of lists of lists 
    of elements into a single nested tuple with
    parallel elements placed together in a common
    inner tuple.

    Returns a tuple of tuples instead of a list
    of lists for safe typing.

    Parameters
    ----------
    corpus : Tuple[Iterable[Iterable[V]],...^n]
        The nested iterables to convert.

    Returns
    -------
    Tuple[Tuple[Tuple[V,...^n],...],...]
        The resulting tuple of tuples of tuples
        where the arity of the inner tuples corresponds
        to the number of input sequences.

    Examples
    -------
    >>> intuple(([["I", "hope"],["You", "go"]], [["1", "4"],["3", "2"]]))
    ((("I", "1"),("hope", "4")), (("You", "3"), ("go", "2")))
    
    See Also
    -------
    outtuple
    """
    return [list(zip(*sentence)) for sentence in zip(*corpora)]


@overload
def outtuple(tupled_corpus : Iterable[Iterable[Tuple[V]]]) -> Tuple[Tuple[Tuple[V], ...]]:
    ...

@overload
def outtuple(tupled_corpus : Iterable[Iterable[Tuple[V, Y]]]) -> Tuple[Tuple[Tuple[V, ...], ...], Tuple[Tuple[Y, ...], ...]]:
    ...

@overload
def outtuple(tupled_corpus : Iterable[Iterable[Tuple[V, Y, Z]]]) -> Tuple[Tuple[Tuple[V, ...], ...], Tuple[Tuple[Y, ...], ...], Tuple[Tuple[Z, ...], ...]]:
    ...

@overload
def outtuple(tupled_corpus : Iterable[Iterable[Tuple[V, ...]]]) -> Tuple[Tuple[Tuple[V, ...], ...], ...]:
    ...

def outtuple(tupled_corpus):
    """"
    Converts a list of lists of tuples with a uniform
    number n of elements to n separate nested
    sequences.

    Returns a tuples of tuples instead of a lists
    of lists for safe typing.

    Parameters
    ----------
    tupled_corpus : Iterable[Iterable[Tuple[V,...^n],...],...]
        The nested iterables with inner tuple to convert.

    Returns
    -------
    Tuple[Tuple[Tuple[V, ...]],...^n]
        The resulting tuples of tuples
        where the number of output tuples corresponds
        to the arity of the input inner tuples.

    Examples
    -------
    >>> outtuple([[("I", "1"),("hope", "4")], [("You", "3"), ("go", "2")]])
    ((("I", "hope"),("You", "go")), (("1", "4"),("3", "2")))

    See Also
    -------
    intuple
    """
    return tuple([list(element) for element in list(zip(*[[list(l) for l in list(zip(*sentence))] for sentence in tupled_corpus]))])


def _listify(t : Tuple[Tuple[V, ...], ...]) -> List[List[V]]:
    """
    Converts a tuple of tuples of elements
    to a list of lists of these elements.
    """
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
    """"
    Converts one or more tuples of tuples of elements
    into lists of lists of elements.

    Due to typing issues there is no specification
    of separate types for each entry.

    Parameters
    ----------
    t1 : Tuple[Tuple[V, ...],...]
        Tuple to convert.
    t2 : None | Tuple[Tuple[V, ...],...], default = None
        Tuple to convert.
    t3 : None | Tuple[Tuple[V, ...],...], default = None
        Tuple to convert.
    *args : Tuple[Tuple[V, ...], ...]
        Additional tuples to convert.

    Attention: do not refer to the
    arguments by name. They only exist
    for mypy to infer types for up to
    three levels.

    Returns
    -------
    List[List[V]]   if only one argument was given
        The converted list.
    Tuple[List[List[V]], ...]  
        The converted lists. Every tuple entry
        corresponds to an input nested tuple.
    """
    if t2 is None and t3 is None:
        return _listify(t1)

    elif t3 is None:
        return (_listify(t1), _listify(t2))

    elif len(args) == 0:
        return (_listify(t1), _listify(t2), _listify(t3))

    else:
        return (_listify(t1), _listify(t2), _listify(t3)) + tuple(_listify(t) for t in args)
    