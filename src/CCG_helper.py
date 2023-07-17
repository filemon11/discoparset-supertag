"""
This module provides functions to
extract features from CCG categories.

Functions
----------
get_sketch
    Converts supertag into sketch feature.
supertag_to_head_arg
    Retrieves functor, argument direction and argument.
supertag_to_arg_list
    Converts supertag to argument list.
supertag_to_arg_struct
    Converts supertag to arg_struct feature.
create_near_action
    TODO
head_arg_functor
    Converts supertag to innermost functor,
    primary argument and functor features.

"""

from typing import List, Tuple
from parsing_typing import AnyCorpus, Corpus, AnySentence, Sentence

Path    = List[Tuple[str, Tuple[int, int]]]
Paths   = List[Path]

IndexPath   = List[List[int]]
IndexPaths  = List[IndexPath]

def get_sketch(supertag : str) -> str:
    """
    Converts supertag into sketch with
    all primtives replaced by "X".

    Parameters
    ----------
    supertag : str
        Supertag to convert

    Returns
    -------
    str
        Extracted sketch

    Examples
    -------
    >>> get_sketch("(S[dcl]\\NP)/NP")
    "(X\\X)/X"
    """
    
    sketch      : str = ""
    in_symbol   : bool = False

    for c in supertag:

        if c not in ("\\", "/", "(", ")"):
            if in_symbol:
                continue

            sketch += "X"
            in_symbol = True

        else:
            sketch += c
            in_symbol = False
    
    return sketch


def supertag_to_functor_arg(supertag : str) -> Tuple[str, str, str]:
    """
    Retrieves the functor, argument direction
    and argument of a supertag. If no argument 
    is present, argument direction
    defaults to "+" and argument to "".

    Parameters
    ----------
    supertag : str
        Supertag to convert

    Returns
    -------
    functor : str
        Extracted functor.
    direction : str
        Direction of argument,
        either "/" or "\\\\".
    argument : str
        Extracted argument.

    Examples
    -------
    >>> supertag_to_functor_arg("A/(B\\C)")
    ("A", "+", "B\\C")

    >>> supertag_to_functor_arg("(A\\B)\\C")
    ("A\\B", "-", "C")
    """

    functor     : str = supertag
    direction   : str = "+"
    arg         : str = ""

    level : int = 0
    for i, c in enumerate(supertag[::-1]):
        if c == ")":
            level += 1
            
        elif c == "(":
            level -= 1
            
        elif level == 0:
            if c == "/":
                direction   = "+"
                functor     = supertag[:-i-1]
                arg         = supertag[-i:]
                break

            if c == "\\":
                direction   = "-"
                functor     = supertag[:-i-1]
                arg         = supertag[-i:]
                break
    
    if len(functor) > 0 and functor[0] == "(" and functor[-1] == ")":
        functor = functor[1:-1]

    if len(arg) > 0 and arg[0] == "(" and arg[-1] == ")":
        arg = arg[1:-1]

    return functor, direction, arg

def supertag_to_arg_list(supertag: str, max_level : int = 50) -> List[Tuple[str, str, str]]:
    """
    Converts supertag to a list of functors, 
    argument directions and arguments retrieved 
    from the argument structure of the supertag.

    The method traverses from right to left through
    the supertag identifying arguments from outside
    inwards.

    Parameters
    ----------
    supertag : str
        Supertag to convert
    max_level : int, default = 50
        Maximum number of arguments
        retrieved from supertag.

    Returns
    -------
    List[Tuple[str, str, str]]
        List of functors, argument directions
        and arguments.

    Examples
    -------
    >>> supertag_to_arg_list("(((A/B)\C)/D)\A")
    [("((A/B)\C)/D", "-", "A"), ("(A/B)\C", "+", "D"), ...]
    """

    arg         : str = " "
    arg_list    : List[Tuple[str, str, str]] = []

    level       : int = 0
    
    while arg != "" and level < max_level:
        supertag, direction, arg = supertag_to_functor_arg(supertag)

        if arg != "":
            arg_list.append((supertag, direction, arg))
        
        level += 1
    
    return arg_list

def supertag_to_arg_struct(supertag : str, max_level : int = 50) -> str:
    """
    Converts supertag to a feature representing
    the number and direction of the supertag's
    arguments from outside inwards.

    Parameters
    ----------
    supertag : str
        Supertag to convert
    max_level : int, default = 50
        Maximum number of arguments
        retrieved from supertag.

    Returns
    -------
    str
        A string sequence where every
        argument is represented with its
        direction (+, -) and an X starting
        from the right of a supertag
        going left (inwards).

    Examples
    -------
    >>> supertag_to_arg_struct("(((A/B)\C)/D)\A")
    "-X+X-X+X"
    """

    arg_list : List[Tuple[str, str, str]] = supertag_to_arg_list(supertag)

    return "".join(["{}X".format(arg[1]) for arg in arg_list])


def create_near_action(sentence : AnySentence) -> List[str]:
    """TODO"""
    def _near_left(arg_list_left : List[Tuple[str, str, str]], arg_list : List[Tuple[str, str, str]], supertag_left : str) -> str:
        
        arguments : List[Tuple[str, str, str]] = arg_list

        arguments_left : List[Tuple[str, str, str]] = arg_list_left

        if not arguments:
            pass

        elif arguments[0][1] == "-":
            if arguments[0][2] == supertag_left:
                return "Arg"

            for left_argument in arguments_left:
                if left_argument[0] == arguments[0][2] and left_argument[1] == "-":
                    return "B"

        elif len(arguments) > 1:
            for inner_arg in arguments[1:]:
                if inner_arg[2] == supertag_left:
                    return "DArg"

                for left_argument in arguments_left:
                    if left_argument[0] == inner_arg[2] and left_argument[1] == "-":
                        return "DB"
        
        return "n"

    def _near_right(arg_list : List[Tuple[str, str, str]], arg_list_right : List[Tuple[str, str, str]], supertag_right : str) -> str:
        
        arguments : List[Tuple[str, str, str]] = arg_list

        arguments_right : List[Tuple[str, str, str]] = arg_list_right
        
        if not arguments:
            pass

        elif arguments[0][1] == "+":
            if arguments[0][2] == supertag_right:
                return "Arg"

            for right_argument in arguments_right:
                if right_argument[0] == arguments[0][2] and right_argument[1] == "+":
                    return "B"

        elif len(arguments) > 1:
            for inner_arg in arguments[1:]:
                if inner_arg[2] == supertag_right:
                    return "DArg"
                for right_argument in arguments_right:
                    if right_argument[0] == inner_arg[2] and right_argument[1] == "+":
                        return "DB"
        
        return "n"

    def _near(arg_list_left : List[Tuple[str, str, str]], arg_list : List[Tuple[str, str, str]], arg_list_right : List[Tuple[str, str, str]], supertag_left : str, supertag_right : str) -> str:
        return "{},{}".format(_near_left(arg_list_left, arg_list, supertag_left), _near_right(arg_list, arg_list_right, supertag_right))
    
    sentence_boundary : str = ""
    sentence_padded : List[str] = [sentence_boundary]
    sentence_padded.extend(sentence)
    sentence_padded.append(sentence_boundary)

    arg_lists : List[List[Tuple[str, str, str]]] = [supertag_to_arg_list(supertag) for supertag in sentence_padded]

    near_action : List[str] = []

    for arg_list_left, arg_list, arg_list_right, supertag_left, supertag_right in zip(arg_lists, arg_lists[1:], arg_lists[2:], sentence_padded, sentence_padded[2:]):
        near_action.append(_near(arg_list_left, arg_list, arg_list_right, supertag_left, supertag_right))
    
    assert(len(near_action) == len(sentence))
    
    return near_action


def head_arg_functor(supertags : AnyCorpus) -> Tuple[Corpus, Corpus, Corpus]:
    """
    Extracts the innermost return category,
    the (first) argument as well as its direction
    and the functor form each entry in a list of
    lists of CCG supertags.

    Parameters
    ----------
    supertags : Sequence[Sequence[str]]
        Corpus of supertags as strings.

    Returns
    -------
    heads : List[List[str]]
        The innermost return type of
        each supertag.
    args : List[List[str]]]
        The argument and argument
        direction of each supertag.
    functor : List[List[str]]]
        The functor of each supertag.

    Examples
    -------
    >>> head_arg_functor([["(A/B)\\(C/D)"]])
    ([["A"]], [["-C/D"]], [["A/B"]])
    """

    heads    : Corpus = []
    args     : Corpus = []
    functors : Corpus = []

    for sentence in supertags:
        sentence_head       : Sentence = []
        sentence_arg        : Sentence = []
        sentence_functor    : Sentence = []

        for tag in sentence:
            arg_list : List[Tuple[str, str, str]] = supertag_to_arg_list(tag)

            if len(arg_list) > 0:

                sentence_head.append(arg_list[-1][0])
                sentence_arg.append(arg_list[0][1]+arg_list[0][2])
                sentence_functor.append(arg_list[0][0])
            
            else:
                sentence_head.append("")
                sentence_arg.append("")
                sentence_functor.append("")
            
        heads.append(sentence_head)
        args.append(sentence_arg)
        functors.append(sentence_functor)
    
    return heads, args, functors