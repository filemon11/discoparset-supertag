"""
This module imports data from the
supertag annotated CCGrebank corpus.

Classes
----------

Functions
----------
construct_tree
import_auto
component_generator
simplify
import_parg
import_complex
import_complex_multi
remove_UB
get_long_distance_sketch

Constants
----------
SPLITS

"""

from collections import defaultdict
from helpers import corpus_apply, intuple, outtuple, listify
import os

from typing import List, Tuple, Dict, Sequence, Optional, Generator

from parsing_typing import Corpus, AnyCorpus, Sentence, AnySentence

from types import MappingProxyType

# standard splits for ccgrebank import
SPLITS : MappingProxyType[str, Tuple[int, int]] = MappingProxyType({"train" : (200,2200), 
                                                                    "dev" : (2201,2300), 
                                                                    "test" : (2301,2400)})
"""TODO"""

class Tree:
    """
    Represents a node in an ordered tree that can have
    zero or one parent and zero or more children.

    Attributes
    ----------
    category : str

    children : List[Tree]
        Zero or more children of the node.
    parent : None | Tree
        Zero or one parent of the node.
    index : Tuple[int, int]
        Start and end index of the span the node dominates.

    Methods
    -------
    get_leaves()
        Retrieves the list of leaves dominated
        by the tree in their tree-order.
    set_index(start_index)
        Method to assign each node its index
        via post-order traversal starting with
        the root.
    
    See Also
    -------
    Leaf
    """
    def __init__(self, category : str, children : List["Tree"] = [], parent : "None | Tree" = None):
        '''
        Initialising method for the ``Tree`` class.
        Represents a node in an ordered tree that can have
        zero or one parent and zero or more children.

        Parameters
        ----------
        category : str
            Supertag assignment of the node.
        children : List[Tree], default = []
            Daughters of the node. Automatically
            assigns the newly initialised node
            as parent of the given daughters.
        parent : None | Tree, default = None
            Parent of the node.
        '''
        
        self.category : str = category

        self.children : List[Tree] = children

        for child in self.children:
            child.parent = self

        self.parent : "None | Tree" = parent

        self.index : Optional[Tuple[int, int]] = None
        "Start and end index of the span the node dominates."

    def get_leaves(self) -> List["Leaf"]:
        '''
        The leaf nodes of this node, i.e.
        the nodes with a path to this node
        that have no daughters themselves.

        Beware: Performs recursive function calls.
        
        Returns
        ----------
        List[Leaf]
            The list of leaves.
        '''

        leaves : List[Leaf] = []
        for child in self.children:
            leaves += child.get_leaves()
        
        return leaves

    def set_index(self, start_index : int = 0) -> int:
        '''
        This method is used to assign this node
        an a start and an end index determining the
        span of the leaves it dominates after constructing 
        the tree. The numbering is done by post-order recursive
        traversal through the tree starting at the root.

        Parameters
        ----------
        start_index : int, default = 0
            Right index of the preceding node.
        
        Returns
        ----------
        int
            Right index of this node.
        '''

        end_index = start_index

        for child in self.children:
            end_index = child.set_index(end_index)

        self.index = (start_index, end_index)

        return end_index


class Leaf(Tree):
    """
    Represents a leaf node in an ordered tree that can have
    zero or one parent and no children. Is a subclass
    of ``Tree``.

    Attributes
    ----------
    category : str
        The supertag.
    word : str
        The token.
    pred_arg_cat : str
        The supertag including co-indices.
    parent : None | Tree
        Inherited, parent of the leaf.
    children : List[Tree]
        Inherited, children of the leaf. Set to an
        empty list at ``__init__``. Should not be changed.
    index : Tuple[int, int]
        Inherited, (n, n+1) if n is the index of
        the leaf.

    Methods
    -------
    get_leaves()
        Retrieves a list with itself as
        the only member.
    set_index(start_index)
        Method to assign each node its index
        via post-order traversal starting with
        the root.
    get_scope(limit)
        Retrieves the indices of the leaf's 
        predecessors and computes their relative
        position to the leaf's index.

    See Also
    -------
    Tree
    """
    def __init__(self, category : str, word : str, pred_arg_cat : str, parent : "None | Tree" = None):
        '''
        Initialising method for the ``Leaf`` class.
        Represents a leaf in an ordered tree that can have
        zero or one parent and zero children. The leaf
        carries additional information associated with
        a token in the sentence.

        Parameters
        ----------
        category : str
            The supertag.
        word : str
            The token associated with the leaf.
        pred_arg_cat : str
            The supertag including co-indices.
        parent : None | Tree, default = None
            Parent of the leaf.
        '''

        super().__init__(category, parent = parent)
        
        self.word : str = word
        self.pred_arg_cat : str = pred_arg_cat

    def get_leaves(self) -> List["Leaf"]:
        '''
        This method is used for recursive
        construction of a list of leaves
        of a node and therefore returns a list
        with only itself. It is called by its
        parent node and overrides its superclass
        ``get_leaves`` method.

        Returns
        ----------
        List[Leaf]
            A list where the only element is the object
            this method was called from.
        '''
        return [self]

    def set_index(self, start_index = 0) -> int:
        '''
        This method is used to assign every node in
        a tree a start and an end index determining the
        span of the leaves it dominates after constructing 
        the tree. This method overrides its superclass
        ``set_index`` method since at a leaf, the index
        counter should be increased by one.

        Parameters
        ----------
        start_index : int, default = 0
            Right index of the preceding node.
        
        Returns
        ----------
        int
            start_index + 1
        '''

        self.index = (start_index, start_index + 1)

        return start_index + 1

    def get_scope(self, depth_limit : int = -1, step_limit : int = 10) -> List[Tuple[int, int]]:
        '''
        This method generates a list of ranges this leaf's
        predecessors span. The first element in the list
        is the span of leaf's parent, the second is the span
        of the parent's parent and so forth. The spans
        are positions relative to the leaf's index (e.g.
        -2 for left index n-2 where n is the leaf's index).
                
        Parameters
        ----------
        depth_limit : int, default = -1, meaning no limit
            Parent level up to which to compute spans.
        step_limit : int, default = 10
            Cap relative distance of range borders to the
            leaf's index at this value. If -1, no capping.

        Returns
        ----------
        List[Tuple[int, int]]
            A list of ranges spanned by the leaf's
            predecessors from bottom to top realtive to the
            leaf's index.
        '''
        
        scope_list : List[Tuple[int, int]] = []

        parent : None | Tree = self.parent

        depth : int = 0

        while parent is not None and (depth_limit == -1 or depth < depth_limit):
            
            assert(parent.index is not None)
            assert(self.index is not None)

            left_index : int = parent.index[0] - self.index[0]      # left index of predecessor relative to this node

            # limit at step_limit
            if abs(left_index) >= step_limit and step_limit != -1:
                left_index = (left_index > 0) * step_limit + (left_index < 0) * -step_limit
            
            right_index : int = parent.index[1] - self.index[0]     # right index of predecessor relative to this node

            # limit at step_limit
            if abs(right_index) >= step_limit and step_limit != -1:
                right_index = (right_index > 0) * step_limit + (right_index < 0) * -step_limit

            scope_list.append((left_index, right_index))

            parent = parent.parent

            depth += 1

        return scope_list
    
def construct_tree(treestring : str) -> Tree:
    '''
    Constructs a tree datastructure out of
    a string bracketed tree in CCGrebank.

    Parameters
    ----------
    treestring : str
        Tree in bracketed string format.

    Returns
    -------
    Tree
        Retrieved tree root.
    '''

    def _construct_tree(treestring : str) -> Tree:
        '''
        Recursively constructs a tree by retrieving the
        node data and then constructing a node for each child
        (and so forth).
        '''

        def _construct_children(children_string : str) -> List[Tree]:
            '''
            Identifies the substring for each child and calls calls
            ``_construct_tree`` for each child.
            '''
            children_strings : List[str] = []

            paren_count : int = 0
            element_start : int = 0

            for i, c in enumerate(children_string):
                if c == "(":
                    paren_count += 1

                    if paren_count == 1:
                        element_start = i

                elif c == ")":
                    paren_count -= 1

                    if paren_count == 0:
                        children_strings.append(children_string[element_start : i+1])

            return [_construct_tree(child) for child in children_strings]


        treestring = treestring[2:-1]

        node : str = ""

        for i, c in enumerate(treestring):
            if c == ">":
                treestring = treestring[i+2:]
                break
            else:
                node += c

        node_parts : List[str] = node.split()

        # "L" denotes a lexical assignment and thus a leaf.
        if node_parts[0] == "L":
            return Leaf(node_parts[1], node_parts[4], node_parts[5])    # 1: supertag, 4: token, 5: supertags including co-indexes

        return Tree(node_parts[1], _construct_children(treestring))


    constructed_tree : Tree = _construct_tree(treestring)

    constructed_tree.set_index(0)       # Initialise indexing

    return constructed_tree

def import_auto(filename : str) -> Tuple[Corpus, Corpus, Corpus]:
    '''
    Imports tokens, supertags and scopes from
    machine-readable derivation file from the
    CCGrebank (CCG annotated Penn Treebank).

    Parameters
    ----------
    filename : str
        Path and filename.

    Returns
    -------
    tokens : List[List[str]]
        Retrieved tokens.
    supertags : List[List[str]]
        Supertag assignments for the tokens.
    scopes : List[List[str]]
        A list of positions of the
        boundaries of the ranges spanned by
        the predecessors of a leaf relative
        to the leaf converted into a string.
    '''
    tokens      : Corpus = []
    supertags   : Corpus = []
    scopes      : Corpus = []

    with open(filename, 'r') as file:

        # one sentence per line
        for line in file:
            if line[0:2] == "ID" or line[0] != "(":
                pass

            else:
                sen_tokens      : Sentence = []
                sen_supertags   : Sentence = []

                first_n : int = 0

                for n, c in enumerate(line):
                    if c == "<":
                        first_n = n

                    elif c == ">":
                        element : str = line[first_n+1:n]

                        element_parts : List[str] = element.split()

                        if len(element_parts) > 4:
                            sen_tokens.append(element_parts[4])
                            sen_supertags.append(remove_UB(simplify(element_parts[5])))        # Removes coindices and mediation indicators

                supertags.append(sen_supertags)
                tokens.append(sen_tokens)

                # Convert scope list into a feature by concatenating the ranges.
                # Since the token and supertag extraction do not rely of the recursive
                # Tree construction algorithm, this can be commented out for faster processing if 
                # scope features are not needed.
                tree : Tree = construct_tree(line)
                scopes.append(["_".join([str(s) for s in leaf.get_scope(2)]) for leaf in tree.get_leaves()])

    assert(len(tokens) == len(supertags) == len(scopes))
    return tokens, supertags, scopes


def simplify(element : str, remove_indices : bool = True) -> str:
    '''
    Converts supertag co-indices into into into variables
    starting at 0. If Optionally removes co-indices entirely.

    Parameters
    ----------
    element : str
        Supertag to convert.
    remove_indices : bool
        Whether to remove co-indices entirely,

    Returns
    -------
    str
        Converted supertag.
    '''

    def _component_generator(element : str, remove_indices : bool = True) -> Generator[Tuple[bool, str], None, None]:
        '''
        Splits the supertag at indices and yields the resulting
        substrings in a lazy manner from left to right indicating
        if it yields an index via a boolean value. Omits underscores.
        If ``remove_indices`` is set to True, indices are not yielded.
        '''
        NUMBERS = "0123456789"

        current_component   : str   = ""
        in_number           : bool  = False

        for c in element:
            if c == "_":
                if not remove_indices:
                    current_component += c

                yield (False, current_component)

                current_component = ""
                in_number = True

            elif in_number and not c in NUMBERS:

                if not remove_indices:
                    yield (True, current_component)

                current_component = c
                in_number = False

            elif c not in NUMBERS:
                current_component += c

        if (not in_number and remove_indices) or not remove_indices:
            yield (in_number, current_component)


    if len(element.split("_")) == 1:
        return element
    
    simple_element : List[str] = []
    num_to_i : defaultdict[str, str] = defaultdict(lambda : str(len(num_to_i)))

    for is_number, component in _component_generator(element, remove_indices):      # iterate through components, i.e. numbers and other substrings
        
        if is_number:                                   # if the component is a number, replace it with its variable
            simple_element.append(num_to_i[component])
        
        else:
            simple_element.append(component)
    
    return "".join(simple_element)


def import_parg(filename : str, limit : int = 5) -> Tuple[List[List[str]], List[List[str]]]:
    '''
    Imports predicate-argument structure 
    file from CCGrebank (CCG annotated Penn Treebank) 
    corpus and converts the information into
    two different features:
        leftaction : TODO
        rightaction : TODO

    Parameters
    ----------
    filename : str
        Path and filename.
    limit : int, default = 5
        Window around word to capture
        relationships in.

    Returns
    -------
    action_left : List[List[str]]
        TODO
    action_right : List[List[str]]
        TODO
    '''

    output_list : Tuple[Corpus, Corpus] = ([], [])

    with open(filename, 'r') as file:

        sentence_number : int = 0

        sentence_left : List[Corpus] = []
        sentence_right : List[Corpus] = []

        for line in file:
            if line[0:2] == "<s":           # When a new sentence header is oberseved, the sentence number variable is incremented
                sentence_number += 1
                sentence_left.append([["0"] for _ in range(int(line.split()[-1]) + 1)])
                sentence_right.append([["0"] for _ in range(int(line.split()[-1]) + 1)])
                pass
            
            elif line == "<\\s>\n" or line == "<\\s> \n":
                pass

            else:
                line_components : List[str] = line.split(" 	 ")
                
                arg_num : str = line_components[3]
                relative_position : int = int(line_components[0]) - int(line_components[1])

                if relative_position < 0:
                    if relative_position > -limit:
                        sentence_left[sentence_number -1][int(line_components[1])].append(arg_num + ":" + str(relative_position))
                else:
                    if relative_position < limit:
                        sentence_right[sentence_number - 1][int(line_components[1])].append(arg_num + ":" + str(relative_position))

        output_list = ( [["_".join(word) for word in sentence] for sentence in sentence_left if len(sentence) > 1 ], \
                        [["_".join(word) for word in sentence] for sentence in sentence_right if len(sentence) > 1 ])
        

    return output_list

def import_complex(num : int, data_dir : str) -> Tuple[Corpus, Corpus, Corpus, Corpus, Corpus, Corpus]:
    '''
    Imports a file from the CCGrebank 
    (CCG annotated Penn Treebank) 
    corpus and converts the information into
    five different features:
        supertags : lexical category assignments
        scopes : range boundaries of the supertag's predecessors in the derivation  
        leftaction : TODO,
        rightaction : TODO
        dependency : obsolete

    Parameters
    ----------
    num : int
        Numbers of the file in the corpus data
        to extract. The file contains a variable
        number of sentences.
    data_dir : str
        Directory of the corpus

    Returns
    -------
    tokens : List[List[str]]
        Retrieved tokens.
    supertags : List[List[str]]
        Supertag assignments.
    scopes : List[List[str]]
        range boundaries of the supertag's predecessors in the derivation 
    action_left : List[List[str]]
        TODO
    action_right : List[List[str]]
        TODO
    dependency : List[List[str]]
        obsolete
    '''

    str_num : str = "0" * (4  - len(str(num))) + str(num)       # Convert the file number into a four-diget number with
                                                                # prepended zeros in string format.

    str_section : str = str_num[0:2]                            # The first two digets represent the section number.
                                                                # DepCCG sorts the files into section folders with
                                                                # a certain number of multiple-sentence files.

    auto_dir : str = f"{data_dir}/AUTO/{str_section}/wsj_{str_num}.auto"
    parg_dir : str = f"{data_dir}/PARG/{str_section}/wsj_{str_num}.parg"

    if not os.path.exists(auto_dir) or not os.path.exists(parg_dir):
        return ([],[],[],[],[],[])

    auto = listify(intuple(import_auto(auto_dir)))  # machine-readable CCG derivation files
    parg = listify(intuple(import_parg(parg_dir)))  # predicate-argument structure file

    max_index : int = max(len(auto), len(parg))

    # This part deals with aligning problems due to PARG containing empty
    # sentences where sentences from the PTB were not translated
    # while they were simply removed from AUTO without a trace.
    for sen_num in range(max_index):
        if sen_num >= max_index:
            break

        if sen_num >= len(auto):
            parg.pop(sen_num)
            break

        elif sen_num >= len(parg):
            auto.pop(sen_num)
            break
        
        if not (len(auto[sen_num]) == len(parg[sen_num])):

            if len(auto[sen_num]) < len(parg[sen_num]):
                auto.pop(sen_num)
                max_index -= 1

    assert(len(auto) == len(parg))

    for i, j in zip(auto, parg):
        assert(len(i) == len(j))

    tokens      : Corpus
    supertags   : Corpus
    scopes      : Corpus

    tokens, supertags, scopes = listify(*outtuple(auto))

    leftaction  : Corpus
    rightaction : Corpus

    leftaction, rightaction = listify(*outtuple(parg))

    # obsolete
    dependency : Corpus = corpus_apply(supertags, get_long_distance_sketch)

    return tokens, supertags, scopes, leftaction, rightaction, dependency
    

def import_complex_multi(nums : Sequence[int], data_dir : str) -> Tuple[Corpus, Corpus, Corpus, Corpus, Corpus, Corpus]:
    '''
    Imports a sequence of files from 
    the CCGrebank (CCG annotated Penn Treebank) 
    corpus and converts the information into
    five different features:
        supertags : lexical category assignments
        scopes : range boundaries of the supertag's predecessors in the derivation 
        leftaction : TODO,
        rightaction : TODO
        dependency : obsolete

    Parameters
    ----------
    nums : Sequence[int]
        Numbers of the files in the corpus data
        to extract. Each file contains several
        sentences.
    data_dir : str
        Directory of the corpus

    Returns
    -------
    tokens : List[List[str]]
        Retrieved tokens.
    supertags : List[List[str]]
        Supertag assignments.
    scopes : List[List[str]]
        range boundaries of the supertag's predecessors in the derivation 
    action_left : List[List[str]]
        TODO
    action_right : List[List[str]]
        TODO
    dependency : List[List[str]]
        obsolete
    '''

    supertags   : Corpus = []
    tokens      : Corpus = []
    action_left : Corpus = []
    action_right    : Corpus = []
    scopes      : Corpus = []
    dependency  : Corpus = []
   
    # For each file number, perform extraction
    for n in nums:
        
        n_tokens, n_supertags, n_scopes, n_action_left, n_action_right, n_dependency = import_complex(n, data_dir)

        supertags   += n_supertags
        tokens      += n_tokens
        action_left += n_action_left
        action_right    += n_action_right
        scopes      += n_scopes
        dependency  += n_dependency

    # assert that the lengths all correspond to each other
    assert(len(supertags) == len(tokens) == len(scopes) == len(action_left) == len(action_right) == len(dependency))

    return tokens, supertags, scopes, action_left, action_right, dependency


# information:
# head position relative to argument (e.g. -1, +3, ...))
# argument number

def remove_UB(supertag : str) -> str:
    '''
    Remove mediation indicator for locally mediated (":U")
    and for long-range ("B") dependencies from
    supertag to retrieve clean supertags.

    Parameters
    ----------
    supertag : str
        Supertag to convert.

    Returns
    -------
    str
        Supertag with removed mediation information
    '''
    in_UB : bool = False
    final_supertag : str = ""

    for c in supertag:

        if in_UB:
            in_UB = False
            continue
        
        elif c == ":":
            in_UB = True
            continue

        else:
            final_supertag += c

    return final_supertag 


def get_long_distance_sketch(supertag : str) -> str:
    '''
    Obsolete

    Converts supertag into sketch with
    all elements replaced by "X".
    Multi-character symbols are treated
    the same as one-character symbols.

    Parameters
    ----------
    supertag : str
        Supertag to convert

    Returns
    -------
    str
        Extracted sketch
    '''
    
    has_dependency : bool = False

    sketch      : str = ""
    in_symbol   : bool = False
    in_UB       : bool = False

    for c in supertag:
        
        if in_UB:
            sketch += c
            in_UB = False

        elif c == ":":
            in_UB = True
            has_dependency = True
            sketch += c

        elif c not in ("\\", "/", "(", ")"):
            if in_symbol:
                continue

            sketch += "X"
            in_symbol = True

        else:
            sketch += c
            in_symbol = False
    
    if has_dependency:
        return sketch
    else:
        return "X"