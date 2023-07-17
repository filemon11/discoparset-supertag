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


from helpers import corpus_apply, intuple, outtuple, listify, mlistify

# standard splits for ccgrebank import
from typing import List, Tuple, Dict, Sequence, Union, Literal, Hashable, Set
import os

from parsing_typing import Corpus, AnyCorpus, Sentence, AnySentence

from types import MappingProxyType

SPLITS : MappingProxyType[str, Tuple[int, int]] = MappingProxyType({"train" : (200,2200), 
                                                                    "dev" : (2201,2300), 
                                                                    "test" : (2301,2400)})
"""TODO"""

class Tree:
    def __init__(self, category : str, children : List["Tree"] = [], parent : "None | Tree" = None):
        self.category : str = category

        self.children : List[Tree] = children

        for child in self.children:
            child.parent = self

        self.parent : None | "Tree" = parent

        self.index : Tuple[int, int]

    @property
    def leafs(self) -> List["Leaf"]:

        leafs : List[Leaf] = []
        for child in self.children:
            leafs += child.leafs
        
        return leafs

    def set_index(self, start_index = 0) -> int:
        end_index = start_index

        for child in self.children:
            end_index += (child.set_index(end_index) - end_index)
            #print(end_index)

        self.index = (start_index, end_index)

        return end_index


class Leaf(Tree):
    def __init__(self, category : str, word : str, pred_arg_cat : str, parent : "None | Tree" = None):
        super().__init__(category, parent = parent)
        
        self.word : str = word
        self.pred_arg_cat : str = pred_arg_cat

    def set_index(self, start_index = 0) -> int:
        self.index = (start_index, start_index + 1)

        return start_index + 1

    def get_scope(self, limit : int = -1) -> List[Tuple[int, int]]:

        scope_list : List[Tuple[int, int]] = []

        parent : None | Tree = self.parent

        step : int = 0

        LIMIT = 10

        while parent is not None and (limit == -1 or step < limit):

            left_index : int = parent.index[0] - self.index[0]
            if abs(left_index) >= LIMIT:
                left_index = (left_index > 0) * LIMIT + (left_index < 0) * -LIMIT
            
            right_index : int = parent.index[1] - self.index[0]
            if abs(right_index) >= LIMIT:
                right_index = (right_index > 0) * LIMIT + (right_index < 0) * -LIMIT

            scope_list.append((left_index, right_index))

            parent = parent.parent

            step += 1

        return scope_list

    @property
    def leafs(self) -> List["Leaf"]:
        return [self]

def construct_tree(treestring : str) -> Tree:
    '''
    TODO

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
        TODO
    '''

    def _construct_tree(treestring : str) -> Tree:

        def construct_children(children_string : str) -> List[Tree]:
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

            return [construct_tree(child) for child in children_strings]


        treestring = treestring[2:-1]

        node : str = ""

        for i, c in enumerate(treestring):
            if c == ">":
                treestring = treestring[i+2:]
                break
            else:
                node += c

        node_parts : List[str] = node.split()

        if node_parts[0] == "L":
            return Leaf(node_parts[1], node_parts[4], node_parts[5])

        return Tree(node_parts[1], construct_children(treestring))

    constructed_tree : Tree = _construct_tree(treestring)
    constructed_tree.set_index(0)
    return constructed_tree

def import_auto(filename : str) -> Tuple[Corpus, Corpus, Corpus]:
    '''
    Imports tokens, supertags and ... from
    machine-readable derivation file from the
    CCGrebank (CCG annotated Penn Treebank).
        leftaction : TODO
        rightaction : TODO

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
        TODO
    '''
    tokens      : Corpus = []
    supertags   : Corpus = []
    scopes      : Corpus = []
    output_list : Tuple[Corpus, Corpus, Corpus] = ([], [], [])

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
                            sen_supertags.append(simplify(element_parts[5]))

                supertags.append(sen_supertags)
                tokens.append(sen_tokens)

                tree : Tree = construct_tree(line)
                scopes.append(["_".join([str(s) for s in leaf.get_scope(2)]) for leaf in tree.leafs])

    assert(len(tokens) == len(supertags) == len(scopes))
    return tokens, supertags, scopes


def simplify(element : str) -> str:
    '''
    What?

    Parameters
    ----------
    element : str
        TODO

    Returns
    -------
    str
        TODO
    '''


    if len(element.split("_")) == 1:
        return element
    
    simple_element : str = ""
    num_to_i : Dict[str, str] = {}

    for is_number, component in component_generator(element):
        
        if is_number:
            if not component in num_to_i:
                num_to_i[component] = str(len(num_to_i))

            simple_element += num_to_i[component]
        
        else:
            simple_element += component

    
    return simple_element


def component_generator(element : str):
    '''
    TODO
    '''
    NUMBERS = "0123456789"
    current_component   : str   = ""
    in_number           : bool  = False
    for c in element:
        if c == "_":
            #current_component += c
            yield (False, current_component)
            current_component = ""
            in_number = True
        
        elif in_number and not c in NUMBERS:
            #yield (True, current_component)
            current_component = c
            in_number = False
             
        
        elif c not in NUMBERS:
            current_component += c
        
    if not in_number:
        yield (in_number, current_component)


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
            if line[0:2] == "<s":
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
        scopes : TODO 
        leftaction : TODO,
        rightaction : TODO
        dependency : TODO obsolete

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
        TODO
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

    auto = listify(intuple(import_auto(auto_dir)))   # machine-readable CCG derivation files
    parg = listify(intuple(import_parg(parg_dir)))     # predicate-argument structure file

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
    
    # obsolete

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

    dependency : Corpus = corpus_apply(supertags, get_long_distance_sketch)

    return tokens, supertags, scopes, leftaction, rightaction, dependency
    

def import_complex_multi(nums : Sequence[int], data_dir : str) -> Tuple[Corpus, Corpus, Corpus, Corpus, Corpus, Corpus]:
    '''
    Imports a sequence of files from 
    the CCGrebank (CCG annotated Penn Treebank) 
    corpus and converts the information into
    five different features:
        supertags : lexical category assignments
        scopes : TODO 
        leftaction : TODO,
        rightaction : TODO
        dependency : TODO obsolete

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
        TODO
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