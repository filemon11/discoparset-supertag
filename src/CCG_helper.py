from collections import defaultdict

from typing import DefaultDict, List, Tuple, Optional, Dict, Set, Any, Hashable

import copy

Path    = List[Tuple[str, Tuple[int, int]]]
Paths   = List[Path]

IndexPath   = List[List[int]]
IndexPaths  = List[IndexPath]

basic_elements : Set[str] = set(('S[pss]', 'conj', 'S[adj]', 'S[intj]', 'S[poss]', 'NP[thr]', 
                                'S[q]', 'NP[expl]', 'N[num]', 'S[qem]', ';', 'S[wq]', 'NP[nb]', 'S', 
                                'PP', 'S[b]', 'S[dcl]', ',', 'RRB', 'S[to]', 'N', 'S[asup]', 'NP', 
                                ':', 'S[em]', 'S[inv]', 'S[for]', 'S[pt]', 'LRB', 'S[ng]', 'S[bem]', 
                                'S[frg]', '.'))


class Node():
    def __init__(self, name : str, is_child : Tuple[int, int] = (0, 0), parent : Optional["Node"] = None, supertag : str = "none") -> None:
        self.supertag   : str               = supertag
        self.name       : str               = name
        self.is_child   : Tuple[int, int]   = is_child # (1, 0) if left of its parent, (0, 1) if right of its parent
        self._left_children   : List[Node]  = []
        self._right_children  : List[Node]  = []
        self.parent     : Optional[Node]    = parent
        self._children_sorted  : List[Node]    = []

    @property
    def children(self) -> List["Node"]:
        return self.left_children + self.right_children

    @property
    def left_children(self) -> List["Node"]:
        return self._left_children

    @property
    def right_children(self) -> List["Node"]:
        return self._right_children[::-1]

    @property
    def left_direct_arg(self) -> "Node":
        return self.left_children[-1] if len(self.left_children) > 0 else Node("none")
    
    @property
    def left_indirect_arg(self) -> "Node":
        return self.left_children[-2] if len(self.left_children) > 1 else Node("none")

    @property
    def right_direct_arg(self) -> "Node":
        return self.right_children[0] if len(self.right_children) > 0 else Node("none")

    @property
    def right_indirect_arg(self) -> "Node":
        return self.right_children[1] if len(self.right_children) > 1 else Node("none")

    @property
    def children_sorted(self) -> List["Node"]:
        return self._children_sorted[::-1]
    
    def add_left_child(self, child : "Node") -> None:
        child.is_child  = (1, 0)
        child.parent    = self
        self._left_children.append(child)
        self._children_sorted.append(child)
    
    def add_right_child(self, child : "Node") -> None:
        child.is_child  = (0, 1)
        child.parent    = self
        self._right_children.append(child)
        self._children_sorted.append(child)
    
    def print(self, level = 0) -> None:
            spacing : str = level*"  "
            prefix : str = ""
            if self.is_child == (1, 0):
                prefix = "< "

            elif self.is_child == (0, 1):
                prefix = "> "
            
            print(f"{spacing}{prefix}{self.name}")

            for child in self.left_children:
                child.print(level + 1)
            
            for child in self.right_children:
                child.print(level + 1)

    def tuple_rep(self) -> Tuple[str, Tuple[int, int]]:
        return self.name, self.is_child

    def list_rep(self, elements_to_index_mapping : Dict[str, int]) -> List[int]:
        def one_hot(len : int, hot : int) -> List[int]:
            l : List[int] = [0]*len
            l[hot] = 1
            return l
        return one_hot(len(elements_to_index_mapping), elements_to_index_mapping[self.name]) + list(self.is_child)


def supertag_to_Node(supertag : str) -> Node:

    def subtag_ident(subtag : str) -> str:

        level : int = 1
        i : int = 0

        for i, c in enumerate(subtag + ")"):
            if c == "(":
                level += 1
            elif c == ")":
                level -= 1
            
            if level <= 0:
                break
        
        return subtag[:i]


    def _get_first_element(supertag : str) -> Tuple[Node, str]:
        
        for i, c in enumerate(supertag):
            if c in ("/", "\\", ")"):
                break
        else: #if for loop exists normally without break
            i += 1
        return Node(supertag[:i], supertag = subtag_ident(supertag)), supertag[i:]
    
    def _supertag_to_Node(supertag : str) -> Tuple[Node, str]:

        head : Node
        rest : str
        if supertag[0] == "(":

            head, rest = _supertag_to_Node(supertag[1:])


            assert rest[0] == ")"
            rest = rest[1:]

        else:

            head, rest = _get_first_element(supertag)

            if len(rest) == 0 or rest[0] == ")":
                return head, rest
        
        if len(rest) == 0:
            return head, rest
            
        if rest[0] == "/":

            right_child : Node
            right_child, rest = _supertag_to_Node(rest[1:])
            head.add_right_child(right_child)


        elif rest[0] == "\\":

            left_child : Node
            left_child, rest = _supertag_to_Node(rest[1:])
            head.add_left_child(left_child)


        if len(rest) == 0:
            return head, rest

        elif rest[0] == ")":
            return head, rest
        
        else:
            raise AssertionError("Problem with supertag format:", supertag)

    node : Node = _supertag_to_Node(supertag)[0]
    node.supertag = supertag
    return node
        
def supertag_to_paths(supertag : str) -> Paths:

    node : Node = supertag_to_Node(supertag)

    return Node_to_paths(node)

def Node_to_paths(node : Node) -> Paths:

    working_list    : List[Tuple[Path, List[Node]]] = [([node.tuple_rep()], node.children)]

    for i1, working_path in enumerate(working_list):
        children    : List[Node]    = working_path[1]
        path        : Path          = working_path[0]
        
        for i2, child in enumerate(children):
            new_path : Path = path.copy()
            new_path.append(child.tuple_rep())
            working_list.insert(i1+i2+1, (new_path, child.children))
    
    final_list : Paths = [path[0][::-1] for path in working_list if len(path[1]) == 0]
    return final_list

def supertag_to_index_paths(supertag : str, elements_to_index_mapping : Dict[str, int]) -> IndexPaths:

    node : Node = supertag_to_Node(supertag)

    working_list    : List[Tuple[IndexPath, List[Node]]] = [([node.list_rep(elements_to_index_mapping)], node.children)]
    final_list      : IndexPaths                         = []

    for working_path in working_list:
        children    : List[Node]    = working_path[1]
        path        : IndexPath     = working_path[0]

        if len(children) == 0:
            final_list.append(path[::-1])
        else:
            for child in children:
                new_path : IndexPath = copy.deepcopy(path)
                new_path.append(child.list_rep(elements_to_index_mapping))
                working_list.append((new_path, child.children))
    
    return final_list


def elements(supertags_to_paths_mapping : Dict[str, Paths]) -> Set[str]:
    els = set()

    for _, paths in supertags_to_paths_mapping.items():
        for path in paths:
            for element, _ in path:
                els.add(element)
    
    return els

def create_elements_to_index_mapping(elements : Set[str] = basic_elements) -> Dict[str, int]:
    mapping : Dict[str, int] = {element : index for index, element in enumerate(elements)}
    return mapping



def Node_info(node : Node) -> Tuple[str, str, str, str, str, bool, bool, bool, bool, str, str, int, int, int, str, str]:
    supertag            : str = node.supertag
    name                : str = node.name
    left_direct_arg     : str = node.left_direct_arg.supertag
    left_indirect_arg   : str = node.left_indirect_arg.supertag
    right_direct_arg    : str = node.right_direct_arg.supertag
    right_indirect_arg  : str = node.right_indirect_arg.supertag

    is_reducing_left    : bool = False
    is_deleting_left    : bool = False
    
    for left_child in node.left_children:
        
        if len(left_child.left_children) > 0 or len(left_child.right_children) > 0:
            is_reducing_left = True
        if len(left_child.right_children) > 0:
            is_deleting_left = True
    
    is_reducing_right   : bool = False
    is_deleting_right   : bool = False
    for right_child in node.right_children:
        if len(right_child.left_children) > 0 or len(right_child.right_children) > 0:
            is_reducing_right = True
        if len(right_child.left_children) > 0:
            is_deleting_right = True

    first_sorted    : str = node.children_sorted[0].supertag if len(node.children_sorted) > 0 else "none"
    second_sorted   : str = node.children_sorted[1].supertag if len(node.children_sorted) > 1 else "none"

    num_left_args   : int = len(node.left_children)
    num_right_args  : int = len(node.right_children)
    num_total_args  : int = len(node.children)

    sketch          : str = get_sketch(node.supertag)
    
    def get_tree_sketch(node : Node) -> str:
        sketch : str = "X" + "("
        first : bool = True
        for child in node.children_sorted:
            if first:
                first = False
            else:
                sketch += ","
            sketch += get_tree_sketch(child)
        sketch += ")"
        return sketch

    tree_sketch     : str = get_tree_sketch(node)

    return (name, left_direct_arg, left_indirect_arg, right_direct_arg, right_indirect_arg, is_reducing_left, is_deleting_left, is_reducing_right, is_deleting_right, first_sorted, second_sorted, num_left_args, num_right_args, num_total_args, sketch, tree_sketch)

def get_sketch(supertag : str) -> str:
    '''Converts supertag into sketch with
    all elements replaced by "X"

    Multi-character symbols are treated
    the same as one-character symbols.

    :param supertag: supertag
    :type supertag: str
    :return: sketch
    :rtype: str
    '''
    
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

def get_type_sketch(supertag : str) -> str:
    '''Converts supertag into sketch with
    identical elements replaced by the 
    same symbol.

    Multi-character symbols are treated
    the same as one-character symbols.

    example:
    "(A/B)\A" -> "(0/1)\0"
    
    :param elements: supertag
    :type supertag: str

    :returns: type sketch
    :rtype: str
    '''

    def get_symbol(s : str) -> str:
        '''Identify CCG constituent
        symbol that may be followed
        by arguments and parentheses.

        :param s: supertag fragment
        :type s: str

        :returns: constituent symbol
        :rtype: str
        '''
        i : int = 0
        s += ")"    # concatenate in order for 
                    # index to shift past the last
                    # element when the string contains
                    # just a constituent

        # iterate through the given string
        # until you find slash or a parenthesis

        for i, c in enumerate(s):
            if c in ("\\", "/", "(", ")"):
                break

        return s[:i]

    class CountingDict(dict):
        '''Default dictionary that
        sets the value for new unknow
        elements as the current
        dictionary length
        '''

        def __init__(self, *args):
            '''Constructor'''
            dict.__init__(self, *args)

        def __missing__(self, key : Hashable) -> int:
            '''Called when retrieving an element not
            in the dictionary. Adds the element,
            assigns it the current length of the
            dictionary and returns the length.
            
            :param key:
            :type key: Hashable
            
            :returns: length of self
            :rtype: int'''
            
            value : int = len(self)
            self[key]   = value

            return value
        
    symbol_dict : CountingDict = CountingDict()

    sketch      : str = ""
    in_symbol   : bool = False

    for i, c in enumerate(supertag):

        if c not in ("\\", "/", "(", ")"):
            if in_symbol:
                continue
            
            sketch += str(symbol_dict[get_symbol(supertag[i:])])
            in_symbol = True

        else:
            sketch += c
            in_symbol = False

    return sketch

def get_num_labels(elements : List[Tuple[Hashable, ...]], add_standard : int = 0) -> Tuple[Tuple[Set[Hashable], ...]  , Tuple[int, ...]]:
    '''Converts label tuples of same length
    to label sets and a tuple giving the number
    of different label tokens seen.
    
    :param elements: list of label tuples with len X
    :type supertag: List[Tuple[Hashable, ...]]
    :param add_standard: to add to each label number,
        defaults to 0
    :type add_standard: int
    :returns: labels sets (len X), labels count (len X)
    :rtype: Tuple[Tuple[Set[Hashable], ...]  , Tuple[int, ...]]
    '''

    assert(len(elements) > 0)
    labels_set_list : Tuple[Set[Hashable], ...] = tuple([set() for _ in elements[0]])

    for el in elements:
        for labels_set, category in zip(labels_set_list, el):
            labels_set.add(category)
    
    return labels_set_list, tuple([len(category) + add_standard for category in labels_set_list])
    
def get_num_labels_single(elements : List[Hashable]) -> Tuple[Set[Hashable], int]:
    '''Converts label tuples of same length
    to label sets and a tuple giving the number
    of different label tokens seen.
    
    :param elements: list of label tuples with len X
    :type supertag: List[Tuple[Hashable, ...]]
    :param add_standard: to add to each label number,
        defaults to 0
    :type add_standard: int
    :returns: labels sets (len X), labels count (len X)
    :rtype: Tuple[Tuple[Set[Hashable], ...]  , Tuple[int, ...]]
    '''

    assert(len(elements) > 0)

    labels_set : Set[Hashable] = {*elements}
    
    return labels_set, len(labels_set)

def replace_rare(sentences : List[List[Hashable]], min = 400) -> List[List[Hashable]]:

    frequencies : DefaultDict[Hashable, int] = defaultdict(int)

    for sentence in sentences:
        for word in sentence:
            frequencies[word] += 1

    threshold : float = 0.00001 * sum([len(sentence) for sentence in sentences])

    print(f"threshold = {threshold}, total labels = {len(frequencies)}, rare labels = {len([label for label, frequency in frequencies.items() if frequency < threshold])}")


    replacement_dict : Dict[Hashable, Hashable] = {}

    for i, (label, frequency) in enumerate(sorted(frequencies.items(), key = lambda x : x[1], reverse = True)):
        #print(label, frequency)

        if frequency >= threshold or i < min:
            replacement_dict[label] = label
        
        else:
            replacement_dict[label] = "unknown"


    return [[replacement_dict[label] for label in sentence] for sentence in sentences]

def supertag_to_head_arg(supertag : str) -> Tuple[str, str, str]:
    '''Converts supertag to head, argument direction
    and argument.

    If no argument is present, argument direction
    defaults to "+" and argument to "".

    Example input -> output:
    "A/(B\C)" -> "A", "+", "B\C"
    "(A\B)\C" -> "A\B", "-", "C"
    
    :param supertag: CCG supertag
    :type supertag: str
    :returns: head, argument direction, argument
    :rtype: Tuple[str, str, str]
    '''

    head        : str = supertag
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
                head        = supertag[:-i-1]
                arg         = supertag[-i:]
                break

            if c == "\\":
                direction   = "-"
                head        = supertag[:-i-1]
                arg         = supertag[-i:]
                break
    
    if len(head) > 0 and head[0] == "(" and head[-1] == ")":
        head = head[1:-1]

    if len(arg) > 0 and arg[0] == "(" and arg[-1] == ")":
        arg = arg[1:-1]

    return head, direction, arg

def supertag_to_arg_list(supertag: str, max_level : int = 50) -> Tuple[str, List[Tuple[str, str, str]]]:
    '''Converts supertag to total head and list of
    heads, argument directions and arguments.

    The method traverses from right to left through
    the supertag.

    Example output for "(((A/B)\C)/D)\A":
    ("A", [("((A/B)\C)/D", "-", "A"), ("(A/B)\C", "+", "D"), ...])
    
    :param supertag: CCG supertag
    :type supertag: str
    :param max_level: max number of iterations for 
        argument identification, defaults to 50
    :type max_level: int
    :returns: total_head, list of heads, 
        argument directions and arguments
    :rtype: Tuple[str, List[Tuple[str, str, str]]]
    '''

    arg         : str = " "
    arg_list    : List[Tuple[str, str, str]] = []

    level       : int = 0
    
    while arg != "" and level < max_level:
        supertag, direction, arg = supertag_to_head_arg(supertag)

        if arg != "":
            arg_list.append((supertag, direction, arg))
        
        level += 1
    
    return supertag, arg_list

def supertag_to_arg_struct(supertag : str) -> str:

    arg_list : List[Tuple[str, str, str]]

    _, arg_list = supertag_to_arg_list(supertag)

    return "".join(["{}X".format(arg[1]) for arg in arg_list])

def pure_info(supertag : str) -> List[str]:
    '''Converts supertag to tuple of relevant component information:

    examples for "(((A/B)\C)/D)\A":
    1. total head:  A
    2. head1:       ((A/B)\C)/D
    3. arg1:        -A
    4. head2:       (A/B)\C
    5. arg2:        +D
    6. inner_arg:   +B              # TODO: NEW: second inner arg
    7. sketch:      (((X/X)\X)/X)\X
    8. type_sketch: (((0/1)\2)/3)\0
    
    :param supertag: CCG supertag
    :type supertag: str
    :returns: relevant component information (see above)
    :rtype: Tuple[str, str, str, str, str, str, str, str]
    '''


    arg_list    : List[Tuple[str, str, str]]
    total_head  : str

    total_head, arg_list = supertag_to_arg_list(supertag)

    head1       : str = total_head
    arg1_dir    : str = "+"
    arg1        : str = ""

    if len(arg_list) > 0:
        head1       = arg_list[0][0]
        arg1_dir    = arg_list[0][1]
        arg1        = arg_list[0][2]

    head2       : str = total_head
    arg2_dir    : str = "+"
    arg2        : str = ""

    if len(arg_list) > 1:
        head2       = arg_list[1][0]
        arg2_dir    = arg_list[1][1]
        arg2        = arg_list[1][2]

    inner_dir : str = "+"
    inner_arg : str = ""

    if len(arg_list) > 0:
        _, inner_dir, inner_arg = arg_list[-1]

    second_inner_dir : str = "+"
    second_inner_arg : str = ""
    
    if len(arg_list) > 1:
        _, second_inner_dir, second_inner_arg = arg_list[-2]

    sketch      : str = get_sketch(supertag)

    type_sketch : str = get_type_sketch(supertag)

    return [supertag, total_head, head1, arg1_dir + arg1, head2, arg2_dir + arg2, inner_dir + inner_arg, second_inner_dir + second_inner_arg, sketch, type_sketch]


def action_labels(supertag_list : List[str]) -> List[Tuple[str, str]]:

    def identify_action(head1, dir1, arg1, supertag1, head2, dir2, arg2, supertag2) -> str:
        if dir1 == "+":
            if arg1 == "":
                return "n"
            elif arg1 == supertag2:
                return "r"
            elif arg1 == head2 and dir2 == "+":
                return "fr"
            else:
                return "nr"
        
        else:
            if arg2 == supertag1:
                return "l"
            elif arg2 == head1 and dir1 == "-":
                return "fl"
            else:
                return "nl"

    
    action_list_r : List[str] = []
    #action_list_l : List[str] = []

    #long_dependency : List[str] = []    #categories : existent_r, non_existent_r, existent_l, non_existent_l, none

    head_arg_list : List[Tuple[str, str, str, str]] = [supertag_to_head_arg(supertag) + (supertag,) for supertag in supertag_list]

    if len(head_arg_list) > 0:    
        for n, ((head1, dir1, arg1, supertag1), tag2) in enumerate(zip(head_arg_list, head_arg_list)):
            
            #print(tag1, tag2)

            #if dir1 == "+":
            #    if arg1 == "":
            #        long_dependency.append("none")
            #
            #    elif len(supertag_list) > n+1 and arg1 in supertag_list[n+1:] or arg1 in supertag:
            #        long_dependency.append("existent_r")
            #
            #    else:
            #        long_dependency.append("non_existent_r")
            #else:
            #    if n > 0 and arg1 in supertag_list[:n]:
            #        long_dependency.append("existent_l")
            #    
            #    else:
            #        long_dependency.append("non_existent_l")

            if n + 1 == len(head_arg_list):
                break

            action : str = identify_action(head1, dir1, arg1, supertag1, *tag2)
            
            action_list_r.append(action)
        
        #action_list_l = ["START"] + action_list_r

        action_list_r.append("EOL")
    
    #print(action_list_r, action_list_l, long_dependency)
    
    return [labels for labels in zip(action_list_r)]


    # X/Y   Y   -> r
    #Y   X\Y   -> l
    #X/Y   Y/Z   -> fr
    #Y\Z   X\Y   -> fl

def create_near_action(sentence : List[str]) -> List[str]:
    def _near_left(arg_list_left : Tuple[str, List[Tuple[str, str, str]]], arg_list : Tuple[str, List[Tuple[str, str, str]]], supertag_left : str) -> str:
        
        arguments : List[Tuple[str, str, str]] = arg_list[1]

        arguments_left : List[Tuple[str, str, str]] = arg_list_left[1]

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

    def _near_right(arg_list : Tuple[str, List[Tuple[str, str, str]]], arg_list_right : Tuple[str, List[Tuple[str, str, str]]], supertag_right : str) -> str:
        
        arguments : List[Tuple[str, str, str]] = arg_list[1]

        arguments_right : List[Tuple[str, str, str]] = arg_list_right[1]
        
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

    def _near(arg_list_left : Tuple[str, List[Tuple[str, str, str]]], arg_list : Tuple[str, List[Tuple[str, str, str]]], arg_list_right : Tuple[str, List[Tuple[str, str, str]]], supertag_left : str, supertag_right : str) -> str:
        return "{},{}".format(_near_left(arg_list_left, arg_list, supertag_left), _near_right(arg_list, arg_list_right, supertag_right))
    
    sentence_boundary : str = ""
    sentence_padded : List[str] = [sentence_boundary]
    sentence_padded.extend(sentence)
    sentence_padded.append(sentence_boundary)

    arg_lists : List[Tuple[str, List[Tuple[str, str, str]]]] = [supertag_to_arg_list(supertag) for supertag in sentence_padded]

    near_action : List[str] = []

    for arg_list_left, arg_list, arg_list_right, supertag_left, supertag_right in zip(arg_lists, arg_lists[1:], arg_lists[2:], sentence_padded, sentence_padded[2:]):
        near_action.append(_near(arg_list_left, arg_list, arg_list_right, supertag_left, supertag_right))
    
    assert(len(near_action) == len(sentence))
    
    return near_action
