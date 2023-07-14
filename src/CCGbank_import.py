from typing import List, Tuple, Dict, Sequence, Union, Literal, Hashable, Set
import os

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

def import_auto(filename : str) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:

    output_list : Tuple[List[List[str]], List[List[str]], List[List[str]]] = ([], [], [])

    with open(filename, 'r') as file:


        for line in file:
            if line[0:2] == "ID" or line[0] != "(":
                pass
            else:
                
                tokens : List[str] = []
                supertags : List[str] = []

                first_n : int = 0

                for n, c in enumerate(line):
                    if c == "<":
                        first_n = n
                    elif c == ">":
                        
                        element : str = line[first_n+1:n]

                        element_parts : List[str] = element.split()

                        if len(element_parts) > 4:
                            tokens.append(element_parts[4])

                            #supertags.append(element_parts[1])
                            supertags.append(simplify(element_parts[5]))

                output_list[0].append(supertags)
                output_list[1].append(tokens)
                tree : Tree = construct_tree(line)
                output_list[2].append(["_".join([str(s) for s in leaf.get_scope(2)]) for leaf in tree.leafs])

                if len(tree.leafs) != len(tokens):
                    print([leaf.word for leaf in tree.leafs], tokens)
                #tree : Tree = construct_tree(line)
                #for leaf, supertag in zip(tree.leafs, supertags):
                #    print("_".join([str(s) for s in leaf.get_scope(2)]), supertag)
    assert(len(output_list[0]) == len(output_list[1]) == len(output_list[2]))
    return output_list

#def component_generator(element : str):
#
#        NUMBERS = "0123456789"
#
#        current_component   : str   = ""
#        in_number           : bool  = False
#
#        for c in element:
#
#            if c == "_":
#                current_component += c
#                yield (False, current_component)
#                current_component = ""
#
#                in_number = True
#            
#            elif in_number and not c in NUMBERS:
#                yield (True, current_component)
#                current_component = c
#                in_number = False
#            
#            else:
#                current_component += c
# 
#        yield (in_number, current_component)

def component_generator(element : str):

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


def simplify(element : str) -> str:


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




def import_parg(filename : str) -> Tuple[List[List[str]], List[List[str]]]:

    output_list : Tuple[List[List[str]], List[List[str]]] = ([], [])

    LIMIT : Literal[5] = 5

    with open(filename, 'r') as file:

        sentence_number : int = 0
        
        sentence_dict_left : Dict[int, List[List[str]]] = {}
        sentence_dict_right : Dict[int, List[List[str]]] = {}

        for line in file:
            if line[0:2] == "<s":
                sentence_number += 1
                sentence_dict_left[sentence_number -1] = [["0"] for _ in range(int(line.split()[-1]) + 1)]
                sentence_dict_right[sentence_number -1] = [["0"] for _ in range(int(line.split()[-1]) + 1)]
                pass
            
            elif line == "<\\s>\n" or line == "<\\s> \n":
                pass

            else:
                line_components : List[str] = line.split(" 	 ")
                
                
                arg_num : str = line_components[3]
                relative_position : int = int(line_components[0]) - int(line_components[1])

                

                if relative_position < 0:
                    if relative_position > -LIMIT:
                    #relative_position = max(relative_position, -5)
                        sentence_dict_left[sentence_number - 1][int(line_components[1])].append(arg_num + ":" + str(relative_position))
                else:
                    if relative_position < LIMIT:
                    #relative_position = min(relative_position, 5)
                        sentence_dict_right[sentence_number - 1][int(line_components[1])].append(arg_num + ":" + str(relative_position))

        output_list = ( [["_".join(word) for word in sentence] for sentence in sentence_dict_left.values() if len(sentence) > 1 ], \
                        [["_".join(word) for word in sentence] for sentence in sentence_dict_right.values() if len(sentence) > 1 ])
        

    return output_list

def import_complex(num : int, data_dir : str) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:

    str_num : str = "0" * (4  - len(str(num))) + str(num)
    str_section : str = str_num[0:2]

    auto_dir : str = f"{data_dir}/AUTO/{str_section}/wsj_{str_num}.auto"
    parg_dir : str = f"{data_dir}/PARG/{str_section}/wsj_{str_num}.parg"

    if not os.path.exists(auto_dir) or not os.path.exists(parg_dir):
        return ([],[],[],[],[])

    auto : Tuple[List[List[str]], List[List[str]], List[List[str]]] = import_auto(auto_dir)
    parg : Tuple[List[List[str]], List[List[str]]] = import_parg(parg_dir)

    
    #include = [comb for comb in zip(*auto, *parg) if len(comb[1]) != 1]
    #print(auto, parg)

    max_index : int = max(len(auto[0]), len(parg[0]))

    for sen_num in range(max_index):
        if sen_num >= max_index:
            break
        
        supertags = auto[0]
        tokens = auto[1]
        scopes = auto[2]

        left = parg[0]
        right = parg[1]

        if sen_num >= len(auto[0]):
            left.pop(sen_num)
            right.pop(sen_num)
            break

        elif sen_num >= len(parg[0]):
            supertags.pop(sen_num)
            tokens.pop(sen_num)
            scopes.pop(sen_num)
            break

        supertags_len = len(auto[0][sen_num])
        tokens_len = len(auto[1][sen_num])
        scopes_len = len(auto[2][sen_num])

        left_len = len(parg[0][sen_num])
        right_len = len(parg[1][sen_num])

        if not (supertags_len == tokens_len and tokens_len == scopes_len and scopes_len == left_len and left_len == right_len):

            if supertags_len < left_len:

                auto[0].pop(sen_num)
                auto[1].pop(sen_num)
                auto[2].pop(sen_num)

                max_index -= 1
    
    return auto + parg

    

def import_complex_multi(nums : Sequence[int], data_dir : str, limit : int = -1) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:

    supertags : List[List[str]] = []
    tokens : List[List[str]] = []
    action_left : List[List[str]] = []
    action_right : List[List[str]] = []
    scopes : List[List[str]] = []
   
        
    for n in nums:
        
        #print(type(import_complex(n, data_dir)))
        
        n_supertags, n_tokens, n_scopes, n_action_left, n_action_right = import_complex(n, data_dir)

        if not n_supertags == []:

            supertags += n_supertags
            tokens += n_tokens
            action_left += n_action_left
            action_right += n_action_right
            scopes += n_scopes
        
        if limit != -1 and len(supertags) >= limit:
            break
    
    dependency_supertags : List[List[str]] = [[get_long_distance_sketch(supertag) for supertag in sentence] for sentence in supertags]
    #supertags_org = supertags
    #supertags = [[remove_UB(supertag) for supertag in sentence] for sentence in supertags]
    #
    ##for sentence1, sentence2, sentence3 in zip(supertags_org, supertags, dependency_supertags):
    ##    for s1, s2, s3 in zip(sentence1, sentence2, sentence3):
    ##        print(s1, s2, s3)
    #sum_X = sum([1 for sentence in dependency_supertags for tag in sentence if tag == "X"])
    #print("X %", sum_X / sum([len(sentence) for sentence in dependency_supertags]))

    #for sentence1, sentence2, sentence3 in zip(scopes, action_left, action_right):
    #    for w1, w2, w3 in zip(sentence1, sentence2, sentence3):
    #        print(w1, w2, w3) 


    if not len(supertags) == len(tokens) == len(scopes) == len(action_left) == len(action_right) == len(dependency_supertags):
        print(len(supertags), len(tokens), len(scopes), len(action_left), len(action_right), len(dependency_supertags))
        raise Exception

    for i in range(len(supertags)):
        if not len(supertags[i]) == len(tokens[i]) == len(scopes[i]) == len(action_left[i]) == len(action_right[i]) == len(dependency_supertags[i]):
            print(supertags[i], tokens[i], scopes[i], action_left[i], action_right[i], dependency_supertags[i])
            raise Exception

    if limit == -1:
        return supertags, tokens, scopes, action_left, action_right, dependency_supertags
    

    else: 
        return supertags[:limit], tokens[:limit], scopes[:limit], action_left[:limit], action_right[:limit], dependency_supertags[:limit]


# information:
# head position relative to argument (e.g. -1, +3, ...))
# argument number

def remove_UB(supertag : str) -> str:
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
    '''Converts supertag into sketch with
    all elements replaced by "X"

    Multi-character symbols are treated
    the same as one-character symbols.

    :param supertag: supertag
    :type supertag: str
    :return: sketch
    :rtype: str
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