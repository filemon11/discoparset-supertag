
import tree as T

class State:
    """Class for parsing configurations and transition applications"""
    STRUCT, LABEL = 0, 1
    memory_sizes = []
    def __init__(self, tokens):
        self.memory = []
        self.focus = None
        self.j = 0
        self.i = 0
        self.buffer = tokens

    def print(self):
        print("size memory", len(self.memory))
        if self.focus is not None:
            print("focus", [s.get_span() for s in self.focus])
        else:
            print("focus: None")
        print("i={}".format(self.i))
        print("j={}".format(self.j))
        print("buffer size ={}".format(len(self.buffer)))

    def next_action_type(self):
        # Returns STRUCT or LABEL depending on type of the next action
        return self.j % 2

    def shift(self):
        # Apply shift
        if self.focus is not None:
            self.memory.append(self.focus)
        self.focus = [self.buffer[self.i]]
        self.i += 1
        self.j += 1

    def combine(self, s):
        # Apply combine
        # s: int
        assert(s < len(self.memory))
        self.focus = self.focus + self.memory[s]
        self.memory = self.memory[:s] + self.memory[s+1:]
        self.j += 1

    def labelX(self, X):
        # Apply label-X
        # X: str
        self.focus = [T.Tree(X, self.focus)]
        self.j += 1

    def nolabel(self):
        # Apply no-Label
        self.j += 1

    def is_prefinal(self):
        # Returns True if configuration is final
        # or would be final after a labelling action
        if self.focus is None:  return False
        if self.memory != []: return False
        if self.i != len(self.buffer): return False
        return True

    def is_final(self):
        # Returns True if the configuration is final
        # (and there is a single full tree in the focus)
        return self.is_prefinal() and self.next_action_type() == State.STRUCT

    def can_shift(self):
        # Returns True if shift is possible in current configuration
        return self.i != len(self.buffer) and self.next_action_type() == self.STRUCT

    def can_combine(self):
        # Returns True if shift is possible in current configuration
        return len(self.memory) > 0 and self.next_action_type() == self.STRUCT

    def get_tree(self):
        # Assumes that configuration is final and returns the predicted tree
        assert(self.is_final())
        return self.focus[0]

    def oracle(self):
        # Assumes that the configuration is built upon a gold tree
        # Returns a training example: next action + input from which it should be predicted
        # Side effect: apply the gold action
        if self.next_action_type() == State.LABEL:
            input_res = self.get_labelling_step_input()
            self.j += 1
            gold_idxs = self.focus[0].parent.get_span()
            current_idxs = set()
            for s in self.focus:
                current_idxs |= s.get_span()
            if current_idxs == gold_idxs:
                self.focus = [self.focus[0].parent]
                return ("label", self.focus[0].label), input_res
            return ("nolabel", "nolabel"),  input_res

        else:
            State.memory_sizes.append(len(self.memory))
            input_res = self.get_structural_step_input()
            if self.focus is None:
                self.shift()
                return ("shift", None), input_res
            p = self.focus[0]
            for i, s in reversed(list(enumerate(self.memory))):
                s_ = s[0]
                if s_.parent == p.parent:
                    self.combine(i)
                    return ("combine", i), input_res
            self.shift()
            return ("shift", None), input_res


    def dyn_oracle(self, gold_constituents):
        if self.next_action_type() == State.LABEL:
            input_res = self.get_labelling_step_input()
            input_tuple = tuple(sorted(input_res))
            if input_tuple in gold_constituents:
                return ("label", gold_constituents[input_tuple])
            else:
                return ("nolabel", "nolabel")
        else:
            # Here are the tricky cases
            # Look for the smallest reachable constituent
            # and return an action that constructs it
            potential_reachable = [set(k) for k, v in gold_constituents.items() if max(k) >= self.i -1]
            potential_reachable.sort(key=lambda x: (max(x), len(x)))

            reachable = None
            memory_sets = []
            for s in self.memory:
                s_0 = s[0].get_span()
                for s_i in s[1:]:
                    s_0 |= s_i.get_span()
                memory_sets.append(s_0)
            
            focus_set = set()
            for s in self.focus:
                focus_set |= s.get_span()

            for c in potential_reachable:
                keep = True

                for s in memory_sets + [focus_set]:
                    # s is a subset of s_g  or   s and s_g are disjoint
                    if all([i in c for i in s]) or not any([i in c for i in s]):
                        continue
                    else:
                        keep = False
                        break
                if keep:
                    reachable = c
                    break
            if reachable is None:
                return ("shift", None)

            #for i, s in reversed(list(enumerate(memory_sets))):
            for i, s in sorted(list(enumerate(memory_sets)), key = lambda x: max(x[1]), reverse=True):
                union = s | focus_set
                if all([i in reachable for i in union]):
                    return ("combine", i)
            
            return ("shift", None)




    def get_structural_step_input(self):
        # Returns representation of the current configuration
        # mem_sets: list of sets of int
        # focus_set: set of int
        # buf_set: set of int  (singleton)
        mem_sets = []
        for l in self.memory:
            current_set = set()
            for s in l:
                current_set |= s.get_span()
            mem_sets.append(current_set)
        
        focus_set = self.get_labelling_step_input()
        
        buf_set = None
        if self.can_shift():
            buf_set = self.buffer[self.i].get_span()
        return mem_sets, focus_set, buf_set


    def get_labelling_step_input(self):
        # Returns set of indices dominated by s_f
        if self.focus is None:
            return None
        focus_set = set()
        for s in self.focus:
            focus_set |= s.get_span()
        return focus_set




