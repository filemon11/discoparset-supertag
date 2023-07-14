
class Token:
    # Leaf of a tree
    header = None
    def __init__(self, token, i, features=None):
        self.token = token
        self.features = features # Only used for POS tags for now which should be self.features[0]
        self.i = i
        self.parent = None
    
    def get_tag(self):
        if len(self.features) > 0:
            return self.features[0]
        return None

    def set_tag(self, tag):
        self.features[0] = tag

    def is_leaf(self):
        return True

    def get_span(self):
        return {self.i}

    def __str__(self):
        return "({} {}={})".format(self.features[0], self.i, self.token)

class Tree:
    def __init__(self, label, children):
        self.label = label
        self.children = sorted(children, key = lambda x: min(x.get_span()))
        self.span = {i for c in self.children for i in c.get_span()}
        self.parent = None
        for c in self.children:
            c.parent = self

    def is_leaf(self):
        assert(self.children != [])
        return False

    def get_span(self):
        return self.span
   
    def get_yield(self, tokens):
        # Updates list of tokens
        for c in self.children:
            if c.is_leaf():
                tokens.append(c)
            else:
                c.get_yield(tokens)
    
    def merge_unaries(self):
        # Collapse unary nodes
        for c in self.children:
            if not c.is_leaf():
                c.merge_unaries()

        if len(self.children) == 1 and not self.children[0].is_leaf():
            c = self.children[0]
            self.label = "{}@{}".format(self.label, c.label)
            self.children = c.children
            for c in self.children:
                c.parent = self

    def expand_unaries(self):
        # Cancel unary node collapse
        for c in self.children:
            if not c.is_leaf():
                c.expand_unaries()

        if "@" in self.label:
            split_labels = self.label.split("@")
            t = Tree(split_labels[-1], self.children)
            for l in reversed(split_labels[1:-1]):
                t = Tree(l, [t])
            self.label = split_labels[0]
            self.children = [t]
            t.parent = self

    def get_constituents(self, constituents):
        # Update set of constituents
        constituents.add((self.label, tuple(sorted(self.span))))
        for c in self.children:
            if not c.is_leaf():
                c.get_constituents(constituents)

    def __str__(self):
        return "({} {})".format(self.label, " ".join([str(c) for c in self.children]))

def get_yield(tree):
    # Returns list of tokens in the tree (in surface order)
    tokens = []
    tree.get_yield(tokens)
    return sorted(tokens, key = lambda x: min(x.get_span()))

def get_constituents(tree, filter_root=False):
    # Returns a set of constituents in the tree
    # Ignores root labels (from PTB, Negra, and Tiger corpora) if filter_root
    constituents = set()
    tree.get_constituents(constituents)
    if filter_root:
        constituents = {(c, i) for c, i in constituents if c not in {'ROOT', 'VROOT', 'TOP'}}
    return constituents


