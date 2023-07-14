from tree import Token, Tree, get_yield

ID,FORM,LEMMA,CPOS,FPOS,MORPH,HEAD,DEPREL,PHEAD,PDEPREL=range(10)

def is_xml(s) : return s[0] == "<" and s[-1] == ">"
def is_xml_beg(s) : return is_xml(s) and s[1] != "/"
def is_xml_end(s) : return is_xml(s) and not is_xml_beg(s)
def is_head(s) : return is_xml(s) and "^head" in s
def get_nt_from_xml(s) :
    if is_xml_beg(s) :
        s = s[1:-1]
    elif is_xml_end(s) :
        s = s[2:-1]
    else : assert(False)
    if s[-5:] == "^head" :
        return s[:-5]
    return s

def parse_token(line) :
    idx, token, line = line[0],line[1],line[2:]
    idx = int(idx.split("^")[0]) # in case head is on idx
    tok = Token(token, idx-1, line[:-1])
    return tok

def read_tbk_tree_rec(lines, beg, end, headersize) :
    if len(lines[beg]) == 1 :
        assert(is_xml_beg(lines[beg][0]))
        assert(is_xml_end(lines[end-1][0]))
        label = get_nt_from_xml(lines[beg][0])
        assert(label == get_nt_from_xml(lines[end-1][0]))
        i = beg + 1
        c_beg = []
        counter = 0
        while i < end :
            if counter == 0 :
                c_beg.append(i)
            if is_xml_beg(lines[i][0]) :
                counter += 1
            elif is_xml_end(lines[i][0]) :
                counter -= 1
            i += 1
        children = [ read_tbk_tree_rec(lines, i, j, headersize) for i,j in zip(c_beg[:-1], c_beg[1:]) ]
        #is_head = "^head" in lines[beg][0]
        subtree = Tree(label, children)
        #node = CtbkTree(label, children)
        #node.head = is_head
        #node.idx = min([c.idx for c in node.children])
        #node.children = sorted(node.children, key = lambda x : x.idx)
        return subtree
    else :
        assert(len(lines[beg]) == headersize + 1)
        assert(end == beg + 1)
        return parse_token(lines[beg])

def read_tbk_tree(string, headersize) :
    lines = [ line.strip().split("\t") for line in string.split("\n") if line.strip()]
    return read_tbk_tree_rec(lines, 0, len(lines), headersize)

def read_ctbk_corpus(filename) :
    instream = open(filename, "r")
    header = instream.readline().strip().split()
    assert(header[-1] == "gdeprel")
    Token.header = header[2:-1]
    sentences = instream.read().split("\n\n")

    return [ read_tbk_tree(s, len(header)) for s in sentences if s.strip() ]

def get_conll(tree):

    tokens = get_yield(tree)

    conll_tokens = []
    for tok in tokens :
        newtok = ["_" for i in range(10)]
        newtok[ID]   = str(tok.i)
        newtok[FORM] = tok.token
        newtok[CPOS] = newtok[FPOS] = tok.features[0]
        newtok[MORPH] = "|".join(sorted(["{}={}".format(a,v) for a,v in zip( Token.header, tok.features[1:] ) if v != "UNDEF"]))
        conll_tokens.append(newtok)
    return conll_tokens

def write_conll(ctree, out):
    for tok in ctree :
        out.write("{}\n".format("\t".join(tok)))


if __name__ == "__main__":
    import sys
    treebank = read_ctbk_corpus("../multilingual_disco_data/data/tiger_spmrl/dev.ctbk")

    for t in treebank:
        conll = get_conll(t)
        write_conll(conll, sys.stdout)
        print()



