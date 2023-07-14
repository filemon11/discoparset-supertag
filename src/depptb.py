from conllu import parse
from conllu.models import TokenList, Token
from io import open

from typing import List, Tuple, Dict, Set

#dir = "/run/media/lukas/Data/owncloud/Dokumente/Heinrich-Heine-Universität/Bachelorarbeit/Supertags_in_discontinous_constituent_parsing/new/discoparset-v1.0/multilingual_disco_data/data/dptb/train_silver.conll"
#def import_data(dir = dir) -> None:
#    data_file = open(dir, "r", encoding="utf-8")

train_split : slice = slice(3915, 43746 + 1)
dev_split   : slice = slice(43747, 45446 + 1)
test_split  : slice = slice(45447, 47862 + 1)
splits : Dict[str, slice] = {"train" : train_split, "dev" : dev_split, "test" : test_split}

def sentence_parse(sentence : TokenList) -> Tuple[List[str], List[str]]:
    tokens : List[str] = []
    pos : List[str] = []
    in_dep : List[str] = []
    in_deptype : List[str] = []
    out_dep : List[Tuple[List[int], List[int]]] = [([],[]) for _ in range(len(sentence))]
    total_dep : List[Tuple[List[int], List[int]]] = [([],[]) for _ in range(len(sentence))]
    out_deptypes : List[List[str]] = [[] for _ in range(len(sentence))]

    max_diff = 5
    max_diff_out = 2
    for token in sentence:
        tokens.append(token['form'])
        pos.append(token["upos"])
        if True:
            diff : int = token['id']-token["head"]
            if diff > max_diff_out:
                diff = max_diff_out
            elif diff < -max_diff_out:
                diff = -max_diff_out
            if token["head"] > 0:
                out_dep[token["head"] - 1][0 if diff < 0 else 1].append(diff) #str(token["deprel"]).split(":")[0])#+ ":" + str(token["upos"]))
                total_dep[token["head"] -1][0 if diff < 0 else 1].append(token['id'] -1)
                out_deptypes[token["head"] -1].append(str(token["deprel"]).split(":")[0])
        in_diff : int = token["head"] - token["id"]
        if in_diff > max_diff:
            in_diff = max_diff
        elif in_diff < -max_diff:
            in_diff = -max_diff

        in_dep.append("" if token["deprel"] == "punct" and False else token["deprel"].split(":")[0] + "/" + str(in_diff))# + ":" + ("ROOT" if token["head"] == 0 else sentence[token["head"] - 1]["upos"]))
        in_deptype.append(token["deprel"].split(":")[0])

    #return tokens, [f"{i}+{max(set(o[0])) if len(o[0]) > 0 else ''}_{min(set(o[1])) if len(o[1]) > 0 else ''}" for p, i, o in zip(pos, in_dep, out_dep)]
    
    #return tokens, [f"{i}_{sentence[min(set(o_idx[0]))]['deprel'] if len(o_idx[0]) > 0 else 'null'}_{sentence[max(set(o_idx[0]))]['deprel'] if len(o_idx[0]) > 0 else 'null'}_{sentence[min(set(o_idx[1]))]['deprel'] if len(o_idx[1]) > 0 else 'null'}_{sentence[max(set(o_idx[1]))]['deprel'] if len(o_idx[1]) > 0 else 'null'}" for i, o_idx in zip(in_deptype, total_dep)]
    #return tokens, [f"{i}_{sentence[max(set(o_idx[0]))]['deprel'].split(':')[0] if len(o_idx[0]) > 0 else 'null'}_{sentence[min(set(o_idx[1]))]['deprel'].split(':')[0] if len(o_idx[1]) > 0 else 'null'}" for i, o_idx in zip(in_deptype, total_dep)]
    return tokens, [f"{i}_{str(max(set(o[0]))) if len(o[0]) > 0 else '0'}_{str(min(set(o[1]))) if len(o[1]) > 0 else '0'}" for i, o in zip(in_deptype, out_dep)]

def corpus_parse(filename : str, split) -> Tuple[List[List[str]], List[List[str]]]:
    data_file = open(filename, "r", encoding="utf-8").read()
    parselist = parse(data_file)[splits[split]]
    corpus_tokens       : List[List[str]] = []
    corpus_supertags    : List[List[str]] = []

    for sentence in parselist:
        tokens      : List[str]
        supertags   : List[str]

        tokens, supertags = sentence_parse(sentence)

        corpus_tokens.append(tokens)
        corpus_supertags.append(supertags)

    return corpus_tokens, corpus_supertags

def build_set(corpus_supertags : List[List[str]]) -> Tuple[Set[str], Dict[str, int]]:
    flat_list : List[str] = [supertag for sentence in corpus_supertags for supertag in sentence]
    frequencies : Dict[str, int] = {supertag : 0 for supertag in flat_list}
    for sentence in corpus_supertags:
        for supertag in sentence:
            frequencies[supertag] += 1
    return set(flat_list), frequencies