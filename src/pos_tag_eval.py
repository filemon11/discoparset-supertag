import corpus_reader
import subprocess
import sys

def get_wordpos(filename):

    params = ["discodop", "treetransforms", filename, "--inputfmt=discbracket", "--outputfmt=wordpos"]
    result = subprocess.check_output(params)

    result = str(result).split("\\n")

    sentences = []
    for line in result:
        line = line.split()
        word_pos = [tok.rsplit("/", 1) for tok in line]
        sentences.append(word_pos)
    if len(sentences[-1]) == 1 and len(sentences[-1][0]) == 1:
        sentences.pop()
    return sentences

if __name__ == "__main__":
    import argparse
    usage=""" """
    parser = argparse.ArgumentParser(description=usage, formatter_class = argparse.RawTextHelpFormatter)

    parser.add_argument("goldtbk", help="Gold corpus tbk format")
    parser.add_argument("preddiscbracket", help="Pred corpus discbracket format")
    
    args = parser.parse_args()

    CPOS = 3

    gold_corpus = corpus_reader.read_ctbk_corpus(args.goldtbk)

    gold_conll = [corpus_reader.get_conll(t) for t in gold_corpus]

    pred_corpus = get_wordpos(args.preddiscbracket)

    # replacements problems
    paren={"$LRB", "$["}

    assert(len(gold_conll) == len(pred_corpus))

    acc = 0
    tot = 0
    for golds, preds in zip(gold_conll, pred_corpus):
        assert(len(golds) == len(preds))
        
        for gtok, ptok in zip(golds, preds):
            if gtok[1] != ptok[0]:
                sys.stderr.write("Warning: {} {}\n".format(gtok[1], ptok[0]))

            if gtok[CPOS] == ptok[1] or (gtok[CPOS] in paren and ptok[1] in paren):
                acc += 1
            tot += 1
    

    print(round(acc / tot * 100, 1))



