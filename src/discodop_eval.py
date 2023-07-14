
import subprocess

def call_eval(gold_file, pred_file, disconly=False):
    """Just calls discodop eval and returns Prec, Recall, Fscore as floats"""
    
    params = ["discodop", "eval", gold_file, pred_file, "proper.prm", "--fmt=discbracket"]
    if disconly:
        params.append("--disconly")
        
    result = subprocess.check_output(params)

    result = str(result).split("\\n")

    recall = result[-6].split()[-1]
    prec = result[-5].split()[-1]
    fscore = result[-4].split()[-1]
    if "nan" not in [prec, recall, fscore]:
        return float(prec), float(recall), float(fscore)
    return 0, 0, 0

if __name__ == "__main__":
    print(call_eval("adev.discbracket", "MOD/tmp_dev.discbracket"))
