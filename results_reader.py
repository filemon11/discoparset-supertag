import numpy as np
import matplotlib.pyplot as plt

from cycler import cycler
monochrome = (cycler('color', ['k']) * cycler('marker', ['', '.']) *
              cycler('linestyle', ['-', '--', '-.', ':']))
plt.rc('axes', prop_cycle=monochrome)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "lmodern",
})


from typing import List, Dict, Tuple, Optional

def print_read(filename : str, column = int, zero_lines = 0):
    with open(filename, mode = "r") as file:
        for n, line in enumerate(file):
            if n == 0:
                continue
            print(line.split()[column])
            for _ in range(zero_lines):
                print()


def read(filename : str, column = int) -> List[float]:
    result_list : List[float] = []
    with open(filename, mode = "r") as file:
        for n, line in enumerate(file):
            if n == 0:
                continue
            result_list.append(float(line.split()[column]))

    return result_list

def get_name(filename : str) -> str:
    return filename.split("/")[-2]

def plot(filenames : List[str], column : int | List[int], img_file : str, dpi : float, xlabel : str = "Epochs", ylabel : str = "F-score", legend_entries : None | List[str] = None):
    
    if not isinstance(column, list):
        column = [column] * len(filenames)

    if legend_entries == None:
        legend_entries = [get_name(file) for file in filenames]
    assert(isinstance(legend_entries, list))
    
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for file, legend_entry, c in zip(filenames, legend_entries, column):
        plt.plot(read(file, 0), read(file, c), label = legend_entry)

    plt.xlim(xmin=0, xmax=100)
    plt.ylim(ymax=100)
    plt.grid()

    leg = plt.legend(loc = 'right', facecolor='white', framealpha=1)

    plt.savefig(img_file, dpi = dpi)

def make_dir(*args : str) -> str:
    return "/".join(args)

LOG = "learning_log"
MODEL_DIR = "/run/media/lukas/Data/owncloud/Dokumente/Heinrich-Heine-Universität/Bachelorarbeit/Supertags_in_discontinous_constituent_parsing/clean/discoparset-supertag/pretrained_final_models/"

#to_compare : Dict[str, Tuple[str, Optional[Dict[str, int]]]] = {"\\textsc{CCG}" : ("CCG", {"train" : 4, "dev" : 12, "f1" : 20, "disc_f1" : 21}), "\\textsc{CCG}$^{\\textrm{gate}}$" : ("CCG_gate", {"train" : 4, "dev" : 12, "f1" : 20, "disc_f1" : 21})}
#to_compare : Dict[str, Tuple[str, Optional[Dict[str, int]]]] = {"\\textsc{CCG}$^{\\textrm{gate}}$" : ("CCG_gate", {"train" : 4, "dev" : 12, "f1" : 20, "disc_f1" : 21}), "\\textsc{CCG}$_{600}^{\\textrm{gate}}$" : ("CCG_600_gate", {"train" : 4, "dev" : 12, "f1" : 20, "disc_f1" : 21})}
#to_compare : Dict[str, Tuple[str, Optional[Dict[str, int]]]] = {"\\textsc{CTR}$_3^{\\textrm{gate}}$" : ("control_3_gate", {"train" : 4, "dev" : 12, "f1" : 20, "disc_f1" : 21}), "\\textsc{CTR}$_{3,600}^{\\textrm{gate}}$" : ("control_3_gate_600", {"train" : 4, "dev" : 12, "f1" : 20, "disc_f1" : 21})}
to_compare : Dict[str, Tuple[str, Optional[Dict[str, int]]]] = {"\\textsc{Ctr}$_{3,600}^{\\textrm{gate}}$" : ("control_3_gate_600", {"train" : 4, "dev" : 12, "f1" : 20, "disc_f1" : 21}), "\\textsc{CCG}$_{600}^{\\textrm{gate}}$" : ("CCG_600_gate", {"train" : 4, "dev" : 12, "f1" : 20, "disc_f1" : 21})}
#"control 3" : ("control_3", {"train" : 3, "dev" : 10, "f1" : 17, "disc_f1" : 18})
#"control 3 gate" : ("control_3_gate", {"train" : 3, "dev" : 10, "f1" : 17, "disc_f1" : 18})

def plot_comparisons(save_dir : str = "..", dpi = 300, to_compare : Dict[str, Tuple[str, Optional[Dict[str, int]]]] = to_compare, 
                        params : Dict[str, int] = {"train" : 3, "dev" : 10}, img_format = "png",
                        model_dir = MODEL_DIR, log = LOG):
    

    #for param, column in params.items():
    plot([make_dir(model_dir, model, log) for model, _ in to_compare.values() for _ in params],
             column = [(additional[param] if additional is not None else column) for _, additional in to_compare.values() for param in params],
             img_file = make_dir(save_dir, f"comparison.{img_format}"),
             dpi = dpi,
             legend_entries = [f"{key} {param}" for key in to_compare.keys() for param in params])
        