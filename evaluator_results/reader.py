import os
from typing import List, Dict, Tuple, TypeVar, Type, Literal, Iterable, Set
from copy import copy

baseline = ("baseline", "baseline")
to_compare = {"pipeline_bank" : "pipeline_bank", "auxiliary" : "auxiliary"}

ranges : Dict[str, Tuple[int, int]] = {"labelled" : (8, 26), "unlabelled" : (30, 48), "errors" : (5, 28), "labelled_wh" : (8, 46), "unlabelled_wh" : (50, 88), "labelled_dep_gap" : (8, 35), "unlabelled_dep_gap" : (38, 66), "labelled_dep_right" : (8, 34), "unlabelled_dep_right" : (37, 64)}

# discontinuous phenomena

def get_lines(path : str) -> List[str]:
    outlist : List[str] = []
    with open(path, "r") as file:
        for line in file:
            outlist.append(line.strip())
    return outlist

def get_results(lines : List[str], fromto: Tuple[int, int], separator : str = "\t", min_at : Tuple[int, float] | None = (0, 5), use_as_index : int | Tuple[int, Literal["+", "-"]] = 0) -> Dict[str, List[float]]:
    outlines : Dict[str, List[float]] = {}

    for line in lines[fromto[0]:fromto[1]]:
        split_line : List[str] = [entry.strip() for entry in line.split(separator)]

        float_line : List[float]
        if isinstance(use_as_index, int):
            float_line = [float(entry) for entry in split_line[:use_as_index] + split_line[use_as_index+1:]]
        elif use_as_index[1] == "+":
            float_line = [float(entry) for entry in split_line[:use_as_index[0]]]
        else:
            raise Exception("- not implemented")

        if min_at is None or float_line[min_at[0]] >= min_at[1]:
            if isinstance(use_as_index, int):
                outlines[split_line[use_as_index]] = float_line

            elif use_as_index[1] == "+":
                outlines[" ".join(split_line[use_as_index[0]:])] = float_line

            else:
                raise Exception("- not implemented")

    return outlines

def difference_formatter(baseline : float, new : float, format : Type, hundred_without_zero : bool = True, mode : Literal["diff", "errorred"] = "diff") -> str:
    if mode == "diff":
        diff : float = new - baseline
        return f"{100 if new == 100 and hundred_without_zero else format(new)} ({'+' if diff >= 0 else ''}{round(diff, 1)})"
    else:
        red : str
        if baseline != 0:
            fred : float = round((new/baseline-1)*100, 1)
            red = f"{'+' if fred >= 0 else ''}{fred}"
        elif new != 0:
            red = "+\\infty"
        else:
            red = "\pm 0"
        return f"{100 if new == 100 and hundred_without_zero else format(new)} ({red}\%)"

def line_difference_formatter(baseline : List[float], new : List[float], spare : Dict[int, type] = {0 : int}, out_formats : Dict[int, type] = {}, hundred_without_zero : bool = True,
                              mode : Literal["diff", "errorred"] = "diff") -> List[str]:
    outline : List[str] = []
    for num, (baseline_entry, new_entry) in enumerate(zip(baseline, new)):
        if num in spare.keys():
            outline.append(str(spare[num](new_entry)))
        else:
            format : Type = out_formats[num] if num in out_formats.keys() else float
            outline.append(difference_formatter(baseline_entry, new_entry, format, hundred_without_zero, mode))
    return outline

def table_difference_formatter(baseline : Dict[str, List[float]], new : Dict[str, List[float]], spare : Dict[int, type] = {0 : int}, out_formats : Dict[int, type] = {}, 
                               hundred_without_zero : bool = True, mode : Literal["diff", "errorred"] = "diff", sort : bool = False) \
        -> List[List[str]]:
    outtable : List[List[str]] = []

    for line_name, baseline_line in (sorted(baseline.items(), key = lambda x : x[0]) if sort else baseline.items()):
        outline : List[str] = [line_name] + line_difference_formatter(baseline_line, new[line_name], spare, out_formats, hundred_without_zero, mode)
        outtable.append(outline)
    return outtable

def line_normal_formatter(line : List[float], spare : Dict[int, type] = {0 : int}, out_formats : Dict[int, type] = {}, hundred_without_zero : bool = True) -> List[str]:
    return [(str(spare[num](entry)) if num in spare.keys() else str(100 if entry == 100 and hundred_without_zero else out_formats[num](entry) if num in out_formats.keys() else entry)) for num, entry in enumerate(line)]
    
def table_normal_formatter(table : Dict[str, List[float]], spare : Dict[int, type] = {0 : int}, out_formats : Dict[int, type] = {}, hundred_without_zero : bool = True, sort : bool = False) -> List[List[str]]:
    return [[line_name] + line_normal_formatter(line, spare, out_formats, hundred_without_zero) for line_name, line in (sorted(table.items(), key = lambda x : x[0]) if sort else table.items())]
    
def outprocessor(table : List[List[str]], entry_sep : str = "&", line_sep : str = "\\\n") -> str:
    fancy_rows : List[str] = [entry_sep.join(row) for row in table]
    fancy_table : str = line_sep.join(fancy_rows)
    return fancy_table

def outprocessor_multi(tables : List[List[List[str]]], entry_sep : str = " & ", line_sep : str = " \\\\ \n", start_with : int = 2) -> str:
    big_table : List[List[str]] = []
    for t0, *tothers in zip(*tables):
        line : List[str] = copy(t0)
        for table in tothers:
            line += copy(table[start_with:])
        big_table.append(line)
    return outprocessor(big_table, entry_sep, line_sep)

# process discontinuous phenomena
disco_name : str = "_disco_errors"
build_disco_path = lambda name : os.path.join(name, name + disco_name)

baseline_path : str = build_disco_path(baseline[1])

to_compare_paths : Dict[str, str] = {key : build_disco_path(name) for key, name in to_compare.items()}

baseline_labelled : Dict[str, List[float]] = get_results(get_lines(baseline_path), ranges["labelled"])
baseline_unlabelled : Dict[str, List[float]] = get_results(get_lines(baseline_path), ranges["unlabelled"])

to_compare_labelled : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["labelled"]) for key, path in to_compare_paths.items()}
to_compare_unlabelled : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["unlabelled"]) for key, path in to_compare_paths.items()}

labelled_table : str = outprocessor_multi([table_normal_formatter(baseline_labelled)] +
                                           [table_difference_formatter(baseline_labelled, to_compare) for to_compare in to_compare_labelled.values()])

unlabelled_table : str = outprocessor_multi([table_normal_formatter(baseline_unlabelled)] +
                                           [table_difference_formatter(baseline_unlabelled, to_compare) for to_compare in to_compare_unlabelled.values()])

print(labelled_table)
print("\n")
print(unlabelled_table)
print("\n")

# errors specific functions

def add_elementwise(l1 : List[float], l2 : List[float]) -> List[float]:
    return [e1 + e2 for e1, e2 in zip(l1, l2)]

def add_subscores(results : Dict[str, List[float]], prefix : str = "disco", separator : str = "-") -> None:
    """ATTENTION: adds in-place"""
    for key in results.keys():
        split_key : List[str] = key.split(separator, 1)
        if split_key[0] == prefix:
            results[split_key[1]] = add_elementwise(results[split_key[1]], results[key])

def add_missing(all_results : Iterable[Dict[str, List[float]]], standard : float = 0.0) -> None:
    """ATTENTION: in-place"""
    all_keynames : Set[str] = set([entry for results in all_results for entry in list(results.keys())])

    table_len : int = len(next(iter(all_results))[next(iter(all_keynames))])

    for keyname in all_keynames:
        for results in all_results:
            if keyname not in results.keys():
                results[keyname] = [standard] * table_len
            

# process errors
errors_name : str = "bpa/full.error_counts"
build_errors_path = lambda name : os.path.join(name, errors_name)

baseline_errors_path : str = build_errors_path(baseline[1])
to_compare_errors_paths : Dict[str, str] = {key : build_errors_path(name) for key, name in to_compare.items()}

baseline_errors : Dict[str, List[float]] = get_results(get_lines(baseline_errors_path), ranges["errors"], separator = " ", min_at = None, use_as_index = (2, "+"))
add_subscores(baseline_errors)

to_compare_errors : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["errors"], separator = " ", min_at = None, use_as_index = (2, "+")) for key, path in to_compare_errors_paths.items()}
for errors in to_compare_errors.values():
    add_subscores(errors)

add_missing([baseline_errors] + list(to_compare_errors.values()))

errors_table : str = outprocessor_multi([table_normal_formatter(baseline_errors, spare = {}, out_formats = {0 : int, 1 : int}, sort = True)] +
                                            [table_difference_formatter(baseline_errors, to_compare, spare = {}, out_formats = {0 : int, 1 : int}, mode = "errorred", sort = True) for to_compare in to_compare_errors.values()], start_with = 1)


print(errors_table)
print("\n")

# wh extraction specific
disco_name_wh : str = "_whextraction_separate_disco_errors"
build_disco_path_wh = lambda name : os.path.join(name, name + disco_name_wh)

baseline_path_wh : str = build_disco_path_wh(baseline[1])

to_compare_paths_wh : Dict[str, str] = {key : build_disco_path_wh(name) for key, name in to_compare.items()}

baseline_labelled_wh : Dict[str, List[float]] = get_results(get_lines(baseline_path_wh), ranges["labelled_wh"])
baseline_unlabelled_wh : Dict[str, List[float]] = get_results(get_lines(baseline_path_wh), ranges["labelled_wh"])

to_compare_labelled_wh : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["labelled_wh"]) for key, path in to_compare_paths_wh.items()}
to_compare_unlabelled_wh : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["labelled_wh"]) for key, path in to_compare_paths_wh.items()}

labelled_table_wh : str = outprocessor_multi(#[table_normal_formatter(baseline_labelled_wh)] +
                                           [table_difference_formatter(baseline_labelled_wh, to_compare) for to_compare in to_compare_labelled_wh.values()])

unlabelled_table_wh : str = outprocessor_multi(#[table_normal_formatter(baseline_unlabelled_wh)] +
                                           [table_difference_formatter(baseline_unlabelled_wh, to_compare) for to_compare in to_compare_unlabelled_wh.values()])

print(labelled_table_wh)
print("\n")
print(unlabelled_table_wh)
print("\n")

# dep specific gap
disco_name_dep_gap : str = "_extraposed_dependents_gap_disco_errors"
build_disco_path_dep_gap = lambda name : os.path.join(name, name + disco_name_dep_gap)

baseline_path_dep_gap : str = build_disco_path_dep_gap(baseline[1])

to_compare_paths_dep_gap : Dict[str, str] = {key : build_disco_path_dep_gap(name) for key, name in to_compare.items()}

baseline_labelled_dep_gap : Dict[str, List[float]] = get_results(get_lines(baseline_path_dep_gap), ranges["labelled_dep_gap"])
baseline_unlabelled_dep_gap : Dict[str, List[float]] = get_results(get_lines(baseline_path_dep_gap), ranges["labelled_dep_gap"])

to_compare_labelled_dep_gap : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["labelled_dep_gap"]) for key, path in to_compare_paths_dep_gap.items()}
to_compare_unlabelled_dep_gap : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["labelled_dep_gap"]) for key, path in to_compare_paths_dep_gap.items()}

labelled_table_dep_gap : str = outprocessor_multi(#[table_normal_formatter(baseline_labelled_dep_gap)] +
                                           [table_difference_formatter(baseline_labelled_dep_gap, to_compare) for to_compare in to_compare_labelled_dep_gap.values()])

unlabelled_table_dep_gap : str = outprocessor_multi(#[table_normal_formatter(baseline_unlabelled_dep_gap)] +
                                           [table_difference_formatter(baseline_unlabelled_dep_gap, to_compare) for to_compare in to_compare_unlabelled_dep_gap.values()])

print(labelled_table_dep_gap)
print("\n")
print(unlabelled_table_dep_gap)
print("\n")

# dep specific right
disco_name_dep_right : str = "_extraposed_dependents_right_disco_errors"
build_disco_path_dep_right = lambda name : os.path.join(name, name + disco_name_dep_right)

baseline_path_dep_right : str = build_disco_path_dep_right(baseline[1])

to_compare_paths_dep_right : Dict[str, str] = {key : build_disco_path_dep_right(name) for key, name in to_compare.items()}

baseline_labelled_dep_right : Dict[str, List[float]] = get_results(get_lines(baseline_path_dep_right), ranges["labelled_dep_right"])
baseline_unlabelled_dep_right : Dict[str, List[float]] = get_results(get_lines(baseline_path_dep_right), ranges["labelled_dep_right"])

to_compare_labelled_dep_right : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["labelled_dep_right"]) for key, path in to_compare_paths_dep_right.items()}
to_compare_unlabelled_dep_right : Dict[str, Dict[str, List[float]]] = {key : get_results(get_lines(path), ranges["labelled_dep_right"]) for key, path in to_compare_paths_dep_right.items()}

labelled_table_dep_right : str = outprocessor_multi(#[table_normal_formatter(baseline_labelled_dep_right)] +
                                           [table_difference_formatter(baseline_labelled_dep_right, to_compare) for to_compare in to_compare_labelled_dep_right.values()])

unlabelled_table_dep_right : str = outprocessor_multi(#[table_normal_formatter(baseline_unlabelled_dep_right)] +
                                           [table_difference_formatter(baseline_unlabelled_dep_right, to_compare) for to_compare in to_compare_unlabelled_dep_right.values()])

print(labelled_table_dep_right)
print("\n")
print(unlabelled_table_dep_right)
print("\n")
