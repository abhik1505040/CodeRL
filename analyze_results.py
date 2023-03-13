import pickle
import glob
import os
import pprint
import numpy as np
import argparse
import json
from itertools import chain
import random

RESULT_KEYS = ["compileError", "runtimeError", "failedTest", "passedTest"]

def determine_error(result):
    result = np.asarray(result)
    if result.size == 0:
        print("Empty result array! This shouldn't happen!!!")

    # Compile error
    if np.any(result == -2):
        return RESULT_KEYS[0], (result == -2).nonzero()[0][0]
    # Runtime error
    elif np.any(result == -1):
        return RESULT_KEYS[1], (result == -1).nonzero()[0][0]
    # Failed unit tests
    elif np.any(result == False):
        return RESULT_KEYS[2], -1
    # Passed all unit tests
    else:
        return RESULT_KEYS[3], -1


def extract_error_name(stacktrace):
    return stacktrace[0].__class__.__name__
    

def print_results(root_dir, test_dir):
    f_iterator = glob.glob(
        os.path.join(root_dir, "*.pkl")
    )
    problem_roots = sorted(glob.glob(test_dir + '/*'))
    results_summary = {}
    error_summary = {}

    for path in f_iterator:
        data = pickle.load(open(path, 'rb'))
        results = list(data.values())[0]

        
        index = os.path.basename(path).rsplit(".pkl", 1)[0]
        meta_path = os.path.join(problem_roots[int(index)], "metadata.json")
        difficulty = json.load(open(meta_path))["difficulty"]
        
        compiler_outputs = results['results']
        compiler_errors = results['errors']
        summary_dict = results_summary.get(difficulty, {})
        
        problem_dict = {k: 0 for k in RESULT_KEYS}
        for i, k in enumerate(compiler_outputs):
            e, first_idx = determine_error(k)
            # print(first_idx)
            if e in RESULT_KEYS[:2]:
                compiler_error = extract_error_name(compiler_errors[i][first_idx])
                error_dict = error_summary.get(e, {})
                error_dict[compiler_error] = error_dict.get(compiler_error, 0) + 1
                error_summary[e] = error_dict

            problem_dict[e] = problem_dict[e] + 1

        for e, v in problem_dict.items():
            result_list = summary_dict.get(e, [])
            # len(compiler_outputs) may not be
            # the same for all problems
            result_list.append(v / len(compiler_outputs))
            summary_dict[e] = result_list
        
        results_summary[difficulty] = summary_dict

    def _get_combined_list(list_of_dicts, key):
        output = []
        for d in list_of_dicts: output += d.get(key, [])
        return output 

    results_summary['all'] = {
        k: _get_combined_list(results_summary.values(), k) for k in RESULT_KEYS
    }

    print("=" * 100)
    format_dict = {}

    # reformat the dict to show info like Figure 8
    for e in RESULT_KEYS:
        result_dict = format_dict.get(e, {})
        for k, v in results_summary.items():
            result_dict[k] = round(np.mean(v.get(e, [0])) * 100, 2)
        format_dict[e] = result_dict
    
    print("*" * 10, "Hidden test result distribution:", "*" * 10)
    print(json.dumps(format_dict, indent=4))

    # normalize the dict
    format_dict = {}
    for k, v in error_summary.items():
        total = sum(v.values())
        updated_dict = {k_n: round((v_n / total) * 100, 4) for k_n, v_n in v.items()}
        format_dict[k] = updated_dict
            
    print("*" * 10, "Error type distribution:", "*" * 10)
    print(json.dumps(format_dict, indent=4))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("--root_dir", default="outputs/test_results_actor", type=str, help="where the results are stored.")
    parser.add_argument("--test_dir", default="data/APPS/test", type=str, help="Test dataset root.")
    
    args = parser.parse_args()

    print_results(args.root_dir, args.test_dir)
