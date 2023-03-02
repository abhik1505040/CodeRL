import pickle
import glob
import os
import numpy as np
import argparse
from itertools import chain

def print_results(root_dir):
    f_iterator = glob.glob(
        os.path.join(root_dir, "*.pkl")
    )

    results_summary = {}
    generated_solutions = -1

    for path in f_iterator:
        data = pickle.load(open(path, 'rb'))
        results = list(data.values())[0]
        
        compiler_outputs = results['results']
        difficulty = results['difficulty']
        generated_solutions = len(compiler_outputs)
        
        summary_list = results_summary.get(difficulty, [])
        summary_list.append(
            any(np.all(np.asarray(k) > 0) for k in compiler_outputs)
        )
        results_summary[difficulty] = summary_list

    results_summary["all"] = list(chain(*(list(k) for k in results_summary.values())))
    
    print("=" * 100)
    print(f"pass@{generated_solutions} (percentage) summary")
    for k, v in results_summary.items():
        print(f"\t{k}:", round(np.mean(v) * 100, 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("--root_dir", default="outputs/test_results", type=str, help="where the results are stored.")
    args = parser.parse_args()

    print_results(args.root_dir)
