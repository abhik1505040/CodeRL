import glob
import os
import json
import sys

def get_formats(root_dir, split="train"):
    iterator = glob.glob(os.path.join(root_dir, "**", "input_output.json"), recursive=True)
    outliers = []
    for path in iterator:
        with open(path) as f:
            data = json.load(f)
            if not data or "fn_name" in data: continue
            length = len(data['inputs'])
            condition = length > 1 if split =="train" else length == 1
            if isinstance(data["inputs"], list) and condition:
                outliers.append(os.path.basename(os.path.dirname(path)))

    print(sorted(outliers))


def read_json(path):
    with open(path) as f: return json.load(f)

def write_json(obj, path):
    with open(path, 'w') as f: json.dump(obj, f)

def test_sample_code(sample_problem_path):
    sys.path.append("../")
    from generate import generate_compiler_result
    solutions = read_json(os.path.join(sample_problem_path, "solutions.json"))
    
    compiler_result = generate_compiler_result(solutions[4], sample_problem_path)
    print(compiler_result)

def get_ground_truth_results(root_dir):
    sys.path.append("../")
    from generate import generate_compiler_result
    iterator = glob.glob(os.path.join(root_dir, "**", "solutions.json"), recursive=True)

    program_count, passed_count = 0, 0
    problematic_paths = set()
    for path in sorted(iterator)[4095:]:
        solutions = read_json(path)
        problem_path = os.path.dirname(path)
        problem_base = os.path.basename(problem_path)

        for i, sol in enumerate(solutions):
            program_count += 1
            result = generate_compiler_result(sol, problem_path)
            
            if result > 0: passed_count += 1
            else: problematic_paths.add(problem_base)
            
            print("passed:total", f"{passed_count}:{program_count}", f"{problem_base}:{i}")
            print(problematic_paths)
    
    print(problematic_paths)

def rewrite_results(root_dir, func="rewrite"):
    sys.path.append("../")
    from generate import generate_compiler_result

    iterator = glob.glob(os.path.join(root_dir, "**", "gen_solutions.json"), recursive=True)
    ne_count = 0

    def _process_obj(obj, problem_path):
        nonlocal ne_count
        for sol in obj:
            new_result = generate_compiler_result(sol["code"], problem_path)
            if sol["result"] != new_result: ne_count += 1
            sol["result"] = new_result
        
        return sol

    for gen_path in iterator:
        problem_path = os.path.dirname(gen_path)
        baseline_path = os.path.join(problem_path, "baseline_solutions.json")
        critic_scores_path = os.path.join(problem_path, 'gen_solutions_critic_scores.pkl')

        if func == "remove":
            os.remove(gen_path)
            os.remove(baseline_path)
            os.remove(critic_scores_path)
        else:
            print(problem_path)
            gen_obj = _process_obj(read_json(gen_path), problem_path)
            baseline_obj = _process_obj(read_json(baseline_path), problem_path)
            
            write_json(gen_obj, gen_path)
            write_json(baseline_obj, baseline_path)

    print("Total:", ne_count)


if __name__ == "__main__":
    # get_formats("APPS/train", "train")
    # get_formats("APPS/test", "test")
    # test_sample_code("APPS/train/0000/")
    # rewrite_results("APPS/train", "remove")
    get_ground_truth_results("APPS/train")

