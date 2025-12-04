import jsonlines
import os
import copy
# import pymysql
import numpy as np
import pandas as pd
import regex as re

import traceback
from typing import List, Union, Iterable, Dict
import itertools


def strip_test_example(model_response):
    """Remove test cases from the model response"""
    kuo_start = []
    kuo_end = []
    kuo_left, kuo_right = 0, 0
    kuo_flag = False
    model_response = model_response.split("\n")
    for seg_id, _ in enumerate(model_response):
        # remove common test-case prefixes
        if _.startswith("if __name__ ==") or _.startswith("# Test code") or _.startswith("# Example test case") or _.startswith("# Example code") or _.lower().startswith("# test case") or _.startswith("# Unit test") or _.startswith("# Test case") or _.lower().startswith("# test cases") or _.startswith("# Validate using the given test cases"):
            model_response = model_response[:seg_id]
            break

    # Record the positions of parentheses
    for seg_id, _ in enumerate(model_response):
        if _.startswith("assert") or _.startswith("print"):
            kuo_flag = True
            kuo_start.append(seg_id)
            kuo_left, kuo_right = len(re.findall("\(", _)), len(re.findall("\)", _))
            if kuo_left == kuo_right and kuo_left != 0 and kuo_flag:
                kuo_end.append(seg_id + 1)
                kuo_left, kuo_right = 0, 0
                kuo_flag = False

        if kuo_flag:
            for seg_id_1 in range(seg_id + 1, len(model_response)):
                kuo_left += len(re.findall("\(", model_response[seg_id_1]))
                kuo_right += len(re.findall("\)", model_response[seg_id_1]))
                if kuo_left == kuo_right and kuo_left != 0 and kuo_flag:
                    kuo_end.append(seg_id_1 + 1)
                    kuo_left, kuo_right = 0, 0
                    kuo_flag = False

    if len(kuo_start) == len(kuo_end) + 1:
        kuo_end.append(len(model_response))

    model_response_final = model_response
    for n in range(len(kuo_start)):
        model_response_final = [_ for _ in model_response_final if _ not in model_response[kuo_start[n]:kuo_end[n]]]

    return "\n".join(model_response_final).rstrip()
            

def extract_standard(response):
    """Extract standard answer and imports"""
    try:
        model_response = re.findall("【Code Start】((?:[\s\S]*?))【Code End】", response)
        entry_point = "merge_nested_dicts"

        if len(model_response) > 1:
            model_response_with_entry_point = [_ for _ in model_response if f"def {entry_point}" in _]
            if len(model_response_with_entry_point) != 0:
                model_response = model_response_with_entry_point[0]
            else:
                model_response = model_response[0]
        else:
            model_response = model_response[0]

        if len(re.findall("```((?:[\s\S]*?))```", model_response)) != 0:
            if len(re.findall("```((?:[\s\S]*?))```", model_response)) == 1:
                model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
            else:
                model_response_with_entry_point = [_ for _ in re.findall("```((?:[\s\S]*?))```", model_response) if f"def {entry_point}" in _]
                if len(model_response_with_entry_point) != 0:
                    model_response = model_response_with_entry_point[0]
                else:
                    model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]

        print("……" * 20)

        model_response = model_response.strip("\n`【Code Start】End")
        model_response = strip_test_example(model_response)

        import_code = ""
        completion = ""
        completion_segs = model_response.split("\n")
        print(completion_segs)

        idx_def_in = []
        for idx, seg in enumerate(completion_segs):
            if "def " in seg:
                idx_def_in.append(idx)

        for idx, seg in enumerate(completion_segs):
            if seg.strip().lower() in ["python", "python code"] or "print" in seg or "assert" in seg:
                continue

            if len(idx_def_in) == 0 or entry_point not in model_response:
                completion += f"\n{seg}"

            else:
                if "import " in seg:
                    if idx < idx_def_in[0]:
                        import_code += f"{seg}\n"
                    else:
                        completion += f"\n{seg}"

                elif "def " in seg and entry_point.strip() in seg:
                    completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                    break

                else:
                    completion += "\n" + f"    {seg}"

    except:
        print("………………………………………………………………………………" * 20)

        model_response = response
        if len(re.findall("```((?:[\s\S]*?))```", model_response)) != 0:
            if len(re.findall("```((?:[\s\S]*?))```", model_response)) == 1:
                model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
            else:
                model_response_with_entry_point = [_ for _ in re.findall("```((?:[\s\S]*?))```", model_response) if f"def {entry_point}" in _]
                if len(model_response_with_entry_point) != 0:
                    model_response = model_response_with_entry_point[0]
                else:
                    model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]

        model_response = model_response.strip("\n`【Code Start】End")
        model_response = strip_test_example(model_response)

        import_code = ""
        completion = ""
        completion_segs = model_response.split("\n")
        print(completion_segs)

        idx_def_in = []
        for idx, seg in enumerate(completion_segs):
            if "def " in seg:
                idx_def_in.append(idx)

        for idx, seg in enumerate(completion_segs):
            if seg.strip() in ["python", "python code"] or "print" in seg or "assert" in seg:
                continue

            if len(idx_def_in) == 0 or entry_point not in model_response:
                completion += f"\n{seg}"

            else:
                if "import " in seg:
                    if idx < idx_def_in[0]:
                        import_code += f"{seg}\n"
                    else:
                        completion += f"\n{seg}"

                elif "def " in seg and entry_point.strip() in seg:
                    print(1)
                    completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                    break

                else:
                    completion += "\n" + f"    {seg}"

    return import_code, completion
    


def code3_standard_with_answer_2_human_eval_formal(table_name, model, problem_file, sample_file, dropped_task_ids=[]):
    """Convert code3 results into human-eval standard format"""
    result = pd.read_excel(f"./data/model_response/{table_name}.xlsx").to_dict("records")
    
    exception_n = 0
    total_answer_completed = 0

    with jsonlines.open(problem_file, mode="w") as ouf1, jsonlines.open(sample_file, mode="w") as ouf2:
        for line in result:
            task_id = line["task_id"]
            if task_id in dropped_task_ids or pd.isnull(line["question_turns_1"]):
                continue

            total_answer_completed += 1
            prompt = line["question_turns_1"]
            reference = line["reference_turns_1"] if "reference" in line else None
            test = line["test"]

            entry_point = re.findall("def (.*)?\(", line["question_turns_1"])[0]
            entry_point = entry_point.strip()

            try:
                if not line[f"model_answer_{model}_turns_1"] or line[f"model_answer_{model}_turns_1"].strip() == "lock":
                    continue

                model_response = re.findall("【代码开始】((?:[\s\S]*?))【代码结束】", line[f"model_answer_{model}_turns_1"])

                if len(model_response) == 0:
                    exception_n += 1
                if len(model_response) > 1:
                    model_response_with_entry_point = [_ for _ in model_response if f"def {entry_point}" in _]
                    if len(model_response_with_entry_point) != 0:
                        model_response = model_response_with_entry_point[0]
                    else:
                        model_response = model_response[0]
                else:
                    model_response = model_response[0]

                if len(re.findall("```((?:[\s\S]*?))```", model_response)) != 0:
                    if len(re.findall("```((?:[\s\S]*?))```", model_response)) == 1:
                        model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
                    else:
                        model_response_with_entry_point = [_ for _ in re.findall("```((?:[\s\S]*?))```", model_response) if f"def {entry_point}" in _]
                        if len(model_response_with_entry_point) != 0:
                            model_response = model_response_with_entry_point[0]
                        else:
                            model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]

                print("……" * 20)

                model_response = model_response.strip("\n`【Code Start】End")
                model_response = strip_test_example(model_response)

                import_code = ""
                completion = ""
                completion_segs = model_response.split("\n")
                idx_def_in = []
                for idx, seg in enumerate(completion_segs):
                    if "def " in seg:
                        idx_def_in.append(idx)

                for idx, seg in enumerate(completion_segs):
                    if seg.strip().lower() in ["python", "python code"] or "print" in seg or "assert" in seg:
                        continue

                    if len(idx_def_in) == 0 or entry_point not in model_response:
                        completion += f"\n{seg}"

                    else:
                        if "import " in seg:
                            if idx < idx_def_in[0]:
                                import_code += f"{seg}\n"
                            else:
                                completion += f"\n{seg}"
                        elif "def " in seg and entry_point.strip() in seg:
                            completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                            break
                        else:
                            completion += "\n" + f"    {seg}"

                ouf1.write({"task_id": task_id, "import_code": import_code, "prompt": prompt, "canonical_solution": reference, "test": test, "entry_point": entry_point, "origin_model_response": model_response})
                ouf2.write({"task_id": task_id, "completion": completion})

            except:
                print("………" * 20)


                model_response = line[f"model_answer_{model}_turns_1"]
                if len(re.findall("```((?:[\s\S]*?))```", model_response)) != 0:
                    if len(re.findall("```((?:[\s\S]*?))```", model_response)) == 1:
                        model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
                    else:
                        model_response_with_entry_point = [_ for _ in re.findall("```((?:[\s\S]*?))```", model_response) if f"def {entry_point}" in _]
                        if len(model_response_with_entry_point) != 0:
                            model_response = model_response_with_entry_point[0]
                        else:
                            model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]

                model_response = model_response.strip("\n`【Code Start】End")
                model_response = strip_test_example(model_response)

                import_code = ""
                completion = ""
                completion_segs = model_response.split("\n")
                idx_def_in = []
                for idx, seg in enumerate(completion_segs):
                    if "def " in seg:
                        idx_def_in.append(idx)

                for idx, seg in enumerate(completion_segs):
                    if seg.strip() in ["python", "python code"] or "print" in seg or "assert" in seg:
                        continue
                    if len(idx_def_in) == 0 or entry_point not in model_response:
                        completion += f"\n{seg}"
                    else:
                        if "import " in seg:
                            if idx < idx_def_in[0]:
                                import_code += f"{seg}\n"
                            else:
                                completion += f"\n{seg}"
                        elif "def " in seg and entry_point.strip() in seg:
                            print(1)
                            completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                            break
                        else:
                            completion += "\n" + f"    {seg}"

                ouf1.write({"task_id": task_id, "import_code": import_code, "prompt": prompt, "canonical_solution": reference, "test": test, "entry_point": entry_point, "origin_model_response": model_response})
                ouf2.write({"task_id": task_id, "completion": completion})

    print(f"Model did not follow instruction: {exception_n} items")
    
    return exception_n, total_answer_completed



def format_and_evaluation(sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 30,
    problem_file: str = None):
    """Extract standardized format file and run sandbox evaluation."""
    from human_eval.evaluate_functional_correctness import entry_point
    pass_k = entry_point(sample_file=sample_file, k=k, n_workers=n_workers, timeout=timeout, problem_file=problem_file)

    return pass_k


def string_match(str1, str2):
    """Check whether two strings are equivalent using MD5"""
    import hashlib
    md5_obj = hashlib.md5()
    
    # Convert string to bytes
    md5_obj.update(str1.encode('utf-8'))
    str1_hash = md5_obj.hexdigest()
    md5_obj = hashlib.md5()
    md5_obj.update(str2.encode('utf-8'))
    str2_hash = md5_obj.hexdigest()

    return str1_hash == str2_hash
    

def caculate_pass_k(table_name, model_names):
    """Calculate pass@k score based on database code answers"""
    model_scores = {}
    for model in model_names:
        problem_file = f"./data/problem_{table_name}_{model}.jsonl"
        sample_file = f"./data/samples_{table_name}_{model}.jsonl"

        model_scores[model] = format_and_evaluation(problem_file=problem_file, sample_file=sample_file)

    return model_scores



def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def calculate_pass_each_test_sample(table_name, selected_models=None, if_split=True):
    """Calculate score for each individual test case."""
    pass
    if if_split:
        new_table_name = f"{table_name}_split"
    else:
        new_table_name = table_name
    abs_path_split = os.path.join('./data', new_table_name)

    print(len(dropped_task_ids))

    result = pd.read_excel(f"./data/model_response/{table_name}.xlsx").to_dict("records")
    count = len(result)

    k = 1
    model_pass_k = {}
    answer_not_completed = {}
    model_not_answer = {}

    if selected_models:
        models = selected_models

    print(models)

    for model in models:
        total_test_sample_num = 0
        pass_num = 0
        
        total_sample_get_answer_completed = []

        result_file = f"./data/{new_table_name}/samples_{new_table_name}_{model}.jsonl_results.jsonl"
        with jsonlines.open(result_file) as inf:
            for line in inf:
                total_test_sample_num += 1
                task_id = line["task_id_old"] if if_split else line["task_id"]

                if task_id in dropped_task_ids:
                    continue
                
                if task_id not in total_sample_get_answer_completed:
                    total_sample_get_answer_completed.append(task_id)

                if_passed = line["passed"]
                if if_passed:
                    pass_num = pass_num + 1

        answer_not_completed[model] = count - len(total_sample_get_answer_completed)
        model_pass_k[model] = pass_num / total_test_sample_num

    return {"model_pass_ratio": model_pass_k, 
            "missing_answer_count": answer_not_completed}



def split_test_sample(source_table, selected_models=None):
    """After generating turbo answers, split each test_sample into an individual unit test for debugging."""
    new_table_name = f"{source_table}_split"
    if f"{source_table}_split" not in os.listdir("./data"):
        os.mkdir(os.path.join("./data", f"{source_table}_split"))
    abs_split_path = os.path.join("./data", f"{source_table}_split")

    result = pd.read_excel(f"./data/model_response/{table_name}.xlsx").to_dict("records")

    # Get model names
    models = []
    
    for k, v in dict(result[0]).items():
        if re.findall("model_answer_.*_turns_1$", k):
            model_name = re.sub('^model_answer_', '', k)
            model_name = re.sub('_turns_1$', '', model_name)
            models.append(model_name)
        
    models = list(set(models))
    if selected_models:
        models = selected_models

    for model in models:
        problem_file = f"./data/{table_name}/problem_{table_name}_{model}.jsonl"
        completion_file = f"./data/{table_name}/samples_{table_name}_{model}.jsonl"

        test_num = {}

        with jsonlines.open(problem_file) as inf, jsonlines.open(os.path.join(abs_split_path, f"problem_{new_table_name}_{model}.jsonl"), "w") as ouf:
            for line in inf:
                test_count = 0
                task_id = line["task_id"]
                test_samples = line["test"]
                for _ in test_samples.split("assert")[1:]:
                    test_count += 1
                    task_id_new = f"{task_id}_{test_count}"
                    
                    test = _.strip()
                    test = "def check(candidate):\n" + f"    assert {test}"
                    print(test)
                    new_row = copy.deepcopy(line)
                    new_row["task_id_old"] = task_id
                    new_row["task_id"] = task_id_new
                    new_row["test"] = test
                    ouf.write(new_row)
                    test_num[task_id] = max(test_num.get(task_id, 0), test_count)

        with jsonlines.open(completion_file) as inf, jsonlines.open(os.path.join(abs_split_path, f"samples_{new_table_name}_{model}.jsonl"), "w") as ouf:
            for line in inf:
                test_count = 0
                task_id = line["task_id"]
                new_row = copy.deepcopy(line)
                for i in range(test_num[task_id]):
                    task_id_new = f"{task_id}_{i+1}"
                    new_row["task_id_old"] = task_id
                    new_row["task_id"] = task_id_new
                    ouf.write(new_row)


def export_pass_ratio(table_name, models, out_file, dropped_task_ids, if_split):
    """Calculate pass@k or per-test-case accuracy and export results."""
    print(len(dropped_task_ids))

    result = calculate_pass_each_test_sample(table_name, selected_models=models, if_split=if_split)

    model_score = result["model_pass_ratio"]
    model_get_answer_failed = result.get("missing_answer_count", None)

    print("Score ranking:")
    print("---" * 10)
    print(sorted(list(model_score.items()), key=lambda x: x[1], reverse=True))
    print("Number of missing answers:")
    print("---" * 10)
    print(model_get_answer_failed)
    

    data = []
    for model in model_score:
        data.append([model, round(model_score[model] * 100, 2), model_get_answer_failed[model]])

    data = pd.DataFrame(data, columns=["model", "score", "missing_answer_count"])
    data.to_excel(out_file)
    print("Scores have been exported to:")
    print(out_file)


def batch_extract_completion(table_name, models, dropped_task_ids, out_file, if_split=True):
    """Batch extract model completions"""
    data = []
    model_not_follow_instruction = {}
    for model in models:
        print(f"Processing {model}")
        problem_file = f"./data/{table_name}/problem_{table_name}_{model}.jsonl"
        sample_file = f"./data/{table_name}/samples_{table_name}_{model}.jsonl"

        exception_n, answer_completed = code3_standard_with_answer_2_human_eval_formal(table_name, model, problem_file, sample_file, dropped_task_ids)
        model_not_follow_instruction[model] = exception_n
        print(f"Finished processing {model}")
        print("--" * 20)
        data.append([model, exception_n, (answer_completed - exception_n) / answer_completed * 100])

    data = pd.DataFrame(data, columns=["model_name", "num_not_follow_instruction", "follow_instruction_ratio"])
    out_file = "./data/eval_result/model_not_follow_instruction.xlsx"
    data.to_excel(out_file)
    print("Model instruction-following ratios exported to:")
    print(out_file)

    """
    Need to split each test_sample into multiple records,
    each containing one test case, and save them under human_eval/data/{table_name}_split.
    When running run.sh, table_name must be set to {table_name}_split.
    """
    if if_split:
        split_test_sample(source_table=table_name, selected_models=models)
        print(f"Selected general monthly leaderboard {table_name}, test case splitting completed")
    print("Export complete:")
    print(out_file)
   


if __name__ == "__main__":
    ## Specify table_name
    table_name = "code_test_result_0629"

    ## Specify model names to process
    models = ["model1"]

    ## Whether to split into one-unit-test per test case (default True)
    if_split = True

    ## Task IDs to exclude from scoring
    dropped_task_ids = []

    ## Create folder human_eval/data/{table_name}, ensure each leaderboard is independent
    if table_name not in os.listdir("./data"):
        os.mkdir(f"./data/{table_name}")

    ## Batch extract model answers and write to files
    out_file = f"./data/eval_result/model_instruction.xlsx"
    batch_extract_completion(table_name, models, dropped_task_ids, out_file, if_split=if_split)
