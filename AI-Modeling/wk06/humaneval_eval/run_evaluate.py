import os

table_name="code_test_result_0629_split"
model="model1"

problem_file=f"./data/{table_name}/problem_{table_name}_{model}.jsonl"
samples_file=f"./data/{table_name}/samples_{table_name}_{model}.jsonl"
print("samples_file: ", samples_file)
eval_cmd = f"python ./human_eval/evaluate_functional_correctness.py \
--n_workers 1 \
--problem_file {problem_file} \
--sample_file {samples_file}"

os.system(eval_cmd)

