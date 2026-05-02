from human_eval.evaluate_functional_correctness import entry_point
def format_and_evaluation(sample_file,
    k="1,10,100",
    n_workers=4,
    # timeout: float = 3.0,
    timeout=60,
    problem_file=None):
    """提取标准化格式文件并且运行沙箱返回分数"""

    print(problem_file, sample_file)
    pass_k = entry_point(sample_file=sample_file, k=k, n_workers=n_workers, timeout=timeout, problem_file=problem_file)

    return pass_k
    
def caculate_pass_k(table_name, model_names):
    """根据数据库code答案，计算指定模型的pass@得分"""
    model_scores =  {}
    for model in model_names:
        print(model)
        problem_file = f"/mnt/SuperCLUE/SuperCLUE-eval/human_eval/data/problem_{table_name}_{model}.jsonl"
        sample_file = f"/mnt/SuperCLUE/SuperCLUE-eval/human_eval/data/samples_{table_name}_{model}.jsonl"

        model_scores[model] = format_and_evaluation(problem_file=problem_file, sample_file=sample_file)

    return model_scores

if __name__=="__main__":
    table_name = "ScoreV3Code3"
    models = ["GPT_4_1106_Preview", "GPT_35_Turbo"]

        
    # 计算pass@k
    model_score = caculate_pass_k(table_name=table_name, model_names=models)

    print(model_score)

