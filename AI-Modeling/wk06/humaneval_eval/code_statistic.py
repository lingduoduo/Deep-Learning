
import jsonlines
import os
import copy
# import pymysql
import numpy as np
import pandas as pd
import regex as re
# from 数据入库阿里云 import select_ali
import traceback
from typing import List, Union, Iterable, Dict
import itertools


def strip_test_example(model_response):
    """去掉模型回复中的测试用例"""
    kuo_start = []
    kuo_end=[]
    kuo_left, kuo_right=0,0
    kuo_flag=False
    model_response = model_response.split("\n")
    for seg_id, _ in enumerate(model_response):
        if _.startswith("if __name__ ==") or _.startswith("# 测试代码") or _.startswith("# 示例测试用例") or _.startswith("# 示例代码") or _.lower().startswith("# test case") or _.startswith("# 单元测试") or _.startswith("# 测试用例") or _.lower().startswith("# test cases") or _.startswith("# 根据给定的测试用例进行验证"):
            
            model_response = model_response[:seg_id]
            break
    #  记录括号开始和结束位置
    
    for seg_id, _ in enumerate(model_response):
        if _.startswith("assert") or _.startswith("print"):
            kuo_flag=True
            kuo_start.append(seg_id)
            kuo_left, kuo_right = len(re.findall("\(", _)), len(re.findall("\)", _))
            if kuo_left==kuo_right and kuo_left!=0 and kuo_flag:
                kuo_end.append(seg_id+1)
                kuo_left, kuo_right=0,0
                kuo_flag=False
        if kuo_flag:
            for seg_id_1 in range(seg_id+1, len(model_response)):
                kuo_left+= len(re.findall("\(", model_response[seg_id_1]))
                kuo_right+=len(re.findall("\)", model_response[seg_id_1]))
                if kuo_left==kuo_right and kuo_left!=0 and kuo_flag:
                    # print(model_response[seg_id], model_response[seg_id_1])
                    kuo_end.append(seg_id_1+1)
                    kuo_left, kuo_right=0,0
                    kuo_flag=False

    if len(kuo_start)==len(kuo_end)+1:
        kuo_end.append(len(model_response))

    model_response_final = model_response
    for n in range(len(kuo_start)):
        model_response_final = [_ for _ in model_response_final if _ not in model_response[kuo_start[n]:kuo_end[n]]]

    return "\n".join(model_response_final).rstrip()
            
def extract_standard(response):
    """提取标准答案和import"""
    try:
        model_response = re.findall("【代码开始】((?:[\s\S]*?))【代码结束】", response)
        entry_point = "merge_nested_dicts"
        # if len(model_response)==0:
            # exception_n+=1
        if len(model_response)>1:
            model_response_with_entry_point = [_ for _ in model_response if f"def {entry_point}" in _]
            if len(model_response_with_entry_point)!=0:
                model_response = model_response_with_entry_point[0]
            else:
                model_response = model_response[0]
        else:
            model_response = model_response[0]

        
        if len(re.findall("```((?:[\s\S]*?))```", model_response))!=0:
            if len(re.findall("```((?:[\s\S]*?))```", model_response))==1:
                model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
            
            else:

                model_response_with_entry_point = [_ for _ in re.findall("```((?:[\s\S]*?))```", model_response) if f"def {entry_point}" in _]
                if len(model_response_with_entry_point)!=0:
                    model_response = model_response_with_entry_point[0]
                else:
                    model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
        print("……"*20)
        


        model_response = model_response.strip("\n`【代码开始】结束")
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
            
            if seg.strip().lower() in ["python", "python code"]  or "print" in seg or "assert" in seg:
                continue
            if len(idx_def_in)==0 or entry_point not in model_response:
                completion += f"\n{seg}"
            
            else:
                if "import " in seg:
                    if idx<idx_def_in[0]:
                        import_code += f"{seg}\n"
                    else:
                        completion += f"\n{seg}"
                elif "def " in seg and entry_point.strip() in seg: #entry_point方法
                    # print(1)
                    completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                    break
                else: 
                    completion += "\n" + f"    {seg}"

        # print(completion)
        if entry_point=="sum_of_multiples":
                    print(completion)
        # if "python" in completion.lower() and entry_point=="sorted_union_of_lists":
        #     print(completion)
    except:
        print("………………………………………………………………………………"*20)
                
        model_response = response
        if len(re.findall("```((?:[\s\S]*?))```", model_response))!=0:
            if len(re.findall("```((?:[\s\S]*?))```", model_response))==1:
                model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
            else:

                model_response_with_entry_point = [_ for _ in re.findall("```((?:[\s\S]*?))```", model_response) if f"def {entry_point}" in _]
                if len(model_response_with_entry_point)!=0:
                    model_response = model_response_with_entry_point[0]
                else:
                    model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]

        
        model_response = model_response.strip("\n`【代码开始】结束")
        model_response = strip_test_example(model_response)
       
        
        import_code = ""
        completion = ""
        completion_segs = model_response.split("\n")
        print(completion_segs)
        idx_def_in = []
        for idx, seg in enumerate(completion_segs):
            if "def " in seg:
                idx_def_in.append(idx)
        print(idx_def_in)
        
        for idx, seg in enumerate(completion_segs):
            if seg.strip() in ["python", "python code"] or "print" in seg or "assert" in seg:
                continue
            if len(idx_def_in)==0 or entry_point not in model_response:
                completion += f"\n{seg}"
                print(entry_point)
            
            else:
                if "import " in seg:
                    print(idx)
                    if idx<idx_def_in[0]:
                        import_code += f"{seg}\n"
                        # completion += f"\n    {seg}"
                    else:
                        completion += f"\n{seg}"
                    # print(completion)
                    # print("……"*20)
                # elif "def" in seg:
                #     completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                #     break
                elif "def " in seg and entry_point.strip() in seg: #entry_point方法
                    print(1)
                    completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                    break
                else: #在entry_point之前定义的小方法,和其他方法内部代码需要缩进
                    completion += "\n" + f"    {seg}"

                


    
    return import_code, completion
    

def code3_standard_with_answer_2_human_eval_formal(table_name, model, problem_file, sample_file, dropped_task_ids=[]):
    """将跑完答案的code3数据转化为human-eval的标准格式"""
    # sql = f"select * from {table_name}"
    # result = select_ali(sql)
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
                if not line[f"model_answer_{model}_turns_1"] or line[f"model_answer_{model}_turns_1"].strip()=="lock":
                    continue
                # print(model_response)
                # model_response = re.findall("```(?:[\s\S]*?)```", line[f"model_answer_{model}_turns_1"])[0]
                model_response = re.findall("【代码开始】((?:[\s\S]*?))【代码结束】", line[f"model_answer_{model}_turns_1"])

                if len(model_response)==0:
                    exception_n+=1
                if len(model_response)>1:
                    model_response_with_entry_point = [_ for _ in model_response if f"def {entry_point}" in _]
                    if len(model_response_with_entry_point)!=0:
                        model_response = model_response_with_entry_point[0]
                    else:
                        model_response = model_response[0]
                else:
                    model_response = model_response[0]

                
                if len(re.findall("```((?:[\s\S]*?))```", model_response))!=0:
                    if len(re.findall("```((?:[\s\S]*?))```", model_response))==1:
                        model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
                    
                    else:

                        model_response_with_entry_point = [_ for _ in re.findall("```((?:[\s\S]*?))```", model_response) if f"def {entry_point}" in _]
                        if len(model_response_with_entry_point)!=0:
                            model_response = model_response_with_entry_point[0]
                        else:
                            model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]

                    # model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
                # print(model_response)
                print("……"*20)
                


                model_response = model_response.strip("\n`【代码开始】结束")
                # 去掉if __name__=="__main__", print(), assert等内容
                model_response = strip_test_example(model_response)

                import_code = ""
                completion = ""
                completion_segs = model_response.split("\n")
                idx_def_in = []
                for idx, seg in enumerate(completion_segs):
                    if "def " in seg:
                        idx_def_in.append(idx)
                
                for idx, seg in enumerate(completion_segs):
                    
                    if seg.strip().lower() in ["python", "python code"]  or "print" in seg or "assert" in seg:
                        continue
                    if len(idx_def_in)==0 or entry_point not in model_response:
                        completion += f"\n{seg}"
                    
                    else:
                        if "import " in seg:
                            if idx<idx_def_in[0]:
                                import_code += f"{seg}\n"
                                # completion += f"\n    {seg}"
                            else:
                                completion += f"\n{seg}"
                        elif "def " in seg and entry_point.strip() in seg: #entry_point方法
                            # print(1)
                            completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                            break
                        else: #在entry_point之前定义的小方法,需要缩进
                            completion += "\n" + f"    {seg}"

                # print(completion)
                if entry_point=="sum_of_multiples":
                            print(completion)
                # if "python" in completion.lower() and entry_point=="sorted_union_of_lists":
                #     print(completion)
                ouf1.write({"task_id":task_id, "import_code":import_code, "prompt":prompt, "canonical_solution":reference, "test":test, "entry_point":entry_point, "origin_model_response":model_response})
                ouf2.write({"task_id":task_id, "completion":completion})

            except:
                
                print("………"*20)

                model_response = line[f"model_answer_{model}_turns_1"]
                if len(re.findall("```((?:[\s\S]*?))```", model_response))!=0:
                    if len(re.findall("```((?:[\s\S]*?))```", model_response))==1:
                        model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
                    else:

                        model_response_with_entry_point = [_ for _ in re.findall("```((?:[\s\S]*?))```", model_response) if f"def {entry_point}" in _]
                        if len(model_response_with_entry_point)!=0:
                            model_response = model_response_with_entry_point[0]
                        else:
                            model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]

                    # model_response = re.findall("```((?:[\s\S]*?))```", model_response)[0]
                
                model_response = model_response.strip("\n`【代码开始】结束")
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
                    if len(idx_def_in)==0 or entry_point not in model_response:
                        completion += f"\n{seg}"
                    
                    else:
                        if "import " in seg:
                            if idx<idx_def_in[0]:
                                import_code += f"{seg}\n"
                                # completion += f"\n    {seg}"
                            else:
                                completion += f"\n{seg}"
                            # print(completion)
                            # print("……"*20)
                        # elif "def" in seg:
                        #     completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                        #     break
                        elif "def " in seg and entry_point.strip() in seg: #entry_point方法
                            print(1)
                            completion += "\n" + "\n".join(model_response.split("\n")[idx+1:])
                            break
                        else: #在entry_point之前定义的小方法,和其他方法内部代码需要缩进
                            completion += "\n" + f"    {seg}"

                # if entry_point=="invert_dictionary":
                #     print(completion)
                ouf1.write({"task_id":task_id, "import_code":import_code, "prompt":prompt, "canonical_solution":reference, "test":test, "entry_point":entry_point, "origin_model_response":model_response})
                ouf2.write({"task_id":task_id, "completion":completion})

    print(f"模型model未按照指令回答：{exception_n}个")
    
    return exception_n, total_answer_completed

def format_and_evaluation(sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    # timeout: float = 3.0,
    timeout: float = 30,
    problem_file: str = None):
    """提取标准化格式文件并且运行沙箱返回分数"""
    from human_eval.evaluate_functional_correctness import entry_point
    pass_k = entry_point(sample_file=sample_file, k=k, n_workers=n_workers, timeout=timeout, problem_file=problem_file)

    return pass_k

def string_match(str1, str2):
    """判断两个字符串是否等价"""
    import hashlib
    md5_obj = hashlib.md5()
    
    # 将字符串转换为字节
    md5_obj.update(str1.encode('utf-8'))
    str1_hash = md5_obj.hexdigest()
    md5_obj = hashlib.md5()
    md5_obj.update(str2.encode('utf-8'))
    str2_hash = md5_obj.hexdigest()

    return str1_hash==str2_hash
    # 返回16进制的md5哈希值
    
def caculate_pass_k(table_name, model_names):
    """根据数据库code答案，计算指定模型的pass@得分"""
    model_scores =  {}
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
    """每个测试用例单独算分，就认为通过了该单元测试"""
    pass
    if if_split:
        new_table_name = f"{table_name}_split"
    else:
        new_table_name=table_name
    abs_path_split = os.path.join('./data', new_table_name)

    print(len(dropped_task_ids))
    # sql = f"select count(*) as count from {table_name}"
    # count = select_ali(sql)[0]["count"]
    result = pd.read_excel(f"./data/model_response/{table_name}.xlsx").to_dict("records")
    count = len(result)

    k=1
    model_pass_k = {}
    answer_not_completed={}
    model_not_answer = {}
    if selected_models:
        models=selected_models
    print(models)
    for model in models:
        # answer_not_completed=0
        total_test_sample_num = 0
        pass_num = 0
        
        total_sample_get_answer_completed = []
        # all_task_ids_for_each_prosblem = []
        result_file = f"./data/{new_table_name}/samples_{new_table_name}_{model}.jsonl_results.jsonl"
        with jsonlines.open(result_file) as inf:
            for line in inf:
                    total_test_sample_num+= 1
                    task_id = line["task_id_old"] if if_split else line["task_id"]
                    
                    if task_id in dropped_task_ids:
                        continue
                    
                    if task_id not in total_sample_get_answer_completed:
                        total_sample_get_answer_completed.append(task_id)

                    # total_test_sample_num[task_id] = total_test_sample_num.get(task_id, 0) +1
                    # total_sample[task_id] = total_sample.get(task_id, 0) + 1
                    # correct_sample_for_each_problem[task_id] = correct_sample_for_each_problem.get(task_id, 0)
                    if_passed = line["passed"]
                    if if_passed:
                        pass_num = pass_num +1
                        # correct_sample_for_each_problem[task_id] = correct_sample_for_each_problem[task_id] + 1
        

        answer_not_completed[model] = count - len(total_sample_get_answer_completed)

        model_pass_k[model] = pass_num/total_test_sample_num


    return {"模型通过比例":model_pass_k, 
            "未获取到答案数量":answer_not_completed}



def split_test_sample(source_table, selected_models=None):
    """实验跑完turbo的答案之后，将每个test_sample拆分作为一个单独的单元测试，导入新的数据库表，方便验证每个测试用例是有问题"""
    new_table_name=f"{source_table}_split"
    if f"{source_table}_split" not in os.listdir("./data"):
        os.mkdir(os.path.join("./data", f"{source_table}_split"))
    abs_split_path = os.path.join("./data", f"{source_table}_split")
    # sql = f"select * from {source_table}"
    # result = select_ali(sql)
    result = pd.read_excel(f"./data/model_response/{table_name}.xlsx").to_dict("records")
    # 获取所有模型名称
    models = []
    
    for k,v in dict(result[0]).items():
        if re.findall("model_answer_.*_turns_1$",k):
            model_name=re.sub('^model_answer_','',k)
            model_name=re.sub('_turns_1$','',model_name)
            models.append(model_name)
        
    models=list(set(models))
    if selected_models:
        models=selected_models
    for model in models:
        problem_file = f"./data/{table_name}/problem_{table_name}_{model}.jsonl"
        completion_file = f"./data/{table_name}/samples_{table_name}_{model}.jsonl"
        # 将promblem_file中的task_id拆分
        # 记录每个task_id的test数量
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
                    new_row  = copy.deepcopy(line)
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
    """计算pass@k，对于code3榜单，是pass@k,对于通用月榜的会将每个测试用例作为一个单元测试，并且将得分数据写进excel"""
    print(len(dropped_task_ids))
    # # 每个测试用例单独算分
    result = calculate_pass_each_test_sample(table_name, selected_models=models, if_split=if_split)

    # # 计算通过80%测试用例的task_id比例
    
    # print(result)
    model_score = result["模型通过比例"]
    model_get_answer_failed = result.get("未获取到答案数量", None)
    print("成绩排名:")
    print("---"*10)
    print(sorted(list(model_score.items()), key=lambda x:x[1], reverse=True))
    print("未获取到答案的数量:")
    print("---"*10)
    print(model_get_answer_failed)
    

    data = []
    for model in model_score:
        data.append([model, round(model_score[model]*100, 2), model_get_answer_failed[model]])


    data = pd.DataFrame(data, columns=["model", "非加权score", "未获取到答案的数量"])
    data.to_excel(out_file)
    print("统计成绩已经导出到下面的文件：")
    print(out_file)

def batch_extract_completion(table_name, models, dropped_task_ids, out_file, if_split=True):
    """批量提取模型的答案"""
    # 提取答案文件，同时输出未遵循指令数量
    data = []
    model_not_follow_instruction = {}
    for model in models:
        print(f"处理{model}")
        problem_file = f"./data/{table_name}/problem_{table_name}_{model}.jsonl"
        sample_file = f"./data/{table_name}/samples_{table_name}_{model}.jsonl"
        exception_n, answer_completed = code3_standard_with_answer_2_human_eval_formal(table_name, model, problem_file, sample_file, dropped_task_ids)
        model_not_follow_instruction[model] = exception_n
        print(f"处理{model}完毕")
        print(f"--"*20)
        data.append([model, exception_n, (answer_completed-exception_n)/answer_completed*100])
    data = pd.DataFrame(data, columns=["model_name", "未遵循指令数量", "遵循指令比例"])
    out_file = "./data/eval_result/model_not_follow_instruction.xlsx"
    data.to_excel(out_file)
    print("模型未遵循指令的比例已经导出到下面的excel中：")
    print(out_file)
    # print(model_not_follow_instruction)

    """
    需要将将每个test_samples拆分为多条记录，
    每条记录包含一个测试用例，并且将拆分后的数据放在文件夹human_eval/data/{table_name}_split中,
    最后运行run.sh文件时，需要将table_name设置为{table_name}_split
    """
    if if_split:
        split_test_sample(source_table=table_name, selected_models=models)
        print(f"选择了通用月榜{table_name}的数据,已经完成测试用例的划分")
    print("模型未遵循指令的比例已经导出到下面的excel中：")
    print(out_file)
   


if __name__=="__main__":
    ## 指定要操作的table_name
    table_name="code_test_result_0629"

    ## 指定本次要处理和统计成绩的model_name
    models=["model1"]
    # print(",".join(models))

    # 是否划分为每个单元测试一个测试用例（默认为False，则每个单元测试包括一个问题和多个测试用例，就像现在code_test里的样子,设置成True则进一步划分为每个单元测试对应一个测试用例）
    if_split = True

    # 不计入成绩的task_ids,不要注释
    dropped_task_ids=[]

    ## human_eval/data路径下为每个榜单创建一个文件夹，文件夹命名为table_name,保证每个榜单的数据独立分开，不要注释掉
    if table_name not in os.listdir("./data"):
        os.mkdir(f"./data/{table_name}")

    ## 批量提取答案文件，同时输出未遵循指令数量，写入到文件，data/model_response/code_test中的一个测试单元对应一个问题和多个测试用例,
    # 可通过if_split参数控制是否划分为每个测试单元一个测试用例,if_split=True表示，进一步划分，默认为True
    # 如果设置if_split=True，则在下一步运行run.sh文件，需要设置table_name=code_test_split，对应data/code_test_split路径
    out_file = f"./data/eval_result/model_instruction.xlsx"
    batch_extract_completion(table_name, models, dropped_task_ids, out_file, if_split=if_split)



    







