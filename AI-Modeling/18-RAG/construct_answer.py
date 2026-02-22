import codecs
import sys
import multiprocessing
import json
import jsonlines
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

file_path = "./data/all_questions_1112.jsonl"
model_path = "/data/zhengwj/model/Qwen/Qwen3-30B-A3B-Thinking-2507"

prompt_template = '''# 角色与任务
你是一个政务及党建相关问题的专家。你能够基于给定的文档，高效、简洁的回答用户提出的问题。

# 文档内容
{doc}

# 用户问题
{question}

请回答：
'''

def inference(model_path, file_path, output_path, start_index, end_index):
    input_list = []
    data_list = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Start to Handler Data From {start_index} To {end_index}")
    
    with open(file_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            doc = data["contentText"]
            question = data["question"]
            
            input_ = prompt_template.replace("{doc}", doc).replace("{question}", question)
            if len(input_)<10:continue
            data_list.append(data)
            input_list.append(input_)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role":"user", "content":prompt}],
                tokenize=False,
                add_generation_prompt = True
            )
            for prompt in input_list ][start_index:end_index]
        print(f"prmpts is {prompts[0:2]}")
        data_list = data_list[start_index:end_index]
        llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=2,gpu_memory_utilization=0.8)
        sampling_params = SamplingParams(
            repetition_penalty = 1.05,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=8192
        )
        print("start Generation!!!!")
        outputs = llm.generate(prompts, sampling_params)

        results = [i.outputs[0].text for i in outputs]
        qwen_input_list = [i.prompt for i in outputs]
        print(f"len of results is {len(results)}, len of data_list is {len(data_list)}")

        print(f"Start To Save Data to file: {output_path}")
        with jsonlines.open(output_path, "w") as writer:
            for i in range(len(data_list)):
                data = data_list[i]
                data["answer"] = results[i]
                data["qwen_input2"] = qwen_input_list[i]
                writer.write(data)

if __name__ == "__main__":
    for i in range(1100):
        output_path = f"./data/answer/doc_answer_{i}.jsonl"
        inference(model_path, file_path, output_path, 1000*i, 1000*(i+1))
