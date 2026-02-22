import codecs
import sys
import multiprocessing
import json
import jsonlines
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

file_path = "data/test_1116.jsonl"
model_path = "./saves/qwen3-14B"

def inference(model_path, file_path, output_path, start_index, end_index):
    input_list = []
    data_list = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Start to Handler Data From {start_index} To {end_index}")
    
    with open(file_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            
            input_ = data["input"].replace("<|im_start|>user", "").replace("<|im_end|> <|im_start|>assistant <think>", "")
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

        result = {
            "input":[],
            "model_answer":[],
            "golden_answer":[]
        }
        for i in range(len(data_list)):
            result["input"].append(data_list[i]["input"])
            result["model_answer"].append(results[i])
            result["golden_answer"].append(data_list[i]["output"])
        df = pd.DataFrame(result)
        df.to_excel(output_path)

if __name__ == "__main__":
    output_path = f"./data/output/gov_v1_checkpoint-qwen3-14B.xlsx"
    inference(model_path, file_path, output_path, 0, 1000000)
