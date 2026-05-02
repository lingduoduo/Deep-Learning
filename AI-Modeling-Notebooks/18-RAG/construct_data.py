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

file_path = "../data_handle/doc1.json"
model_path = "/data/zhengwj/model/Qwen/Qwen3-30B-A3B-Thinking-2507"

prompt_template = '''# 角色与任务
你是一个专业的问答对数据集构建专家。你的核心任务是根据用户提供的一篇文档，生成一系列高质量、多样化的问题。这些问题应基于文档内容，并且其答案能够直接从文档中找到。

# 文档内容
{doc}

# 问题生成要求
请严格按照以下要求生成问题：
1.  **基于文档**：所有问题必须严格基于上述文档内容生成，确保答案明确存在于文档中。
2.  **问题类型多样化**：应混合生成不同类型的问题，例如：
    *   **事实性问题**：关于具体事实、数据、定义（例如：什么是X？事件发生在何时？）。
    *   **理解性问题**：需要理解上下文、原因、影响或过程（例如：为什么X会导致Y？请解释Z的运作机制。）。
    *   **综合性问题**：需要概括总结或连接多个信息点（例如：请总结X的主要特点。A和B之间的共同点是什么？）。
3.  **质量要求**：
    *   **清晰具体**：问题应焦点明确，表述清晰无歧义。
    *   **一句一问**：每个问题应独立且只包含一个核心疑问。
    *   **语言匹配**：问题的语言风格和术语应与原文保持一致。

# 输出格式
你必须以严格的JSON格式输出结果，确保无需额外说明即可被程序解析。格式如下：
json
{

"generated_questions": [

{

"question": "生成的第一个问题内容？",

"type": "问题类型，如：factual/comprehensive/inferential"

},
{

"question": "生成的第二个问题内容？",

"type": "问题类型"

}

// ... 生成更多问题

]

}
请生成大约10个问题。
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

            input_ = prompt_template.replace("{doc}", doc)
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
                data["question_unhandle"] = results[i]
                data["qwen_input"] = qwen_input_list[i]
                writer.write(data)

if __name__ == "__main__":
    for i in range(110):
        output_path = f"./doc_question_{i}.jsonl"
        inference(model_path, file_path, output_path, 1000*i, 1000*(i+1))
