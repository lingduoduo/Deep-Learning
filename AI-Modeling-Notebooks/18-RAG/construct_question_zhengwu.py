import codecs
import sys
import multiprocessing
import json
import jsonlines
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

## 业务有新的需求，需要我们的智能客服 能够回答业务流程 相关的问题
## 大量的用户咨询都是关于具体的业务流程所以希望我们的职能客服，能够回答相关的问题
## 1. RAG的方式（构建本地知识库、实时检索）引进政务服务相关的业务的知识
## 2.我们对比，单纯的RAG和微调（llm）之后两种技术方案那种效果更好一点
## 为什么要微调？
## 微调模型：微调模型的目的，1.学习政务领域的知识； 2.学习解决政务领域问题的方法


os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

file_path = "./data/zhengwu/zhengwu_md.jsonl"
model_path = "/data/zhengwj/model/Qwen/Qwen3-30B-A3B-Thinking-2507"

prompt_template = '''# 角色与任务
你是一个政务服务的工作人员，你能够非常娴熟的回答政务服务相关的问题。
你的核心任务是根据提供的一篇政务服务流程问题，生成一系列高质量、多样化的问题。这些问题市民关于政务服务相关的问题，生成的问题其答案能够从文档中找到。

# 文档内容
{doc}

# 问题生成要求
请严格按照以下要求生成问题：
1.  **基于文档**：所有问题必须严格基于上述文档内容生成，确保答案明确存在于文档中。
2.  **问题类型多样化**：应混合生成不同类型的问题，例如：
    *   **贴合生活**：结合文档内容，生成出来市民可能会问的生活中的问题。
    *   **个性化**：需要拟定一个角色，按照个这个角色生成符合他角色的问题。
    *   **完整性**：是一个完整的问题，不需要依赖文档的信息就能够知道是在询问那个业务流程。
3.  **质量要求**：
    *   **清晰具体**：问题应焦点明确，表述清晰无歧义。
    *   **一句一问**：每个问题应独立且只包含一个核心疑问。
    *   **语言匹配**：问题的语言风格和术语应与原文保持一致。

# 示例：
{
    "generated_questions": [
     "摇号的时候提示审核不能过，审核失败原因，公安交管：有驾照，申请驾照身份证明名称与交管部门登记信息不一致? 是什么登记住处不一致? 是身份证的问题?应该没有问题呀?我都使用几十年的身份证还有问题么?",
    ]
}

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
            doc = data["doc"]

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
    output_path = f"./doc_question_2.jsonl"
    inference(model_path, file_path, output_path,  0, 10000000)
