import os
from elasticsearch import Elasticsearch
import jieba
import math
from torch import Tensor
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 连接远程的ES库
es_tool = Elasticsearch(
    hosts=[
        "http://localhost:9200"
    ],
    verify_certs=True
)

dsl = {
    "query": {
        "bool": {
            "must": [
            ],
            "should": [
            ],
            "must_not": [
            ],
            "minimum_should_match": 1,
            "boost": 1.0
        }
    },
    "size": 30
}


def word_seg(query):
    # 1.加载词典
    jieba.load_userdict("userdict.txt")

    # 2：从文件加载停用词
    with open('stop_words.txt', 'r', encoding='utf-8') as f:
        stop_words = {line.strip() for line in f}

    words = jieba.lcut(query, cut_all=False)
    filtered_words = [word for word in words if word not in stop_words]
    result = []
    for word in filtered_words:
        result.append(word)
    return list(set(result))


def recall_from_es(query):
    dsl = {
        "query": {
            "bool": {
                "must": [
                ],
                "should": [
                ],
                "must_not": [
                ],
                "minimum_should_match": 1,
                "boost": 1.0
            }
        },
        "size": 30
    }
    word_list = word_seg(query)
    should_list = []
    for word in word_list:
        should_list.extend(
            [
                {"match_phrase": {"docContext": {"query": word, "boost": 1}}},
                {"match_phrase": {"title": {"query": word, "boost": 1}}}
            ]
        )
    dsl["query"]["bool"]["should"] = should_list

    result = es_tool.search(index="gov", body=dsl)["hits"]["hits"]
    return result


def recall_from_es_embedding(query, query_embedding):
    dsl = {
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "field": "docEmbedding",
                            "query_vector": []
                        }
                    }
                ],
                "should": [],
                "must_not": [],
                "minimum_should_match": 0,
                "boost": 1.0
            }
        },
        "size": 30
    }
    word_list = word_seg(query)
    should_list = []
    dsl["query"]["bool"]["must"][0]["knn"]["query_vector"] = query_embedding
    for word in word_list:
        should_list.extend(
            [
                {"match_phrase": {"docContext": {"query": word, "boost": 1}}},
                {"match_phrase": {"title": {"query": word, "boost": 5}}}
            ]
        )
    dsl["query"]["bool"]["should"] = should_list

    result = es_tool.search(index="gov", body=dsl)["hits"]["hits"]
    return result


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def batch_embedding(tokenizer, model, texts, batch_size, max_length):
    count = len(texts)
    results = []
    print(f"all Emebedding texts :{count} items")
    for i in range(math.ceil(count / batch_size)):
        print(f"Start to embedding:{i * batch_size}_{(i + 1) * batch_size}")
        input_texts = texts[i * batch_size:(i + 1) * batch_size]
        batch_dict = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        batch_dict.to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

        embeddings = F.normalize(embeddings, p=2, dim=1).detach().numpy()
        results.extend(embeddings)
    return np.asarray(results)


task = 'Given a web search query, retrieve relevant passages that answer the query'


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


tokenizer = AutoTokenizer.from_pretrained("/model/Qwen/Qwen/Qwen3-Embedding-0.6B", padding_side='left')
model = AutoModel.from_pretrained("/model/Qwen/Qwen/Qwen3-Embedding-0.6B")


model_path = "/model/Qwen/Qwen3-14B"

tokenizer_llm = AutoTokenizer.from_pretrained(model_path)
llm = AutoModelForCausalLM.from_pretrained(model_path)


def inference(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)

    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=8192
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    answer = tokenizer.decode(output_ids[index + 1:], skip_special_token=True)
    return thinking_content, answer


def filter_es_result(result_list):
    return [data["_source"]["title"] + "\n" + data["_source"]["docContext"] for data in result_list]


def get_answer(query):
    text = get_detailed_instruct(task, query)
    query_embedding = batch_embedding(tokenizer, model, [text], 16, 8192)[0]
    # 从es库检索相关的内容回来
    recall_result = recall_from_es_embedding(query, query_embedding)

    recall_result = filter_es_result(recall_result)[0:1]
    # print(f"query is {query}, recall result is {recall_result}")
    # 构建一个回答指令
    prompt_template = '''你是一个文档问答助手，你可以基于我给定的参考内容来回答问题，
如果问答的答案无法从参考内容中获取，请你回答"无法从参考答案中获取正确的参考信息，请更改询问的方式，或者致电12345询问"
参考内容：
{reference}
用户问题：
{question}
请回答：
'''
    prompt = prompt_template.replace(
        "{reference}", "\n".join(recall_result)
    ).replace(
        "{question}", query
    )
    print(f"query is {query}, prompt is {prompt}")
    return inference(prompt)

if __name__ == "__main__":
    get_answer("如何推动风电等新能源产业发展加速？")