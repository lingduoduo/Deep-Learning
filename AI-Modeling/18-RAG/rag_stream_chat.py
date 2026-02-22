from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, TextIteratorStreamer
import os
import torch
from torch import Tensor
from threading import Thread
import os
import time, math
import torch.nn.functional as F
import numpy as np
from naive_rag import recall_from_es_embedding, filter_es_result

# 将localhost和本地IP加入不代理列表
os.environ["NO_PROXY"] = "localhost,127.0.0.1,192.168.0.*"
# 清空代理设置，确保请求直连
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

model_name = "/data/zhengwj/model/Qwen/Qwen3-0.6B"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"  # 优先使用GPU
# 重要：确保此处的模型结构与保存ckpt时完全一致
# 例如，如果你使用的是Qwen等预训练模型，通常可以直接从对应的AutoModel加载
try:
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto"
    ).to(device)

except Exception as e:
    # 如果上述方式失败，可能需要你自定义的模型类
    # from my_model import MyModelClass
    # model = MyModelClass().to(device)
    # model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"加载模型时出错: {e}")


def simple_stream_test(message, history):
    response = "这是模拟的回复："
    for char in response + message:
        time.sleep(0.02)
        yield response + message[:message.find(char) + 1]


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


embedding_tokenizer = AutoTokenizer.from_pretrained("/data/zhengwj/model/Qwen/Qwen/Qwen3-Embedding-0.6B",
                                                    padding_side='left')
embedder = AutoModel.from_pretrained("/data/zhengwj/model/Qwen/Qwen/Qwen3-Embedding-0.6B")


def construct_input(query):
    #1. 获取query对应的query embedding
    text = get_detailed_instruct(task, query)
    query_embedding = batch_embedding(embedding_tokenizer, embedder, [text], 16, 8192)[0]
    #2. 从ES库中召回与query相关的内容
    recall_result = recall_from_es_embedding(query, query_embedding)
    #3.内容过滤
    recall_result = filter_es_result(recall_result)[0:1]

    #4.构建一个回答的指令
    prompt_template =  '''你是一个文档问答助手，你可以基于我给定的参考内容来回答问题，
如果问答的答案无法从参考内容中获取，请你回答"无法从参考答案中获取正确的参考信息，请更改询问的方式，或者致电12345询问"
参考内容：
{reference}
用户问题：
{question}
请回答：
'''
    prompt = prompt_template.\
        replace("{reference}", "\n".join(recall_result)).\
        replace("{question}", query)
    return prompt

def stream_chat(message, history, max_new_tokens=8192):
    """
    与模型进行流式对话的核心函数。
    message: 用户当前输入
    history: 对话历史（Gradio自动管理）
    max_new_tokens: 生成文本的最大token数量
    """
    # 1. 构建对话历史提示
    # 将Gradio格式的历史记录转换为模型需要的消息格式
    conversation = history
    message = construct_input(message)
    print(f"Final message is:\n {message}")
    # 加入当前用户问题
    conversation.append({"role": "user", "content": message})

    # 2. 将对话记录转换为模型输入
    # 使用tokenizer的聊天模板（如果模型支持）
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
    else:
        # 简易处理方式
        text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation]) + "\nassistant: "

    inputs = tokenizer([text], return_tensors="pt").to(device)

    # 3. 设置流式输出器
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 4. 配置生成参数
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # 启用采样使输出更多样
        temperature=0.7,  # 控制随机性（0-1，越低越确定）
        top_k=50,  # 顶部k采样
        pad_token_id=tokenizer.eos_token_id
    )

    # 5. 在单独线程中启动生成任务
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    # 6. 通过生成器逐步返回文本
    partial_text = ""
    for new_text in streamer:
        new_text = new_text.replace("<think>", "# 思考内容：\n").replace("</think>", "\n# 答案：\n")
        partial_text += new_text
        # print(new_text)
        yield partial_text


import gradio as gr

# 创建聊天界面
demo = gr.ChatInterface(
    fn=stream_chat,  # 绑定流式聊天函数
    title="京市政务智能大模型",
    description="基于本地CKPT模型搭建的聊天界面，支持流式输出。",
    type="messages",
    examples=["二十二次全省民政会议召开谁出席并讲话？", "省政府党组第36次会议暨省政府第61次常务会议何时召开？"],
)

# 启动服务
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问，如果仅本地使用可设为"127.0.0.1"
        server_port=8000,  # 指定端口
        share=False  # 是否生成公共链接
    )
