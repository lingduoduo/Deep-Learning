from openai import OpenAI, AsyncOpenAI

client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")


def inference_one(query):
    try:
        messages = [
                {"role": "user", "content": query}
            ]
        response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages
            )
    except Exception as e:
        print(f"error is {e}")
        return ""
    return response.choices[0].message.content

prompt = '''代码生成任务
1.任务名称：
根据需求描述和测试用例来生成代码。

2.任务描述：
需求描述定义了需要生成的代码的用途和要求；提供的测试用例包括了代码的输入参数列表、期望的输出，用于测试你生成的代码。
你生成的所有代码需要包含必要的库导入步骤。不允许更改方法的名称和已经给定的形式参数的名称和类型。
除了定义函数的功能体和必要的包导入步骤，你生成的代码不应该包括任何多余内容。你生成的代码需要符合Python代码格式要求，要能正确运行。

3.生成的代码的片段必须符合下面的格式要求：
【代码开始】
your code here
【代码结束】

比如：
【代码开始】
from typing import List

def has_xxx_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
    return False
【代码结束】

4. 需求描述和测试用例
<需求描述和测试用例--开始>
{question_turns_1}
<需求描述和测试用例--结束>
'''
import pandas as pd
from tqdm import tqdm
code_test_data = pd.read_excel("data/model_response/code_test.xlsx")
model_answer_model1_turns_1 = []
for i in tqdm(range(len(code_test_data))):
    code_question = code_test_data.iloc[i]["question_turns_1"]
    print(f"start to inference code_question is {code_question}")
    answer = inference_one(prompt.replace("{question_turns_1}", code_question))
    print(f"answer is {answer}")
    model_answer_model1_turns_1.append(answer)

code_test_data["model_answer_model1_turns_1"] = model_answer_model1_turns_1

code_test_data.to_excel("data/model_response/code_test_result_0629.xlsx")