from openai import OpenAI
from prompt import keyword_prompt

client = OpenAI(
    api_key="sk-a460c92184084b06b638f68a0ae7560e",
    base_url="https://api.deepseek.com"
)


def translate(sentence):
    '''
    将sentence(中文)翻译成英文
    :param sentence: 待翻译的句子
    :return:
    '''
    response = client.chat.completions.create(
        model="deepseek-chat", # "deepseek-reasoner"
        messages=[
            {"role": "system", "content": "你是一个翻译助手，"
                                          "你会将用户输入的句子从中文翻译成英文"},
            {"role": "user", "content": sentence},
        ],
        stream=False
    )

    return response.choices[0].message.content


def keyword(document):
    '''
    将sentence(中文)翻译成英文
    :param sentence: 待翻译的句子
    :return:
    '''
    response = client.chat.completions.create(
        model="deepseek-chat", # "deepseek-reasoner"
        messages=[
            #{"role": "system", "content": keyword_prompt.format(document)},
            {"role": "user", "content": keyword_prompt.format(document)},
        ],
        stream=False
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    sentence = "这部《沐浴之友》是一部由秦宇自导自演的青春剧情片，主要讲述了周明辉作为一名温泉营销顾问，通过自己的宣传，最终带动了销售业绩。温泉的经营越来越好，也吸引到了更多的顾客去光顾，周明辉也继续努力的经营着温泉，他希望通过大家来温泉游玩，都能找到属于自己的快乐与幸福的故事。"
    print(keyword(sentence))