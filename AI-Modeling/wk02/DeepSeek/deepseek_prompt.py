# Please install OpenAI SDK first: `pip3 install openai`
import os
import dotenv
from openai import OpenAI
from prompt import keyword_prompt

dotenv.load_dotenv()

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

def translate(sentence):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant able to translate English to Chinese"},
            {"role": "user", "content": sentence},
        ],
        stream=False
    )
    return response.choices[0].message.content

def keyword_extractor(input_text):
    defaults = {
        "input_text": input_text,
        "max_keywords": 10,
        "keyword_type": "all",
        "output_format": "json",
        "min_relevance": "medium"
    }
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            # {"role": "system", "content": "You are an AI specialist in linguistic analysis and information retrieval. Your primary function is to perform advanced keyword extraction from input text based on configurable parameters."},
            {"role": "user", "content": keyword_prompt.format(**defaults)},
        ],
        stream=False
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test translation
    # while True:
    #     sentence = input("Please enter a sentence: ")
    #     print(translate('Hello DeepSeek'))

    # Test DeepSeek Completions
    text = '''
    What is RAG?
    
    Retrieval-Augmented Generation (RAG) is a method that combines two parts:
    
    Retrieval – finding relevant information from a knowledge base (like documents, databases, or the web).
    
    Generation – using a large language model (LLM) to generate an answer based on both the user’s question and the retrieved information.
    
    Why is RAG Useful?
    
    More accurate answers – It grounds the model’s response with real data instead of relying only on memory.
    
    Reduces hallucinations – Since the answer is supported by retrieved facts, the model is less likely to make things up.
    
    Keeps information up to date – You can update the knowledge base without retraining the model.
    
    Works well for custom or private data – You can retrieve information from your own documents or systems.
    
    How It Works (Simplified Flow)
    
    User asks a question
    
    Retriever searches for related documents/chunks from a database or vector store
    
    LLM reads the retrieved content
    
    LLM generates a grounded answer using both context and its own reasoning
    '''
    res = keyword_extractor(input_text=text)
    print(res)