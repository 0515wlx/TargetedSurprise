import sys
import json
import requests
from tqdm import tqdm
from typing import Optional
from openai import OpenAI

def load_longbench_data():
    """加载本地 LongBench-v2 数据集"""
    with open('datasets/LongBench-v2/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def api_inference(client, data):
    """API版本推理函数"""
    results = []
    
    pbar = tqdm(data[1:2], desc="Processing", unit="sample")
    
    for item in pbar:
        context = item['context'][:8192]  # 截断长上下文
        question = item['question']
        
        # 构建提示（保持原有格式）
        choices = [item[k] for k in ['choice_A','choice_B','choice_C','choice_D']]
        prompt = f"根据上下文回答选择题：\n上下文：{context}\n问题：{question}\n选项：\n"
        prompt += "\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(choices))
        prompt += "\n请逐步分析后给出最终答案，格式为：答案：X"
        
        # 调用API
        completion = client.chat.completions.create(
            model="DeepSeek-R1-Distill-Qwen-1.5B",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}],
        )
        
        # 解析结果
        output_text = completion if completion else "API调用失败"
        
        results.append({
            'question': question,
            'choices': choices,
            'model_output': output_text,
            'correct_answer': item['answer']
        })
    
    return results

if __name__ == "__main__":
    # 初始化API客户端
    client = OpenAI(api_key='sk-EisnS8DwfNbCbpcgbQxXPpBsCEPNdODCBJd0u7ce5Pf3VNcC', base_url='https://openapi.coreshub.cn/v1')
    
    if not client.api_key:
        print("警告：请先在CoreHubAPI类中配置有效的API密钥")
    
    # 加载数据
    dataset = load_longbench_data()
    
    # 进行推理
    results = api_inference(client, dataset)
    
    # 输出结果
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Model Output: {result['model_output']}")
        print(f"Correct Answer: {result['correct_answer']}")
        print("-" * 50)