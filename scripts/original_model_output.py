import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

def load_longbench_data():
    """加载本地 LongBench-v2 数据集"""
    with open('datasets/LongBench-v2/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_deepseek_model():
    """加载本地 DeepSeek-R1-Distill-Qwen-1.5B 模型"""
    model_path = 'huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        offload_folder="offload",
        offload_state_dict=True
    ).eval().to(device)
    return model, tokenizer

def simple_inference(model, tokenizer, data):
    """简化版推理函数"""
    results = []
    
    generation_config = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "max_length": 2048
    }
    
    pbar = tqdm(data[1:2], desc="Processing", unit="sample")
    
    for item in pbar:
        context = item['context']
        question = item['question']
        
        # 构建提示
        choices = [item[k] for k in ['choice_A','choice_B','choice_C','choice_D']]
        prompt = f"[INST]根据上下文回答选择题：\n上下文：{context}\n问题：{question}\n选项：\n"
        prompt += "\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(choices))
        prompt += "\n请逐步分析后给出最终答案，格式为：答案：X [/INST]"
        
        # 准备输入
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
        input_ids = tokenized['input_ids'].to(model.device)
        attention_mask = tokenized['attention_mask'].to(model.device)
        
        # 生成输出
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 存储结果
        results.append({
            'question': question,
            'choices': choices,
            'model_output': output_text,
            'correct_answer': item['answer']
        })
    
    return results

if __name__ == "__main__":
    # 加载数据
    dataset = load_longbench_data()
    
    # 加载模型
    model, tokenizer = load_deepseek_model()
    
    # 进行推理
    with torch.no_grad():
        results = simple_inference(model, tokenizer, dataset)
    
    # 输出结果
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Model Output: {result['model_output']}")
        print(f"Correct Answer: {result['correct_answer']}")
        print("-" * 50)