import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append('.')  # 添加项目根目录到 Python 路径
from src.TargetedSurprise import TargetedSurprise

def load_single_sample():
    """加载单个测试样本"""
    with open('datasets/LongBench-v2/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[0]  # 只返回第一个样本

def load_deepseek_model():
    """加载本地 DeepSeek-R1-Distill-Qwen-1.5B 模型"""
    model_path = 'huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def initialize_targeted_surprise(d_model=64, n_targets=8):
    """初始化 TargetedSurprise 模块"""
    return TargetedSurprise(d_model, n_targets)

def lightweight_inference(model, tokenizer, targeted_surprise, sample):
    """轻量级推理"""
    context = sample['context'][:512]  # 截取前512个字符
    question = sample['question']
    
    input_text = f"Context: {context}\nQuestion: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt")['input_ids']
    
    # 初始化状态
    hidden_state = torch.zeros(n_targets, d_model)
    target_texts = [context]
    
    # 简化推理过程
    with torch.no_grad():
        # 确保tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 生成attention_mask
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        
        # 使用max_new_tokens并添加attention_mask
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=100
        )
    
    model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 应用 TargetedSurprise
    x = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    x = tokenizer(x, return_tensors="pt")['input_ids'].float()
    surprise, _ = targeted_surprise(x, hidden_state, target_texts)
    
    return {
        'question': question,
        'model_output': model_output,
        'surprise_score': surprise.mean().item()
    }

if __name__ == "__main__":
    # 加载单个样本
    sample = load_single_sample()
    
    # 加载模型
    model, tokenizer = load_deepseek_model()
    
    # 初始化 TargetedSurprise
    d_model = 64
    n_targets = 8
    targeted_surprise = initialize_targeted_surprise(d_model, n_targets)
    
    # 进行轻量级推理
    result = lightweight_inference(model, tokenizer, targeted_surprise, sample)
    
    # 输出结果
    print(f"Question: {result['question']}")
    print(f"Model Output: {result['model_output']}")
    print(f"Surprise Score: {result['surprise_score']:.4f}")