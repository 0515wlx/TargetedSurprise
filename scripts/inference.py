import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append('.')  # 添加项目根目录到 Python 路径
from src.TargetedSurprise import TargetedSurprise

def load_longbench_data():
    """加载本地 LongBench-v2 数据集"""
    with open('datasets/LongBench-v2/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_deepseek_model():
    """加载本地 DeepSeek-R1-Distill-Qwen-1.5B 模型"""
    model_path = 'huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def initialize_targeted_surprise(d_model=64, n_targets=8):
    """初始化 TargetedSurprise 模块"""
    return TargetedSurprise(d_model, n_targets)

def inference_with_targeted_surprise(model, tokenizer, targeted_surprise, data):
    """结合 TargetedSurprise 进行推理"""
    results = []
    
    for item in data[:2]:  # 测试前2个样本
        # 准备输入（添加长度检查）
        context = item['context']
        question = item['question']
        
        # 动态截断上下文
        max_context_length = 8192  # 最大上下文长度
        if len(context) > max_context_length:
            context = context[:max_context_length]
            print(f"Warning: Context truncated to {max_context_length} tokens")
            
        input_text = f"Context: {context}\nQuestion: {question}"
        
        # 初始化状态
        hidden_state = torch.zeros(n_targets, d_model)
        target_texts = [context]
        
        # 显存监控
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated()
            free_mem = total_mem - allocated_mem
            print(f"GPU memory: Allocated {allocated_mem / (1024 ** 3):.2f} GB, Free {free_mem / (1024 ** 3):.2f} GB")
        
        # 分步推理（带动态分块）
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=8192, truncation=True)['input_ids']
        output_ids = []
        
        # 动态分块参数
        max_seq_length = 8192
        chunk_size = 2048
        overlap_size = 128
        
        for i in range(512):  # 最大生成长度
            # 检查序列长度，必要时分块处理
            if input_ids.shape[1] > max_seq_length:
                # 分块处理
                chunks = []
                start_idx = max(0, input_ids.shape[1] - chunk_size)
                end_idx = input_ids.shape[1]
                
                while start_idx >= 0:
                    chunk = input_ids[:, start_idx:end_idx]
                    chunks.append(chunk)
                    end_idx = start_idx
                    start_idx = max(0, start_idx - chunk_size + overlap_size)
                
                # 反向处理分块
                chunks = chunks[::-1]
                
                # 分块推理
                for chunk in chunks:
                    with torch.no_grad():
                        outputs = model(input_ids=chunk,
                                     attention_mask=torch.ones_like(chunk))
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1)
            else:
                # 正常推理
                with torch.no_grad():
                    outputs = model(input_ids=input_ids,
                                 attention_mask=torch.ones_like(input_ids))
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
            
            output_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # 显存清理
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 应用 TargetedSurprise
            x = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            x = tokenizer(x, return_tensors="pt")['input_ids'].float()
            surprise, hidden_state = targeted_surprise(x, hidden_state, target_texts)
            
            # 动态更新上下文
            if surprise.abs().max() > 0.5:  # 检测到显著变化
                target_texts.append(x)  # 添加新目标
                # 更新输入以包含新上下文
                input_text = f"Context: {context}\nAdditional Context: {x}\nQuestion: {question}"
                input_ids = tokenizer(input_text, return_tensors="pt")['input_ids']
        
        model_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # 存储结果
        results.append({
            'question': question,
            'model_output': model_output,
            'surprise_score': surprise.mean().item()
        })
    
    return results

if __name__ == "__main__":
    # 加载数据
    dataset = load_longbench_data()
    
    # 加载模型
    model, tokenizer = load_deepseek_model()
    
    # 初始化 TargetedSurprise
    d_model = 64
    n_targets = 8
    targeted_surprise = initialize_targeted_surprise(d_model, n_targets)
    
    # 进行推理
    results = inference_with_targeted_surprise(model, tokenizer, targeted_surprise, dataset)
    
    # 输出结果
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Model Output: {result['model_output']}")
        print(f"Surprise Score: {result['surprise_score']:.4f}")
        print("-" * 50)