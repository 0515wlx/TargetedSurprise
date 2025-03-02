import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append('.')  # 添加项目根目录到 Python 路径
from src.TargetedSurprise import TargetedSurprise

class ChunkOptimizer:
    """动态分块优化类"""
    def __init__(self, model, tokenizer, chunk_size=512, overlap=64):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def process_chunks(self, input_ids, past_key_values=None):
        """使用环形缓冲区实现滑动窗口分块，支持kv_cache"""
        next_token_logits = None
        seq_len = input_ids.size(1)
        window = torch.zeros((1, self.chunk_size), dtype=torch.long, device=input_ids.device)
        
        # 初始化窗口
        start_pos = max(0, seq_len - self.chunk_size)
        window[:, :input_ids.size(1)-start_pos].copy_(input_ids[:, start_pos:])
        
        # 逆向滑动处理
        for pos in range(seq_len - self.chunk_size, -self.chunk_size, -self.chunk_size + self.overlap):
            actual_pos = max(0, pos)
            actual_end = actual_pos + self.chunk_size
            
            # 更新窗口内容
            if pos != start_pos:
                window[:, :-self.overlap] = window[:, self.overlap:].clone()
                window[:, -self.overlap:] = input_ids[:, actual_pos:actual_pos+self.overlap]
            
            # 执行推理，使用kv_cache
            with torch.no_grad():
                outputs = self.model(input_ids=window, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
            
            # 确保始终返回有效的logits
            if outputs.logits is not None:
                next_token_logits = outputs.logits[:, -1, :]
            else:
                # 如果logits为None，使用默认值
                vocab_size = self.model.config.vocab_size
                next_token_logits = torch.zeros(
                    (1, vocab_size),
                    dtype=torch.float32,
                    device=input_ids.device
                )
                # 设置默认token的概率
                default_token_id = self.tokenizer.eos_token_id or 0
                next_token_logits[0, default_token_id] = 1.0
        
        # 确保返回值都不为None
        if next_token_logits is None:
            vocab_size = self.model.config.vocab_size
            next_token_logits = torch.zeros(
                (1, vocab_size),
                dtype=torch.float32,
                device=input_ids.device
            )
            # 设置默认token的概率
            default_token_id = self.tokenizer.eos_token_id or 0
            next_token_logits[0, default_token_id] = 1.0
            
        if past_key_values is None:
            # 初始化空的past_key_values
            past_key_values = tuple([(None, None)] * self.model.config.num_hidden_layers)
            
        return next_token_logits, past_key_values

class IncrementalEncoder:
    """增量编码优化类"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cache = {}
        
    def encode(self, text, prev_ids):
        """增量编码实现"""
        new_text = text[len(self.tokenizer.decode(prev_ids)):]
        new_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
        # 将一维张量转换为二维张量 [seq_len, 1]
        return torch.cat([prev_ids, torch.tensor(new_ids)], dim=-1).unsqueeze(-1)

class AdaptiveMemoryManager:
    """自适应显存管理类"""
    def __init__(self, total_mem):
        self.total_mem = total_mem
        self.safety_margin = 0.2  # 20%安全余量
        
    def should_clean_cache(self):
        """判断是否需要清理缓存"""
        used = torch.cuda.memory_allocated()
        return used > self.total_mem * (1 - self.safety_margin)
    
    def dynamic_chunk_size(self):
        """动态计算分块大小"""
        # 计算实际可用显存（保留内存 - 已分配内存）
        free_mem = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        # 精确估算系数（1.2MB/token包含参数和激活值）
        return int(free_mem // (1.2 * 1024**2))

def load_longbench_data():
    """加载本地 LongBench-v2 数据集"""
    with open('datasets/LongBench-v2/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_deepseek_model():
    """加载本地 DeepSeek-R1-Distill-Qwen-1.5B 模型"""
    model_path = 'huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def initialize_targeted_surprise(d_model=64, n_targets=8):
    """初始化 TargetedSurprise 模块"""
    return TargetedSurprise(d_model, n_targets).to("cpu")

def inference_with_targeted_surprise(model, tokenizer, targeted_surprise, data):
    """结合 TargetedSurprise 进行推理"""
    results = []
    
    # 初始化优化工具
    chunk_optimizer = ChunkOptimizer(model, tokenizer, chunk_size=32, overlap=16)
    encoder = IncrementalEncoder(tokenizer)
    mem_manager = AdaptiveMemoryManager(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None
    
    # 定义最大序列长度
    max_seq_length = 16  # 进一步减小最大序列长度以节省显存
    model.gradient_checkpointing_enable()  # 启用梯度检查点
    # 使用更小的batch size
    batch_size = 1
    
    for item in data[1:2]:  # 测试第2个样本
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
        hidden_state = torch.zeros(n_targets, d_model).to("cpu")
        target_texts = [context]
        
        # 显存监控
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated()
            free_mem = total_mem - allocated_mem
            print(f"GPU memory: Allocated {allocated_mem / (1024 ** 3):.2f} GB, Free {free_mem / (1024 ** 3):.2f} GB")
        
        # 初始化输入
        tokenized = tokenizer(input_text, return_tensors="pt", max_length=max_seq_length, truncation=True, padding='max_length')
        input_ids = tokenized['input_ids'].to("cpu")
        attention_mask = tokenized['attention_mask'].to("cpu")
        print(f"Input IDs shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")
        output_ids = []
        
        # 上下文嵌入缓存
        context_embeddings = model.get_input_embeddings()(input_ids)
        
        for i in range(64):  # 进一步减小最大生成长度以节省显存
            # 动态调整分块大小
            if mem_manager:
                chunk_size = mem_manager.dynamic_chunk_size()
                chunk_optimizer.chunk_size = chunk_size
            
            # 初始化past_key_values
            if i == 0:
                past_key_values = None
            
            # 检查序列长度，必要时分块处理
            if input_ids.shape[1] > max_seq_length:
                next_token_logits, past_key_values = chunk_optimizer.process_chunks(input_ids, past_key_values)
            else:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids,
                                 attention_mask=torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device),
                                 past_key_values=tuple(past_key_values) if past_key_values else None,
                                 use_cache=True)
                    next_token_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
            
            next_token = torch.argmax(next_token_logits, dim=-1)
            output_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # 显存清理
            if mem_manager and mem_manager.should_clean_cache():
                torch.cuda.empty_cache()
                # 按模型层数清理KV Cache (DeepSeek-R1有32层)
                if past_key_values is not None:
                    # 保留最近2层并释放其余层的显存
                    retained_layers = 2
                    past_key_values = [(k[-retained_layers:], v[-retained_layers:]) for (k,v) in past_key_values]
                    # 显式释放被截断层的显存
                    for i in range(retained_layers, len(past_key_values[0][0])):
                        del past_key_values[0][0][i]
                        del past_key_values[0][1][i]
                    torch.cuda.empty_cache()
            
            # 应用 TargetedSurprise
            # 保持int类型输入，通过嵌入层获取向量
            # 优化嵌入向量获取，仅处理最后一个token
            with torch.no_grad():
                x_emb = model.get_input_embeddings()(input_ids[:, -1:]).to("cpu").to(torch.float16)
            surprise, hidden_state = targeted_surprise(x_emb.squeeze(0), hidden_state, target_texts)
            
            # 动态更新上下文（差分更新）
            if surprise.abs().max() > 0.5:
                # 只编码新增部分
                new_embeddings = model.get_input_embeddings()(x.unsqueeze(0))
                context_embeddings = torch.cat([context_embeddings, new_embeddings], dim=1)
                input_ids = torch.cat([input_ids, x.unsqueeze(0)], dim=-1)
        
        model_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # 存储结果
        results.append({
            'question': question,
            'model_output': model_output,
            'surprise_score': surprise.mean().item()
        })
    
    return results

if __name__ == "__main__":
    # 设置CUDA内存分配策略
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 加载数据
    dataset = load_longbench_data()
    
    # 加载模型
    model, tokenizer = load_deepseek_model()
    
    # 初始化 TargetedSurprise
    d_model = 64
    n_targets = 8
    targeted_surprise = initialize_targeted_surprise(d_model, n_targets)
    
    # 启用混合精度训练
    from torch.cuda.amp import autocast
    
    # 进行推理
    with autocast():
        results = inference_with_targeted_surprise(model, tokenizer, targeted_surprise, dataset)
    
    # 输出结果
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Model Output: {result['model_output']}")
        print(f"Surprise Score: {result['surprise_score']:.4f}")
        print("-" * 50)