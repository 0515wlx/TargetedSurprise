import sys
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
import transformers
sys.path.append('.')  # 添加项目根目录到 Python 路径
from src.TargetedSurprise import TargetedSurprise

class IncrementalEncoder:
    """增量编码优化类"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cache = {}
        
    def encode(self, text, prev_ids):
        """增量编码实现（改进版）"""
        # 初始化processed_chars（如果不存在）
        if not hasattr(self, 'processed_chars'):
            self.processed_chars = 0
            
        # 获取新增文本（考虑可能的删除操作）
        new_text = text[self.processed_chars:]
        
        # 编码新文本并更新处理进度
        new_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
        self.processed_chars += len(new_text)  # 更新已处理字符数
        
        # 处理可能的删除操作：当检测到文本缩短时重置进度
        if len(text) < self.processed_chars:
            self.processed_chars = len(text)
            new_text = text[self.processed_chars:]
            new_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
            self.processed_chars += len(new_text)
            
        # 转换张量形状并拼接
        new_ids_tensor = torch.tensor(new_ids).to(prev_ids.device)
        if prev_ids.dim() == 1:
            prev_ids = prev_ids.unsqueeze(-1)
        return torch.cat([prev_ids, new_ids_tensor.unsqueeze(-1)], dim=0)

def load_longbench_data():
    """加载本地 LongBench-v2 数据集"""
    with open('datasets/LongBench-v2/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_deepseek_model():
    """加载本地 DeepSeek-R1-Distill-Qwen-1.5B 模型"""
    model_path = 'huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float32,  # 强制使用float32
        low_cpu_mem_usage=True,
        offload_folder="offload",
        offload_state_dict=True
    ).eval().to(device)
    return model, tokenizer

def initialize_targeted_surprise(d_model=64, max_targets=8):
    """初始化 TargetedSurprise 模块"""
    return TargetedSurprise(d_model, max_targets).to(model.device)

def inference_with_targeted_surprise(model, tokenizer, targeted_surprise, data):
    """结合 TargetedSurprise 进行推理"""
    results = []
    
    # 初始化优化工具
    encoder = IncrementalEncoder(tokenizer)
    
    # 定义最大序列长度
    max_seq_length = 2048  # 增大序列长度保证完整上下文
    # 禁用梯度检查点，推理模式下不需要
    # 配置生成参数
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
         
    generation_config = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 256,  # 增加生成token数量以支持完整推理
        "temperature": 0.6,  # 根据DeepSeek推荐设置为0.6
        "top_p": 0.9,
        "top_k": 50,  # 增加top_k以增强多样性
        "do_sample": True,
        "repetition_penalty": 1.2,  # 防止重复
        "max_length": 2048  # 增加最大序列长度
    }
    
    # 初始化进度条
    pbar = tqdm(data[3:4], desc="Processing", unit="sample")
    
    def get_gpu_memory():
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return {
                'total': total / (1024 ** 3),
                'allocated': allocated / (1024 ** 3),
                'reserved': reserved / (1024 ** 3)
            }
        return None
    
    for item in pbar:  # 测试第2个样本
        # 更新进度条显示GPU内存信息
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            pbar.set_postfix({
                'GPU Alloc': f"{gpu_mem['allocated']:.2f}GB",
                'GPU Resv': f"{gpu_mem['reserved']:.2f}GB"
            })
        # 准备输入（添加长度检查）
        context = item['context']
        question = item['question']
        
        # 动态截断上下文
        max_context_length = 8192  # 最大上下文长度
        if len(context) > max_context_length:
            context = context[:max_context_length]
            print(f"Warning: Context truncated to {max_context_length} tokens")
            
        # 构建新提示
        choices = [item[k] for k in ['choice_A','choice_B','choice_C','choice_D']]
        prompt = f"[INST]根据上下文回答选择题：\n上下文：{context}\n问题：{question}\n选项：\n"
        prompt += "\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(choices))
        prompt += "\n请用<think>标签包裹你的思考过程，并在最后给出答案，格式为：\n<think>\n...\n</think>\n答案：X [/INST]"
        input_text = prompt
        
        # 初始化状态
        hidden_state = torch.zeros(max_targets, d_model).to(model.device)
        # 使用问题和选项作为目标文本，确保长度不超过最大目标数
        # 根据配置生成目标文本
        target_texts = []
        if targeted_surprise.enabled:
            # 使用TF-IDF动态生成目标文本
            target_texts = targeted_surprise.tfidf_analysis(context, max_keywords=targeted_surprise.max_targets)
            # 添加问题和选项
            target_texts.extend([question] + choices)
            # 截断到最大目标数
            target_texts = target_texts[:targeted_surprise.max_targets]
        
        # 显存监控
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated()
            free_mem = total_mem - allocated_mem
            print(f"GPU memory: Allocated {allocated_mem / (1024 ** 3):.2f} GB, Free {free_mem / (1024 ** 3):.2f} GB")
        
        # 初始化输入
        tokenized = tokenizer(input_text, return_tensors="pt", truncation=True, padding=False)
        input_ids = tokenized['input_ids'].to(model.device)
        attention_mask = tokenized['attention_mask'].to(model.device)

        # print(f"Input IDs shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")
        # print(f"Input IDs: {input_ids}")
        # print(f"Attention mask: {attention_mask}")

        output_ids = []
        
        # 上下文嵌入缓存
        context_embeddings = model.get_input_embeddings()(input_ids)
        
        # 使用model.generate()进行完整推理
        with torch.no_grad():
            # 先运行一次forward获取logits
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits = outputs.logits
            # print(f"Logits shape: {logits.shape}")
            # print(f"Logits min: {logits.min().item()}, max: {logits.max().item()}")
            # print(f"Logits contains NaN: {torch.isnan(logits).any().item()}")
            # print(f"Logits contains Inf: {torch.isinf(logits).any().item()}")
            
            # 生成输出
            # 获取<think>的token id
            think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generation_config["max_new_tokens"],
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                do_sample=True,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # forced_bos_token_id=think_token_id  # 移除强制开始符号
            )
            output_ids = outputs[0].tolist()
            
            # StaticCache会自动管理显存，无需手动清理
            
            # 应用 TargetedSurprise
            # 保持int类型输入，通过嵌入层获取向量
            # 优化嵌入向量获取，仅处理最后一个token
            with torch.no_grad():
                x_emb = model.get_input_embeddings()(input_ids[:, -1:]).to(model.device)
                if targeted_surprise.enabled:
                    surprise, hidden_state = targeted_surprise(x_emb.squeeze(0).float(), hidden_state.float(), target_texts)
                else:
                    surprise = None
            
            # 动态更新上下文（差分更新）
            if surprise is not None and surprise.abs().max() > 0.5:
                # 只编码新增部分
                new_embeddings = model.get_input_embeddings()(input_ids[:, -1:])
                context_embeddings = torch.cat([context_embeddings, new_embeddings], dim=1)
        
        # 解码并清理模型输出
        model_output = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        # 提取答案
        def extract_answer(text):
            # 增强版答案提取，处理更多格式情况
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL|re.IGNORECASE)
            answer_part = text[think_match.end():] if think_match else text
            
            # 尝试多种匹配模式
            patterns = [
                r'答案\s*[:：]\s*([ABCD])',        # 标准格式
                r'[\(（]([ABCD])[\)）]',           # 括号格式
                r'\n\s*([ABCD])\s*\n',            # 独立行格式
                r'最终答案\s*[:：]\s*([ABCD])',    # 明确标注格式
                r'选项\s*([ABCD])\s*正确'          # 描述性格式
            ]
            
            for pattern in patterns:
                match = re.search(pattern, answer_part, re.IGNORECASE)
                if match:
                    return match.group(1).upper()  # 统一返回大写字母
            
            # 如果所有模式都失败，尝试最后一个字母
            last_letter = re.findall(r'[ABCD]', answer_part)
            return last_letter[-1] if last_letter else None
            
        predicted_answer = extract_answer(model_output)
        if not predicted_answer:
            print(f"警告：未检测到有效答案格式 - {model_output}")
            predicted_answer = "N/A"
        
        # 计算surprise score
        if targeted_surprise.enabled:
            surprise_score = surprise.mean().item() if surprise is not None else 0.0
        else:
            surprise_score = None  # 未激活时设为None
            
        # 存储完整结果
        results.append({
            'question': question,
            'choices': [item[k] for k in ['choice_A','choice_B','choice_C','choice_D']],
            'model_output': model_output,
            'predicted_answer': predicted_answer,
            'correct_answer': item['answer'],
            'is_correct': predicted_answer == item['answer'],
        })
        # 仅在激活时添加surprise_score
        if targeted_surprise.enabled:
            results[-1]['surprise_score'] = surprise_score
    
    return results

if __name__ == "__main__":
    # 加载数据
    dataset = load_longbench_data()
    
    # 加载模型
    model, tokenizer = load_deepseek_model()
    
    # 初始化 TargetedSurprise
    d_model = 64
    max_targets = 8
    targeted_surprise = initialize_targeted_surprise(d_model, max_targets)
    
    # 启用混合精度训练
    from torch.cuda.amp import autocast
    
    # 进行推理
    with torch.no_grad():
        results = inference_with_targeted_surprise(model, tokenizer, targeted_surprise, dataset)
    
    # 输出结果
    # for result in results:
    #     print(f"Question: {result['question']}")
    #     print(f"Model Output: {result['model_output']}")
    #     print(f"Surprise Score: {result['surprise_score']:.4f}")
    #     print("-" * 50)
    
    # 保存报告
    import os
    from datetime import datetime
    
    # 创建reports目录
    os.makedirs('reports', exist_ok=True)
    
    # 生成报告文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'reports/inference_report_{timestamp}.json'
    
    # 准备报告内容
    report_data = {
        'timestamp': timestamp,
        'environment': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'transformers_version': transformers.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'
        },
        'results': results
    }
    
    # 保存报告
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport saved to {report_filename}")