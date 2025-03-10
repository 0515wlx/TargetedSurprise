import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

class TargetedSurprise(nn.Module):
    def __init__(self, d_model, max_targets):
        super().__init__()
        # 启用标志
        self.enabled = True
        # 可学习的目标查询向量
        self.target_queries = nn.Parameter(torch.randn(max_targets, d_model))
        
        self.d_model = d_model
        self.max_targets = max_targets  # 最大目标数
        
        # 初始化TF-IDF分析器
        self.tfidf = TfidfVectorizer(
            max_features=512,
            tokenizer=jieba.cut,
            stop_words=['的','是','在','和','了','有']
        )
        
        # 初始化上下文记忆
        self.context_memory = nn.Parameter(torch.zeros(max_targets, d_model))
        
        # 初始化decay_gate
        self.decay_gate = None
        self.input_dim = None
        
    def tfidf_analysis(self, text, max_keywords=8):
        """使用TF-IDF提取关键词"""
        # 分词并计算TF-IDF
        tfidf_matrix = self.tfidf.fit_transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        
        # 获取TF-IDF权重
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # 按权重排序并选择前max_keywords个关键词
        sorted_indices = tfidf_scores.argsort()[::-1]
        keywords = [feature_names[i] for i in sorted_indices[:max_keywords]]
        
        return keywords
        
        # 删除重复的参数注册代码
        
    def _create_sliding_window_mask(self, seq_len, window_size):
        """创建滑动窗口注意力掩码"""
        # 创建全1矩阵
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        # 设置滑动窗口
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            end = i + 1
            mask[i, :start] = 0
            mask[i, end:] = 0
        return mask
        # 动态衰减系数生成器（延迟初始化）
        self.decay_gate = None
        self.input_dim = None
        
    def _init_decay_gate(self, input_dim, device):
        """根据输入维度初始化decay_gate"""
        if self.decay_gate is None or self.input_dim != input_dim:
            self.input_dim = input_dim
            self.decay_gate = nn.Sequential(
                nn.Linear(input_dim, 512, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(512, 1, dtype=torch.float32)
            ).to(device)
            # 保持float32类型
            for param in self.decay_gate.parameters():
                param.data = param.data.float()
        
    def forward(self, x_emb, position_state, target_texts):
        """
        x_emb: [seq_len, d_model] 输入嵌入向量
        position_state: [n_targets, d_model] 位置状态矩阵
        target_texts: List[str] 激活的目标原文
        """
        # 标准化输入形状和类型
        x_emb = x_emb.float()
        if x_emb.dim() == 3:
            x_emb = x_emb.squeeze(0)
        seq_len, _ = x_emb.shape
        
        # 目标文本编码
        # 使用TF-IDF分析更新目标文本
        active_targets = []
        for text in target_texts:
            keywords = self.tfidf_analysis(text)
            active_targets.extend(keywords)
        
        # 动态调整目标数量
        active_targets = active_targets[:self.max_targets]
        self.n_targets = len(active_targets)
        
        # 更新目标编码
        target_embeddings = []
        for i, keyword in enumerate(active_targets):
            embedding = self.target_queries[i].clone()
            # 将关键词特征编码到embedding
            chars = list(keyword)
            for j in range(min(len(chars), self.d_model//2)):
                embedding[j*2] = len(chars[j])  # 字符长度特征
                embedding[j*2+1] = hash(chars[j]) % 100  # 哈希特征
            # 更新上下文记忆
            self.context_memory[i] = embedding.detach()
            target_embeddings.append(embedding)
        
        # 将目标文本编码转换为tensor
        target_embeddings = torch.stack(target_embeddings).to(x_emb.device)
        
        # 初始化decay_gate
        if x_emb.dim() == 1:
            x_emb = x_emb.unsqueeze(0)
        self._init_decay_gate(x_emb.size(1), x_emb.device)
        
        # 分块处理设置
        # 使用固定窗口大小
        window_size = 4096
        # 初始化缓存
        past_key_values = None
        # 使用模型自带的缓存管理
        if past_key_values is not None and window_size > 0:
            past_key_values = self._truncate_cache(past_key_values, window_size)
        # 生成符合窗口大小的注意力掩码
        attention_mask = self._create_sliding_window_mask(x_emb.size(0), window_size)
        # 分块处理设置
        chunks = x_emb.split(window_size, dim=0)
        sim_chunks = []
        
        # 目标相似度计算（分块处理）
        target_embeddings_normalized = target_embeddings / target_embeddings.norm(dim=1, keepdim=True)
        
        for chunk in chunks:
            # 确保输入维度正确
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
            chunk_normalized = chunk / chunk.norm(dim=1, keepdim=True)
            
            # 调整目标编码维度以匹配输入
            if chunk_normalized.size(1) != target_embeddings_normalized.size(1):
                # 如果维度不匹配，使用线性投影对齐
                projection = nn.Linear(chunk_normalized.size(1), target_embeddings_normalized.size(1), dtype=torch.float32).to(x_emb.device)
                chunk_normalized = projection(chunk_normalized)
                
            # 计算与目标文本的相似度
            chunk_sim = torch.matmul(chunk_normalized, target_embeddings_normalized.t())
            sim_chunks.append(chunk_sim)
            
        sim = torch.cat(sim_chunks, dim=0)  # [seq_len, n_targets]
        
        # 动态重要性权重（分块计算）
        alpha_chunks = []
        for chunk in chunks:
            chunk_alpha = torch.sigmoid(self.decay_gate(chunk)).unsqueeze(-1)
            alpha_chunks.append(chunk_alpha)
        alpha = torch.cat(alpha_chunks, dim=0)  # [seq_len, 1]
        
        # 缓存截断（确保不超过窗口大小）
        # 严格保持缓存对象类型（使用DynamicCache接口）
        if past_key_values is not None and window_size > 0:
            # 创建新的DynamicCache实例
            new_cache = DynamicCache()
            # 逐层复制并截断缓存
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                new_cache.update(
                    key_states[..., -window_size:, :],
                    value_states[..., -window_size:, :],
                    layer_idx,
                )
            past_key_values = new_cache

        # 显存监控与自动调整
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated()
            free_mem = (total_mem - allocated_mem) / (1024 ** 3)
            if free_mem < 1.0:  # 显存不足1GB时自动调整
                core_threshold = 0.9
            else:
                core_threshold = 0.95
        else:
            core_threshold = 0.95
            
        # 目标导向惊喜计算（优化内存使用）
        core_mask = (alpha > core_threshold).float()  # 动态保留系数阈值
        position_state = position_state.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, n_targets, d_model]
        
        # 分块计算target_surprise
        surprise_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_sim = sim_chunks[i]
            chunk_alpha = alpha_chunks[i]
            max_chunk_size = window_size
            chunk_core_mask = core_mask[i*chunk.size(0):(i+1)*chunk.size(0)].unsqueeze(-1)
            
            # 使用目标文本编码计算surprise
            target_weights = torch.softmax(chunk_sim, dim=-1)  # [seq_len, n_targets]
            target_weights = target_weights.unsqueeze(-1).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, n_targets, 1]
            chunk_alpha = chunk_alpha.unsqueeze(-1).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, 1, 1]
            chunk_core_mask = chunk_core_mask.unsqueeze(-1).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, 1, 1]
            
            # 重新计算chunk_surprise
            # 调整position_state维度以匹配target_weights
            pos_state = position_state.expand(-1, target_weights.size(1), -1, -1, -1)
            chunk_surprise = (target_weights * chunk_alpha * chunk_core_mask) - \
                           (pos_state * (1 - chunk_alpha))
            surprise_chunks.append(chunk_surprise)
            
        target_surprise = torch.cat(surprise_chunks, dim=0)
        
        # 添加TargetSurprise检测机制
        surprise_detector = torch.where(
            target_surprise.abs() > 0.5,
            torch.ones_like(target_surprise),
            torch.zeros_like(target_surprise)
        )
        
        # 记忆更新规则
        new_hidden = position_state.squeeze(0) + target_surprise.mean(dim=0) 
        
        # 调整输出形状为 [seq_len, n_targets]
        target_surprise = target_surprise.mean(dim=-1)  # 对最后一个维度取平均
        return target_surprise, new_hidden