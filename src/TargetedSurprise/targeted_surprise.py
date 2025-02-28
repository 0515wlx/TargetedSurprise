import torch
import torch.nn as nn

class TargetedSurprise(nn.Module):
    def __init__(self, d_model, n_targets):
        super().__init__()
        # 可学习的目标查询向量
        self.target_queries = nn.Parameter(torch.randn(n_targets, d_model))
        # 动态衰减系数生成器（延迟初始化）
        self.decay_gate = None
        self.input_dim = None
        
    def _init_decay_gate(self, input_dim):
        """根据输入维度初始化decay_gate"""
        if self.decay_gate is None or self.input_dim != input_dim:
            self.input_dim = input_dim
            self.decay_gate = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        
    def forward(self, x, position_state, target_texts):
        """
        x: [seq_len, d_model] 输入序列
        position_state: [n_targets, d_model] 位置状态矩阵
        target_texts: List[str] 激活的目标原文
        """
        seq_len, _ = x.shape
        
        # 初始化decay_gate
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self._init_decay_gate(x.size(1))
        
        # 分块处理设置
        max_chunk_size = 4096  # 每个分块最大长度
        chunks = torch.split(x, max_chunk_size, dim=0)
        sim_chunks = []
        
        # 目标相似度计算（分块处理）
        target_queries_normalized = self.target_queries / self.target_queries.norm(dim=1, keepdim=True)
        
        for chunk in chunks:
            # 确保输入维度正确
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
            chunk_normalized = chunk / chunk.norm(dim=1, keepdim=True)
            
            # 调整target_queries_normalized维度以匹配输入
            if chunk_normalized.size(1) != target_queries_normalized.size(1):
                # 如果维度不匹配，使用线性投影对齐
                projection = nn.Linear(chunk_normalized.size(1), target_queries_normalized.size(1))
                chunk_normalized = projection(chunk_normalized)
                
            chunk_sim = torch.matmul(chunk_normalized, target_queries_normalized.t())
            sim_chunks.append(chunk_sim)
            
        sim = torch.cat(sim_chunks, dim=0)  # [seq_len, n_targets]
        
        # 动态重要性权重（分块计算）
        alpha_chunks = []
        for chunk in chunks:
            chunk_alpha = torch.sigmoid(self.decay_gate(chunk))
            alpha_chunks.append(chunk_alpha)
        alpha = torch.cat(alpha_chunks, dim=0)  # [seq_len, 1]
        
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
        position_state = position_state.unsqueeze(0)  # [1, n_targets, d_model]
        
        # 分块计算target_surprise
        surprise_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_sim = sim_chunks[i]
            chunk_alpha = alpha_chunks[i]
            chunk_core_mask = core_mask[i*max_chunk_size:(i+1)*max_chunk_size]
            
            chunk_surprise = (chunk_sim.unsqueeze(-1) * chunk_alpha.unsqueeze(1) * chunk_core_mask.unsqueeze(1)) - \
                           (position_state * (1-chunk_alpha.unsqueeze(1)))
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
        
        return target_surprise.squeeze(-1), new_hidden