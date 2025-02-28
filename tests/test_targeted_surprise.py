import torch
import unittest
from src.TargetedSurprise import TargetedSurprise

class TestTargetedSurprise(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_targets = 8
        self.seq_len = 128
        self.model = TargetedSurprise(self.d_model, self.n_targets)
        
    def test_forward_pass(self):
        # 测试前向传播
        x = torch.randn(self.seq_len, self.d_model)
        position_state = torch.randn(self.n_targets, self.d_model)
        target_texts = ["text1", "text2", "text3", "text4", 
                       "text5", "text6", "text7", "text8"]
        
        surprise, new_hidden = self.model(x, position_state, target_texts)
        
        # 验证输出形状
        self.assertEqual(surprise.shape, (self.seq_len, self.n_targets))
        self.assertEqual(new_hidden.shape, (self.n_targets, self.d_model))
        
    def test_initialization(self):
        # 验证参数初始化
        self.assertEqual(self.model.target_queries.shape, 
                        (self.n_targets, self.d_model))
        self.assertEqual(self.model.decay_gate.in_features, self.d_model)
        self.assertEqual(self.model.decay_gate.out_features, 1)

if __name__ == '__main__':
    unittest.main()