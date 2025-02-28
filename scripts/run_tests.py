import sys
import unittest
import yaml
sys.path.append('.')  # 添加项目根目录到 Python 路径
from tests.test_targeted_surprise import TestTargetedSurprise

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:  # 指定 UTF-8 编码
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    # 加载配置
    config = load_config('configs/targeted_surprise_config.yaml')
    print("Loaded config:", config)
    
    # 运行测试
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTargetedSurprise)
    unittest.TextTestRunner(verbosity=2).run(test_suite)