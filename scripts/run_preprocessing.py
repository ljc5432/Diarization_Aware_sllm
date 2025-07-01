# scripts/run_preprocessing.py

import yaml
import os
import sys

# 将项目根目录添加到Python路径中，以便导入我们自己的模块
# 假设脚本在 scripts/ 目录下，项目根目录是其父目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from diarization_aware_asr.data.preprocessor import Preprocessor

def main():
    # 加载配置文件
    config_path = os.path.join(project_root, 'configs', 'data_config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件未找到于 {config_path}")
        print("请确保您已创建 configs/data_config.yaml 文件。")
        return

    # 实例化并运行预处理器
    preprocessor = Preprocessor(config)
    preprocessor.process_directory()

if __name__ == "__main__":
    main()