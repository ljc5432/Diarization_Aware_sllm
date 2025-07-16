# diarization_aware_asr/utils/config_loader.py

import yaml
import os

def load_config(config_name: str) -> dict:
    """
    加载指定名称的配置文件。

    Args:
        config_name (str): 'data_config.yaml' 或 'model_config.yaml'

    Returns:
        dict: 加载后的配置字典。
    """
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'configs', config_name)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：配置文件未找到于 {config_path}")