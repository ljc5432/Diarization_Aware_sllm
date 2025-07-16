# scripts/test_dataset.py (更新版)

import torch
import os
import sys
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from diarization_aware_asr.data.dataset import DiarizationAwareDataset, DataCollator
from diarization_aware_asr.models.s_llm import DiarizationAwareSLLM
from diarization_aware_asr.utils.config_loader import load_config

def main():
    print("--- 开始测试数据集与数据整理器 ---")
    
    data_config = load_config('data_config.yaml')
    model_config = load_config('model_config.yaml')
    
    # 1. 实例化模型以获取tokenizer和feature_extractor
    # 这只是为了获取它们，不是为了训练
    print("正在实例化模型以获取工具...")
    model = DiarizationAwareSLLM(model_config)
    tokenizer = model.tokenizer
    feature_extractor = model.feature_extractor
    
    # 2. 实例化数据集
    dataset = DiarizationAwareDataset(
        metadata_path=data_config['metadata_path'],
        data_config=data_config,
        model_config=model_config
    )
    
    # 3. 实例化新的 DataCollator
    data_collator = DataCollator(
        model_config=model_config,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor
    )
    
    # 4. 创建 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2, # 重要：当前collator只支持batch_size=1
        collate_fn=data_collator,
        num_workers=0
    )
    
    # 5. 取一个批次的数据并检查
    print("\n--- 从DataLoader中取出一个批次并检查 ---")
    try:
        batch = next(iter(dataloader))
        
        print("批次包含的键:", batch.keys())
        print("\n--- 维度检查 ---")
        print(f"audio_features shape: {batch['audio_features'].shape}")
        print(f"input_ids shape:      {batch['input_ids'].shape}")
        print(f"labels shape:         {batch['labels'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")

        print("\n--- 内容检查 (以第一个样本为例) ---")
        print("input_ids[0]:\n", batch['input_ids'][0])
        print("可以看到左边的填充token ID:", tokenizer.pad_token_id)
        
        print("\nlabels[0]:\n", batch['labels'][0])
        print("可以看到右边的填充值: -100")

        print("\nattention_mask[0]:\n", batch['attention_mask'][0])
        print("可以看到左边对应padding位置为0")

    except Exception as e:
        print(f"\n从 DataLoader 中取数据时发生错误: {e}")

if __name__ == "__main__":
    main()