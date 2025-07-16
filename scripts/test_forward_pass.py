# scripts/test_forward_pass.py

import torch
import os
import sys
from torch.utils.data import DataLoader
import math

# --- 设置环境 ---
# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# --- 导入我们自己的模块 ---
from diarization_aware_asr.data.dataset import DiarizationAwareDataset, DataCollator
from diarization_aware_asr.models.s_llm import DiarizationAwareSLLM
from diarization_aware_asr.utils.config_loader import load_config

def main():

    print("="*50)
    print("  开始测试模型前向传播 (Forward Pass)  ")
    print("="*50)
    
    # 1. 加载配置
    print("\n[1/5] 正在加载配置...")
    data_config = load_config('data_config.yaml')
    model_config = load_config('model_config.yaml')
    print("配置加载成功。")

    # 2. 实例化模型
    # 注意：为了快速测试，可以考虑在config中用一个更小的模型，如 'Qwen/Qwen2-0.5B-Instruct'
    # 但使用目标模型可以更准确地测试内存和维度
    print("\n[2/5] 正在实例化模型...")
    # 将模型移动到CPU进行测试，避免GPU依赖
    model = DiarizationAwareSLLM(model_config).to('cpu')
    model.eval() # 设置为评估模式，关闭dropout等
    print("模型实例化成功。")

    # 3. 准备数据加载器
    print("\n[3/5] 正在准备数据...")
    dataset = DiarizationAwareDataset(
        metadata_path=data_config['metadata_path'],
        data_config=data_config,
        model_config=model_config
    )
    
    data_collator = DataCollator(
        model_config=model_config,
        tokenizer=model.tokenizer,
        # 特征提取器只需要配置，不需要模型本身
        feature_extractor=model.feature_extractor
    )
    
    # 使用一个小的batch_size
    dataloader = DataLoader(
        dataset, 
        batch_size=2,
        collate_fn=data_collator,
        num_workers=0
    )
    print("数据准备完成。")

    # 4. 取一个批次并执行前向传播
    print("\n[4/5] 正在执行前向传播...")
    try:
        # 使用 with torch.no_grad() 可以在评估模式下节省内存和计算
        with torch.no_grad():
            batch = next(iter(dataloader))
            
            # 将批次数据移动到与模型相同的设备 (CPU)
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to('cpu')
                elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
                    # 处理 speaker_embeddings_list
                    batch_on_device[k] = [t.to('cpu') for t in v]
                else:
                    # 处理 triplets_list
                    batch_on_device[k] = v
            
            # 执行模型的前向传播
            outputs = model(**batch_on_device)
        
        print("前向传播成功！未发生运行时错误。")
    except Exception as e:
        print("\n" + "="*20 + " 错误 " + "="*20)
        print("前向传播失败！")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        print("="*47)
        return # 提前退出

    # 5. 验证输出
    print("\n[5/5] 正在验证输出...")
    
    # --- 验证1: 输出类型和存在性 ---
    print("\n--- 验证1: 输出类型 ---")
    assert outputs is not None, "模型输出为 None"
    assert 'loss' in outputs, "模型输出中缺少 'loss'"
    assert 'logits' in outputs, "模型输出中缺少 'logits'"
    print("Loss 和 Logits 均存在于输出中。 [OK]")
    
    # --- 验证2: Loss 的形状和值 ---
    print("\n--- 验证2: 损失 (Loss) ---")
    loss = outputs.loss
    assert loss.dim() == 0, f"Loss 应该是一个标量, 但其维度为 {loss.dim()}"
    print(f"Loss 是一个标量。 [OK]")
    
    loss_value = loss.item()
    vocab_size = model.llm.config.vocab_size
    expected_loss = math.log(vocab_size)
    
    print(f"  - 损失值: {loss_value:.4f}")
    print(f"  - 词汇表示大小: {vocab_size}")
    print(f"  - 理论初始损失 (log(vocab_size)): {expected_loss:.4f}")
    
    assert not math.isnan(loss_value) and loss_value > 0, "损失值为 NaN 或非正数，这不正确。"
    if loss_value > expected_loss * 2:
        print("  - 警告: 损失值远高于理论初始值，可能存在问题，但并非绝对错误。")
    else:
        print("损失值在合理范围内。 [OK]")

    # --- 验证3: Logits 和 Labels 的形状对齐 ---
    print("\n--- 验证3: 形状对齐 ---")
    logits = outputs.logits
    # 在 forward 方法中，我们最终的 labels 是 padded_final_labels
    # 我们需要从 model 的 forward 方法中获取它的形状，或者在这里重新计算
    # 但一个简单的检查是 logits 的第二维应该和 padded_final_labels 的第二维相同
    # 这里的 batch['padded_labels'] 是 collator 的输出，不是最终喂给模型的，所以不能直接用
    # 但我们可以检查 logits 的形状
    batch_size, seq_len, voc_size = logits.shape
    print(f"Logits 形状: ({batch_size}, {seq_len}, {voc_size})")
    
    assert batch_size == dataloader.batch_size, "Logits 的 batch size 与输入不匹配。"
    assert voc_size == vocab_size, "Logits 的 vocab size 与模型配置不匹配。"
    print("Logits 形状符合预期。 [OK]")
    
    # 这一步是最关键的隐式验证：如果 forward 能够成功计算出 loss，
    # 那么 PyTorch 内部已经确认了 logits (B, L, V) 和 labels (B, L) 的形状是兼容的。
    # L (sequence_length) 必须一致。
    print("能够成功计算Loss，意味着Logits和内部最终的Labels形状对齐。 [OK]")

    print("\n" + "="*50)
    print("  所有检查通过！实现基本正确，可以准备训练。  ")
    print("="*50)


if __name__ == "__main__":
    main()