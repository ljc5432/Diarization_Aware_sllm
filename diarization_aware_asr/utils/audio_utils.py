# diarization_aware_asr/utils/audio_utils.py

import torch

def create_energy_weighted_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    通过计算每个声道的能量，加权混合成一个优化的单声道。
    如果已经是单声道，则直接返回。
    """
    if waveform.shape[0] <= 1:
        return waveform

    # print(f"  - 检测到多声道音频 (通道数: {waveform.shape[0]})，正在进行能量加权混合...")
    
    # 计算每个声道的能量 (RMS)
    # 添加一个小的epsilon防止能量为0时除以0
    energies = torch.sqrt(torch.mean(waveform**2, dim=1)) + 1e-9
    
    # 如果所有通道能量都极低，直接平均混合
    if torch.sum(energies) < 1e-6:
        return torch.mean(waveform, dim=0, keepdim=True)
        
    weights = energies / torch.sum(energies)
    
    # unsqueeze(1) 将权重从 [C] 变成 [C, 1] 以便广播
    weighted_waveform = torch.sum(waveform * weights.unsqueeze(1), dim=0, keepdim=True)
    
    return weighted_waveform