# diarization_aware_asr/models/components.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCrossAttention(nn.Module):
    """
    门控交叉注意力模块。
    用于将说话人嵌入信息融合到语音特征中。
    """
    def __init__(self, speech_dim: int, speaker_dim: int, num_heads: int = 8):
        """
        初始化模块。

        Args:
            speech_dim (int): 语音特征的维度 (Query dimension)。
            speaker_dim (int): 说话人嵌入的维度 (Key/Value dimension)。
            num_heads (int, optional): 多头注意力的头数。默认为 8。
        """
        super().__init__()
        
        # 确保语音维度可以被头数整除
        if speech_dim % num_heads != 0:
            raise ValueError(f"speech_dim ({speech_dim}) 必须能被 num_heads ({num_heads}) 整除。")

        self.speech_dim = speech_dim
        self.speaker_dim = speaker_dim
        self.num_heads = num_heads
        
        # 1. 交叉注意力层
        # 我们使用 PyTorch 内置的 MultiheadAttention，它非常高效。
        # embed_dim 是 Query 的维度，kdim/vdim 是 Key/Value 的维度。
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=speech_dim,
            kdim=speaker_dim,
            vdim=speaker_dim,
            num_heads=num_heads,
            batch_first=True  # 非常重要，让输入/输出的批次维度在第一位
        )

        # 2. 门控机制 (Gating Mechanism)
        # 门是一个学习到的权重，决定融合多少信息。
        # 输入是原始语音特征和注意力输出的拼接。
        self.gate_linear = nn.Linear(speech_dim * 2, speech_dim)
        
        # 3. 层归一化 (Layer Normalization)
        # 增加训练稳定性
        self.layer_norm = nn.LayerNorm(speech_dim)

    def forward(self, speech_features: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            speech_features (torch.Tensor): 语音特征张量。
                                            形状: (Batch, SeqLen, SpeechDim)
            speaker_embedding (torch.Tensor): 说话人嵌入张量。
                                              形状: (Batch, SpeakerDim)

        Returns:
            torch.Tensor: 融合了说话人信息的语音特征。
                          形状: (Batch, SeqLen, SpeechDim)
        """
        # --- 准备输入 ---
        # speaker_embedding 的形状是 (Batch, SpeakerDim)，我们需要将其适配注意力层
        # unsqueeze(1) -> (Batch, 1, SpeakerDim)
        # 这样它就可以作为交叉注意力的 Key 和 Value
        speaker_kv = speaker_embedding.unsqueeze(1)
        
        # --- 交叉注意力计算 ---
        # Query: speech_features
        # Key: speaker_kv
        # Value: speaker_kv
        # attn_output 的形状是 (Batch, SeqLen, SpeechDim)
        attn_output, _ = self.cross_attention(
            query=speech_features,
            key=speaker_kv,
            value=speaker_kv
        )
        
        # --- 门控计算 ---
        # 拼接原始语音特征和注意力输出
        gate_input = torch.cat([speech_features, attn_output], dim=-1)
        
        # 计算门控值 g，并通过 sigmoid 压缩到 0-1 之间
        # g 的形状是 (Batch, SeqLen, SpeechDim)
        g = torch.sigmoid(self.gate_linear(gate_input))
        
        # --- 应用门控和残差连接 ---
        # 融合后的特征 = g * 注意力输出 + (1 - g) * 原始语音特征
        # 这种方式被称为 "soft-gating"，它允许模型动态地在原始特征和融合特征之间插值。
        fused_features = g * attn_output + (1 - g) * speech_features
        
        # --- 应用层归一化 ---
        # 对融合后的特征进行归一化，然后返回
        return self.layer_norm(fused_features)