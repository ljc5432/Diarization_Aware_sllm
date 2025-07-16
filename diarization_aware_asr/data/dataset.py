# diarization_aware_asr/data/dataset.py

import torch
import torchaudio
import json
import os
from torch.utils.data import Dataset
from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperFeatureExtractor, AutoTokenizer

class DiarizationAwareDataset(Dataset):
    """
    用于 Diarization-Aware S-LLM 的 PyTorch 数据集。
    """
    def __init__(self, metadata_path: str, data_config: dict, model_config: dict):
        """
        初始化数据集。
        """
        self.data_config = data_config
        self.model_config = model_config
        
        # 在初始化时计算并存储项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        metadata_full_path = os.path.join(self.project_root, metadata_path)
        print(f"正在从 {metadata_full_path} 加载元数据...")
        with open(metadata_full_path, 'r', encoding='utf-8') as f:
            self.metadata = [json.loads(line) for line in f]
        print(f"元数据加载完成，共 {len(self.metadata)} 个样本。")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个数据样本。
        """
        sample_meta = self.metadata[idx]
        
        # 1. 加载音频
        # sample_meta['audio_path'] 是相对于项目根目录的路径
        audio_full_path = os.path.join(self.project_root, sample_meta['audio_path'])
        
        try:
            waveform, sample_rate = torchaudio.load(audio_full_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != self.data_config['sample_rate']:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.data_config['sample_rate'])
                waveform = resampler(waveform)
        except Exception as e:
            print(f"警告：加载音频文件 {audio_full_path} 失败: {e}")
            return None

        # 2. 加载说话人嵌入向量
        speaker_embeddings = {}
        for speaker_id, emb_rel_path in sample_meta['speaker_embeddings'].items():
            # emb_rel_path 也是相对于项目根目录的路径
            emb_full_path = os.path.join(self.project_root, emb_rel_path)
            speaker_embeddings[speaker_id] = torch.load(emb_full_path)

        triplets = sample_meta['triplets']
        
        return {
            "waveform": waveform,
            "triplets": triplets,
            "speaker_embeddings": speaker_embeddings
        }

class DataCollator:
    """
    数据整理器最终版。
    - 准备所有原始数据，包括文本ID、音频特征和指令。
    - 负责Padding。
    - 将最终的多模态拼接任务交给模型的forward方法。
    """
    def __init__(self, model_config: dict, tokenizer: AutoTokenizer, feature_extractor: WhisperFeatureExtractor):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.audio_chunk_token_id = self.tokenizer.convert_tokens_to_ids('<audio_chunk>')
        self.audio_time_per_feature = 0.02 # 30s / 1500 frames

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [f for f in features if f is not None]
        if not features:
            return {}

        # 1. 提取并填充音频特征
        batch_audio_features = [
            self.feature_extractor(
                f["waveform"].squeeze(0).numpy(), 
                sampling_rate=self.feature_extractor.sampling_rate, 
                return_tensors="pt"
            ).input_features.squeeze(0) for f in features
        ]
        padded_audio_features = pad_sequence(
            [feat.transpose(0, 1) for feat in batch_audio_features],
            batch_first=True, padding_value=0.0
        ).transpose(1, 2)

        # 2. 构建 Prompt 和 Label 的 Token ID 列表
        batch_prompt_ids = []
        batch_label_ids = []
        # --- 新增 ---
        batch_speaker_embeddings = [] # 用于存储每个语音片段对应的说话人嵌入
        # ------------
        eos_token = self.tokenizer.eos_token

        for sample in features:
            # --- 构建 Prompt ---
            # 这是输入给模型的指令部分
            prompt_parts = ["请根据提供的音频片段和说话人信息，转写以下对话。\n"]
            # --- 新增 ---
            current_sample_speaker_embeds = []
            # ------------
            for triplet in sample["triplets"]:
                prompt_parts.append(f"说话人: {triplet['speaker_id']}, 时间: {triplet['start']:.2f}s - {triplet['end']:.2f}s <audio_chunk>\n")
                # --- 新增 ---
                # 为每个triplet(即每个<audio_chunk>)记录其对应的说话人嵌入
                speaker_id = triplet['speaker_id']
                current_sample_speaker_embeds.append(sample['speaker_embeddings'][speaker_id])
                # ------------
            prompt_parts.append("转写结果为：")
            prompt_text = "".join(prompt_parts)
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            batch_prompt_ids.append(torch.tensor(prompt_ids))

            # --- 新增 ---
            # 将当前样本的所有说话人嵌入堆叠成一个Tensor
            if current_sample_speaker_embeds:
                batch_speaker_embeddings.append(torch.stack(current_sample_speaker_embeds))
            else: # 处理没有triplets的边缘情况
                batch_speaker_embeddings.append(torch.empty(0))
            # ------------

            # --- 构建 Label ---
            # 这是模型需要生成的目标部分
            texts_to_join = [t['text'] for t in sample["triplets"] if t['text']]
            if not texts_to_join: # 如果没有任何文本标注
                label_text = ""
            else:
                label_text = eos_token.join(texts_to_join) + eos_token # 在末尾也加上eos
            
            label_ids = self.tokenizer(label_text, add_special_tokens=False).input_ids
            batch_label_ids.append(torch.tensor(label_ids))

        # 3. 对文本ID进行Padding
        # Prompt (左填充)
        reversed_prompt_ids = [torch.flip(ids, dims=[0]) for ids in batch_prompt_ids]
        padded_prompt_ids = pad_sequence(reversed_prompt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_prompt_ids = torch.flip(padded_prompt_ids, dims=[1])

        # Label (右填充, -100)
        padded_label_ids = pad_sequence(batch_label_ids, batch_first=True, padding_value=-100)

        return {
            "prompt_ids": padded_prompt_ids,
            "label_ids": padded_label_ids,
            "audio_features": padded_audio_features,
            "triplets_list": [f['triplets'] for f in features],
            # --- 新增返回项 ---
            "speaker_embeddings_list": batch_speaker_embeddings
        }

