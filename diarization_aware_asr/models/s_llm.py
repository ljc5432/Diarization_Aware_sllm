# diarization_aware_asr/models/s_llm.py (完整修正版)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel
from torch.nn.utils.rnn import pad_sequence
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Dict, Any
from .components import GatedCrossAttention

class ModalityAdapter(nn.Module):
    """一个简单的MLP，用于将音频特征投影到LLM的嵌入维度。"""
    def __init__(self, audio_feature_dim, llm_hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(audio_feature_dim, llm_hidden_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(llm_hidden_dim, llm_hidden_dim)

    def forward(self, audio_features):
        return self.layer2(self.activation(self.layer1(audio_features)))

class DiarizationAwareSLLM(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        self.config = model_config

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['llm_backbone'], trust_remote_code=True)

        # 2. 加载 LLM 基座
        print(f"正在加载LLM: {self.config['llm_backbone']}")
        # 指定 trust_remote_code=True 是加载Qwen等模型所必需的
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config['llm_backbone'],
            trust_remote_code=True 
        )
        # 3. 加载语音编码器 (Whisper)
        self.speech_encoder = WhisperModel.from_pretrained(self.config['speech_encoder'])
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.config['speech_encoder'])
        
        # 4. 添加和设置Tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<audio_chunk>']})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.audio_chunk_token_id = self.tokenizer.convert_tokens_to_ids('<audio_chunk>')

        # 5. 冻结模块
        if self.config['freeze_speech_encoder']:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

        # --- 实例化新模块 ---
        # Whisper-large-v3 encoder 输出维度是 1280
        # pyannote/embedding 输出维度是 512
        self.gated_attention = GatedCrossAttention(speech_dim=1280, speaker_dim=512)

        # 6. 初始化新模块 (模态适配器)
        # Whisper-large-v3的encoder输出维度是1280
        self.adapter = ModalityAdapter(1280, self.config['adapter_output_dim'])
        
        # 7. 使用LoRA微调，设置LoRA配置
        # --- 应用LoRA ---
        if self.config.get('use_lora', False):
            print("正在应用LoRA配置...")
            lora_config = LoraConfig(
                r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=self.config['lora_target_modules'],
                lora_dropout=self.config['lora_dropout'],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            # 使用 get_peft_model 包装 LLM
            self.llm = get_peft_model(self.llm, lora_config)
            print("LoRA应用完成。可训练参数如下：")
            self.llm.print_trainable_parameters()

    def forward(self, prompt_ids: torch.Tensor, label_ids: torch.Tensor, 
                  audio_features: torch.Tensor, triplets_list: List[List[Dict[str, Any]]],
                  speaker_embeddings_list: List[torch.Tensor]):
        
        # 1. 获取文本部分的词嵌入
        word_embeddings = self.llm.get_input_embeddings()

        # 2. 准备拼接后的输入嵌入和对齐后的标签
        batch_size = prompt_ids.shape[0]
        final_inputs_embeds = []
        final_labels = []

        for i in range(batch_size):
            # --- 准备当前样本的数据 ---
            current_prompt_ids = prompt_ids[i]
            current_label_ids = label_ids[i]
            current_audio_features = audio_features[i]
            current_triplets = triplets_list[i]
            current_speaker_embeds = speaker_embeddings_list[i] # <--- 获取当前样本的说话人嵌入

            # --- 提取并投影音频特征 ---
            with torch.no_grad():
                # 注意：这里我们假设 speech_encoder.encoder 的输入不需要 padding mask
                # 对于 Whisper Encoder，通常是这样
                encoder_output = self.speech_encoder.encoder(current_audio_features.unsqueeze(0)).last_hidden_state.squeeze(0)
            
            # projected_audio_embeds = self.adapter(encoder_output)

            # --- 查找 <audio_chunk> 的位置 ---
            audio_chunk_indices_in_prompt = torch.where(current_prompt_ids == self.audio_chunk_token_id)[0]

            # --- 开始拼接 ---
            current_input_embeds_parts = []
            current_labels_parts = []
            last_text_idx = 0
            audio_time_per_feature = 0.02 # 这个值应该从config读取

            for j, chunk_idx in enumerate(audio_chunk_indices_in_prompt):
                # a. 添加文本部分 (Prompt)
                text_part_ids = current_prompt_ids[last_text_idx:chunk_idx]
                text_part_embeds = word_embeddings(text_part_ids)
                current_input_embeds_parts.append(text_part_embeds)
                current_labels_parts.append(torch.full_like(text_part_ids, -100))
                
                # b. 添加音频部分
                triplet = current_triplets[j]
                start_frame = int(triplet['start'] / audio_time_per_feature)
                end_frame = int(triplet['end'] / audio_time_per_feature)
                # 确保裁剪不越界
                if end_frame > encoder_output.shape[0]:
                    end_frame = encoder_output.shape[0]
                # audio_segment_embeds = projected_audio_embeds[start_frame:end_frame]
                # current_input_embeds_parts.append(audio_segment_embeds)
                # current_labels_parts.append(torch.full((audio_segment_embeds.shape[0],), -100, dtype=torch.long, device=prompt_ids.device))
                
                # last_text_idx = chunk_idx + 1

                # 裁剪出原始的音频特征片段
                speech_segment_features = encoder_output[start_frame:end_frame]
                
                # 获取对应的说话人嵌入
                speaker_embed = current_speaker_embeds[j]
                
                # --- 调用 GatedCrossAttention ---
                # 输入: (Batch=1, SeqLen, Dim), (Batch=1, Dim)
                fused_segment_features = self.gated_attention(
                    speech_features=speech_segment_features.unsqueeze(0),
                    speaker_embedding=speaker_embed.unsqueeze(0)
                ).squeeze(0) # 移除Batch维度
                
                # 将融合后的特征通过适配器
                projected_audio_embeds = self.adapter(fused_segment_features)
                
                current_input_embeds_parts.append(projected_audio_embeds)
                current_labels_parts.append(torch.full((projected_audio_embeds.shape[0],), -100, dtype=torch.long, device=prompt_ids.device))
                
                last_text_idx = chunk_idx + 1

            # 添加 "转写结果为：" 这部分文本
            final_prompt_part_ids = current_prompt_ids[last_text_idx:]
            final_prompt_part_embeds = word_embeddings(final_prompt_part_ids)
            current_input_embeds_parts.append(final_prompt_part_embeds)
            current_labels_parts.append(torch.full_like(final_prompt_part_ids, -100))

            # --- 拼接输入和标签 ---
            inputs_embeds = torch.cat(current_input_embeds_parts, dim=0)
            labels_prompt_part = torch.cat(current_labels_parts, dim=0)
            
            valid_labels = current_label_ids[current_label_ids != -100]
            
            final_inputs_embeds.append(torch.cat([inputs_embeds, word_embeddings(valid_labels)], dim=0))
            final_labels.append(torch.cat([labels_prompt_part, valid_labels], dim=0))

        # 3. 对最终的序列进行Padding
        padded_inputs_embeds = pad_sequence(final_inputs_embeds, batch_first=True, padding_value=0.0)
        padded_final_labels = pad_sequence(final_labels, batch_first=True, padding_value=-100)

        # 4. 创建 Attention Mask
        attention_mask = torch.ones(padded_inputs_embeds.shape[:2], dtype=torch.long, device=padded_inputs_embeds.device)
        # 找到padding的位置 (检查嵌入向量的第一个维度是否全为0)
        padding_positions = (padded_inputs_embeds.sum(dim=-1) == 0)
        attention_mask[padding_positions] = 0

        # 5. 将最终输入送入LLM
        outputs = self.llm(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            labels=padded_final_labels,
            return_dict=True
        )
        
        return outputs