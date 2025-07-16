# diarization_aware_asr/data/preprocessor.py

import torch
import torchaudio
import numpy as np
from collections import defaultdict
import os
import json
from tqdm import tqdm
from pyannote.audio import Pipeline, Model
from pyannote.core import Annotation

# 从我们自己的工具包中导入
from diarization_aware_asr.utils.audio_utils import create_energy_weighted_mono

class Preprocessor:
    """
    负责处理整个数据集，从原始音频生成用于训练的元数据。
    """
    def __init__(self, config: dict):
        """
        初始化 Preprocessor.

        Args:
            config (dict): 从 data_config.yaml 加载的配置字典。
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 在初始化时计算并存储项目根目录，以供后续使用
        # __file__ -> preprocessor.py
        # dirname -> .../data/
        # dirname -> .../diarization_aware_asr/
        # dirname -> project_root
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        print("正在加载 Pyannote 模型...")
        # 1. 加载 Diarization 流水线
        # 注意: 如果你的模型需要授权，请确保已通过 huggingface-cli login 登录
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        ).to(self.device)

        # 2. 加载 Embedding 模型
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding"
        ).to(self.device)
        print(f"Pyannote 模型已加载到 {self.device}")

    def _get_speaker_embeddings(self, waveform: torch.Tensor, sample_rate: int, diarization: Annotation) -> dict:
        """
        为Diarization结果中的每个说话人提取平均嵌入向量。
        """
        speaker_embeddings = {}
        speaker_segments = defaultdict(list)
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments[speaker].append(segment)

        with torch.no_grad():
            for speaker, segments in speaker_segments.items():
                embeddings = []
                for segment in segments:
                    if (segment.end - segment.start) < self.config['min_segment_duration_s']:
                        continue
                    
                    start_frame = int(segment.start * sample_rate)
                    end_frame = int(segment.end * sample_rate)
                    
                    if end_frame > waveform.shape[1]:
                        end_frame = waveform.shape[1]
                    if start_frame >= end_frame:
                        continue
                        
                    segment_waveform = waveform[:, start_frame:end_frame]
                    if segment_waveform.shape[1] > 0:
                        segment_waveform = segment_waveform.to(self.device)
                        embedding = self.embedding_model(segment_waveform)
                        embedding = embedding.squeeze().detach().cpu().numpy()
                        embeddings.append(embedding)

                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                    norm_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-6)
                    speaker_embeddings[speaker] = torch.from_numpy(norm_embedding)
        
        return speaker_embeddings

    def process_single_audio(self, audio_path: str) -> (dict, dict):
        """
        处理单个音频文件。
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != self.config['sample_rate']:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.config['sample_rate'])
                waveform = resampler(waveform)
            sample_rate = self.config['sample_rate']

            mono_waveform = create_energy_weighted_mono(waveform)

            diarization_result = self.diarization_pipeline(
                {"waveform": mono_waveform, "sample_rate": sample_rate}
            )

            speaker_embeddings = self._get_speaker_embeddings(mono_waveform, sample_rate, diarization_result)

            triplets = []
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                if (segment.end - segment.start) < self.config['min_segment_duration_s']:
                    continue
                if speaker in speaker_embeddings:
                    triplets.append({
                        'speaker_id': speaker,
                        'start': round(segment.start, 3),
                        'end': round(segment.end, 3),
                        'text': "" # 文本转写留空，等待标注
                    })
            
            triplets.sort(key=lambda x: x['start'])

            # 使用 self.project_root 来计算相对路径
            relative_audio_path = os.path.relpath(audio_path, self.project_root)

            metadata = {
                'audio_path': relative_audio_path,
                'duration': round(mono_waveform.shape[1] / sample_rate, 3),
                'triplets': triplets,
                'speaker_embeddings': {}
            }
            
            return metadata, speaker_embeddings

        except Exception as e:
            print(f"  - 错误：处理文件 {os.path.basename(audio_path)} 时发生错误: {e}")
            return None, None

    def process_directory(self):
        """
        处理配置中指定的整个音频目录，并生成 metadata.jsonl 和嵌入文件。
        """
        # 从项目根目录构建绝对路径
        raw_audio_dir = os.path.join(self.project_root, self.config['raw_audio_dir'])
        processed_dir = os.path.join(self.project_root, self.config['processed_dir'])
        metadata_path = os.path.join(self.project_root, self.config['metadata_path'])
        embeddings_dir = os.path.join(self.project_root, self.config['speaker_embeddings_dir'])

        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)

        audio_files = [os.path.join(raw_audio_dir, f) for f in os.listdir(raw_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        
        print(f"找到 {len(audio_files)} 个音频文件，开始处理...")

        with open(metadata_path, 'w', encoding='utf-8') as f:
            for audio_path in tqdm(audio_files, desc="Processing audio files"):
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                
                metadata, speaker_embeddings = self.process_single_audio(audio_path)
                
                if metadata is None:
                    continue

                for speaker_id, embedding_tensor in speaker_embeddings.items():
                    emb_filename = f"{base_name}_{speaker_id}.pt"
                    emb_path_abs = os.path.join(embeddings_dir, emb_filename)
                    torch.save(embedding_tensor, emb_path_abs)
                    
                    # 记录相对于项目根目录的路径
                    metadata['speaker_embeddings'][speaker_id] = os.path.relpath(emb_path_abs, self.project_root)

                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        
        print(f"\n预处理完成！元数据已保存至: {metadata_path}")
        print(f"说话人嵌入向量已保存至: {embeddings_dir}")