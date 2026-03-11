#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import subprocess
import soundfile as sf
import tempfile
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
import argparse
import shlex
import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model
import pyloudnorm as pyln
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.io import wavfile
import noisereduce as nr

# Модули для распознавания, диаризации и перевода
import whisper
import whisper.tokenizer
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)

# Resemblyzer для извлечения голосовых эмбеддингов (не требует токенов)
from resemblyzer import VoiceEncoder, preprocess_wav

# Silero VAD для обнаружения речи (загружается через torch.hub)
import torch.hub

# Для кластеризации
from sklearn.cluster import AgglomerativeClustering

# Ваш существующий класс для клонирования голоса
from TTS.api import TTS

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
#  Конфигурация
# ------------------------------------------------------------
@dataclass
class Config:
    """Конфигурация программы"""
    # Модели
    whisper_model: str = "large-v3"  # или "medium", "small" для слабых ПК
    nllb_model: str = "facebook/nllb-200-distilled-600M"  # или "facebook/nllb-200-1.3B"
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    # Аудио параметры
    target_sample_rate: int = 16000
    tts_sample_rate: int = 24000

    # Нормализация громкости
    target_snr_db: float = 30.0           # увеличили с 20 до 30 дБ
    normalize_lufs_target: Optional[float] = -16.0  # целевой уровень LUFS (None = использовать старую пиковую нормализацию)
    normalize_speech_before_mix: bool = True  # нормализовать каждую реплику перед вставкой
    use_master_compressor: bool = True   # применять ли компрессор к итоговому миксу
    compressor_threshold: float = -12.0  # порог компрессии (dB)
    compressor_ratio: float = 2.0        # коэффициент сжатия (2:1)

    # Параметры скорости синтеза
    tts_speed_auto: bool = True   # автоматически подбирать скорость под тайминг
    tts_default_speed: float = 1.0  # скорость по умолчанию (если авто отключено)
    tts_speed_fine_tune: bool = True  # итеративно уточнять скорость
    tts_speed_tolerance: float = 0.1   # допустимое отклонение длительности (сек)
    
    # Диаризация
    min_speech_duration: float = 1.0  # минимальная длительность сегмента речи (сек)
    max_segment_duration: float = 30.0  # максимальная длительность сегмента для TTS
    
    # Микширование
    preserve_background_noise: bool = True
    target_snr_db: float = 20.0  # целевое отношение сигнал/шум для синтезированной речи
    
    # Пути
    temp_dir: str = "temp"
    output_dir: str = "output"
    reference_samples_dir: str = "reference_samples"
    
    def __post_init__(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reference_samples_dir, exist_ok=True)


# ------------------------------------------------------------
#  Модуль для работы с видео (FFmpeg)
# ------------------------------------------------------------
class VideoProcessor:
    """Работа с видеофайлами через FFmpeg"""
    
    def __init__(self, config: Config):
        self.config = config
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Проверка наличия FFmpeg"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg не найден. Установите FFmpeg:\n"
                "Windows: скачайте с https://www.gyan.dev/ffmpeg/builds/ и добавьте в PATH\n"
                "Linux: sudo apt install ffmpeg\n"
                "MacOS: brew install ffmpeg"
            )
    
    def extract_audio(self, video_path: str) -> str:
        """
        Извлекает аудио из видео в формате WAV
        Возвращает путь к аудиофайлу
        """
        audio_path = os.path.join(self.config.temp_dir, f"{Path(video_path).stem}_audio.wav")
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',                       # без видео
            '-acodec', 'pcm_s16le',       # PCM 16-bit
            '-ar', str(self.config.target_sample_rate),
            '-ac', '1',                    # моно
            '-y',
            audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path
    
    def replace_audio(self, video_path: str, new_audio_path: str, output_path: str):
        """Заменяет аудиодорожку в видео"""
        cmd = [
            'ffmpeg', '-i', video_path,
            '-i', new_audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def get_video_duration(self, video_path: str) -> float:
        """Получает длительность видео в секундах"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())


# ------------------------------------------------------------
#  Модуль диаризации на основе Silero VAD + Resemblyzer (без токенов!)
# ------------------------------------------------------------
class SileroVAD:
    """Обёртка для Silero VAD (детекция речи)"""
    
    def __init__(self, use_onnx: bool = True):
        self.use_onnx = use_onnx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка модели Silero VAD
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=use_onnx
        )
        
        (self.get_speech_timestamps, self.save_audio, self.read_audio, 
         self.VADIterator, self.collect_chunks) = self.utils
        
        # Перемещаем модель на устройство
        if not use_onnx:
            self.model = self.model.to(self.device)
    
    def get_speech_segments(self, audio_path: str, 
                           min_speech_duration_ms: int = 1500,
                           threshold: float = 0.5) -> List[Tuple[float, float]]:
        """
        Возвращает список (start_sec, end_sec) речевых сегментов
        """
        # Загружаем аудио как тензор
        wav = self.read_audio(audio_path, sampling_rate=16000)
        
        # Получаем таймстемпы речи
        speech_timestamps = self.get_speech_timestamps(
            wav, 
            self.model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=500,
            speech_pad_ms=200
        )
        
        # Преобразуем в секунды
        segments = [(ts['start'] / 16000, ts['end'] / 16000) for ts in speech_timestamps]
        return segments


class ResemblyzerDiarization:
    """
    Диаризация на основе Resemblyzer (эмбеддинги) + Silero VAD
    Не требует токенов, совместима с новыми версиями torch
    """
    
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Инициализация VAD
        self.vad = SileroVAD(use_onnx=True)  # ONNX быстрее на CPU
        
        # Инициализация VoiceEncoder от Resemblyzer
        self.encoder = VoiceEncoder(device=self.device)
        print("Resemblyzer VoiceEncoder загружен")
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict[str, any]]:
        """
        Диаризация с автоматическим определением числа спикеров.
        Возвращает список сегментов с ключами 'start', 'end', 'speaker'.
        """
        # Шаг 1: Получаем речевые сегменты через VAD
        speech_segments = self.vad.get_speech_segments(
            audio_path,
            min_speech_duration_ms=int(self.config.min_speech_duration * 1000)
        )
        if len(speech_segments) < 1:
            return []
        
        print(f"   VAD: найдено {len(speech_segments)} речевых сегментов")
        
        # Шаг 2: Загружаем аудио для извлечения эмбеддингов
        wav, sr = librosa.load(audio_path, sr=16000)
        
        # Шаг 3: Для каждого сегмента извлекаем эмбеддинг
        embeddings = []
        valid_segments = []
        
        for start, end in speech_segments:
            start_sample = int(start * 16000)
            end_sample = int(end * 16000)
            segment_wav = wav[start_sample:end_sample]
            segment_wav = segment_wav.astype(np.float32)
            
            # Минимальная длина для эмбеддинга (~1 секунда)
            if len(segment_wav) < 16000:
                continue
            
            emb = self.encoder.embed_utterance(segment_wav)
            embeddings.append(emb)
            valid_segments.append((start, end))
        
        if len(embeddings) < 1:
            return []
        
        embeddings = np.array(embeddings)
        
        # Шаг 4: Определение числа спикеров и кластеризация
        if num_speakers is not None:
            # Если число спикеров задано явно, используем его
            n_clusters = num_speakers
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
        else:
            # Автоматическое определение числа спикеров
            n_samples = len(embeddings)
            
            # Если сегментов слишком мало, назначаем одного спикера
            if n_samples <= 3:
                labels = np.zeros(n_samples, dtype=int)
            else:
                # Проверим, не являются ли все эмбеддинги почти идентичными
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(embeddings)
                # Берём среднюю попарную схожесть (без диагонали)
                upper_tri = np.triu_indices_from(sim_matrix, k=1)
                mean_sim = np.mean(sim_matrix[upper_tri])
                
                if mean_sim > 0.9:  # порог схожести – можно настроить
                    labels = np.zeros(n_samples, dtype=int)
                else:
                    # Перебираем количество кластеров от 2 до min(5, n_samples-1)
                    best_score = -1
                    best_labels = None
                    max_k = min(5, n_samples - 1)
                    
                    for k in range(2, max_k + 1):
                        clustering = AgglomerativeClustering(
                            n_clusters=k,
                            metric='cosine',
                            linkage='average'
                        )
                        labels_tmp = clustering.fit_predict(embeddings)
                        
                        # Убедимся, что получилось больше одного кластера (могло быть, что все отнесли к одному)
                        unique_labels = np.unique(labels_tmp)
                        if len(unique_labels) < 2:
                            continue
                        
                        # Вычисляем силуэт
                        from sklearn.metrics import silhouette_score
                        score = silhouette_score(embeddings, labels_tmp, metric='cosine')
                        
                        if score > best_score:
                            best_score = score
                            best_labels = labels_tmp
                    
                    if best_labels is None:
                        # Если ни одна кластеризация не дала приемлемого результата, считаем одного спикера
                        labels = np.zeros(n_samples, dtype=int)
                    else:
                        labels = best_labels
        
        # Шаг 5: Формирование результата
        result = []
        for (start, end), label in zip(valid_segments, labels):
            result.append({
                'start': start,
                'end': end,
                'speaker': f"SPEAKER_{label:02d}"
            })
        
        # Шаг 6: Объединение соседних сегментов одного спикера
        merged = self._merge_adjacent_segments(result)
        return merged
        
    def _merge_adjacent_segments(self, segments: List[Dict], max_gap: float = 2.0) -> List[Dict]:
        """Объединяет сегменты одного спикера с небольшим разрывом"""
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            if (next_seg['speaker'] == current['speaker'] and 
                next_seg['start'] - current['end'] <= max_gap):
                current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged


# ------------------------------------------------------------
#  Модуль распознавания речи (Whisper)
# ------------------------------------------------------------
class ASREngine:
    def __init__(self, config: Config):
        self.config = config
        print(f"Загрузка Whisper ({config.whisper_model})...")
        self.model = whisper.load_model(config.whisper_model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # Создаём токенизатор для преобразования ID языка в строку
        self._tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        print("Whisper загружен")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Транскрибирует аудио и возвращает сегменты с таймкодами
        """
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        segments = []
        for seg in result['segments']:
            # Разбиваем длинные сегменты для лучшей обработки TTS
            text = seg['text'].strip()
            if not text:
                continue
            
            duration = seg['end'] - seg['start']
            if duration <= self.config.max_segment_duration:
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': text
                })
            else:
                # Разбиваем на подсегменты по предложениям
                self._split_long_segment(seg, segments)
        
        return segments
    
    def _split_long_segment(self, segment: Dict, output: List):
        """
        Разбивает длинный сегмент на части по предложениям
        """
        # Простое разбиение по знакам препинания
        import re
        sentences = re.split(r'(?<=[.!?])\s+', segment['text'])
        
        if len(sentences) <= 1:
            # Если не удалось разбить, оставляем как есть
            output.append(segment)
            return
        
        # Равномерно распределяем время между предложениями
        duration = segment['end'] - segment['start']
        chars_per_second = len(segment['text']) / duration
        
        current_time = segment['start']
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Оцениваем длительность предложения
            sent_duration = len(sentence) / chars_per_second
            sent_end = min(current_time + sent_duration, segment['end'])
            
            output.append({
                'start': current_time,
                'end': sent_end,
                'text': sentence.strip()
            })
            
            current_time = sent_end
    
    def detect_language_from_file(self, audio_path: str) -> str:
        """
        Определяет язык аудиофайла, возвращает строковый код (например, 'en', 'ru').
        """
        # Загружаем аудио и приводим к нужной длине
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # Получаем число мел-каналов из параметров модели
        n_mels = self.model.dims.n_mels
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(self.model.device)
        mel = mel.unsqueeze(0)

        # Определяем язык (detect_language возвращает кортеж (языковой токен, вероятности))
        language_token, _ = self.model.detect_language(mel)

        # Извлекаем ID токена
        if torch.is_tensor(language_token):
            lang_id = language_token.item()
        else:
            lang_id = language_token

        # Преобразуем ID в строку через decode токенизатора (получаем токен вида '<|en|>')
        token_str = self._tokenizer.decode([lang_id])
        # Извлекаем код языка, убирая обрамление '<|' и '|>'
        if token_str.startswith('<|') and token_str.endswith('|>'):
            lang_code = token_str[2:-2]
        else:
            lang_code = token_str.strip()
        return lang_code

# ------------------------------------------------------------
#  Модуль перевода (NLLB)
# ------------------------------------------------------------
class TranslationEngine:
    """Перевод текста с использованием NLLB"""
    
    # Соответствие между кодами языков Whisper/TTS и NLLB
    LANG_MAP = {
        'ru': 'rus_Cyrl',
        'en': 'eng_Latn',
        'de': 'deu_Latn',
        'fr': 'fra_Latn',
        'es': 'spa_Latn',
        'it': 'ita_Latn',
        'pt': 'por_Latn',
        'pl': 'pol_Latn',
        'uk': 'ukr_Cyrl',
        'zh': 'zho_Hans',
        'ja': 'jpn_Jpan',
        'ko': 'kor_Hang',
        'ar': 'ara_Arab',
        'hi': 'hin_Deva',
        'tr': 'tur_Latn',
        'nl': 'nld_Latn',
        'sv': 'swe_Latn',
        'da': 'dan_Latn',
        'fi': 'fin_Latn',
        'cs': 'ces_Latn',
        'hu': 'hun_Latn',
        'ro': 'ron_Latn',
        'bg': 'bul_Cyrl',
        'el': 'ell_Grek',
        'he': 'heb_Hebr',
        'id': 'ind_Latn',
        'ms': 'zsm_Latn',
        'th': 'tha_Thai',
        'vi': 'vie_Latn',
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Загрузка модели перевода {config.nllb_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.nllb_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.nllb_model).to(self.device)
        print("Модель перевода загружена")
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Переводит текст с одного языка на другой
        """
        # Конвертируем коды языков
        src_nllb = self.LANG_MAP.get(source_lang, source_lang)
        tgt_nllb = self.LANG_MAP.get(target_lang, target_lang)

        self.tokenizer.src_lang = src_nllb

        # Разбиваем длинный текст на части для перевода
        if len(text) > 500:
            return self._translate_long_text(text, src_nllb, tgt_nllb)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        # Получаем ID начального токена целевого языка через convert_tokens_to_ids
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    def _translate_long_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Переводит длинный текст по частям"""
        # Разбиваем на предложения
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        translated_parts = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 500:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    translated_parts.append(self._translate_chunk(current_chunk, src_lang, tgt_lang))
                current_chunk = sentence
        
        if current_chunk:
            translated_parts.append(self._translate_chunk(current_chunk, src_lang, tgt_lang))
        
        return " ".join(translated_parts)
    
    def _translate_chunk(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Переводит один чанк текста"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# ------------------------------------------------------------
#  Модуль клонирования голоса (XTTS v2)
# ------------------------------------------------------------
class VoiceCloningEngine:
    """Синтез речи с клонированием голоса на основе XTTS v2"""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Загрузка XTTS v2...")
        self.tts = TTS(model_name=config.tts_model, progress_bar=True).to(self.device)
        print("XTTS загружен")

    def trim_silence(self, audio_path: str, output_path: str = None, top_db: int = 25) -> Optional[str]:
        """Обрезает тишину в начале и конце аудиофайла (более агрессивно)."""
        audio, sr = librosa.load(audio_path, sr=None)
        intervals = librosa.effects.split(audio, top_db=top_db)
        if len(intervals) == 0:
            return audio_path
        start, end = intervals[0][0], intervals[-1][1]
        trimmed = audio[start:end]
        if output_path is None:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_trimmed{ext}"
        sf.write(output_path, trimmed, sr)
        return output_path

    def time_stretch(self, input_path: str, output_path: str, target_duration: float) -> Optional[str]:
        """
        Растягивает/сжимает аудио до заданной длительности с высоким качеством.
        Использует pyrubberband, если доступен, иначе librosa с контролем длины.
        """
        try:
            y, sr = librosa.load(input_path, sr=None)
            current_duration = librosa.get_duration(y=y, sr=sr)
            rate = current_duration / target_duration
            rate = np.clip(rate, 0.8, 1.5)  # щадящий диапазон

            # Пытаемся использовать pyrubberband для лучшего качества
            try:
                import pyrubberband as pyrb
                y_stretched = pyrb.time_stretch(y, sr, rate)
            except ImportError:
                # Fallback на librosa
                y_stretched = librosa.effects.time_stretch(y, rate=rate)

            # Убеждаемся, что длина совпадает с target_duration с точностью до сэмпла
            target_len = int(target_duration * sr)
            if len(y_stretched) > target_len:
                y_stretched = y_stretched[:target_len]
            elif len(y_stretched) < target_len:
                # Дополняем нулями (тишиной) – это лучше, чем обрезка, но пользователь просил без тишины в конце.
                # Однако если мы растягиваем, то тишина появляется только если исходный файл был короче.
                # Мы предпочтём оставить как есть, без дополнения, но тогда длительность не совпадёт.
                # Выбираем компромисс: дополняем очень малым хвостом (макс 50 мс), чтобы избежать кликов.
                if target_len - len(y_stretched) < 0.05 * sr:
                    y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)))
                else:
                    # Если разница большая, лучше пересчитать rate точнее и повторить?
                    # Для простоты пока дополняем.
                    y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)))

            sf.write(output_path, y_stretched, sr)
            return output_path
        except Exception as e:
            print(f"Ошибка растяжения: {e}")
            return None

    def synthesize(self, text: str, reference_audio: str, language: str,
                   output_path: str, speed: float = 1.0, trim: bool = True,
                   split_sentences: bool = False) -> Optional[str]:
        """
        Синтезирует речь с клонированием голоса.
        split_sentences=False для предотвращения лишних пауз.
        """
        try:
            if not os.path.exists(reference_audio):
                raise FileNotFoundError(f"Референсное аудио не найдено: {reference_audio}")

            if len(text) > 500:
                return self._synthesize_long_text(text, reference_audio, language, output_path, speed, trim, split_sentences)

            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=language,
                file_path=output_path,
                split_sentences=split_sentences,
                speed=speed
            )

            if trim and os.path.exists(output_path):
                trimmed_path = self.trim_silence(output_path)
                if trimmed_path != output_path:
                    os.replace(trimmed_path, output_path)

            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            print(f"Ошибка синтеза: {e}")
            return None

    def _synthesize_long_text(self, text: str, reference_audio: str, language: str,
                              output_path: str, speed: float = 1.0, trim: bool = True,
                              split_sentences: bool = False) -> Optional[str]:
        """Синтезирует длинный текст по частям и объединяет."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        temp_files = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            temp_file = os.path.join(self.config.temp_dir, f"tts_part_{i:04d}.wav")
            result = self.synthesize(sentence.strip(), reference_audio, language, temp_file, speed, trim=True, split_sentences=False)
            if result:
                temp_files.append(result)

        if not temp_files:
            return None

        combined, sr = librosa.load(temp_files[0], sr=None)
        for temp_file in temp_files[1:]:
            audio, _ = librosa.load(temp_file, sr=sr)
            combined = np.concatenate([combined, audio])

        sf.write(output_path, combined, sr)
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        if trim:
            trimmed_path = self.trim_silence(output_path)
            if trimmed_path != output_path:
                os.replace(trimmed_path, output_path)
        return output_path if os.path.exists(output_path) else None

    def estimate_duration(self, text: str, language: str, ref_audio: str, speed: float = 1.0) -> Optional[float]:
        """Оценивает длительность текста при заданной скорости."""
        if len(text) > 500:
            sample_text = text[:500] + "..."
        else:
            sample_text = text
        temp_file = os.path.join(self.config.temp_dir, f"calib_{hash(text)}.wav")
        result = self.synthesize(sample_text, ref_audio, language, temp_file, speed=speed, trim=True, split_sentences=False)
        if result and os.path.exists(result):
            duration = librosa.get_duration(filename=result)
            os.remove(result)
            if len(text) > 500:
                duration = duration * (len(text) / len(sample_text))
            return duration
        return None
    
# ------------------------------------------------------------
#  Модуль интеллектуального микширования
# ------------------------------------------------------------
import pyloudnorm as pyln

class AudioMixer:
    """
    Микширование с сохранением фона и наложением речи.
    После наложения применяется мощная нормализация LUFS и лимитер.
    """

    def __init__(self, config: Config):
        self.config = config
        self.crossfade = 0.05  # 50 мс
        self.target_lufs = -16.0  # итоговый уровень громкости
        self.background_lufs = -23.0  # уровень фона перед наложением (вещательный стандарт)

    def extract_background_track_demucs_api(self, original_audio_path: str, output_background_path: str) -> str:
        """Извлекает фоновую дорожку через Demucs."""
        import torch
        import torchaudio
        from demucs import pretrained
        from demucs.apply import apply_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Загрузка модели Demucs на {device}...")
        model = pretrained.get_model('htdemucs_ft')
        model.to(device)
        model.eval()

        wav, sr = torchaudio.load(original_audio_path)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        wav = wav.unsqueeze(0).to(device)

        print("   Применение модели Demucs...")
        with torch.no_grad():
            sources = apply_model(model, wav, device=device, shifts=1)[0]
        accompaniment = sources[:3].sum(dim=0).cpu()
        torchaudio.save(output_background_path, accompaniment, sr)
        print(f"   Фоновая дорожка сохранена: {output_background_path}")
        return output_background_path

    def mix_with_background(self,
                        original_audio_path: str,
                        synthesized_segments: List[Dict],
                        output_path: str) -> str:
        """
        Микширование без Demucs и без лишней обработки:
        - фон остаётся оригинальным,
        - речевые сегменты заменяются синтезированной речью,
        - только пиковая нормализация в конце.
        """
        # Загружаем оригинальное аудио (именно оно будет основой)
        original, sr = librosa.load(original_audio_path, sr=self.config.target_sample_rate)
        result = original.copy()

        for i, seg in enumerate(synthesized_segments):
            start = seg['start']
            synth_path = seg['audio_path']

            # Загружаем синтезированную речь
            synth, synth_sr = librosa.load(synth_path, sr=None)
            if synth_sr != sr:
                synth = librosa.resample(synth, orig_sr=synth_sr, target_sr=sr)

            # Усиливаем на 30%
            synth = synth * 1.3
            peak = np.max(np.abs(synth))
            if peak > 0.95:
                synth = synth / peak * 0.95

            start_sample = int(start * sr)
            end_sample = start_sample + len(synth)

            # Если результат слишком короткий, расширяем его (добавляем тишину)
            if end_sample > len(result):
                result = np.pad(result, (0, end_sample - len(result)))

            # Плавное скрещивание на границах (50 мс)
            cf = int(0.05 * sr)  # 50 мс
            if cf > 0 and len(synth) > 2 * cf:
                fade_in = np.linspace(0, 1, cf)
                fade_out = np.linspace(1, 0, cf)

                # Левая граница
                result[start_sample:start_sample+cf] = (
                    result[start_sample:start_sample+cf] * (1 - fade_in) +
                    synth[:cf] * fade_in
                )
                # Середина – полная замена
                result[start_sample+cf:end_sample-cf] = synth[cf:-cf]
                # Правая граница
                result[end_sample-cf:end_sample] = (
                    result[end_sample-cf:end_sample] * fade_out +
                    synth[-cf:] * (1 - fade_out)
                )
            else:
                # Если сегмент слишком короткий для кроссфейда, просто заменяем
                result[start_sample:end_sample] = synth

            print(f"   Сегмент {i+1} заменён, позиция {start:.2f}с")

        # Пиковая нормализация для защиты от клиппинга
        peak = np.max(np.abs(result))
        if peak > 0.95:
            result = result / peak * 0.95
            print("   Применена пиковая нормализация (запас 5%)")

        sf.write(output_path, result, sr)
        print(f"   Микширование завершено, длина файла: {len(result)/sr:.2f} с")
        return output_path
    
# ------------------------------------------------------------
#  Модуль для автоматического извлечения образцов голоса
# ------------------------------------------------------------
class VoiceSampleExtractor:
    """Автоматическое извлечение образцов голоса из видео"""
    
    def __init__(self, config: Config, asr_engine: ASREngine):
        self.config = config
        self.asr = asr_engine 
    
    def extract_samples(self, 
                    audio_path: str, 
                    speaker_segments: List[Dict],
                    min_duration: float = 5.0,
                    max_duration: float = 15.0) -> Dict[str, str]:
        """
        Извлекает образцы голоса для каждого спикера из видео.
        Гарантированно возвращает какой-либо образец для каждого спикера (даже если очень короткий).
        """
        audio, sr = librosa.load(audio_path, sr=self.config.target_sample_rate)
        samples = {}
        
        # Группируем сегменты по спикерам
        speaker_groups = defaultdict(list)
        for seg in speaker_segments:
            speaker_groups[seg['speaker']].append((seg['start'], seg['end']))
        
        for speaker, segments in speaker_groups.items():
            # Сортируем по длительности (от больших к меньшим)
            segments.sort(key=lambda x: x[1] - x[0], reverse=True)
            
            best_sample = None
            best_duration = 0
            best_path = None
            
            # --- Попытка 1: ищем качественный сегмент достаточной длины ---
            for start, end in segments:
                duration = end - start
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Если сегмент длиннее max_duration, обрезаем до середины
                if duration > max_duration:
                    mid = len(segment_audio) // 2
                    half_len = int(max_duration * sr / 2)
                    start_mid = max(0, mid - half_len)
                    end_mid = min(len(segment_audio), mid + half_len)
                    segment_audio = segment_audio[start_mid:end_mid]
                    duration = max_duration
                
                output_path = os.path.join(
                    self.config.reference_samples_dir,
                    f"{speaker}_sample.wav"
                )
                
                # Запоминаем лучший по длительности (на случай, если качество плохое)
                if duration > best_duration:
                    best_duration = duration
                    best_sample = segment_audio.copy()
                    best_path = output_path
                
                # Проверяем качество
                if duration >= min_duration and self._check_sample_quality(segment_audio, sr):
                    sf.write(output_path, segment_audio, sr)
                    samples[speaker] = output_path
                    print(f"   Извлечён качественный образец для {speaker}: {duration:.1f} сек")
                    break
            
            if speaker in samples:
                continue
            
            # --- Попытка 2: объединение коротких сегментов ---
            print(f"   {speaker}: нет одного длинного сегмента, пробуем объединить короткие...")
            segments_sorted = sorted(segments, key=lambda x: x[0])
            
            combined = []
            current_len = 0
            target_len = int(min_duration * sr)
            
            for start, end in segments_sorted:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Мягкая проверка качества (меньший порог RMS)
                if self._check_sample_quality(segment_audio, sr, min_rms=0.005):
                    combined.append(segment_audio)
                    current_len += len(segment_audio)
                    if current_len >= target_len:
                        break
            
            if combined:
                combined_audio = np.concatenate(combined)
                if len(combined_audio) > int(max_duration * sr):
                    combined_audio = combined_audio[:int(max_duration * sr)]
                
                output_path = os.path.join(
                    self.config.reference_samples_dir,
                    f"{speaker}_sample_combined.wav"
                )
                sf.write(output_path, combined_audio, sr)
                
                if self._check_sample_quality(combined_audio, sr):
                    samples[speaker] = output_path
                    print(f"   Извлечён объединённый образец для {speaker}: {len(combined_audio)/sr:.1f} сек")
                    continue
                else:
                    # Если объединённый не прошёл проверку, но он лучше, чем существующий best_sample?
                    if len(combined_audio) > best_duration * sr:
                        best_duration = len(combined_audio) / sr
                        best_sample = combined_audio
                        best_path = output_path
            
            # --- Попытка 3: используем лучший из найденных (даже если короткий или плохой) ---
            if best_sample is not None:
                sf.write(best_path, best_sample, sr)
                samples[speaker] = best_path
                print(f"Используется запасной образец для {speaker} длительностью {best_duration:.1f} сек (качество может быть низким)")
            else:
                # Если вообще нет сегментов (такого быть не должно), создаём заглушку?
                # Создаём крошечный образец из первого сегмента (если есть)
                if segments:
                    start, end = segments[0]
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    segment_audio = audio[start_sample:end_sample]
                    output_path = os.path.join(
                        self.config.reference_samples_dir,
                        f"{speaker}_sample_fallback.wav"
                    )
                    sf.write(output_path, segment_audio, sr)
                    samples[speaker] = output_path
                    print(f"Используется первый попавшийся сегмент для {speaker} длительностью {end-start:.1f} сек (качество не гарантируется)")
                else:
                    # Если сегментов действительно нет, пропускаем спикера
                    print(f"Нет сегментов для спикера {speaker}, он будет пропущен")
        
        return samples

    def _check_sample_quality(self, audio: np.ndarray, sr: int, min_rms: float = 0.01) -> bool:
        """
        Проверка качества образца.
        min_rms можно понизить для коротких или тихих фрагментов.
        """
        # Проверяем, что не слишком тихо
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < min_rms:
            return False
        
        # Проверяем, что нет длительных пауз
        energy = librosa.feature.rms(y=audio, frame_length=int(0.025*sr), hop_length=int(0.010*sr))[0]
        silence_ratio = np.sum(energy < np.percentile(energy, 20)) / len(energy)
        if silence_ratio > 0.5:
            return False
        
        return True

# ------------------------------------------------------------
#  Модуль генерации субтитров
# ------------------------------------------------------------
class SubtitleGenerator:
    """Генерация субтитров в формате SRT"""
    
    @staticmethod
    def segments_to_srt(segments: List[Dict], output_path: str, prefix: str = ""):
        """
        Конвертирует сегменты в SRT формат
        """
        def time_to_srt(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            millis = int((secs - int(secs)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start = time_to_srt(seg['start'])
                end = time_to_srt(seg['end'])
                
                text = seg.get('translated_text', seg.get('text', ''))
                if prefix and 'speaker' in seg:
                    text = f"{prefix}{seg['speaker']}: {text}"
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")


# ------------------------------------------------------------
#  Основной класс программы - УЛУЧШЕННЫЙ
# ------------------------------------------------------------
class VideoTranslator:
    """
    Главный класс, объединяющий все компоненты
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.video_processor = VideoProcessor(self.config)
        self.diarization = ResemblyzerDiarization(self.config)
        self.asr = ASREngine(self.config)
        self.translator = TranslationEngine(self.config)
        self.tts = VoiceCloningEngine(self.config)
        self.mixer = AudioMixer(self.config)
        self.sample_extractor = VoiceSampleExtractor(self.config, self.asr)
        self.subtitle_gen = SubtitleGenerator()

    def process_video(self,
                     video_path: str,
                     target_lang: str,
                     source_lang: Optional[str] = None,
                     speaker_samples: Optional[Dict[str, str]] = None,
                     num_speakers: Optional[int] = None,
                     output_path: Optional[str] = None,
                     generate_subtitles: bool = True,
                     subtitle_path: Optional[str] = None,
                     generate_subs_only: bool = False) -> Dict[str, Any]:
        """
        Основной метод обработки видео
        """
        start_time = time.time()
        results = {
            'video_path': video_path,
            'target_lang': target_lang,
            'source_lang': source_lang,
            'status': 'processing',
            'steps': {}
        }
        
        # Инициализация переменных, которые могут остаться неопределёнными в некоторых ветках
        asr_segments = None
        translated_segments = None
        speaker_segments = None
        
        try:
            # Шаг 1: Извлечение аудио
            print("\n1. Извлечение аудио из видео...")
            audio_path = self.video_processor.extract_audio(video_path)
            results['steps']['audio_extraction'] = {'status': 'success', 'path': audio_path}

            # Если передан файл субтитров, загружаем сегменты из него и пропускаем ASR/перевод
            if subtitle_path is not None:
                print("\n2. Загрузка субтитров...")
                aligned_segments = self._load_subtitles(subtitle_path)
                print(f"   Загружено {len(aligned_segments)} сегментов")
                asr_segments = None
                translated_segments = aligned_segments  # для синтеза используем текст из субтитров
                source_lang = None
                # Создаём speaker_segments для извлечения образцов
                speaker_segments = [{'start': s['start'], 'end': s['end'], 'speaker': s['speaker']} for s in aligned_segments]
                results['steps']['diarization'] = {'status': 'skipped', 'reason': 'using external subtitles'}
                results['steps']['asr'] = {'status': 'skipped'}
                results['steps']['translation'] = {'status': 'skipped'}
                
                # --- Добавлено: извлечение образцов голоса ---
                print("\n5. Автоматическое извлечение образцов голоса...")
                speaker_samples = self.sample_extractor.extract_samples(
                    audio_path, 
                    speaker_segments
                )
                results['steps']['sample_extraction'] = {
                    'status': 'success',
                    'samples': speaker_samples
                }
                
                # Проверяем, что есть образцы для всех спикеров
                speakers_in_video = set(seg['speaker'] for seg in aligned_segments)
                missing = speakers_in_video - set(speaker_samples.keys())
                if missing:
                    print(f"Предупреждение: нет образцов голоса для спикеров: {missing}")
                    print("Сегменты этих спикеров будут пропущены.")
                    aligned_segments = [seg for seg in aligned_segments if seg['speaker'] in speaker_samples]
                    translated_segments = aligned_segments
                    if not aligned_segments:
                        raise ValueError("Не осталось сегментов для обработки после фильтрации")
            else:
                # Обычный поток: диаризация, ASR, перевод
                # Шаг 2: Диаризация
                print("\n2. Диаризация (определение спикеров)...")
                speaker_segments = self.diarization.diarize(audio_path, num_speakers)
                results['steps']['diarization'] = {
                    'status': 'success',
                    'num_segments': len(speaker_segments),
                    'speakers': list(set(s['speaker'] for s in speaker_segments))
                }
                
                # Шаг 3: Распознавание речи
                print("\n3. Распознавание речи...")
                asr_segments = self.asr.transcribe(audio_path, source_lang)
                results['steps']['asr'] = {
                    'status': 'success',
                    'num_segments': len(asr_segments)
                }
                
                # Шаг 4: Совмещение ASR с диаризацией
                print("\n4. Совмещение текста со спикерами...")
                aligned_segments = self._align_asr_with_speakers(asr_segments, speaker_segments)
                
                # Если режим только субтитров – генерируем оригинальные субтитры и завершаем
                if generate_subs_only:
                    print("\n5. Генерация оригинальных субтитров...")
                    base_name = Path(video_path).stem
                    orig_srt = os.path.join(self.config.output_dir, f"{base_name}_original.srt")
                    self.subtitle_gen.segments_to_srt(asr_segments, orig_srt)
                    print(f"   Оригинальные субтитры сохранены: {orig_srt}")
                    results['steps']['subtitles'] = {'status': 'success', 'original': orig_srt}
                    results['status'] = 'success'
                    results['elapsed_time'] = time.time() - start_time
                    results['output_video'] = None
                    return results
                
                # Шаг 5: Извлечение образцов голоса (если не предоставлены)
                if not speaker_samples:
                    print("\n5. Автоматическое извлечение образцов голоса...")
                    speaker_samples = self.sample_extractor.extract_samples(
                        audio_path, 
                        speaker_segments
                    )
                    results['steps']['sample_extraction'] = {
                        'status': 'success',
                        'samples': speaker_samples
                    }
                
                # Проверяем, что есть образцы для всех спикеров
                speakers_in_video = set(seg['speaker'] for seg in aligned_segments)
                missing = speakers_in_video - set(speaker_samples.keys())
                if missing:
                    print(f"  Предупреждение: нет образцов голоса для спикеров: {missing}")
                    print("   Сегменты этих спикеров будут пропущены.")
                    aligned_segments = [seg for seg in aligned_segments if seg['speaker'] in speaker_samples]
                    if not aligned_segments:
                        raise ValueError("Не осталось сегментов для обработки после фильтрации")
                
                # Шаг 6: Перевод текста
                print("\n6. Перевод текста...")
                if source_lang is None and asr_segments:
                    source_lang = self.asr.detect_language_from_file(audio_path)
                    print(f"   Определён язык оригинала: {source_lang}")
                
                translated_segments = []
                for seg in aligned_segments:
                    if seg['text'].strip():
                        translated_text = self.translator.translate(seg['text'], source_lang, target_lang)
                        seg['translated_text'] = translated_text
                        translated_segments.append(seg)
                
                results['steps']['translation'] = {
                    'status': 'success',
                    'source_lang': source_lang,
                    'num_segments': len(translated_segments)
                }
            
            # Шаг 7: Синтез речи для каждого сегмента
            print("\n7. Синтез речи с клонированием голоса и точной подгонкой длительности...")
            synthesized_segments = []

            for i, seg in enumerate(translated_segments):
                speaker = seg['speaker']
                ref_audio = speaker_samples[speaker]
                original_duration = seg['end'] - seg['start']
                text_to_speak = seg.get('translated_text', seg['text'])  # ← универсальный доступ

                # Оценка длительности при скорости 1.0
                est_duration = self.tts.estimate_duration(text_to_speak, target_lang, ref_audio, speed=1.0)
                if self.config.tts_speed_auto and est_duration and est_duration > 0:
                    desired_speed = est_duration / original_duration
                    desired_speed = max(0.7, min(2.0, desired_speed))
                else:
                    desired_speed = 1.0

                max_attempts = 7
                tolerance = 0.1
                current_speed = desired_speed
                best_audio = None
                best_duration = None
                best_delta = float('inf')

                for attempt in range(max_attempts):
                    temp_audio = os.path.join(self.config.temp_dir, f"synth_{i:06d}_a{attempt}.wav")
                    result = self.tts.synthesize(
                        text_to_speak,
                        ref_audio,
                        target_lang,
                        temp_audio,
                        speed=current_speed,
                        trim=True,
                        split_sentences=False
                    )
                    if not result or not os.path.exists(result):
                        continue

                    actual_duration = librosa.get_duration(filename=result)
                    delta = abs(actual_duration - original_duration)

                    if delta < best_delta:
                        if best_audio is not None and best_audio != result:
                            try:
                                os.remove(best_audio)
                            except:
                                pass
                        best_audio = result
                        best_duration = actual_duration
                        best_delta = delta

                    if delta <= tolerance:
                        print(f"   Сегмент {i+1}: точное попадание с {attempt+1} попытки")
                        break
                    else:
                        correction = original_duration / actual_duration
                        current_speed *= correction
                        current_speed = max(0.7, min(2.0, current_speed))
                        if abs(correction - 1.0) < 0.01:
                            break
                        if result != best_audio:
                            try:
                                os.remove(result)
                            except:
                                pass

                if best_audio:
                    # Применяем растяжение только если ошибка > 0.2 с
                    if abs(best_duration - original_duration) > 0.2:
                        stretched_path = os.path.join(self.config.temp_dir, f"synth_{i:06d}_stretched.wav")
                        stretched = self.tts.time_stretch(best_audio, stretched_path, original_duration)
                        if stretched:
                            os.remove(best_audio)
                            best_audio = stretched
                            best_duration = original_duration
                            print(f"   Сегмент {i+1}: применено мягкое растяжение")
                        else:
                            print(f"   Сегмент {i+1}: растяжение не удалось, ошибка {best_delta:.2f}с")

                    synthesized_segments.append({
                        'start': seg['start'],
                        'end_original': seg['end'],
                        'end': seg['start'] + best_duration,
                        'text': seg['text'],
                        'translated_text': seg.get('translated_text', seg['text']),
                        'speaker': speaker,
                        'audio_path': best_audio
                    })
                    print(f"   Сегмент {i+1}/{len(translated_segments)}: "
                        f"ориг.длит.={original_duration:.2f}с, факт.={best_duration:.2f}с, "
                        f"скорость={current_speed:.2f}, ошибка={best_delta:.2f}с")
                else:
                    print(f"   Не удалось синтезировать сегмент {i+1}, пропускаем...")

            # Шаг 8: Микширование
            print("\n8. Микширование аудио с сохранением фонового шума...")
            mixed_audio = os.path.join(self.config.temp_dir, "mixed_audio.wav")
            mixed_path = self.mixer.mix_with_background(
                audio_path,
                synthesized_segments,
                mixed_audio
            )
            
            # Проверка, что mixed_path не None
            if mixed_path is None:
                print("   Предупреждение: mix_with_background вернул None, используем путь по умолчанию")
                mixed_path = mixed_audio
            
            if not os.path.exists(mixed_path):
                raise FileNotFoundError(f"Файл со смешанным аудио не найден: {mixed_path}")

            # Шаг 9: Генерация субтитров
            if generate_subtitles:
                print("\n9. Генерация субтитров...")
                base_name = Path(video_path).stem
                
                # Оригинальные субтитры (если есть)
                if asr_segments:
                    orig_srt = os.path.join(self.config.output_dir, f"{base_name}_original.srt")
                    self.subtitle_gen.segments_to_srt(asr_segments, orig_srt)
                    print(f"   Оригинальные субтитры сохранены: {orig_srt}")
                
                # Переведённые субтитры (всегда есть в этом режиме)
                if translated_segments:
                    trans_srt = os.path.join(self.config.output_dir, f"{base_name}_translated.srt")
                    self.subtitle_gen.segments_to_srt(translated_segments, trans_srt)
                    print(f"   Переведённые субтитры сохранены: {trans_srt}")
                
                results['steps']['subtitles'] = {'status': 'success'}
            
            # Шаг 10: Создание итогового видео
            print("\n10. Создание итогового видео...")
            if output_path is None:
                output_path = os.path.join(
                    self.config.output_dir,
                    f"{Path(video_path).stem}_{target_lang}.mp4"
                )
            
            # Проверка, что все пути корректны
            if not isinstance(video_path, (str, bytes, os.PathLike)):
                raise ValueError(f"video_path имеет неверный тип: {type(video_path)}")
            if not isinstance(mixed_path, (str, bytes, os.PathLike)):
                raise ValueError(f"mixed_path имеет неверный тип: {type(mixed_path)}")
            if not isinstance(output_path, (str, bytes, os.PathLike)):
                raise ValueError(f"output_path имеет неверный тип: {type(output_path)}")
            
            if not os.path.exists(mixed_path):
                raise FileNotFoundError(f"Файл со смешанным аудио не найден: {mixed_path}")
            
            self.video_processor.replace_audio(video_path, mixed_path, output_path)
            results['output_video'] = output_path
            results['steps']['video_creation'] = {'status': 'success', 'path': output_path}
            
            # Итог
            elapsed = time.time() - start_time
            results['status'] = 'success'
            results['elapsed_time'] = elapsed

            print(f"\n{'='*60}")
            print(f"ОБРАБОТКА ЗАВЕРШЕНА ЗА {elapsed:.1f} СЕКУНД")
            print(f"Результат: {output_path}")
            print('='*60)

            return results

        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            print(f"\n  Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return results
        
    def _load_subtitles(self, srt_path: str) -> List[Dict]:
        """
        Загружает субтитры из SRT-файла.
        Возвращает список словарей с ключами: start, end, text, speaker.
        """
        import re
        segments = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Регулярное выражение для парсинга SRT
        pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\Z)', re.DOTALL)
        
        def time_to_seconds(t):
            h, m, s = t.replace(',', '.').split(':')
            return int(h)*3600 + int(m)*60 + float(s)
        
        for match in pattern.finditer(content):
            start = time_to_seconds(match.group(2))
            end = time_to_seconds(match.group(3))
            text = match.group(4).strip().replace('\n', ' ')
            
            # Пытаемся извлечь спикера (если текст начинается с SPEAKER_XX:)
            speaker = 'SPEAKER_00'
            speaker_match = re.match(r'(SPEAKER_\d+):\s*(.*)', text)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = speaker_match.group(2)
            
            segments.append({
                'start': start,
                'end': end,
                'text': text,
                'speaker': speaker
            })
        
        return segments
    
    def _align_asr_with_speakers(self, asr_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
        """
        Совмещает сегменты ASR с сегментами диаризации
        """
        aligned = []
        
        for asr in asr_segments:
            asr_start = asr['start']
            asr_end = asr['end']
            
            # Находим спикера с максимальным перекрытием
            best_speaker = None
            max_overlap = 0.0
            
            for spk in speaker_segments:
                overlap_start = max(asr_start, spk['start'])
                overlap_end = min(asr_end, spk['end'])
                
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = spk['speaker']
            
            # Если перекрытия нет, используем ближайшего по времени
            if best_speaker is None:
                min_distance = float('inf')
                for spk in speaker_segments:
                    distance = min(abs(asr_start - spk['end']), abs(asr_end - spk['start']))
                    if distance < min_distance:
                        min_distance = distance
                        best_speaker = spk['speaker']
            
            aligned.append({
                'start': asr_start,
                'end': asr_end,
                'text': asr['text'],
                'speaker': best_speaker or 'SPEAKER_00'
            })
        
        return aligned
    
    def _split_text_into_parts(self, text: str, max_parts: int = 3) -> List[str]:
        """
        Пытается разбить текст на логические части (по знакам препинания).
        Возвращает список частей (не более max_parts).
        """
        import re
        # Сначала пробуем разбить по точке с запятой, двоеточию, тире
        sentences = re.split(r'(?<=[.!?;:])\s+', text)
        if len(sentences) > 1:
            # Если предложений больше max_parts, группируем их
            if len(sentences) <= max_parts:
                return sentences
            else:
                # Объединяем предложения в группы примерно равного размера
                avg_per_part = len(sentences) // max_parts
                parts = []
                for i in range(0, len(sentences), avg_per_part):
                    part = ' '.join(sentences[i:i+avg_per_part])
                    parts.append(part)
                    if len(parts) >= max_parts:
                        break
                return parts
        else:
            # Если не удалось разбить, возвращаем исходный текст как одну часть
            return [text]
    
    def batch_process(self, video_paths: List[str], target_lang: str, **kwargs) -> List[Dict]:
        """
        Пакетная обработка нескольких видео
        """
        results = []
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n{'='*60}")
            print(f"ОБРАБОТКА ФАЙЛА {i}/{len(video_paths)}: {Path(video_path).name}")
            print('='*60)
            
            result = self.process_video(video_path, target_lang, **kwargs)
            results.append(result)
        
        return results

# ------------------------------------------------------------
#  Интерфейс командной строки
# ------------------------------------------------------------
def main_cli():
    """Точка входа для командной строки"""
    
    parser = argparse.ArgumentParser(
        description="Видео-переводчик с клонированием голоса",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обработать одно видео
  python video_translator.py video.mp4 --target-lang ru
  
  # Указать исходный язык и количество спикеров
  python video_translator.py video.mp4 --target-lang en --source-lang ru --num-speakers 2
  
  # Предоставить образцы голосов
  python video_translator.py video.mp4 --target-lang de --speaker SPEAKER_00 voice1.wav --speaker SPEAKER_01 voice2.wav
  
  # Пакетная обработка
  python video_translator.py video1.mp4 video2.mp4 --target-lang fr
        """
    )
    
    parser.add_argument("videos", nargs="*", help="Видеофайлы для обработки")
    parser.add_argument("--target-lang", "-t", required=True, help="Целевой язык")
    parser.add_argument("--source-lang", "-s", help="Исходный язык (если не указан, определяется автоматически)")
    parser.add_argument("--speaker", action="append", nargs=2, metavar=("ID", "PATH"), 
                        help="Образец голоса для спикера (можно несколько)")
    parser.add_argument("--num-speakers", type=int, help="Количество спикеров (если известно)")
    parser.add_argument("--output", "-o", help="Имя выходного видеофайла (только для одного видео)")
    parser.add_argument("--no-subtitles", action="store_true", help="Не генерировать субтитры")
    parser.add_argument("--no-noise", action="store_true", help="Не сохранять фоновый шум")
    parser.add_argument("--whisper-model", default="large-v3", 
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Модель Whisper")
    parser.add_argument("--nllb-model", default="facebook/nllb-200-distilled-600M",
                        help="Модель перевода")
    parser.add_argument("--fine-tune", action="store_true", 
                        help="Итеративно уточнять скорость для точного попадания в тайминг")
    parser.add_argument("--subtitle", "-sub", 
                        help="Путь к файлу субтитров в формате SRT (озвучить готовые субтитры вместо распознавания)")
    parser.add_argument("--generate-subs-only", action="store_true", 
                        help="Только распознать речь и создать субтитры (без синтеза и озвучки)")
    
    args = parser.parse_args()

    if not args.videos:
        parser.error("Укажите видеофайлы")
    
    if args.generate_subs_only and args.subtitle:
        print("Предупреждение: --generate-subs-only и --subtitle несовместимы. Игнорируем --generate-subs-only.")
        args.generate_subs_only = False

    # Конфигурация
    config = Config(
        whisper_model=args.whisper_model,
        nllb_model=args.nllb_model,
        preserve_background_noise=not args.no_noise
    )
    
    # Обработка образцов голосов
    speaker_samples = {}
    if args.speaker:
        for spk_id, path in args.speaker:
            if not os.path.exists(path):
                print(f"Ошибка: файл {path} не найден")
                sys.exit(1)
            speaker_samples[spk_id] = path
    
    # Создаём переводчик
    translator = VideoTranslator(config)
    
    # Обработка
    if len(args.videos) == 1 and args.output:
        # Одно видео с указанным выходным файлом
        result = translator.process_video(
            video_path=args.videos[0],
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            speaker_samples=speaker_samples if speaker_samples else None,
            num_speakers=args.num_speakers,
            output_path=args.output,
            generate_subtitles=not args.no_subtitles,
            subtitle_path=args.subtitle,
            generate_subs_only=args.generate_subs_only
        )
    else:
        # Пакетная обработка
        results = translator.batch_process(
            video_paths=args.videos,
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            speaker_samples=speaker_samples if speaker_samples else None,
            num_speakers=args.num_speakers,
            generate_subtitles=not args.no_subtitles,
            subtitle_path=args.subtitle,
            generate_subs_only=args.generate_subs_only
        )
        
        # Вывод статистики
        successful = sum(1 for r in results if r['status'] == 'success')
        print(f"\n{'='*60}")
        print(f"ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ:")
        print(f"Всего: {len(results)}, Успешно: {successful}, Ошибок: {len(results)-successful}")
        print('='*60)


# ------------------------------------------------------------
#  Точка входа
# ------------------------------------------------------------
if __name__ == "__main__":
    main_cli()