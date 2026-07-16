#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Видео-переводчик с клонированием голоса (v2.1)
- Улучшенный VAD (чувствительный, находит больше речи)
- Исправление ошибки LUFS для коротких сегментов
- Подгонка длительности пересинтезом (без обрезки)
- Удаление оригинального голоса через Demucs
"""

import os, sys, warnings, subprocess, tempfile, shutil, time, argparse, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model
import pyloudnorm as pyln

import whisper
import whisper.tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

import torch.hub

# Патч transformers
import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
    transformers.pytorch_utils.isin_mps_friendly = torch.isin
    print("✓ Added isin_mps_friendly stub")

from TTS.api import TTS
warnings.filterwarnings('ignore')


# ------------------------------------------------------------
#  Конфигурация
# ------------------------------------------------------------
@dataclass
class Config:
    whisper_model: str = "medium"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    target_sample_rate: int = 16000
    tts_sample_rate: int = 24000

    # Громкость
    normalize_lufs_target: Optional[float] = -16.0
    normalize_speech_before_mix: bool = True
    use_master_compressor: bool = False

    # Параметры скорости синтеза
    tts_speed_auto: bool = True
    tts_min_speed: float = 0.5
    tts_max_speed: float = 2.0            # обычный режим
    tts_speed_tolerance: float = 0.1      # секунд
    tts_max_attempts: int = 12

    # Агрессивный режим (--aggressive-speed)
    tts_aggressive_speed: bool = False
    tts_aggressive_max_speed: float = 2.5

    # Диаризация – улучшенные параметры VAD
    min_speech_duration: float = 1.0      # было 2.5, теперь ловим короткие фразы
    max_segment_duration: float = 30.0
    vad_threshold: float = 0.4            # пониженный порог (было 0.6)

    # Микширование
    preserve_background_noise: bool = True
    crossfade_ms: int = 50

    temp_dir: str = "temp"
    output_dir: str = "output"
    reference_samples_dir: str = "reference_samples"

    def __post_init__(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reference_samples_dir, exist_ok=True)


# ------------------------------------------------------------
#  Работа с видео
# ------------------------------------------------------------
class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except:
            raise RuntimeError("FFmpeg не найден.")

    def extract_audio(self, video_path: str) -> str:
        audio_path = os.path.join(self.config.temp_dir, f"{Path(video_path).stem}_audio.wav")
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(self.config.target_sample_rate), '-ac', '1', '-y', audio_path
        ], check=True, capture_output=True)
        return audio_path

    def replace_audio(self, video_path: str, new_audio_path: str, output_path: str):
        subprocess.run([
            'ffmpeg', '-i', video_path, '-i', new_audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', '-y', output_path
        ], check=True, capture_output=True)

    def get_video_duration(self, video_path: str) -> float:
        res = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ], capture_output=True, text=True, check=True)
        return float(res.stdout.strip())


# ------------------------------------------------------------
#  Silero VAD (улучшенная чувствительность)
# ------------------------------------------------------------
class SileroVAD:
    def __init__(self, use_onnx: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=use_onnx
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.utils
        if not use_onnx:
            self.model = self.model.to(self.device)

    def get_speech_segments(self, audio_path: str,
                            min_speech_duration_ms: int = 1000,   # 1 секунда
                            threshold: float = 0.4) -> List[Tuple[float, float]]:
        wav = self.read_audio(audio_path, sampling_rate=16000)
        ts = self.get_speech_timestamps(wav, self.model, sampling_rate=16000,
                                        threshold=threshold,
                                        min_speech_duration_ms=min_speech_duration_ms,
                                        min_silence_duration_ms=300,  # меньше пауза = больше сегментов
                                        speech_pad_ms=100)
        return [(t['start'] / 16000, t['end'] / 16000) for t in ts]


# ------------------------------------------------------------
#  Диаризация
# ------------------------------------------------------------
class ResemblyzerDiarization:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vad = SileroVAD(use_onnx=True)
        self.encoder = VoiceEncoder(device=self.device)
        print("Resemblyzer VoiceEncoder загружен")

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        segments = self.vad.get_speech_segments(
            audio_path,
            min_speech_duration_ms=int(self.config.min_speech_duration * 1000),
            threshold=self.config.vad_threshold
        )
        if not segments:
            return []
        print(f"   VAD: найдено {len(segments)} речевых сегментов")

        wav, sr = librosa.load(audio_path, sr=16000)
        embeddings, valid_segments = [], []
        for start, end in segments:
            s = int(start * 16000)
            e = int(end * 16000)
            seg_wav = wav[s:e].astype(np.float32)
            if len(seg_wav) < 8000:  # минимум 0.5 сек для эмбеддинга
                continue
            emb = self.encoder.embed_utterance(seg_wav)
            embeddings.append(emb)
            valid_segments.append((start, end))
        if not embeddings:
            return []

        embeddings = np.array(embeddings)
        if num_speakers is not None:
            clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
            labels = clustering.fit_predict(embeddings)
        else:
            n = len(embeddings)
            if n <= 3:
                labels = np.zeros(n, dtype=int)
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity(embeddings)
                if np.mean(sim[np.triu_indices_from(sim, k=1)]) > 0.9:
                    labels = np.zeros(n, dtype=int)
                else:
                    dist = squareform(pdist(embeddings, metric='cosine'))
                    best_score, best_labels = -1, None
                    for th in np.linspace(0.1, 0.9, 9):
                        clust = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                                        metric='cosine', linkage='average')
                        lbs = clust.fit_predict(embeddings)
                        if len(np.unique(lbs)) < 2:
                            continue
                        try:
                            sc = silhouette_score(embeddings, lbs, metric='cosine')
                        except:
                            sc = 0
                        if sc > best_score:
                            best_score, best_labels = sc, lbs
                    if best_labels is None:
                        clust = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3,
                                                        metric='cosine', linkage='average')
                        lbs = clust.fit_predict(embeddings)
                        best_labels = lbs if len(np.unique(lbs)) >= 2 else np.zeros(n, dtype=int)
                    labels = best_labels

        result = []
        for (start, end), lbl in zip(valid_segments, labels):
            result.append({'start': start, 'end': end, 'speaker': f"SPEAKER_{lbl:02d}"})
        return self._merge_adjacent(result, max_gap=0.5)   # уменьшаем разрыв

    def _merge_adjacent(self, segments: List[Dict], max_gap: float = 0.5) -> List[Dict]:
        if not segments:
            return []
        merged = [segments[0].copy()]
        for nxt in segments[1:]:
            if nxt['speaker'] == merged[-1]['speaker'] and nxt['start'] - merged[-1]['end'] <= max_gap:
                merged[-1]['end'] = nxt['end']
            else:
                merged.append(nxt.copy())
        return merged


# ------------------------------------------------------------
#  Распознавание речи
# ------------------------------------------------------------
class ASREngine:
    def __init__(self, config: Config):
        print(f"Загрузка Whisper ({config.whisper_model})...")
        self.model = whisper.load_model(config.whisper_model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self._tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        print("Whisper загружен")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> List[Dict]:
        res = self.model.transcribe(audio_path, language=language, word_timestamps=True,
                                    verbose=False, condition_on_previous_text=True,
                                    compression_ratio_threshold=2.4, logprob_threshold=-1.0,
                                    no_speech_threshold=0.6)
        segments = []
        for seg in res['segments']:
            text = seg['text'].strip()
            if not text:
                continue
            dur = seg['end'] - seg['start']
            if dur <= 30.0:
                segments.append({'start': seg['start'], 'end': seg['end'], 'text': text})
            else:
                self._split_long(seg, segments)
        return segments

    def _split_long(self, seg: Dict, out: List):
        sentences = re.split(r'(?<=[.!?])\s+', seg['text'])
        if len(sentences) <= 1:
            out.append(seg)
            return
        dur = seg['end'] - seg['start']
        cps = len(seg['text']) / dur
        t = seg['start']
        for s in sentences:
            if not s.strip():
                continue
            d = len(s) / cps
            out.append({'start': t, 'end': min(t + d, seg['end']), 'text': s.strip()})
            t += d

    def detect_language_from_file(self, audio_path: str) -> str:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        n_mels = self.model.dims.n_mels
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(self.model.device).unsqueeze(0)
        lang_token, _ = self.model.detect_language(mel)
        lang_id = lang_token.item() if torch.is_tensor(lang_token) else lang_token
        token_str = self._tokenizer.decode([lang_id])
        return token_str[2:-2] if token_str.startswith('<|') else token_str.strip()


# ------------------------------------------------------------
#  Перевод
# ------------------------------------------------------------
class TranslationEngine:
    LANG_MAP = {
        'ru': 'rus_Cyrl', 'en': 'eng_Latn', 'de': 'deu_Latn', 'fr': 'fra_Latn',
        'es': 'spa_Latn', 'it': 'ita_Latn', 'pt': 'por_Latn', 'pl': 'pol_Latn',
        'uk': 'ukr_Cyrl', 'zh': 'zho_Hans', 'ja': 'jpn_Jpan', 'ko': 'kor_Hang',
        'ar': 'ara_Arab', 'hi': 'hin_Deva', 'tr': 'tur_Latn',
    }

    def __init__(self, config: Config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Загрузка модели перевода {config.nllb_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.nllb_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.nllb_model).to(self.device)
        print("Модель перевода загружена")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src = self.LANG_MAP.get(src_lang, src_lang)
        tgt = self.LANG_MAP.get(tgt_lang, tgt_lang)
        self.tokenizer.src_lang = src
        if len(text) > 500:
            return self._translate_long(text, src, tgt)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        forced = self.tokenizer.convert_tokens_to_ids(tgt)
        out = self.model.generate(**inputs, forced_bos_token_id=forced, max_length=512, num_beams=5)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    def _translate_long(self, text: str, src: str, tgt: str) -> str:
        parts = re.split(r'(?<=[.!?])\s+', text)
        chunks, cur = [], ""
        for p in parts:
            if len(cur) + len(p) < 500:
                cur += " " + p
            else:
                if cur:
                    chunks.append(self._translate_chunk(cur, src, tgt))
                cur = p
        if cur:
            chunks.append(self._translate_chunk(cur, src, tgt))
        return " ".join(chunks)

    def _translate_chunk(self, text: str, src: str, tgt: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        forced = self.tokenizer.convert_tokens_to_ids(tgt)
        out = self.model.generate(**inputs, forced_bos_token_id=forced, max_length=512, num_beams=5)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]


# ------------------------------------------------------------
#  Синтез речи (XTTS v2) с исправлением LUFS
# ------------------------------------------------------------
class VoiceCloningEngine:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Загрузка XTTS v2...")
        self.tts = TTS(model_name=config.tts_model, progress_bar=True).to(self.device)
        print("XTTS загружен")

    def synthesize(self, text: str, ref_wav: str, lang: str, out_path: str,
                   speed: float = 1.0, trim: bool = True, split_sentences: bool = False) -> Optional[str]:
        try:
            if not os.path.exists(ref_wav):
                raise FileNotFoundError(ref_wav)
            if len(text) > 500:
                return self._synthesize_long(text, ref_wav, lang, out_path, speed, trim)
            self.tts.tts_to_file(text=text, speaker_wav=ref_wav, language=lang,
                                 file_path=out_path, split_sentences=split_sentences, speed=speed)
            if trim and os.path.exists(out_path):
                self._trim_silence(out_path)
            return out_path if os.path.exists(out_path) else None
        except Exception as e:
            print(f"Ошибка синтеза: {e}")
            return None

    def _synthesize_long(self, text: str, ref: str, lang: str, out: str, speed: float, trim: bool) -> Optional[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        tmp_files = []
        for i, s in enumerate(sentences):
            if not s.strip():
                continue
            tmp = os.path.join(self.config.temp_dir, f"long_{i:04d}.wav")
            if self.synthesize(s.strip(), ref, lang, tmp, speed, trim=True, split_sentences=False):
                tmp_files.append(tmp)
        if not tmp_files:
            return None
        combined, sr = librosa.load(tmp_files[0], sr=None)
        for f in tmp_files[1:]:
            a, _ = librosa.load(f, sr=sr)
            combined = np.concatenate([combined, a])
        sf.write(out, combined, sr)
        for f in tmp_files:
            try: os.remove(f)
            except: pass
        if trim:
            self._trim_silence(out)
        return out if os.path.exists(out) else None

    def _trim_silence(self, path: str, top_db: int = 25):
        y, sr = librosa.load(path, sr=None)
        intervals = librosa.effects.split(y, top_db=top_db)
        if len(intervals) == 0:
            return
        y = y[intervals[0][0]:intervals[-1][1]]
        sf.write(path, y, sr)

    def estimate_duration(self, text: str, lang: str, ref: str, speed: float = 1.0) -> Optional[float]:
        if len(text) <= 400:
            tmp = os.path.join(self.config.temp_dir, f"est_{hash(text)}.wav")
            if self.synthesize(text, ref, lang, tmp, speed=speed, trim=True, split_sentences=False):
                dur = librosa.get_duration(filename=tmp)
                try: os.remove(tmp)
                except: pass
                return dur
            return None
        else:
            head, tail = text[:200], text[-200:]
            tmp_h, tmp_t = os.path.join(self.config.temp_dir, "est_h.wav"), os.path.join(self.config.temp_dir, "est_t.wav")
            d_h = d_t = None
            if self.synthesize(head, ref, lang, tmp_h, speed=speed, trim=True): d_h = librosa.get_duration(filename=tmp_h)
            if self.synthesize(tail, ref, lang, tmp_t, speed=speed, trim=True): d_t = librosa.get_duration(filename=tmp_t)
            if d_h and d_t:
                total_chars = len(text)
                mid_chars = total_chars - 200 - 200
                d_mid = (d_h / 200 + d_t / 200) / 2 * mid_chars
                return d_h + d_mid + d_t
            return None

    def match_lufs(self, audio_path: str, ref_lufs: float) -> str:
        y, sr = librosa.load(audio_path, sr=None)
        # Пропускаем, если аудио короче 0.4 секунды (минимальный размер блока pyloudnorm)
        if len(y) < int(sr * 0.4):
            return audio_path
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if loudness > -70:
            gain = 10 ** ((ref_lufs - loudness) / 20)
            y *= gain
            peak = np.max(np.abs(y))
            if peak > 0.98:
                y /= peak * 0.98
            sf.write(audio_path, y, sr)
        return audio_path


# ------------------------------------------------------------
#  Микширование (Demucs + выравнивание длины)
# ------------------------------------------------------------
class AudioMixer:
    def __init__(self, config: Config):
        self.config = config

    def extract_background(self, audio_path: str, out_path: str) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Загрузка Demucs на {device}...")
        model = pretrained.get_model('htdemucs_ft')
        model.to(device).eval()
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        wav = wav.unsqueeze(0).to(device)
        print("   Применение Demucs...")
        with torch.no_grad():
            sources = apply_model(model, wav, device=device, shifts=1)[0]
        accompaniment = sources[:3].sum(dim=0).cpu()
        torchaudio.save(out_path, accompaniment, sr)
        return out_path

    def mix(self, original_audio: str, synth_segments: List[Dict], out_path: str) -> str:
        bg_path = os.path.join(self.config.temp_dir, "bg_no_vocals.wav")
        if self.config.preserve_background_noise:
            self.extract_background(original_audio, bg_path)
            bg, sr = librosa.load(bg_path, sr=self.config.target_sample_rate)
        else:
            dur = librosa.get_duration(filename=original_audio)
            bg = np.zeros(int(dur * self.config.target_sample_rate))
            sr = self.config.target_sample_rate

        speech = np.zeros_like(bg)
        crossfade_samples = int(self.config.crossfade_ms / 1000 * sr)

        for seg in synth_segments:
            synth, synth_sr = librosa.load(seg['audio_path'], sr=None)
            if synth_sr != sr:
                synth = librosa.resample(synth, orig_sr=synth_sr, target_sr=sr)
            start_s = int(seg['start'] * sr)
            end_s = start_s + len(synth)
            if end_s > len(speech):
                speech = np.pad(speech, (0, end_s - len(speech)))

            if crossfade_samples > 0 and len(synth) > 2 * crossfade_samples:
                fade_in = np.linspace(0, 1, crossfade_samples)
                fade_out = np.linspace(1, 0, crossfade_samples)
                speech[start_s:start_s+crossfade_samples] += synth[:crossfade_samples] * fade_in
                speech[start_s+crossfade_samples:end_s-crossfade_samples] += synth[crossfade_samples:-crossfade_samples]
                speech[end_s-crossfade_samples:end_s] += synth[-crossfade_samples:] * fade_out
            else:
                speech[start_s:end_s] += synth

        # Выравнивание длины
        if len(speech) > len(bg):
            speech = speech[:len(bg)]
        elif len(speech) < len(bg):
            speech = np.pad(speech, (0, len(bg) - len(speech)))

        result = bg + speech
        peak = np.max(np.abs(result))
        if peak > 0.95:
            result /= peak * 0.95
        sf.write(out_path, result, sr)
        print(f"   Микширование завершено, длительность: {len(result)/sr:.2f} с")
        return out_path


# ------------------------------------------------------------
#  Извлечение образцов голоса
# ------------------------------------------------------------
class VoiceSampleExtractor:
    def __init__(self, config: Config, asr: ASREngine):
        self.config = config
        self.asr = asr

    def extract(self, audio_path: str, speaker_segs: List[Dict],
                min_dur: float = 5.0, max_dur: float = 15.0) -> Dict[str, str]:
        audio, sr = librosa.load(audio_path, sr=self.config.target_sample_rate)
        groups = defaultdict(list)
        for s in speaker_segs:
            groups[s['speaker']].append((s['start'], s['end']))
        samples = {}
        for spk, segs in groups.items():
            segs.sort(key=lambda x: x[1]-x[0], reverse=True)
            best_seg, best_dur, best_path = None, 0, None
            for start, end in segs:
                dur = end - start
                seg_audio = audio[int(start*sr):int(end*sr)]
                if dur > max_dur:
                    mid = len(seg_audio)//2
                    half = int(max_dur*sr/2)
                    seg_audio = seg_audio[max(0,mid-half):min(len(seg_audio),mid+half)]
                    dur = max_dur
                path = os.path.join(self.config.reference_samples_dir, f"{spk}_sample.wav")
                if dur > best_dur:
                    best_dur, best_seg, best_path = dur, seg_audio.copy(), path
                if dur >= min_dur and self._quality_ok(seg_audio, sr):
                    sf.write(path, seg_audio, sr)
                    samples[spk] = path
                    break
            if spk in samples:
                continue
            # объединение коротких
            combined, cur_len = [], 0
            for start, end in sorted(segs, key=lambda x: x[0]):
                seg_audio = audio[int(start*sr):int(end*sr)]
                if self._quality_ok(seg_audio, sr, min_rms=0.005):
                    combined.append(seg_audio)
                    cur_len += len(seg_audio)
                    if cur_len >= min_dur * sr:
                        break
            if combined:
                combined_audio = np.concatenate(combined)[:int(max_dur*sr)]
                path = os.path.join(self.config.reference_samples_dir, f"{spk}_combined.wav")
                sf.write(path, combined_audio, sr)
                if self._quality_ok(combined_audio, sr):
                    samples[spk] = path
                    continue
            if best_seg is not None:
                sf.write(best_path, best_seg, sr)
                samples[spk] = best_path
                print(f"   Запасной образец для {spk} ({best_dur:.1f}с)")
        return samples

    def _quality_ok(self, audio: np.ndarray, sr: int, min_rms: float = 0.01) -> bool:
        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False
        energy = librosa.feature.rms(y=audio, frame_length=int(0.025*sr), hop_length=int(0.01*sr))[0]
        return np.sum(energy < np.percentile(energy, 20)) / len(energy) <= 0.5


# ------------------------------------------------------------
#  Субтитры
# ------------------------------------------------------------
class SubtitleGenerator:
    @staticmethod
    def to_srt(segments: List[Dict], path: str, include_speaker: bool = True):
        def fmt(sec: float) -> str:
            h, m = int(sec//3600), int((sec%3600)//60)
            s = sec % 60
            ms = int((s - int(s))*1000)
            return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"
        speakers = {s.get('speaker', '') for s in segments}
        multi = len(speakers) > 1 and include_speaker
        with open(path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                text = seg.get('translated_text', seg.get('text', ''))
                if multi and 'speaker' in seg:
                    text = f"{seg['speaker']}: {text}"
                f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{text}\n\n")


# ------------------------------------------------------------
#  Главный класс VideoTranslator
# ------------------------------------------------------------
class VideoTranslator:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.video = VideoProcessor(self.config)
        self.diarization = ResemblyzerDiarization(self.config)
        self.asr = ASREngine(self.config)
        self.translator = TranslationEngine(self.config)
        self.tts = VoiceCloningEngine(self.config)
        self.mixer = AudioMixer(self.config)
        self.samples = VoiceSampleExtractor(self.config, self.asr)
        self.subs = SubtitleGenerator()

    def process_video(self, video_path, target_lang, source_lang=None,
                      speaker_samples=None, num_speakers=None, output_path=None,
                      generate_subtitles=True, subtitle_path=None, generate_subs_only=False):
        start_time = time.time()
        res = {'video_path': video_path, 'target_lang': target_lang, 'status': 'processing'}

        print("\n1. Извлечение аудио...")
        audio_path = self.video.extract_audio(video_path)

        if subtitle_path:
            print("\n2. Загрузка субтитров...")
            aligned = self._load_subtitles(subtitle_path)
            asr_segs = None
            translated = aligned
            speaker_segs = [{'start': s['start'], 'end': s['end'], 'speaker': s['speaker']} for s in aligned]
            if not speaker_samples:
                print("\n3. Извлечение образцов...")
                speaker_samples = self.samples.extract(audio_path, speaker_segs)
            present = set(speaker_samples.keys())
            aligned = [s for s in aligned if s['speaker'] in present]
            translated = aligned
            if not aligned:
                raise RuntimeError("Нет образцов для спикеров в субтитрах")
        else:
            print("\n2. Диаризация...")
            speaker_segs = self.diarization.diarize(audio_path, num_speakers)
            print("\n3. Распознавание речи...")
            asr_segs = self.asr.transcribe(audio_path, source_lang)
            if generate_subs_only:
                base = Path(video_path).stem
                self.subs.to_srt(asr_segs, os.path.join(self.config.output_dir, f"{base}_original.srt"))
                res['status'] = 'success'
                res['elapsed_time'] = time.time() - start_time
                return res
            print("\n4. Совмещение текста и спикеров...")
            aligned = self._align(asr_segs, speaker_segs)
            print("\n5. Извлечение образцов...")
            if not speaker_samples:
                speaker_samples = self.samples.extract(audio_path, speaker_segs)
            present = set(speaker_samples.keys())
            aligned = [s for s in aligned if s['speaker'] in present]
            if not aligned:
                raise RuntimeError("Нет образцов для найденных спикеров")
            print("\n6. Перевод...")
            if source_lang is None:
                source_lang = self.asr.detect_language_from_file(audio_path)
                print(f"   Определён язык: {source_lang}")
            translated = []
            for seg in aligned:
                if seg['text'].strip():
                    seg['translated_text'] = self.translator.translate(seg['text'], source_lang, target_lang)
                    translated.append(seg)

        print("\n7. Синтез речи с точной подгонкой длительности...")
        ref_lufs = self._calc_ref_lufs(audio_path, speaker_segs)
        synth_segments = []
        tol = self.config.tts_speed_tolerance
        max_attempts = self.config.tts_max_attempts
        min_sp, max_sp = self.config.tts_min_speed, self.config.tts_max_speed
        aggressive = self.config.tts_aggressive_speed
        agg_max = self.config.tts_aggressive_max_speed

        for i, seg in enumerate(translated):
            spk = seg['speaker']
            ref = speaker_samples[spk]
            target_dur = seg['end'] - seg['start']
            text = seg.get('translated_text', seg['text'])

            # Начальная скорость
            est = self.tts.estimate_duration(text, target_lang, ref, 1.0)
            if self.config.tts_speed_auto and est and est > 0:
                speed = est / target_dur
                speed = max(min_sp, min(max_sp, speed))
            else:
                speed = 1.0

            # Хранилище всех временных файлов этой попытки
            temp_files = []
            best_audio = None
            best_shorter_audio = None
            best_dur = None
            best_delta = float('inf')
            best_shorter_dur = None
            best_shorter_delta = float('inf')

            # Основной цикл подбора скорости
            for attempt in range(max_attempts):
                tmp = os.path.join(self.config.temp_dir, f"synth_{i:06d}_{attempt}.wav")
                temp_files.append(tmp)
                if not self.tts.synthesize(text, ref, target_lang, tmp, speed=speed, trim=True):
                    continue
                dur = librosa.get_duration(filename=tmp)
                delta = abs(dur - target_dur)

                # абсолютный лучший
                if delta < best_delta:
                    best_audio = tmp
                    best_dur = dur
                    best_delta = delta

                # лучший среди коротких (длительность строго меньше целевой)
                if dur < target_dur and delta < best_shorter_delta:
                    best_shorter_audio = tmp
                    best_shorter_dur = dur
                    best_shorter_delta = delta
                elif dur < target_dur and best_shorter_audio is None:
                    best_shorter_audio = tmp
                    best_shorter_dur = dur
                    best_shorter_delta = delta

                if delta <= tol:
                    break

                if dur > 0:
                    correction = target_dur / dur
                    speed *= correction
                    speed = max(min_sp, min(max_sp, speed))
                    if abs(correction - 1.0) < 0.01:
                        break

            # Агрессивный режим (если разрешён и допуск не достигнут)
            if aggressive and best_delta > tol and best_dur:
                print(f"   Сегмент {i+1}: попытка агрессивного ускорения...")
                for attempt2 in range(4):
                    speed_agg = min(agg_max, target_dur / best_dur * speed)
                    tmp = os.path.join(self.config.temp_dir, f"synth_{i:06d}_agg{attempt2}.wav")
                    temp_files.append(tmp)
                    if self.tts.synthesize(text, ref, target_lang, tmp, speed=speed_agg, trim=True):
                        dur = librosa.get_duration(filename=tmp)
                        delta = abs(dur - target_dur)

                        if delta < best_delta:
                            best_audio = tmp
                            best_dur = dur
                            best_delta = delta
                        if dur < target_dur and delta < best_shorter_delta:
                            best_shorter_audio = tmp
                            best_shorter_dur = dur
                            best_shorter_delta = delta
                        elif dur < target_dur and best_shorter_audio is None:
                            best_shorter_audio = tmp
                            best_shorter_dur = dur
                            best_shorter_delta = delta

                        if delta <= tol:
                            break
                    else:
                        break

            # Выбор финального файла: при отсутствии точного попадания предпочитаем короткий
            final_audio = None
            if best_shorter_audio is not None and best_delta > tol:
                final_audio = best_shorter_audio
                final_dur = best_shorter_dur
                final_delta = best_shorter_delta
                print(f"   Сегмент {i+1}: выбран более короткий вариант (ошибка {final_delta:.3f}с)")
            else:
                final_audio = best_audio
                final_dur = best_dur
                final_delta = best_delta

            # Удаляем все временные файлы, кроме выбранного
            for tmp in temp_files:
                if tmp != final_audio and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except:
                        pass

            if final_audio and os.path.exists(final_audio):
                if ref_lufs is not None:
                    final_audio = self.tts.match_lufs(final_audio, ref_lufs)
                synth_segments.append({
                    'start': seg['start'],
                    'end': seg['start'] + final_dur,
                    'text': seg['text'],
                    'translated_text': seg.get('translated_text', ''),
                    'speaker': spk,
                    'audio_path': final_audio
                })
                print(f"   Сегмент {i+1}/{len(translated)}: длит.={final_dur:.2f}с (цель {target_dur:.2f}с), "
                      f"ошибка={final_delta:.3f}с")
            else:
                print(f"   Сегмент {i+1}: синтез не удался")

        print("\n8. Микширование (удаление оригинального голоса)...")
        mixed = os.path.join(self.config.temp_dir, "mixed.wav")
        self.mixer.mix(audio_path, synth_segments, mixed)

        if generate_subtitles:
            print("\n9. Генерация субтитров...")
            base = Path(video_path).stem
            if asr_segs:
                self.subs.to_srt(asr_segs, os.path.join(self.config.output_dir, f"{base}_original.srt"))
            if translated:
                self.subs.to_srt(translated, os.path.join(self.config.output_dir, f"{base}_translated.srt"))

        print("\n10. Создание финального видео...")
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, f"{Path(video_path).stem}_{target_lang}.mp4")
        self.video.replace_audio(video_path, mixed, output_path)
        res['output_video'] = output_path
        res['status'] = 'success'
        res['elapsed_time'] = time.time() - start_time
        print(f"\n{'='*60}\nГОТОВО за {res['elapsed_time']:.1f}с\nРезультат: {output_path}\n{'='*60}")
        return res

    def _calc_ref_lufs(self, audio_path, speaker_segs):
        y, sr = librosa.load(audio_path, sr=self.config.target_sample_rate)
        meter = pyln.Meter(sr)
        vals = []
        for seg in speaker_segs:
            s = int(seg['start']*sr)
            e = int(seg['end']*sr)
            if e - s < sr*0.5:
                continue
            try:
                l = meter.integrated_loudness(y[s:e])
                if l > -70: vals.append(l)
            except:
                pass
        return np.mean(vals) if vals else -16.0

    def _load_subtitles(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        pat = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\Z)', re.DOTALL)
        def t2s(t):
            h,m,s = t.replace(',','.').split(':')
            return int(h)*3600+int(m)*60+float(s)
        segs = []
        for m in pat.finditer(raw):
            start, end = t2s(m.group(2)), t2s(m.group(3))
            text = m.group(4).strip().replace('\n',' ')
            spk = 'SPEAKER_00'
            m2 = re.match(r'(SPEAKER_\d+):\s*(.*)', text)
            if m2:
                spk = m2.group(1)
                text = m2.group(2)
            segs.append({'start': start, 'end': end, 'text': text, 'speaker': spk})
        return segs

    def _align(self, asr_segs, spk_segs):
        aligned = []
        for a in asr_segs:
            best_spk, max_ov = None, 0
            for s in spk_segs:
                ov = min(a['end'], s['end']) - max(a['start'], s['start'])
                if ov > max_ov:
                    max_ov, best_spk = ov, s['speaker']
            if best_spk is None:
                best_spk = min(spk_segs, key=lambda s: min(abs(a['start']-s['end']), abs(a['end']-s['start'])))['speaker']
            aligned.append({**a, 'speaker': best_spk})
        return aligned

    def batch_process(self, videos, target_lang, **kwargs):
        results = []
        for v in videos:
            results.append(self.process_video(v, target_lang, **kwargs))
        return results


# ------------------------------------------------------------
#  CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Видео-переводчик с клонированием голоса")
    parser.add_argument("videos", nargs="*", help="Видеофайлы")
    parser.add_argument("--target-lang", "-t", required=True)
    parser.add_argument("--source-lang", "-s")
    parser.add_argument("--speaker", action="append", nargs=2, metavar=("ID","PATH"))
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--output", "-o")
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument("--no-noise", action="store_true", help="Убрать фоновый шум")
    parser.add_argument("--whisper-model", default="small", choices=["tiny","base","small","medium","large-v3"])
    parser.add_argument("--nllb-model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--subtitle", "-sub", help="SRT файл с субтитрами")
    parser.add_argument("--generate-subs-only", action="store_true")
    parser.add_argument("--aggressive-speed", action="store_true",
                        help="Разрешить экстремальное ускорение речи для предотвращения наложений")
    args = parser.parse_args()

    if not args.videos:
        parser.error("Укажите видеофайлы")
    if args.generate_subs_only and args.subtitle:
        print("Предупреждение: --generate-subs-only и --subtitle несовместимы")
        args.generate_subs_only = False

    config = Config(
        whisper_model=args.whisper_model,
        nllb_model=args.nllb_model,
        preserve_background_noise=not args.no_noise,
        tts_aggressive_speed=args.aggressive_speed
    )
    speaker_samples = {}
    if args.speaker:
        for sid, path in args.speaker:
            if not os.path.exists(path):
                sys.exit(f"Файл {path} не найден")
            speaker_samples[sid] = path

    translator = VideoTranslator(config)
    if len(args.videos) == 1 and args.output:
        translator.process_video(
            args.videos[0], args.target_lang, args.source_lang,
            speaker_samples or None, args.num_speakers, args.output,
            not args.no_subtitles, args.subtitle, args.generate_subs_only
        )
    else:
        results = translator.batch_process(
            args.videos, args.target_lang, source_lang=args.source_lang,
            speaker_samples=speaker_samples or None, num_speakers=args.num_speakers,
            generate_subtitles=not args.no_subtitles,
            subtitle_path=args.subtitle, generate_subs_only=args.generate_subs_only
        )
        ok = sum(1 for r in results if r['status'] == 'success')
        print(f"\nИтого: {len(results)} файлов, успешно {ok}")

if __name__ == "__main__":
    main()
