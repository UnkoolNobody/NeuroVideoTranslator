# Video Translator with Voice Cloning / Видео-переводчик с клонированием голоса

This program translates videos into another language while preserving the original speakers' voices. It uses open-source models for speech recognition, speaker diarization, translation, and voice cloning. The program can also work with external subtitle files to synthesize speech from them.

Программа переводит видео на другой язык с сохранением голосов оригинальных спикеров. Используются открытые модели для распознавания речи, диаризации спикеров, перевода и клонирования голоса. Программа также может работать с внешними файлами субтитров для синтеза речи из них.

## Features / Функциональные возможности

- Extracts audio from video (FFmpeg required).
- Speech recognition with timestamps (Whisper).
- Speaker diarization without API tokens (Silero VAD + Resemblyzer).
- Text translation (NLLB-200).
- Speech synthesis with voice cloning (XTTS v2).
- Intelligent mixing preserving background noise.
- Subtitle generation in original and target languages.
- Automatic voice sample extraction from video.
- Batch processing of multiple videos.
- Support for external subtitle files (SRT) to synthesize speech from them.
- Mode to generate only subtitles without synthesis.

- Извлечение аудио из видео (требуется FFmpeg).
- Распознавание речи с таймкодами (Whisper).
- Диаризация спикеров без токенов (Silero VAD + Resemblyzer).
- Перевод текста (NLLB-200).
- Синтез речи с клонированием голоса (XTTS v2).
- Интеллектуальное микширование с сохранением фонового шума.
- Генерация субтитров на исходном и целевом языках.
- Автоматическое извлечение образцов голоса из видео.
- Пакетная обработка нескольких видео.
- Поддержка внешних файлов субтитров (SRT) для синтеза речи из них.
- Режим генерации только субтитров без синтеза.

## Supported Languages / Поддерживаемые языки

The program uses language codes compatible with Whisper, NLLB, and XTTS. Below is a list of common languages and their codes for command-line use.

Программа использует коды языков, совместимые с Whisper, NLLB и XTTS. Ниже приведён список распространённых языков и их коды для использования в командной строке.

| Language / Язык       | Code / Код |
|-----------------------|------------|
| English / Английский  | en         |
| Russian / Русский     | ru         |
| German / Немецкий     | de         |
| French / Французский  | fr         |
| Spanish / Испанский   | es         |
| Italian / Итальянский | it         |
| Portuguese / Португальский | pt   |
| Polish / Польский     | pl         |
| Ukrainian / Украинский| uk         |
| Chinese / Китайский   | zh         |
| Japanese / Японский   | ja         |
| Korean / Корейский    | ko         |
| Arabic / Арабский     | ar         |
| Hindi / Хинди         | hi         |
| Turkish / Турецкий    | tr         |
| Dutch / Голландский   | nl         |
| Swedish / Шведский    | sv         |
| Danish / Датский      | da         |
| Finnish / Финский     | fi         |
| Czech / Чешский       | cs         |
| Hungarian / Венгерский| hu         |
| Romanian / Румынский  | ro         |
| Bulgarian / Болгарский| bg         |
| Greek / Греческий     | el         |
| Hebrew / Иврит        | he         |
| Indonesian / Индонезийский | id   |
| Malay / Малайский     | ms         |
| Thai / Тайский        | th         |
| Vietnamese / Вьетнамский | vi      |

If your language is not listed, try its ISO 639-1 code (two letters) or ISO 639-3 code (three letters). The program will attempt to map it automatically.

Если ваш язык не указан, попробуйте его код ISO 639-1 (две буквы) или ISO 639-3 (три буквы). Программа попытается сопоставить его автоматически.

## Command Line Arguments / Параметры командной строки
usage: video_translator.py [-h]/ 
[--target-lang TARGET_LANG]/
[--source-lang SOURCE_LANG]/
[--speaker ID PATH]/
[--num-speakers NUM_SPEAKERS]/
[--output OUTPUT]/
[--no-subtitles]/
[--no-noise]/
[--whisper-model {tiny,base,small,medium,large-v3}]/
[--nllb-model NLLB_MODEL]/
[--subtitle SUBTITLE]/
[--generate-subs-only]/
[videos ...]


| Argument / Параметр | Description / Описание |
|---------------------|------------------------|
| `videos` | Video files to process. / Видеофайлы для обработки. |
| `--target-lang`, `-t` | **Required.** Target language code (e.g., ru, en). / **Обязательный.** Код целевого языка (например, ru, en). |
| `--source-lang`, `-s` | Source language code. If omitted, detected automatically. / Код исходного языка. Если не указан, определяется автоматически. |
| `--speaker` | Provide voice samples for speakers. Can be used multiple times. Format: `SPEAKER_ID PATH`. / Предоставить образцы голоса для спикеров. Можно использовать несколько раз. Формат: `SPEAKER_ID PATH`. |
| `--num-speakers` | Number of speakers (if known). / Количество спикеров (если известно). |
| `--output`, `-o` | Output video file name (for single video). / Имя выходного видеофайла (для одного видео). |
| `--no-subtitles` | Do not generate subtitle files. / Не генерировать файлы субтитров. |
| `--no-noise` | Do not preserve background noise (simpler mixing). / Не сохранять фоновый шум (упрощённое микширование). |
| `--whisper-model` | Whisper model size: tiny, base, small, medium, large-v3 (default: large-v3). / Размер модели Whisper. |
| `--nllb-model` | NLLB translation model (default: facebook/nllb-200-distilled-600M). / Модель перевода NLLB. |
| `--subtitle`, `-sub` | Path to an external SRT subtitle file. The program will use its text for synthesis, skipping recognition and translation. / Путь к внешнему файлу субтитров SRT. Программа использует его текст для синтеза, пропуская распознавание и перевод. |
| `--generate-subs-only` | Only generate original subtitles from the video and exit (no synthesis, no mixing). / Только создать оригинальные субтитры из видео и завершиться (без синтеза и микширования). |

## Examples / Примеры использования

**1. Basic translation and dubbing / Базовый перевод и озвучка**
python video_translator.py video.mp4 --target-lang ru

**2. Specify source language and number of speakers / Указать исходный язык и количество спикеров**
python video_translator.py video.mp4 --target-lang en --source-lang ru --num-speakers 2

**3. Provide custom voice samples / Предоставить собственные образцы голоса**
python video_translator.py video.mp4 --target-lang de --speaker SPEAKER_00 voice1.wav --speaker SPEAKER_01 voice2.wav

**4. Batch processing / Пакетная обработка**
python video_translator.py video1.mp4 video2.mp4 --target-lang fr

**5. Use external subtitles for dubbing / Использовать внешние субтитры для озвучки**
python video_translator.py video.mp4 --target-lang ru --subtitle translated.srt

**6. Generate only original subtitles (no dubbing) / Сгенерировать только оригинальные субтитры (без озвучки)**
python video_translator.py video.mp4 --target-lang ru --generate-subs-only


## Requirements / Требования

- Python 3.8 or higher.
- FFmpeg installed and accessible in PATH.
- NVIDIA GPU with CUDA support (optional, but recommended for speed).
- Python packages: torch, torchaudio, whisper, transformers, resemblyzer, librosa, soundfile, scipy, noisereduce, scikit-learn, TTS, demucs, pyloudnorm, pyrubberband (optional for better stretching).

Install dependencies with / Установить зависимости командой:
pip install torch torchaudio whisper transformers resemblyzer librosa soundfile scipy noisereduce scikit-learn TTS demucs pyloudnorm pyrubberband


## Notes / Примечания

- The first run may download several large models (Whisper, NLLB, XTTS, Demucs). Ensure sufficient disk space and a stable internet connection.
- Voice cloning works best with at least 5–10 seconds of clean reference audio per speaker.
- For videos with multiple speakers, the program attempts to automatically extract voice samples. If it fails, provide them manually via `--speaker`.
- When using external subtitles, the program expects the text to be in the target language (no translation is performed). Speaker labels in the subtitles (e.g., "SPEAKER_00: text") are used to assign voices.
- The `--generate-subs-only` mode produces two SRT files: `video_original.srt` and `video_translated.srt`. The translated one is ready for editing and later use with `--subtitle`.

- При первом запуске могут загружаться несколько больших моделей (Whisper, NLLB, XTTS, Demucs). Убедитесь в достаточном месте на диске и стабильном интернет-соединении.
- Клонирование голоса работает лучше всего с 5–10 секундами чистого референсного аудио на спикера.
- Для видео с несколькими спикерами программа пытается автоматически извлечь образцы голоса. При неудаче укажите их вручную через `--speaker`.
- При использовании внешних субтитров программа ожидает, что текст уже на целевом языке (перевод не выполняется). Метки спикеров в субтитрах (например, "SPEAKER_00: текст") используются для назначения голосов.
- Режим `--generate-subs-only` создаёт два файла SRT: `video_original.srt` и `video_translated.srt`. Переведённый файл готов к редактированию и последующему использованию с `--subtitle`.

## Troubleshooting / Устранение неполадок

- **FFmpeg not found**: Install FFmpeg and add it to your PATH.
- **Out of memory**: Reduce Whisper model size (e.g., `--whisper-model small`).
- **No voice samples extracted**: Provide samples manually with `--speaker`.
- **Poor audio quality**: Ensure background noise preservation is enabled (default). If using Demucs, try the `htdemucs_6s` model for better separation.
- **Slow processing**: Use a GPU if available. Set `--whisper-model tiny` for faster (but less accurate) transcription.

- **FFmpeg не найден**: Установите FFmpeg и добавьте его в PATH.
- **Не хватает памяти**: Уменьшите размер модели Whisper (например, `--whisper-model small`).
- **Не удалось извлечь образцы голоса**: Укажите образцы вручную через `--speaker`.
- **Плохое качество звука**: Убедитесь, что сохранение фонового шума включено (по умолчанию). При использовании Demucs попробуйте модель `htdemucs_6s` для лучшего разделения.
- **Медленная обработка**: Используйте GPU, если доступно. Установите `--whisper-model tiny` для более быстрого (но менее точного) распознавания.
