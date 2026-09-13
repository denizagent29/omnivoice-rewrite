#!/usr/bin/env python3
"""Голосовая переписка: транскрипт → правка у LLM → синтез тем же голосом.

Три шага, каждый можно звать отдельно:

    python rewrite.py transcribe --audio story.ogg
    python rewrite.py edit --text "..." --instruction "..."
    python rewrite.py say --text "..." --ref story.ogg --out result.wav

или всё сразу:

    python rewrite.py run --audio story.ogg --instruction "перепиши как полную противоположность" --out result.wav

Голос клонирует OmniVoice (k2-fsa, 600+ языков, zero-shot по образцу 3–10 с).
Транскрибирует faster-whisper — локально и бесплатно. Правку делает LLM: по
умолчанию DeepSeek, но годится любой OpenAI-совместимый адрес, Gemini или
локальная Ollama (см. переменные ниже).

Ключи и настройки — переменными окружения, чтобы не таскать их в командной строке:

    LLM_PROVIDER   deepseek | moonshot | gemini | openai | ollama | none
    LLM_API_KEY    ключ провайдера (для ollama не нужен)
    LLM_BASE_URL   свой адрес, если провайдер не из списка
    LLM_MODEL      имя модели, если дефолт не подходит
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

SAMPLE_RATE = 24000  # OmniVoice отдаёт 24 кГц
REF_MIN_S, REF_MAX_S = 3.0, 10.0  # сладкое пятно клонирования по докам OmniVoice

PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "moonshot": ("https://api.moonshot.cn/v1", "kimi-k2-0905-preview"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "ollama": ("http://localhost:11434/v1", "llama3.1"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash"),
}

EDIT_SYSTEM = (
    "Ты редактор устной речи. Переписываешь расшифровку голосового сообщения по "
    "инструкции автора. Сохраняй язык оригинала, живую разговорную интонацию и "
    "примерную длину. Числа пиши словами, а не цифрами — так их правильнее прочтёт "
    "синтезатор речи. В ответ верни только готовый текст, без пояснений и кавычек."
)


# ---------------------------------------------------------------- утилиты

def die(msg: str, code: int = 1):
    print("ошибка: " + msg, file=sys.stderr)
    raise SystemExit(code)


def need_ffmpeg():
    if not shutil.which("ffmpeg"):
        die("нужен ffmpeg в PATH — без него не нарезать образец голоса")


def run_ffmpeg(args: list[str]):
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def read_wav(path: str | Path):
    """WAV → (numpy float32 моно, sr). numpy тянется вместе с omnivoice."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 2:
        data /= 32768.0
    elif width == 4:
        data /= 2147483648.0
    elif width == 1:
        data = (data - 128.0) / 128.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def write_wav(path: str | Path, samples, sr: int = SAMPLE_RATE):
    import numpy as np

    data = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def split_sentences(text: str, limit: int = 220) -> list[str]:
    """Режем на фразы: длинный кусок диффузионный синтез пересказывает и глотает."""
    import re

    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) + 1 > limit:
            out.append(buf)
            buf = p
        else:
            buf = (buf + " " + p).strip()
    if buf:
        out.append(buf)
    return out


# ---------------------------------------------------------------- шаг 1: транскрипт

def transcribe(audio: str, model: str = "small", lang: str | None = None,
               device: str = "auto") -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        die("нет faster-whisper — поставь: pip install faster-whisper")
    if device == "auto":
        device = "cpu"
        try:
            import torch  # noqa: F401

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
    compute = "int8" if device == "cpu" else "float16"
    m = WhisperModel(model, device=device, compute_type=compute)
    segments, info = m.transcribe(audio, language=lang, vad_filter=True)
    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    return {
        "text": " ".join(s["text"] for s in segs).strip(),
        "language": info.language,
        "segments": segs,
    }


# ---------------------------------------------------------------- шаг 2: правка

def _http_json(url: str, body: dict, headers: dict, timeout: int = 180) -> dict:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        die(f"{url} ответил HTTP {e.code}: {e.read().decode()[:300]}")
    except Exception as e:
        die(f"{url} недоступен: {type(e).__name__}: {e}")


def edit(text: str, instruction: str, provider: str | None = None,
         model: str | None = None, base_url: str | None = None,
         key: str | None = None) -> str:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "deepseek").lower()
    if provider == "none":
        return text
    default_url, default_model = PROVIDERS.get(provider, (None, None))
    base_url = base_url or os.environ.get("LLM_BASE_URL") or default_url
    model = model or os.environ.get("LLM_MODEL") or default_model
    key = key or os.environ.get("LLM_API_KEY") or ""
    if not base_url:
        die(f"неизвестный провайдер {provider}: задай LLM_BASE_URL")
    if provider != "ollama" and not key:
        die(f"нет ключа для {provider}: задай LLM_API_KEY")

    prompt = f"Инструкция автора: {instruction}\n\nРасшифровка:\n{text}"
    if provider == "gemini":
        url = f"{base_url}/models/{model}:generateContent?key={key}"
        data = _http_json(url, {
            "systemInstruction": {"parts": [{"text": EDIT_SYSTEM}]},
            "contents": [{"parts": [{"text": prompt}]}],
        }, {})
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            die("Gemini вернул не текст: " + json.dumps(data)[:300])

    url = base_url.rstrip("/") + "/chat/completions"
    data = _http_json(url, {
        "model": model,
        "messages": [{"role": "system", "content": EDIT_SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": 0.7,
    }, {"Authorization": f"Bearer {key}"} if key else {})
    try:
        return data["choices"][0]["message"]["content"].strip().strip('"')
    except (KeyError, IndexError):
        die("LLM вернул не текст: " + json.dumps(data)[:300])


# ---------------------------------------------------------------- шаг 3: синтез

def pick_ref(audio: str, segments: list[dict], workdir: Path) -> tuple[Path, str]:
    """Образец голоса 3–10 с: берём самый длинный кусок речи в этих рамках.

    OmniVoice просит короткий образец: на длинном он медленнее и клонирует хуже.
    Возвращаем и транскрипт именно этого куска — образец и его текст должны
    совпадать, иначе модель тянет интонацию не туда.
    """
    if not segments:
        candidates = [(0.0, min(REF_MAX_S, wav_duration(audio)), "")]
    else:
        candidates = [(s["start"], s["end"], s["text"]) for s in segments]
    best = None
    for start, end, text in candidates:
        if end - start < REF_MIN_S:
            continue
        end = min(end, start + REF_MAX_S)
        if best is None or (end - start) > (best[1] - best[0]):
            best = (start, end, text)
    if best is None:  # короткая запись целиком
        best = (0.0, min(REF_MAX_S, wav_duration(audio)), " ".join(s["text"] for s in segments))
    start, end, text = best
    ref = workdir / "ref.wav"
    run_ffmpeg(["-i", audio, "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                "-ac", "1", "-ar", str(SAMPLE_RATE), str(ref)])
    if not text:  # таймкодов нет — распознаём нарезанный образец
        text = transcribe(str(ref), "small", None, "cpu")["text"]
    return ref, text


_MODELS: dict = {}


def load_model(device: str = "cuda"):
    """Модель грузится один раз на процесс: с диска это десятки секунд.

    В окне иначе каждый синтез начинался бы с повторной загрузки весов.
    """
    if device not in _MODELS:
        from omnivoice import OmniVoice

        _MODELS[device] = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device)
    return _MODELS[device]


def say_omnivoice(text: str, out: Path, ref: Path | None = None, ref_text: str = "",
                  instruct: str = "", device: str = "cuda", num_step: int = 32,
                  speed: float | None = None, duration: float | None = None,
                  prompt_path: str = "", save_prompt: str = "",
                  chunk_duration: float | None = None, guidance_scale: float | None = None):
    """Синтез: клон по образцу, дизайн по описанию или авто-голос.

    Три режима одной функции — они и есть весь OmniVoice:
      ref       указан → клонирование голоса по образцу;
      instruct  указан → дизайн голоса («female, low pitch, british accent»);
      ничего    не указано → модель сама выберет голос.
    """
    try:
        import numpy as np
    except ImportError:
        die("нет numpy — поставь: pip install numpy")
    try:
        import omnivoice  # noqa: F401
    except ImportError:
        die("нет omnivoice — поставь: pip install omnivoice")

    model = load_model(device)
    kw = {"text": text, "num_step": num_step}
    if prompt_path:
        from omnivoice import VoiceClonePrompt

        kw["voice_clone_prompt"] = VoiceClonePrompt.load(prompt_path)
    elif ref:
        # Голос кодируем один раз и, если попросили, сохраняем: следующий запуск
        # обойдётся без образца — тембр тот же, возни меньше.
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref), ref_text=ref_text or None)
        kw["voice_clone_prompt"] = prompt
        if save_prompt:
            prompt.save(save_prompt)
    elif instruct:
        kw["instruct"] = instruct

    if speed is not None:
        kw["speed"] = speed
    if duration is not None:
        kw["duration"] = duration
    if guidance_scale is not None:
        kw["guidance_scale"] = guidance_scale
    if chunk_duration is not None:
        # Длинный текст модель сама режет на куски — это штатный режим на много
        # абзацев с почти постоянной памятью. Значение — целевая длина куска.
        kw["audio_chunk_duration"] = chunk_duration

    audio = model.generate(**kw)
    samples = np.asarray(audio[0], dtype=np.float32).reshape(-1)
    write_wav(out, samples)


def split_long_run(text: str, limit: int = 220) -> list[str]:
    """Разрезать слишком длинную фразу: её модель любит пересказать своими словами."""
    return split_sentences(text, limit) if len(text) > limit else [text]


def say_edge(text: str, out: Path, voice: str = "ru-RU-DmitryNeural"):
    """Запасной путь без видеокарты: бесплатный edge-tts, но голос не твой."""
    if not shutil.which("edge-tts") and not _module("edge_tts"):
        die("нет edge-tts — поставь: pip install edge-tts")
    cmd = [sys.executable, "-m", "edge_tts", "--voice", voice, "--text", text,
           "--write-media", str(out)]
    subprocess.run(cmd, check=True)


def _module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Перепиши голосовое сообщение своим же голосом")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="расшифровать запись")
    t.add_argument("--audio", required=True)
    t.add_argument("--model", default="small", help="размер faster-whisper (tiny…large-v3)")
    t.add_argument("--lang", default=None, help="язык, если автоопределение ошибается")
    t.add_argument("--asr-device", default="auto", choices=["auto", "cpu", "cuda"])
    t.add_argument("--json", default="", help="сохранить расшифровку с таймкодами в файл")

    e = sub.add_parser("edit", help="отредактировать текст у LLM")
    e.add_argument("--text", required=True)
    e.add_argument("--instruction", required=True)
    e.add_argument("--provider", default=None)
    e.add_argument("--model", default=None)

    s = sub.add_parser("say", help="озвучить текст клонированным голосом")
    s.add_argument("--text", required=True)
    s.add_argument("--ref", default="", help="запись-образец голоса (3–10 с)")
    s.add_argument("--ref-text", default="", help="расшифровка образца (пусто — распознает сам)")
    s.add_argument("--instruct", default="", help="дизайн голоса без образца, напр. 'male, low pitch'")
    s.add_argument("--voice-prompt", default="", help="сохранённый голос (.pt) вместо образца")
    s.add_argument("--save-voice", default="", help="сохранить голос из образца в .pt")
    s.add_argument("--out", default="out.wav")
    s.add_argument("--engine", default="omnivoice", choices=["omnivoice", "edge"])
    s.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps", "xpu"])
    s.add_argument("--num-step", type=int, default=32, help="шаги диффузии (16 быстрее, 32 чище)")
    s.add_argument("--speed", type=float, default=None, help=">1 быстрее, <1 медленнее")
    s.add_argument("--duration", type=float, default=None, help="жёсткая длина, секунды")
    s.add_argument("--guidance-scale", type=float, default=None)
    s.add_argument("--chunk-duration", type=float, default=None, help="целевая длина куска длинного текста")

    r = sub.add_parser("run", help="всё сразу: транскрипт → правка → синтез")
    r.add_argument("--audio", required=True, help="исходное голосовое")
    r.add_argument("--instruction", required=True, help="что сделать с текстом")
    r.add_argument("--out", default="out.wav")
    r.add_argument("--ref", default="", help="другой образец голоса (по умолчанию — сам --audio)")
    r.add_argument("--voice-prompt", default="", help="сохранённый голос (.pt) вместо образца")
    r.add_argument("--engine", default="omnivoice", choices=["omnivoice", "edge"])
    r.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps", "xpu"])
    r.add_argument("--num-step", type=int, default=32)
    r.add_argument("--speed", type=float, default=None)
    r.add_argument("--model", default="small")
    r.add_argument("--lang", default=None)
    r.add_argument("--provider", default=None)
    r.add_argument("--asr-device", default="auto", choices=["auto", "cpu", "cuda"])
    r.add_argument("--dry-run", action="store_true", help="только транскрипт и правка, без синтеза")
    r.add_argument("--keep-text", default="", help="сохранить расшифровку и правку в файл")

    args = ap.parse_args()

    if args.cmd == "transcribe":
        res = transcribe(args.audio, args.model, args.lang, args.asr_device)
        print(res["text"])
        if args.json:
            Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
            print(f"таймкоды → {args.json}", file=sys.stderr)
        return

    if args.cmd == "edit":
        print(edit(args.text, args.instruction, args.provider, args.model))
        return

    if args.cmd == "say":
        out = Path(args.out)
        if args.engine == "edge":
            say_edge(args.text, out)
            print(f"готово (edge-tts, голос не клонирован) → {out}")
            return
        need_ffmpeg()
        ref = Path(args.ref) if args.ref else None
        ref_text = args.ref_text
        if ref and not ref_text:
            ref_text = transcribe(str(ref), "small", None, "cpu")["text"]
        say_omnivoice(args.text, out, ref=ref, ref_text=ref_text, instruct=args.instruct,
                      device=args.device, num_step=args.num_step, speed=args.speed,
                      duration=args.duration, prompt_path=args.voice_prompt,
                      save_prompt=args.save_voice, chunk_duration=args.chunk_duration,
                      guidance_scale=args.guidance_scale)
        print(f"готово → {out}")
        return

    # run
    print("[1/3] расшифровка", flush=True)
    tr = transcribe(args.audio, args.model, args.lang, args.asr_device)
    print("  " + tr["text"], flush=True)
    print("[2/3] правка у LLM", flush=True)
    new_text = edit(tr["text"], args.instruction, args.provider)
    print("  " + new_text, flush=True)
    if args.keep_text:
        Path(args.keep_text).write_text(
            f"Расшифровка:\n{tr['text']}\n\nПосле правки:\n{new_text}\n", encoding="utf-8")
    if args.dry_run:
        print("dry-run: синтез пропущен")
        return
    print("[3/3] синтез клонированным голосом", flush=True)
    need_ffmpeg()
    out = Path(args.out)
    if args.engine == "edge":
        say_edge(new_text, out)
        print(f"готово (edge-tts, голос не клонирован) → {out}")
        return
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        if args.ref:
            ref = Path(args.ref)
            ref_text = transcribe(str(ref), "small", tr["language"], "cpu")["text"]
        elif args.voice_prompt:
            ref, ref_text = None, ""
        else:
            ref, ref_text = pick_ref(args.audio, tr["segments"], work)
        say_omnivoice(new_text, out, ref=ref, ref_text=ref_text, device=args.device,
                      num_step=args.num_step, speed=args.speed,
                      prompt_path=args.voice_prompt)
    print(f"готово → {out}")


if __name__ == "__main__":
    main()
