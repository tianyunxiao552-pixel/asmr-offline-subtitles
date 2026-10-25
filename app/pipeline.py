"""Local Japanese ASMR -> Japanese/Chinese/bilingual SRT and Chinese LRC.

The coordinator never loads CUDA models. Each audio stage has its own process;
process exit releases CTranslate2/ONNX allocations as well as PyTorch's cache.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from collections import deque
from urllib import request, error
import zipfile

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / 'app'
sys.path.insert(0, str(APP))  # Also works with isolated embedded Python's _pth.
import packages

MODELS = ROOT / '.models'
RUNTIME = ROOT / '.runtime'
STATE = ROOT / '.state'
LOGS = ROOT / '.logs'
TEMP = ROOT / '.temp'
UVR_MODEL = 'UVR-MDX-NET-Inst_HQ_3.onnx'
WHISPER_REPO = 'Systran/faster-whisper-large-v3'
SAKURA14 = 'hf.co/SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF:Q4_K_M'
# The official 7B repository has no Q4_K_M. Select its exact Q6_K filename.
SAKURA7 = ('hf.co/SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF:'
           'sakura-7b-qwen2.5-v1.0-q6k.gguf')
SYSTEM = ('你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，'
          '并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。'
          '本次输入为日语口语字幕，中文应自然简洁，保留口语语气、真实重复和省略。'
          '不要补写剧情、动作、人物关系或解释；听不清的标记必须保留。')
VAD = dict(threshold=0.3, min_speech_duration_ms=100,
           min_silence_duration_ms=1000, speech_pad_ms=400)
OPTIONS = dict(temperature=0.1, top_p=0.3, frequency_penalty=0.2,
               repeat_penalty=1.0, num_ctx=6144, num_predict=2048,
               seed=42, stop=['<|im_end|>', '<|im_start|>', '<|endoftext|>'])
LOG = logging.getLogger('asmr')
DLL_HANDLES = []
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
HTTP_LOCAL = request.build_opener(request.ProxyHandler({}))


def atomic_text(path: Path, content: str, bom: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('w', encoding='utf-8-sig' if bom else 'utf-8', newline='\n') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def setup_environment(offline=False):
    for p in (RUNTIME, MODELS, STATE, LOGS, TEMP):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        'HF_HOME': str(MODELS / 'huggingface'),
        'TORCH_HOME': str(MODELS / 'torch'),
        'AUDIO_SEPARATOR_MODEL_DIR': str(MODELS / 'uvr'),
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
        'PYTHONUTF8': '1',
    })
    (MODELS / 'uvr').mkdir(exist_ok=True)
    if os.name == 'nt':
        # Rust's Xet downloader does not read Windows' registry proxy settings.
        # Bridge the user's existing settings into this process only. Never
        # change global Windows settings or expose credentials in logs.
        proxies = request.getproxies()
        for scheme in ('http', 'https'):
            if not os.environ.get(scheme.upper() + '_PROXY') and not os.environ.get(scheme + '_proxy'):
                if proxies.get(scheme):
                    os.environ[scheme.upper() + '_PROXY'] = proxies[scheme]
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    # Prepend the CUDA/cuDNN DLLs shipped inside the pinned PyTorch wheel.
    torch_lib = Path(sys.prefix) / 'Lib' / 'site-packages' / 'torch' / 'lib'
    bin_dir = RUNTIME / 'bin'
    bin_dir.mkdir(exist_ok=True)
    dll_dirs = [torch_lib, bin_dir]
    os.environ['PATH'] = os.pathsep.join(map(str, dll_dirs)) + os.pathsep + os.environ.get('PATH', '')
    if os.name == 'nt':
        for p in dll_dirs:
            if p.is_dir():
                DLL_HANDLES.append(os.add_dll_directory(str(p)))


def configure_logging():
    stamp = time.strftime('%Y%m%d-%H%M%S') + '-' + str(os.getpid())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS / (stamp + '.log'), encoding='utf-8'),
    ])


def download(url: str, target: Path, offline=False, sha256=None):
    if target.is_file() and target.stat().st_size:
        if not sha256 or file_hash(target) == sha256:
            return
        raise RuntimeError(f'文件校验失败，请删除后重试：{target}')
    if offline:
        raise RuntimeError(f'离线模式缺少文件：{target.name}；请先联网运行 run.bat -Prepare。')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    LOG.info('下载 %s', target.name)
    req = request.Request(url, headers={'User-Agent': 'ASMR-Offline-Subtitles/1.0'})
    with request.urlopen(req, timeout=120) as response, partial.open('wb') as f:
        expected = int(response.headers.get('Content-Length', 0))
        copied = 0
        last_report = time.monotonic()
        while chunk := response.read(4 * 1024 * 1024):
            f.write(chunk)
            copied += len(chunk)
            if time.monotonic() - last_report > 8:
                LOG.info('已下载 %.1f / %.1f MiB', copied / 2**20, expected / 2**20)
                last_report = time.monotonic()
    if not copied or (expected and copied != expected):
        raise RuntimeError(f'下载不完整：{target.name}；重新运行即可重试。')
    if sha256 and file_hash(partial) != sha256:
        raise RuntimeError(f'SHA256 校验失败：{target.name}')
    os.replace(partial, target)


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Cue:
    start: int  # milliseconds
    end: int
    text: str


def srt_time(ms: int):
    h, rest = divmod(ms, 3600000)
    m, rest = divmod(rest, 60000)
    s, millis = divmod(rest, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{millis:03d}'


def parse_time(value: str):
    h, m, s, ms = map(int, re.split('[:,.]', value))
    if m >= 60 or s >= 60:
        raise ValueError('非法 SRT 时间')
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def parse_srt(path: Path):
    raw = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').strip()
    if not raw:
        return []
    cues = []
    previous = -1
    for block in re.split(r'\n\s*\n', raw):
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError(f'字幕格式不正确：{path.name}')
        match = re.fullmatch(r'(\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*'
                             r'(\d{2,}:\d{2}:\d{2}[,.]\d{3})', lines[1].strip())
        if not match:
            raise ValueError(f'字幕时间轴不正确：{path.name}')
        start, end = map(parse_time, match.groups())
        text = '\n'.join(lines[2:]).strip()
        if start < previous or end <= start or not text:
            raise ValueError(f'字幕时间或文本不正确：{path.name}')
        previous = start
        cues.append(Cue(start, end, text))
    return cues


def srt_text(cues):
    return ''.join(f'{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{c.text}\n\n'
                   for i, c in enumerate(cues, 1))


def export_subtitles(wav: Path, japanese: list[Cue], chinese: list[str]):
    if len(japanese) != len(chinese) or any(not x.strip() for x in chinese):
        raise ValueError('译文行数错误；为避免字幕错位，不输出成品。')
    zh = [Cue(c.start, c.end, t) for c, t in zip(japanese, chinese)]
    # Only the Chinese SRT is a public product and completion boundary.
    atomic_text(wav.with_suffix('.srt'), srt_text(zh), bom=True)


def has_subtitle(wav):
    path = wav.with_suffix('.srt')
    if path.exists() and path.stat().st_size > 3:
        # Validate rather than silently accept a corrupt result.
        parse_srt(path)
        return True
    marker = state_dir(wav) / 'empty.done.json'
    return bool(path.exists() and marker.exists() and read_json(marker) == source_signature(wav))


def state_dir(wav: Path):
    key = hashlib.sha256(packages.logical_name(wav, ROOT).encode('utf-8')).hexdigest()[:20]
    return STATE / key


def japanese_path(wav: Path):
    return state_dir(wav) / 'japanese.srt'


def source_signature(wav: Path):
    stat = wav.stat()
    return dict(name=wav.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def gpu_info(report=True):
    cmd = shutil.which('nvidia-smi')
    if not cmd:
        p = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'System32' / 'nvidia-smi.exe'
        cmd = str(p) if p.exists() else None
    if not cmd:
        raise RuntimeError('未找到 NVIDIA 驱动。工具固定使用 CUDA GPU，不会自动降级模型或改用 CPU 听写。')
    r = subprocess.run([cmd, '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True, creationflags=NO_WINDOW)
    rows = [line.split(',') for line in r.stdout.splitlines() if line.strip()]
    index, name, total, free = [v.strip() for v in rows[0]]
    if report:
        LOG.info('GPU %s: %s，总显存 %s MiB，空闲 %s MiB', index, name, total, free)
    return dict(index=int(index), name=name, total=int(total), free=int(free))


def wait_gpu_recovery(baseline_mib, timeout=20):
    """Wait for this stage's allocations to return, without touching other apps."""
    deadline = time.monotonic() + timeout
    while True:
        info = gpu_info(report=False)
        if info['free'] >= baseline_mib - 256:
            LOG.info('翻译进程树已退出，空闲显存 %d MiB（翻译前 %d MiB）。', info['free'], baseline_mib)
            return info
        if time.monotonic() >= deadline:
            LOG.warning('翻译进程树已退出，但空闲显存尚未回到原水平：%d / %d MiB；下一阶段将等待并检查实际可用量。',
                        info['free'], baseline_mib)
            return info
        time.sleep(0.5)


def select_sakura(total_mib):
    # The quality profile uses the full 14B; Ollama may offload layers to CPU.
    return SAKURA14


def ensure_ffmpeg():
    destination = RUNTIME / 'bin' / 'ffmpeg.exe'
    if not destination.exists():
        import imageio_ffmpeg
        source = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if not source.is_file():
            raise RuntimeError('FFmpeg 未正确安装，请删除 .runtime/dependencies.ready 后重新运行。')
        shutil.copyfile(source, destination)
    return destination


def cuda_dependencies():
    import torch  # Import before ONNX Runtime / CTranslate2 for CUDA DLL loading.
    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls(directory=str(Path(torch.__file__).parent / 'lib'))
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA 不可用。请检查 NVIDIA 驱动；不会改用轻量模型。')
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('ONNX CUDA provider 不可用，拒绝静默使用 CPU 分离。')
    return torch, ort


def empty_cuda(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chunk_ranges(frames, sample_rate, seconds=300, overlap=2):
    step = round(sample_rate * seconds)
    pad = round(sample_rate * overlap)
    for start in range(0, frames, step):
        end = min(start + step, frames)
        yield start, end, max(0, start - pad), min(frames, end + pad)


def separate_long_wav(separator, wav: Path, output: Path):
    """Bound RAM for multi-hour WAVs; trim only overlapping chunk margins.

    The model sees 2 seconds on each side of a chunk boundary. The final voice
    track retains the exact original timeline, with no concatenated overlaps.
    """
    import numpy as np
    import soundfile as sf
    combined = output / 'vocals.wav'
    with sf.SoundFile(str(wav)) as source:
        if not source.frames:
            raise ValueError('WAV 没有音频采样。')
        ranges = list(chunk_ranges(source.frames, source.samplerate))
        with contextlib.ExitStack() as stack:
            sink = None
            for i, (start, end, read_start, read_end) in enumerate(ranges, 1):
                LOG.info('分离音频块 %d / %d（每块 5 分钟，边界保留 2 秒上下文）', i, len(ranges))
                chunk_file = output / 'chunk-input.wav'
                results = []
                try:
                    source.seek(read_start)
                    block = source.read(read_end - read_start, dtype='float32', always_2d=True)
                    sf.write(str(chunk_file), block, source.samplerate, subtype='FLOAT')
                    del block
                    results = separator.separate(str(chunk_file))
                    if len(results) != 1:
                        raise RuntimeError(f'预期只生成 Vocals，实际：{results}')
                    vocal = Path(results[0])
                    if not vocal.is_absolute():
                        vocal = output / vocal
                    if not vocal.resolve().is_relative_to(output.resolve()):
                        raise RuntimeError('分离输出路径越界。')
                    data, rate = sf.read(str(vocal), dtype='float32', always_2d=True)
                    crop = round((start - read_start) * rate / source.samplerate)
                    count = round(end * rate / source.samplerate) - round(start * rate / source.samplerate)
                    data = data[crop:crop + count]
                    if data.shape[0] < count:
                        missing = count - data.shape[0]
                        if missing > 2:
                            raise RuntimeError('分离音频长度异常，拒绝改变字幕时间轴。')
                        data = np.pad(data, ((0, missing), (0, 0)))
                    if sink is None:
                        sink = stack.enter_context(sf.SoundFile(str(combined), mode='w', samplerate=rate,
                                                               channels=data.shape[1], subtype='FLOAT', format='RF64'))
                    elif sink.samplerate != rate or sink.channels != data.shape[1]:
                        raise RuntimeError('不同音频块的分离输出格式不一致。')
                    sink.write(data)
                    del data
                finally:
                    if chunk_file.exists():
                        chunk_file.unlink()
                    for filename in results:
                        path = Path(filename)
                        if not path.is_absolute():
                            path = output / path
                        if path.resolve().is_relative_to(output.resolve()) and path.is_file():
                            path.unlink()
    return combined


def separate_stage(wav: Path | None, output: Path, offline: bool):
    torch, ort = cuda_dependencies()
    from audio_separator.separator import Separator
    from audio_separator.separator.common_separator import CommonSeparator
    from audio_io import write_stem
    original_writer = CommonSeparator.write_audio_soundfile
    CommonSeparator.write_audio_soundfile = write_stem
    # Make all upstream downloads atomic, and explicitly forbid downloads offline.
    def checked_download(self, url, output_path):
        download(url, Path(output_path), offline)
    Separator.download_file_if_not_exists = checked_download
    original_session = ort.InferenceSession

    def gpu_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        if 'CUDAExecutionProvider' not in session.get_providers():
            raise RuntimeError('ONNX 实际加载 CUDA 失败。请检查 cuDNN/CUDA DLL；已阻止 CPU 回退。')
        session.disable_fallback()
        return session
    ort.InferenceSession = gpu_session
    separator = None
    try:
        separator = Separator(model_file_dir=str(MODELS / 'uvr'), output_dir=str(output),
                              output_format='WAV', output_single_stem='Vocals',
                              use_soundfile=True, sample_rate=44100,
                              mdx_params=dict(hop_length=1024, segment_size=256,
                                              overlap=0.5, batch_size=1, enable_denoise=True))
        separator.load_model(model_filename=UVR_MODEL)
        if wav is None:
            return
        vocal = separate_long_wav(separator, wav, output)
        if not vocal.is_file() or vocal.stat().st_size < 44:
            raise RuntimeError('分离未生成有效人声音频。')
        save_json(output / 'vocals.json', dict(path=str(vocal.resolve())))
    finally:
        if separator is not None:
            del separator
        CommonSeparator.write_audio_soundfile = original_writer
        ort.InferenceSession = original_session
        empty_cuda(torch)


def whisper_model_path(offline):
    from hub_download import download_files
    target = MODELS / 'whisper-large-v3'
    required = ('config.json', 'model.bin', 'tokenizer.json', 'preprocessor_config.json')
    marker = target / '.download-complete'
    if marker.exists() and all((target / f).is_file() for f in required):
        return target
    if offline:
        raise RuntimeError('离线模式缺少完整 large-v3。请先联网运行 run.bat -Prepare。')
    LOG.info('下载完整 large-v3（CTranslate2 格式，运行精度 FP16），不会使用 turbo/distil/int8。')
    download_files(WHISPER_REPO, list(required) + ['vocabulary.json'], target)
    if not all((target / f).is_file() for f in required):
        raise RuntimeError('large-v3 下载不完整。')
    save_json(marker, dict(repository=WHISPER_REPO, files=list(required)))
    return target


def whisper_transcribe_stage(vocal: Path | None, ja_path: Path | None, offline):
    torch, _ = cuda_dependencies()
    import ctranslate2
    from faster_whisper import WhisperModel
    if 'float16' not in ctranslate2.get_supported_compute_types('cuda', 0):
        raise RuntimeError('本 GPU 不支持 CTranslate2 float16；拒绝降低运行精度。')
    model = None
    segments = None
    try:
        model = WhisperModel(str(whisper_model_path(offline)), device='cuda', device_index=0,
                             compute_type='float16', local_files_only=True, num_workers=1)
        if model.model.compute_type != 'float16':
            raise RuntimeError('CTranslate2 实际运行精度不是 float16，停止处理。')
        if vocal is None:
            return
        segments, info = model.transcribe(str(vocal), language='ja', task='transcribe',
                                          beam_size=5, condition_on_previous_text=False,
                                          vad_filter=True, vad_parameters=VAD.copy(),
                                          temperature=0.0, word_timestamps=True,
                                          hallucination_silence_threshold=2.0)
        LOG.info('音频 %.1f 分钟；开始听写。', info.duration / 60)
        cues = []
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            start = max(0, round(s.start * 1000))
            end = max(start + 1, round(s.end * 1000))
            if cues and start < cues[-1].start:
                raise RuntimeError('Whisper 返回逆序时间轴，停止写入。')
            cues.append(Cue(start, end, text))
            LOG.info('[%s] %s', srt_time(start), text)
        atomic_text(ja_path, srt_text(cues), bom=True)
    finally:
        # The generator may keep references to the model, so close it first.
        if segments is not None and hasattr(segments, 'close'):
            segments.close()
        del segments
        del model
        gc.collect()
        torch.cuda.empty_cache()
        # CTranslate2 owns its CUDA allocator; process exit is the final release.


def transcribe_stage(source: Path | None, ja_path: Path | None, offline):
    import quality_asr
    checkpoint = state_dir(source) / 'asr-v2.json' if source else LOGS / 'preflight-asr.json'
    with tempfile.TemporaryDirectory(prefix='raw-asr-', dir=TEMP) as tmp:
        quality_asr.transcribe(source, ja_path, MODELS, Path(tmp), checkpoint,
                               ensure_ffmpeg(), offline, prepare_only=(source is None))


class ChildJob:
    """Windows job kills only this run's descendants if the console is closed."""
    def __init__(self):
        self.handle = None
        if os.name != 'nt':
            return
        from ctypes import wintypes
        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                        ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]
        class Counters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ('ReadOperationCount', 'WriteOperationCount',
                        'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', Counters),
                        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
        class Accounting(ctypes.Structure):
            _fields_ = [(n, ctypes.c_int64) for n in ('TotalUserTime', 'TotalKernelTime',
                        'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')] + [
                       (n, wintypes.DWORD) for n in ('TotalPageFaultCount', 'TotalProcesses',
                        'ActiveProcesses', 'TotalTerminatedProcesses')]
        self.accounting_type = Accounting
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                         ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.handle or not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def add(self, process):
        if self.handle and not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            process.kill()
            raise ctypes.WinError(ctypes.get_last_error())

    def active_processes(self):
        if not self.handle:
            return 0
        info = self.accounting_type()
        if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.ActiveProcesses

    def close(self):
        if self.handle:
            try:
                if self.active_processes():
                    if not self.kernel.TerminateJobObject(self.handle, 1):
                        raise ctypes.WinError(ctypes.get_last_error())
                    deadline = time.monotonic() + 15
                    while self.active_processes():
                        if time.monotonic() >= deadline:
                            raise RuntimeError('模型进程树未能在 15 秒内完全退出，停止队列以免显存叠加。')
                        time.sleep(0.05)
            finally:
                self.kernel.CloseHandle(self.handle)
                self.handle = None


class StageError(RuntimeError):
    """A model worker failed; stop the queue instead of repeating on all tracks."""


def run_worker(job: ChildJob, stage: str, offline, wav=None, output=None):
    args = [sys.executable, str(APP / 'pipeline.py'), 'worker-' + stage]
    if offline:
        args.append('--offline')
    if wav is not None:
        args.extend(['--input', str(wav)])
    if output is not None:
        args.extend(['--output', str(output)])
    p = subprocess.Popen(args, cwd=ROOT, creationflags=NO_WINDOW,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding='utf-8', errors='replace', bufsize=1)
    job.add(p)
    tail = deque(maxlen=8)
    try:
        with p.stdout:
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    tail.append(line)
                    LOG.info('[%s] %s', stage, line)
        code = p.wait()
    except BaseException:
        p.terminate()
        p.wait()
        raise
    if code:
        detail = '\n'.join(tail)
        raise StageError(f'{stage} 阶段失败（退出码 {code}）；已保留完成阶段。\n{detail}')


def get_ollama(offline):
    portable = RUNTIME / 'ollama' / 'ollama.exe'
    if portable.exists() and (portable.parent / '.extracted').exists():
        return portable
    system = shutil.which('ollama')
    if not system:
        candidate = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Ollama' / 'ollama.exe'
        system = str(candidate) if candidate.is_file() else None
    if system:
        return Path(system)
    if offline:
        raise RuntimeError('离线模式缺少 Ollama 程序，请先联网运行 run.bat -Prepare。')
    # Fixed official standalone release; no installer, registry or system service.
    version = 'v0.40.4'
    url = f'https://api.github.com/repos/ollama/ollama/releases/tags/{version}'
    with request.urlopen(request.Request(url, headers={'User-Agent': 'ASMR-Offline-Subtitles'}), timeout=60) as f:
        release = json.load(f)
    asset = next(x for x in release['assets'] if x['name'] == 'ollama-windows-amd64.zip')
    digest = asset.get('digest', '')
    archive = RUNTIME / 'ollama.zip'
    download(asset['browser_download_url'], archive, sha256=digest.removeprefix('sha256:') or None)
    portable.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            path = (portable.parent / name).resolve()
            if not path.is_relative_to(portable.parent.resolve()):
                raise RuntimeError('Ollama ZIP 内含非法路径。')
        z.extractall(portable.parent)
    if not portable.exists():
        raise RuntimeError('Ollama 官方压缩包中未找到 ollama.exe。')
    atomic_text(portable.parent / '.extracted', version)
    archive.unlink()
    return portable


class Ollama:
    def __init__(self, job, model, offline):
        self.model = model
        self.offline = offline
        self.process = None
        self.log = None
        self.scope = None
        self.memory_before = None
        exe = get_ollama(offline)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.base = f'http://127.0.0.1:{port}'
        env = os.environ.copy()
        env.update(OLLAMA_HOST=f'127.0.0.1:{port}', OLLAMA_MODELS=str(MODELS / 'ollama'),
                   OLLAMA_NO_CLOUD='1', OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_NUM_PARALLEL='1',
                   OLLAMA_FLASH_ATTENTION='1', OLLAMA_CONTEXT_LENGTH='8192',
                   CUDA_VISIBLE_DEVICES='0', OLLAMA_GPU_OVERHEAD=str(512 * 2**20))
        self.environment = env
        self.executable = exe
        self.job = job
        self.log = (LOGS / f'ollama-{os.getpid()}.log').open('a', encoding='utf-8')
        try:
            self.memory_before = gpu_info(report=False)['free']
            # An independent nested job owns this service AND llama-server.
            # Its lifetime ends after every track, not only after the whole queue.
            self.scope = ChildJob()
            self.process = subprocess.Popen([str(exe), 'serve'], env=env, stdout=self.log,
                                             stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            job.add(self.process)
            self.scope.add(self.process)
            for _ in range(120):
                if self.process.poll() is not None:
                    raise RuntimeError('独立 Ollama 服务启动失败，请查看 .logs/ollama 日志。')
                try:
                    self.api('/api/version', timeout=2)
                    break
                except (OSError, ValueError):
                    time.sleep(0.5)
            else:
                raise RuntimeError('Ollama 启动超时。')
            self.ensure_model()
        except BaseException:
            self.close()
            raise

    def api(self, endpoint, body=None, timeout=1800):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
        req = request.Request(self.base + endpoint, data=data,
                              headers={'Content-Type': 'application/json'})
        try:
            with HTTP_LOCAL.open(req, timeout=timeout) as r:
                result = json.load(r)
        except error.HTTPError as e:
            detail = e.read().decode('utf-8', errors='replace')[:1500]
            raise RuntimeError(f'Ollama HTTP {e.code}: {detail}') from e
        if result.get('error'):
            raise RuntimeError(str(result['error']))
        return result

    def ensure_model(self):
        names = {v['name'].casefold() for v in self.api('/api/tags')['models']}
        if self.model.casefold() in names:
            return
        if self.offline:
            raise RuntimeError(f'离线模式缺少 Sakura 模型：{self.model}')
        try:
            self.pull_model()
        except RuntimeError as exc:
            alias_missing = (self.model == SAKURA14 and 'specified tag is not available' in str(exc))
            if not alias_missing and 'redirect target not allowed' not in str(exc) and 'resolves to non-public' not in str(exc):
                raise
            LOG.warning('Ollama 在线标签不可用或代理 DNS 不兼容，改为下载同一官方 GGUF 后本地导入。')
            self.import_official_gguf()

    def pull_model(self):
        LOG.info('拉取官方模型 %s', self.model)
        req = request.Request(self.base + '/api/pull',
                              data=json.dumps(dict(model=self.model, stream=True)).encode(),
                              headers={'Content-Type': 'application/json'})
        last = 0
        try:
            response = HTTP_LOCAL.open(req, timeout=3600)
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')[:1500]
            raise RuntimeError(f'Ollama pull HTTP {exc.code}: {detail}') from exc
        with response:
            for raw in response:
                item = json.loads(raw)
                if item.get('error'):
                    raise RuntimeError(item['error'])
                if time.monotonic() - last > 5 or item.get('status') == 'success':
                    LOG.info('%s %.1f / %.1f MiB', item.get('status'), item.get('completed', 0) / 2**20,
                             item.get('total', 0) / 2**20)
                    last = time.monotonic()
        if self.model.casefold() not in {v['name'].casefold() for v in self.api('/api/tags')['models']}:
            raise RuntimeError('Ollama 拉取未完成；请重新运行。')

    def import_official_gguf(self):
        from hub_download import download_files
        if self.model == SAKURA7:
            repo = 'SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF'
            filename = 'sakura-7b-qwen2.5-v1.0-q6k.gguf'
            quant = 'Q6_K'
        elif self.model == SAKURA14:
            repo = 'SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF'
            filename = 'sakura-14b-qwen2.5-v1.0-q4km.gguf'
            quant = 'Q4_K_M'
        else:
            raise RuntimeError('不支持对非指定官方模型执行本地导入。')
        folder = MODELS / 'sakura-gguf'
        path = download_files(repo, [filename], folder)[filename]
        modelfile = folder / 'Modelfile'
        atomic_text(modelfile, 'FROM ./' + filename + '\n')
        LOG.info('官方 GGUF 已下载，正在导入 Ollama；模型和量化不变。')
        process = subprocess.Popen([str(self.executable), 'create', self.model, '-f', str(modelfile)],
                                   cwd=folder, env=self.environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                   errors='replace', creationflags=NO_WINDOW)
        self.job.add(process)
        tail = deque(maxlen=8)
        with process.stdout:
            for line in process.stdout:
                if line.strip():
                    tail.append(line.strip())
        if process.wait():
            raise RuntimeError('Ollama 本地导入失败：' + '\n'.join(tail))
        details = self.api('/api/show', dict(model=self.model)).get('details', {})
        if details.get('quantization_level') != quant:
            raise RuntimeError(f'导入模型量化校验失败：{details}')
        LOG.info('Ollama 本地导入成功，实际量化 %s。', quant)
        # Ollama now owns the content-addressed blob; don't keep a second GGUF.
        if path.resolve().is_relative_to(folder.resolve()):
            path.unlink()

    def generate(self, texts):
        # Literal official Sakura v1.0 ChatML, bypassing Ollama's own templates.
        prompt = ('<|im_start|>system\n' + SYSTEM + '<|im_end|>\n'
                  '<|im_start|>user\n将下面的日文文本翻译成中文：' + '\n'.join(texts) +
                  '<|im_end|>\n<|im_start|>assistant\n')
        return self.generate_prompt(prompt, texts)

    def generate_contextual(self, texts, context, context_translations):
        if len(context) != len(context_translations):
            raise ValueError('上文与已完成译文数量不一致')
        prompt = '<|im_start|>system\n' + SYSTEM + '<|im_end|>\n'
        if context:
            prompt += ('<|im_start|>user\n将下面的日文文本翻译成中文：' + '\n'.join(context) +
                       '<|im_end|>\n<|im_start|>assistant\n' + '\n'.join(context_translations) + '<|im_end|>\n')
        prompt += ('<|im_start|>user\n将下面的日文文本翻译成中文：' + '\n'.join(texts) +
                   '<|im_end|>\n<|im_start|>assistant\n')
        translated = self.generate_prompt(prompt, texts)
        # One semantic unit may wrap over several display lines. Joining those
        # lines keeps every word and maps only this one unit to its own interval.
        return ' '.join(translated.splitlines()) if len(texts) == 1 else translated

    def generate_focused(self, text, context, context_translations=None):
        # A model can merge an incomplete cue with the preceding context even
        # after splitting a batch down to one target. Put the five context cues
        # into a completed conversation turn, then request only the target cue.
        if context_translations is not None:
            return self.generate_contextual([text], context, context_translations)
        prompt = '<|im_start|>system\n' + SYSTEM + '<|im_end|>\n'
        if context:
            previous = self.generate(context)
            prompt += ('<|im_start|>user\n将下面的日文文本翻译成中文：' + '\n'.join(context) +
                       '<|im_end|>\n<|im_start|>assistant\n' + previous + '<|im_end|>\n')
        prompt += ('<|im_start|>user\n将下面的日文文本翻译成中文：' + text +
                   '<|im_end|>\n<|im_start|>assistant\n')
        return self.generate_prompt(prompt, [text])

    def generate_prompt(self, prompt, texts):
        options = OPTIONS.copy()
        # A short cue should not spend over a minute generating 2048 tokens.
        # The full model and requested sampling settings remain unchanged.
        options['num_predict'] = min(OPTIONS['num_predict'], max(128, 3 * sum(map(len, texts)) + 64))
        value = self.api('/api/generate', dict(model=self.model, prompt=prompt, raw=True,
                                               stream=False, keep_alive='10m', options=options))
        if value.get('done_reason') == 'length':
            raise OutputLimitError(value.get('response', ''), options['num_predict'])
        return value.get('response', '')

    def unload(self):
        if self.process and self.process.poll() is None:
            self.api('/api/generate', dict(model=self.model, keep_alive=0, stream=False), timeout=120)

    def close(self):
        had_scope = self.scope is not None
        try:
            if self.process:
                with contextlib.suppress(Exception):
                    self.unload()
                if self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            try:
                if self.scope:
                    # HTTP unload may leave an idle CUDA context in llama-server.
                    # Also kills descendants if their service parent already exited.
                    self.scope.close()
            finally:
                self.scope = None
                self.process = None
                if self.log:
                    self.log.close()
                    self.log = None
        if had_scope and self.memory_before is not None:
            wait_gpu_recovery(self.memory_before)


class AlignmentError(ValueError):
    pass


class OutputLimitError(AlignmentError):
    def __init__(self, response, limit):
        self.response = response
        super().__init__(f'译文输出达到 {limit} token 上限')


def translated_lines(response: str, expected: int):
    lines = [line.strip() for line in response.replace('\r\n', '\n').strip().splitlines() if line.strip()]
    if len(lines) != expected or any('<|im_' in line or '-->' in line for line in lines):
        raise AlignmentError(f'要求 {expected} 行，收到 {len(lines)} 行')
    return lines


def translate_range(client, texts, start, end, diagnostics, prior=None, on_progress=None):
    context_start = max(0, start - 5)
    sent = texts[context_start:end]

    def split():
        mid = (start + end) // 2
        left = translate_range(client, texts, start, mid, diagnostics, prior, on_progress)
        next_prior = prior + left if prior is not None else None
        right = translate_range(client, texts, mid, end, diagnostics, next_prior, on_progress)
        return left + right

    def accepted(lines):
        if on_progress:
            on_progress(start, lines)
        return lines

    # Keep enough space for the output. Split unusually long subtitle batches.
    if sum(map(len, sent)) > 2800 and end - start > 1:
        return split()
    # Same prompt + fixed seed produces the same failure. Split immediately.
    for attempt in range(1):
        raw = ''
        try:
            contextual = getattr(client, 'generate_contextual', None)
            if prior is not None and len(prior) == start and callable(contextual):
                raw = contextual(texts[start:end], texts[context_start:start], prior[context_start:start])
                return accepted(translated_lines(raw, end - start))
            raw = client.generate(sent)
            lines = translated_lines(raw, len(sent))
            return accepted(lines[start - context_start:])
        except AlignmentError as e:
            raw = getattr(e, 'response', raw)
            save_json(diagnostics / f'alignment-{start}-{end}-{attempt}.json',
                      dict(error=str(e), source=sent, response=raw))
            LOG.debug('翻译对齐检查 [%d:%d]：%s', start, end, e)
            if end - start > 1:
                LOG.info('第 %d—%d 句正在拆成小批重译；已完成译文保留。', start + 1, end)
    if end - start == 1:
        focused = getattr(client, 'generate_focused', None)
        if callable(focused):
            for attempt in range(1):
                raw = ''
                try:
                    if prior is not None and len(prior) == start:
                        raw = focused(texts[start], texts[context_start:start], prior[context_start:start])
                    else:
                        raw = focused(texts[start], texts[context_start:start])
                    return accepted(translated_lines(raw, 1))
                except AlignmentError as e:
                    raw = getattr(e, 'response', raw)
                    save_json(diagnostics / f'focused-{start}-{attempt}.json',
                              dict(error=str(e), source=texts[start], response=raw))
        raise AlignmentError(f'第 {start + 1} 句重试后仍无法对齐。已保存诊断，未生成错位字幕。')
    return split()


def translate_all(client, japanese, checkpoint: Path, ja_hash):
    texts = [re.sub(r'\s+', ' ', cue.text).strip() for cue in japanese]
    key = dict(version=2, ja_sha256=ja_hash, model=client.model,
               prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(), options=OPTIONS)
    translations = []
    if checkpoint.exists():
        saved = read_json(checkpoint)
        if saved.get('key') == key:
            translations = saved.get('translations', [])
            if (not isinstance(translations, list) or len(translations) > len(texts) or
                    any(not isinstance(t, str) or not t.strip() for t in translations)):
                raise ValueError('翻译断点文件损坏，请删除对应 .state 子目录后重试。')
            LOG.info('从已完成的 %d / %d 句继续翻译。', len(translations), len(texts))
    def persist(index, lines):
        if index != len(translations):
            raise ValueError('拒绝保存不连续的字幕断点')
        translations.extend(lines)
        save_json(checkpoint, dict(key=key, translations=translations))
        LOG.info('翻译完成 %d / %d 句', len(translations), len(texts))

    while len(translations) < len(texts):
        start = len(translations)
        if texts[start] == '［聞き取り不明］':
            persist(start, ['［听不清］'])
            continue
        end = start + 1
        characters = len(texts[start])
        while end < min(start + 20, len(texts)):
            if (texts[end] == '［聞き取り不明］' or characters + len(texts[end]) > 600 or
                    japanese[end].start - japanese[end - 1].end > 4000):
                break
            characters += len(texts[end])
            end += 1
        batch = translate_range(client, texts, start, end, checkpoint.parent / 'diagnostics',
                                prior=translations.copy(), on_progress=persist)
        if len(batch) != end - start or len(translations) != end:
            raise AlignmentError('内部译文行数错误')
    return translations


def validate_wav(wav: Path):
    with wav.open('rb') as f:
        header = f.read(16)
    if len(header) >= 12 and header[:4] in (b'RIFF', b'RF64', b'BW64') and header[8:12] == b'WAVE':
        return True
    # Some supplied files are M4A/MP3 renamed to .wav. Decode privately;
    # never rename, overwrite, or re-encode the user's original file.
    if (header[4:8] == b'ftyp' or header[:4] in (b'fLaC', b'OggS') or
            header[:3] == b'ID3' or (len(header) >= 2 and header[0] == 255 and header[1] & 224 == 224)):
        return False
    raise ValueError(f'音频格式无法识别或文件已损坏：{wav.name}')


def decode_audio(wav, output, job):
    if not output.resolve().is_relative_to(TEMP.resolve()) or output.resolve() == wav.resolve():
        raise ValueError('临时解码输出必须位于本工具 .temp 内')
    LOG.info('检测到扩展名与实际音频格式不一致，正在临时解码；原文件保留。')
    command = [str(RUNTIME / 'bin/ffmpeg.exe'), '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-y', '-i', str(wav), '-map', '0:a:0', '-vn', '-c:a', 'pcm_f32le',
               '-rf64', 'auto', str(output)]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               creationflags=NO_WINDOW)
    if job:
        job.add(process)
    _, detail = process.communicate()
    if process.returncode or not output.is_file() or validate_wav(output) is not True:
        raise RuntimeError('音频临时解码失败：' + detail.decode('utf-8', errors='replace')[-2000:])
    return output


def process_audio(wav, job, offline):
    import quality_asr
    native_wave = validate_wav(wav)
    state = state_dir(wav)
    state.mkdir(parents=True, exist_ok=True)
    ja = japanese_path(wav)
    legacy_ja = wav.with_suffix('.ja.srt')
    existing = ja if ja.exists() else legacy_ja
    signature = source_signature(wav)
    ja_meta = state / 'japanese.json'
    profile_file = state / 'japanese.profile.json'
    if existing.exists() and ja_meta.exists() and read_json(ja_meta) != signature:
        raise RuntimeError(f'{wav.name} 的源音频已变更，原日文字幕可能过期。请移走旧字幕后重试。')
    same_profile = profile_file.exists() and read_json(profile_file).get('profile') == quality_asr.PROFILE
    if existing.exists() and (same_profile or not ja_meta.exists()) and (existing.stat().st_size > 3 or ja_meta.exists()):
        cues = parse_srt(existing)
        if existing != ja:
            # Preserve legacy bytes so a valid translation checkpoint keeps its SHA.
            pending = ja.with_name(ja.name + '.writing')
            shutil.copy2(existing, pending)
            os.replace(pending, ja)
        LOG.info('复用内部日文听写缓存；对外仅输出中文字幕。')
        return cues
    if existing.exists():
        backup = state / ('previous-ja-' + time.strftime('%Y%m%d-%H%M%S') + '.srt')
        shutil.copy2(existing, backup)
    LOG.info('阶段 1/3：原音耳语检测；阶段 2/3：完整 Qwen3-ASR 日语听写')
    run_worker(job, 'transcribe', offline, wav, ja)
    save_json(ja_meta, signature)
    save_json(profile_file, dict(profile=quality_asr.PROFILE))
    return parse_srt(ja)


def legacy_process_audio(wav, job, offline):
    """Retained for reproducible UVR/Whisper comparison, not the default path."""
    native_wave = validate_wav(wav)
    ja = wav.with_suffix('.ja.srt')
    state = state_dir(wav)
    state.mkdir(parents=True, exist_ok=True)
    signature = source_signature(wav)
    ja_meta = state / 'japanese.json'
    # Stale files only under our private temp folder; never touch source WAVs.
    with tempfile.TemporaryDirectory(prefix='audio-', dir=TEMP) as tmp:
        stage_dir = Path(tmp)
        source = wav if native_wave else decode_audio(wav, stage_dir / 'decoded-input.wav', job)
        LOG.info('阶段 1/3：Inst_HQ_3 人声分离')
        run_worker(job, 'separate', offline, source, stage_dir)
        vocal = Path(read_json(stage_dir / 'vocals.json')['path'])
        if not vocal.resolve().is_relative_to(stage_dir.resolve()):
            raise RuntimeError('人声音轨路径越界。')
        LOG.info('分离子进程已退出并释放显存。阶段 2/3：large-v3 FP16 听写')
        try:
            run_worker(job, 'whisper', offline, vocal, ja)
        finally:
            # The vocal stem is required by ASR, then removed even if ASR fails.
            if vocal.is_file():
                vocal.unlink()
        save_json(ja_meta, signature)
    LOG.info('听写子进程已退出，临时音频已删除，显存已释放。')
    return parse_srt(ja)


def check_deps():
    import torch
    import torchvision
    import torchaudio
    import onnxruntime as ort
    import ctranslate2
    import faster_whisper
    import audio_separator
    from audio_separator.separator import Separator
    import imageio_ffmpeg
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    import quality_asr
    if importlib.metadata.version('torch').split('+')[0] != '2.8.0' or not torch.version.cuda:
        raise RuntimeError('要求 torch 2.8.0 CUDA build。')
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('GPU ONNX Runtime 安装失败。')
    if importlib.util.find_spec('onnxruntime') is None:
        raise RuntimeError('ONNX Runtime 未安装。')
    ensure_ffmpeg()
    LOG.info('依赖导入检查通过。PyTorch CUDA %s；ONNX %s；CTranslate2 %s',
             torch.version.cuda, ort.__version__, ctranslate2.__version__)


def diagnose():
    info = gpu_info()
    check_deps()
    torch, ort = cuda_dependencies()
    LOG.info('Sakura 选择：%s', select_sakura(info['total']))
    LOG.info('模型目录：%s；剩余磁盘 %.1f GiB', MODELS, shutil.disk_usage(ROOT).free / 2**30)
    for package in ('audio-separator', 'faster-whisper', 'torch', 'onnxruntime-gpu', 'ctranslate2'):
        LOG.info('%s %s', package, importlib.metadata.version(package))
    LOG.info('ONNX providers: %s', ort.get_available_providers())
    empty_cuda(torch)


def prepare_models(job, model, offline, force=False):
    proof_path = MODELS / 'prepared.json'
    import quality_asr
    key = dict(schema=3, asr=quality_asr.PROFILE, sakura=model,
               precision='bfloat16', source='original', translation_prompt=SYSTEM)
    if not force and proof_path.exists():
        try:
            proof = read_json(proof_path)
            files = [MODELS / name for name in proof.get('files', [])]
            if (proof.get('key') == key and files and
                    all(f.resolve().is_relative_to(MODELS.resolve()) and f.is_file() for f in files)):
                return
        except (OSError, ValueError):
            pass
    LOG.info('首次运行先准备并加载全部模型，通过后才开始处理作品。')
    run_worker(job, 'transcribe', offline)
    client = None
    try:
        client = Ollama(job, model, offline)
        client.api('/api/generate', dict(model=model, stream=False, keep_alive='1m', options=OPTIONS))
    finally:
        if client:
            client.close()
    files = [MODELS / 'qwen3-asr-1.7b/model.safetensors', MODELS / 'asmr-vad/model.onnx',
             MODELS / 'qwen3-asr-1.7b/config.json', MODELS / 'qwen3-aligner/model.safetensors']
    files.extend(f for manifests in (MODELS / 'ollama').glob('manifests*')
                 if manifests.is_dir() for f in manifests.rglob('*') if f.is_file())
    save_json(proof_path, dict(key=key, files=[f.relative_to(MODELS).as_posix() for f in files]))
    LOG.info('全部模型已下载并实际加载通过，可断网运行。')


def process_package(package, job, model, offline):
    # Check for an existing completed package before starting expensive work.
    packages.destination(ROOT, package)
    failed = []
    for wav in package.audio:
        client = None
        try:
            if has_subtitle(wav):
                LOG.info('跳过已有中文字幕：%s', wav.relative_to(package.path))
                continue
            LOG.info('========== %s / %s ==========', package.path.name, wav.relative_to(package.path))
            japanese = process_audio(wav, job, offline)
            if not japanese:
                export_subtitles(wav, [], [])
                save_json(state_dir(wav) / 'empty.done.json', source_signature(wav))
                LOG.warning('未检测到语音，已写空字幕，请人工复核是否漏掉耳语。')
                continue
            LOG.info('阶段 3/3：Sakura 本地翻译（20 句/批，前 5 句上下文）')
            client = Ollama(job, model, offline)
            translations = translate_all(client, japanese, state_dir(wav) / 'translation.json',
                                         file_hash(japanese_path(wav)))
            export_subtitles(wav, japanese, translations)
            LOG.info('已生成同名中文字幕 .srt；日文听写保存在内部缓存。')
        except StageError:
            LOG.error('模型子进程失败，立即停止队列，避免对后续音轨重复同一错误。')
            raise
        except Exception:
            LOG.exception('处理失败，保留在待翻译：%s', wav.relative_to(package.path))
            failed.append(wav.relative_to(package.path).as_posix())
        finally:
            if client:
                client.close()
    if failed:
        raise RuntimeError(f'作品「{package.path.name}」未完成：' + '、'.join(failed))
    target = packages.archive(ROOT, package, has_subtitle)
    LOG.info('整包完成，已移入：%s', target)


def backup_for_rebuild(work):
    """Move only managed subtitle products/state aside, never audio or assets."""
    stamp = time.strftime('%Y%m%d-%H%M%S')
    moves = []
    for package in work:
        for wav in package.audio:
            state = state_dir(wav)
            history = state / ('subtitle-history-' + stamp)
            paths = [wav.with_suffix(suffix) for suffix in ('.srt', '.ja.srt', '.bi.srt', '.lrc')]
            paths += [state / name for name in ('japanese.srt', 'japanese.json', 'japanese.profile.json',
                       'translation.json', 'asr-v2.json', 'asr-quality-report.json', 'empty.done.json')]
            for item in paths:
                if item.is_file():
                    target = (history / 'cache' / item.name) if item == japanese_path(wav) else history / item.name
                    if target.exists():
                        raise FileExistsError(f'字幕备份冲突：{target}')
                    moves.append((item, target))
    completed = []
    try:
        for item, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            item.rename(target)
            completed.append((item, target))
    except Exception:
        for item, target in reversed(completed):
            target.rename(item)
        raise
    LOG.info('旧字幕及断点已备份到 .state 对应音轨的 subtitle-history 目录，原音频未改动。')


def coordinate(mode, offline):
    work = []
    failed = []
    if mode != 'prepare':
        packages.layout(ROOT)
        count = packages.import_loose(ROOT, state_dir)
        if count:
            LOG.info('已将 %d 个散放 WAV 连同已有字幕和断点归入待翻译。', count)
        discovered, errors = packages.discover(ROOT)
        for name, message in errors:
            LOG.error('作品「%s」保留在待翻译：%s', name, message)
            failed.append(name)
        for package in discovered:
            try:
                packages.destination(ROOT, package)
                if mode != 'rebuild' and all(has_subtitle(wav) for wav in package.audio):
                    target = packages.archive(ROOT, package, has_subtitle)
                    LOG.info('已有字幕的作品整包移入：%s', target)
                else:
                    work.append(package)
            except Exception as e:
                LOG.error('作品「%s」保留在待翻译：%s', package.path.name, e)
                failed.append(package.path.name)
        if mode == 'organize':
            LOG.info('目录整理完成，待处理 %d 个作品；未运行音频模型。', len(work))
            if failed:
                raise RuntimeError('以下作品需检查：' + '、'.join(failed))
            return
        if not work:
            if failed:
                raise RuntimeError('以下作品需检查：' + '、'.join(failed))
            LOG.info('没有待处理 WAV。请把完整作品文件夹放入：%s', ROOT / packages.INBOX_NAME)
            return
    info = gpu_info()
    model = select_sakura(info['total'])
    LOG.info('质量模式：原音 + ASMR 专用 VAD + 完整 Qwen3-ASR BF16；翻译 %s', model)
    if shutil.disk_usage(ROOT).free < 3 * 2**30:
        raise RuntimeError('可用空间不足 3 GiB；请腾出模型和临时 WAV 所需空间。')
    ensure_ffmpeg()
    job = ChildJob()
    try:
        prepare_models(job, model, offline, force=(mode == 'prepare'))
        if mode == 'prepare':
            return
        if mode == 'rebuild':
            backup_for_rebuild(work)
        for package in work:
            try:
                process_package(package, job, model, offline)
            except StageError:
                raise
            except Exception as e:
                LOG.exception('作品未移出待翻译：%s：%s', package.path.name, e)
                failed.append(package.path.name)
        if failed:
            raise RuntimeError('以下作品未完成，已保留断点：' + '、'.join(failed))
    finally:
        job.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'rebuild', 'prepare', 'organize', 'diagnose', 'check-deps', 'self-test',
                                        'worker-separate', 'worker-transcribe', 'worker-whisper'])
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'self-test':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(APP / 'tests'))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    setup_environment(args.offline)
    configure_logging()
    try:
        if args.mode == 'worker-separate':
            separate_stage(args.input, args.output, args.offline)
        elif args.mode == 'worker-transcribe':
            transcribe_stage(args.input, args.output, args.offline)
        elif args.mode == 'worker-whisper':
            whisper_transcribe_stage(args.input, args.output, args.offline)
        elif args.mode == 'check-deps':
            check_deps()
        elif args.mode == 'diagnose':
            diagnose()
        else:
            coordinate(args.mode, args.offline)
        return 0
    except KeyboardInterrupt:
        LOG.error('已中断；完成阶段会在下次运行时复用。')
        return 130
    except Exception:
        LOG.exception('运行失败')
        return 1


if __name__ == '__main__':
    sys.exit(main())
