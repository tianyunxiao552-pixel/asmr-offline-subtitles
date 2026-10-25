"""Raw-audio Japanese ASR with whisper-sensitive VAD and bounded GPU use.

VAD runs on CPU. Only one complete ASR model occupies CUDA at a time.
No denoising, source separation, or text-only reconstruction of missing speech.
"""
from pathlib import Path
import gc
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time

LOG = logging.getLogger('asmr')
ASR_REPO = 'Qwen/Qwen3-ASR-1.7B-hf'
VAD_REPO = 'TransWithAI/Whisper-Vad-EncDec-ASMR-onnx'
PROFILE = 'qwen3-full-raw-asmrvad-align-v2'
SAMPLE_RATE = 16000


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_models(models, offline):
    asr = models / 'qwen3-asr-1.7b'
    vad = models / 'asmr-vad'
    aligner = models / 'qwen3-aligner'
    asr_names = ['config.json', 'generation_config.json', 'processor_config.json',
                 'chat_template.jinja', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors']
    aligner_names = ['config.json', 'processor_config.json', 'chat_template.jinja',
                     'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']
    needed = ([asr / n for n in asr_names] + [vad / 'model.onnx', vad / 'preprocessor_config.json'] +
              [aligner / n for n in aligner_names])
    if all(path.is_file() for path in needed):
        return asr, vad
    if offline:
        raise RuntimeError('离线模式缺少 Qwen3-ASR 或 ASMR VAD，请先运行 run.bat -Prepare。')
    from hub_download import download_files
    download_files(ASR_REPO, asr_names, asr)
    download_files(VAD_REPO, ['model.onnx', 'model_metadata.json'], vad)
    download_files('openai/whisper-base', ['preprocessor_config.json'], vad)
    download_files('Qwen/Qwen3-ForcedAligner-0.6B-hf', aligner_names, aligner)
    return asr, vad


def mask_regions(mask, offset, seconds_per_frame=0.02, minimum=0.10):
    """Convert speech frames into half-open intervals without inventing text."""
    regions = []
    start = None
    for i, speech in enumerate(list(mask) + [False]):
        if speech and start is None:
            start = i
        elif not speech and start is not None:
            if (i - start) * seconds_per_frame >= minimum:
                regions.append((offset + start * seconds_per_frame, offset + i * seconds_per_frame))
            start = None
    return regions


def speech_windows(regions, duration, gap=0.65, padding=0.25, maximum=12.0):
    """Group nearby speech, preserve pauses, then cap model input duration."""
    merged = []
    for start, end in sorted(regions):
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    windows = []
    for start, end in merged:
        start, end = max(0.0, start - padding), min(duration, end + padding)
        pieces = max(1, math.ceil((end - start) / maximum))
        width = (end - start) / pieces
        windows.extend((start + i * width, min(end, start + (i + 1) * width)) for i in range(pieces))
    # Padding must not make adjacent windows overlap: each sample has one owner.
    for i in range(1, len(windows)):
        if windows[i][0] < windows[i - 1][1]:
            boundary = (windows[i][0] + windows[i - 1][1]) / 2
            windows[i - 1] = (windows[i - 1][0], boundary)
            windows[i] = (boundary, windows[i][1])
    return windows


def sentence_cues(rows, max_seconds=12.0, max_chars=110, max_gap=0.65):
    """Join only adjacent unfinished speech; retain all source text and timing."""
    result = []
    terminal = re.compile(r'[。！？!?][」』）)]?$')
    for row in rows:
        item = dict(row)
        if not item['text'].strip():
            continue
        if result:
            previous = result[-1]
            should_join = (not terminal.search(previous['text']) and
                           0 <= item['start'] - previous['end'] <= round(max_gap * 1000) and
                           item['end'] - previous['start'] <= round(max_seconds * 1000) and
                           len(previous['text']) + len(item['text']) <= max_chars and
                           not previous.get('review') and not item.get('review'))
            if should_join:
                previous['text'] += item['text']
                previous['end'] = item['end']
                continue
        result.append(item)
    return result


def suspicious(text, exhausted=False):
    if exhausted:
        return '生成达到长度上限，可能不完整'
    normalized = re.sub(r'\s|[、。！？!?…ー〜～]', '', text)
    if len(normalized) > 70 and len(set(normalized)) / len(normalized) < 0.12:
        return '长段重复，需与音频复核'
    if normalized and not re.search(r'[ぁ-ゖァ-ヺ一-龯]', normalized):
        return '未返回可识别的日文'
    return None


def aligned_sentences(text, words, offset, duration, max_seconds=12.0, max_chars=90):
    """Retain transcript characters exactly, using aligner words as boundaries."""
    if not words:
        raise ValueError('对齐器未返回词时间')
    matches = []
    cursor = 0
    for word in words:
        token = word['text']
        start = text.find(token, cursor)
        if start < 0 or not token:
            raise ValueError('分词与原始听写不一致，不能安全重切字幕')
        matches.append((start, start + len(token), word))
        cursor = start + len(token)
    units = []
    group_text = ''
    group_start = None
    group_end = None
    last_end = 0
    for index, (start, end, word) in enumerate(matches):
        boundary = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        fragment = text[last_end:boundary]
        last_end = boundary
        relative_start = min(duration, max(0, float(word['start_time'])))
        relative_end = min(duration, max(relative_start + 0.001, float(word['end_time'])))
        start_ms = round((offset + relative_start) * 1000)
        end_ms = max(start_ms + 1, round((offset + relative_end) * 1000))
        if group_start is not None and (end_ms - group_start > max_seconds * 1000 or
                                        len(group_text) + len(fragment) > max_chars):
            units.append(dict(start=group_start, end=max(group_start + 1, group_end), text=group_text))
            group_text, group_start = '', None
        if group_start is None:
            group_start = max(start_ms, units[-1]['end'] if units else start_ms)
        group_end = max(group_start + 1, end_ms)
        group_text += fragment
        if re.search(r'[。！？!?][」』）)]?\s*$', group_text):
            units.append(dict(start=group_start, end=group_end, text=group_text))
            group_text, group_start = '', None
    if group_text:
        units.append(dict(start=group_start, end=max(group_start + 1, group_end), text=group_text))
    if ''.join(unit['text'] for unit in units) != text:
        raise ValueError('字幕重切改变了听写文本，拒绝发布')
    return units


def reserve_cuda_budget(torch, timeout=30):
    """Keep the original model/precision and safety budget; wait for handoff."""
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        free, total = torch.cuda.mem_get_info(0)
        budget = max(0, min(total * 0.90, free - 768 * 2**20))
        if budget >= 4.6 * 2**30:
            LOG.info('听写加载前：CUDA 空闲 %.2f GiB，总计 %.2f GiB，预算 %.2f GiB。',
                     free / 2**30, total / 2**30, budget / 2**30)
            return budget, total
        if time.monotonic() >= deadline:
            raise RuntimeError(f'等待显存交接 {timeout} 秒后仍不足：CUDA 空闲 {free / 2**30:.2f} GiB，'
                               '要求至少 5.35 GiB（完整模型预算 4.6 GiB 加安全余量）；'
                               '已保留断点，未降低模型或精度。')
        if not announced:
            LOG.warning('等待上一阶段显存回收，当前 CUDA 空闲 %.2f GiB，最多等待 %d 秒。', free / 2**30, timeout)
            announced = True
        time.sleep(1)


def transcribe(source, output, models, temporary, checkpoint, ffmpeg, offline=True, prepare_only=False):
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import (AutoProcessor, AutoModelForMultimodalLM,
                              AutoModelForTokenClassification, WhisperFeatureExtractor)

    asr_path, vad_path = prepare_models(models, offline)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen3-ASR 需要可用 NVIDIA CUDA；未切换小模型。')
    budget, total = reserve_cuda_budget(torch)
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    model = None
    audio = None
    decoder = None
    metrics = dict(profile=PROFILE, gpu_budget_bytes=int(budget), source=str(source))
    try:
        LOG.info('加载完整 Qwen3-ASR BF16，原音听写；显存预算 %.2f GiB。', budget / 2**30)
        processor = AutoProcessor.from_pretrained(str(asr_path), local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(str(asr_path), dtype=torch.bfloat16,
                    attn_implementation='sdpa', local_files_only=True).eval().to('cuda:0')
        vad_options = ort.SessionOptions()
        vad_options.intra_op_num_threads = 2
        detector = ort.InferenceSession(str(vad_path / 'model.onnx'), sess_options=vad_options,
                                       providers=['CPUExecutionProvider'])
        extractor = WhisperFeatureExtractor.from_pretrained(str(vad_path), local_files_only=True)
        if prepare_only:
            del model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            aligner_path = models / 'qwen3-aligner'
            AutoProcessor.from_pretrained(str(aligner_path), local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(str(aligner_path), dtype=torch.bfloat16,
                        attn_implementation='sdpa', local_files_only=True).eval().to('cuda:0')
            LOG.info('完整 ASR、ASMR VAD、日语时间对齐器均已实际加载通过。')
            return
        temporary.mkdir(parents=True, exist_ok=True)
        pcm = temporary / 'original-16k-mono.pcm'
        LOG.info('阶段 1/3：解码原音并检测耳语；不做人声分离。')
        decoder = subprocess.Popen([str(ffmpeg), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(source), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
                    '-f', 'f32le', str(pcm)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, detail = decoder.communicate()
        if decoder.returncode:
            raise RuntimeError('原音解码失败：' + detail.decode('utf-8', errors='replace')[-1200:])
        if not pcm.is_file() or pcm.stat().st_size == 0:
            raise ValueError('音频没有可解码的采样')
        audio = np.memmap(pcm, dtype=np.float32, mode='r')
        duration = len(audio) / SAMPLE_RATE
        signature = dict(profile=PROFILE, source_bytes=source.stat().st_size,
                         source_mtime_ns=source.stat().st_mtime_ns)
        saved = {}
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding='utf-8'))
            if saved.get('signature') != signature:
                saved = {}
        if saved.get('windows') is not None:
            windows = [tuple(window) for window in saved['windows']]
            if any(not (0 <= start < end <= duration + 0.02) for start, end in windows):
                raise ValueError('听写断点含非法时间范围')
        else:
            regions = []
            length = 30 * SAMPLE_RATE
            for offset in range(0, len(audio), length):
                fragment = np.asarray(audio[offset:offset + length])
                features = extractor(fragment, sampling_rate=SAMPLE_RATE, return_tensors='np').input_features
                mask = detector.run(None, {detector.get_inputs()[0].name: features.astype(np.float32)})[0]
                mask = np.asarray(mask).reshape(-1)[:math.ceil(len(fragment) / SAMPLE_RATE / 0.02)]
                regions.extend(mask_regions(mask > 0.35, offset / SAMPLE_RATE))
                LOG.info('耳语检测 %.0f / %.0f 秒', min(duration, offset / SAMPLE_RATE + 30), duration)
            windows = speech_windows(regions, duration, gap=2.0, padding=0.4, maximum=30.0)
            saved = dict(signature=signature, windows=windows, rows=[], next_window=0, review=[])
            atomic_json(checkpoint, saved)
        del detector
        LOG.info('阶段 2/3：原音日语听写，共 %d 个语音片段。', len(windows))
        rows = saved.get('rows', [])
        review = saved.get('review', [])
        start_index = saved.get('next_window', 0)
        if not isinstance(start_index, int) or not 0 <= start_index <= len(windows):
            raise ValueError('听写断点损坏')
        torch.cuda.reset_peak_memory_stats()
        for index in range(start_index, len(windows)):
            start, end = windows[index]
            fragment = np.asarray(audio[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]).copy()
            inputs = processor.apply_transcription_request(audio=fragment, language='Japanese').to(model.device, model.dtype)
            limit = min(768, max(96, math.ceil((end - start) * 14)))
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
            generated = ids[:, inputs['input_ids'].shape[1]:]
            text = processor.decode(generated, return_format='transcription_only', skip_special_tokens=True)[0].strip()
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            exhausted = generated.shape[1] >= limit and int(generated[0, -1]) not in eos
            reason = suspicious(text, exhausted)
            if reason:
                review.append(dict(window=index, start=start, end=end, reason=reason, candidate=text))
                text = '［聞き取り不明］'
            if text:
                rows.append(dict(start=round(start * 1000), end=max(round(start * 1000) + 1, round(end * 1000)),
                                 text=text, review=bool(reason)))
            del inputs, ids, generated
            saved.update(rows=rows, next_window=index + 1, review=review)
            atomic_json(checkpoint, saved)
            LOG.info('原音听写 %d / %d 段（%.1f%%）', index + 1, len(windows), 100 * (index + 1) / len(windows))
        # Release the complete ASR model before loading the alignment model.
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        LOG.info('日文语义段听写完成并释放显存；正在按原音定位每个词的时间。')
        aligner_path = models / 'qwen3-aligner'
        align_processor = AutoProcessor.from_pretrained(str(aligner_path), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(str(aligner_path), dtype=torch.bfloat16,
                    attn_implementation='sdpa', local_files_only=True).eval().to('cuda:0')
        final_rows = saved.get('aligned_rows', [])
        aligned_count = saved.get('aligned_count', 0)
        if not isinstance(aligned_count, int) or not 0 <= aligned_count <= len(rows):
            raise ValueError('时间对齐断点损坏')
        for index in range(aligned_count, len(rows)):
            row = rows[index]
            if row.get('review'):
                final_rows.append(row.copy())
            else:
                start, end = row['start'] / 1000, row['end'] / 1000
                fragment = np.asarray(audio[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]).copy()
                align_inputs, word_lists = align_processor.prepare_forced_aligner_inputs(
                    audio=fragment, transcript=row['text'], language='Japanese')
                align_inputs = align_inputs.to(model.device, model.dtype)
                with torch.inference_mode():
                    logits = model(**align_inputs).logits
                words = align_processor.decode_forced_alignment(logits, align_inputs['input_ids'], word_lists,
                            timestamp_token_id=model.config.timestamp_token_id)[0]
                try:
                    final_rows.extend(aligned_sentences(row['text'], words, start, end-start))
                except ValueError as exc:
                    final_rows.append(dict(row, review=True))
                    review.append(dict(start=start, end=end, reason='时间对齐需复核：' + str(exc)))
                del align_inputs, logits
            saved.update(aligned_rows=final_rows, aligned_count=index+1, review=review)
            atomic_json(checkpoint, saved)
            LOG.info('时间定位 %d / %d 段', index+1, len(rows))
        # Adjacent incomplete clauses can now be safely reconstructed with word times.
        final_rows = sentence_cues(final_rows)
        text = ''
        for index, row in enumerate(final_rows, 1):
            def stamp(ms):
                h, r = divmod(ms, 3600000)
                m, r = divmod(r, 60000)
                s, t = divmod(r, 1000)
                return f'{h:02d}:{m:02d}:{s:02d},{t:03d}'
            text += f"{index}\n{stamp(row['start'])} --> {stamp(row['end'])}\n{row['text']}\n\n"
        pending = output.with_suffix(output.suffix + '.writing')
        pending.write_text(text, encoding='utf-8-sig')
        os.replace(pending, output)
        metrics.update(duration=duration, windows=len(windows), cues=len(final_rows), review=review,
                       cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved=torch.cuda.max_memory_reserved())
        atomic_json(checkpoint.with_name('asr-quality-report.json'), metrics)
        LOG.info('原音听写完成：%d 条字幕，%d 段需复核；CUDA 峰值 %.2f GiB。',
                 len(final_rows), len(review), metrics['cuda_peak_reserved'] / 2**30)
    finally:
        if decoder is not None and decoder.poll() is None:
            decoder.kill()
            decoder.wait()
        if audio is not None:
            audio._mmap.close()
            del audio
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
