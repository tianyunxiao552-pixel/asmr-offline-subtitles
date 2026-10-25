"""Lossless stem writer for audio-separator 0.30.2's soundfile backend.

That backend ignores output_dir and flattens/interleaves floating point stereo
into int16 without scaling. Keep frames/channels/amplitude and propagate errors.
"""
from pathlib import Path


def write_stem(separator, stem_path, stem_source):
    import numpy as np
    import soundfile as sf
    directory = Path(separator.output_dir).resolve()
    target = Path(stem_path)
    if not target.is_absolute():
        target = directory / target
    target = target.resolve()
    if not target.is_relative_to(directory):
        raise ValueError(f'分离音轨输出路径越界：{target}')
    data = np.asarray(stem_source, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] not in (1, 2) or not len(data):
        raise ValueError(f'分离音频形状不正确：{data.shape}')
    if not np.isfinite(data).all():
        raise ValueError('分离音频包含 NaN 或无穷值。')
    directory.mkdir(parents=True, exist_ok=True)
    # Float WAV retains quiet whispers and avoids both integer truncation and
    # clipping. Chunk boundaries and frame counts remain untouched.
    sf.write(str(target), np.ascontiguousarray(data), separator.sample_rate,
             format='WAV', subtype='FLOAT')
    info = sf.info(str(target))
    if info.frames != len(data) or info.channels != data.shape[1]:
        raise RuntimeError('分离音频写出后，帧数或声道不匹配。')
