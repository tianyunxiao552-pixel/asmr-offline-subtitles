"""Verified, resumable parallel HTTP downloads from the official HF Hub.

Uses the user's normal HTTP proxy, avoiding native Xet proxy incompatibilities.
Every range is checked; the assembled weight must match HF's LFS SHA256.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from urllib import request

LOG = logging.getLogger('asmr')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def range_download(url, target: Path, size: int, sha256: str, workers=16, chunk_size=8 * 1024 * 1024):
    if target.is_file() and target.stat().st_size == size and digest(target) == sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = target.with_name(target.name + '.parts')
    if not parts.resolve().is_relative_to(target.parent.resolve()) or parts.is_symlink():
        raise ValueError('Invalid download cache path')
    parts.mkdir(exist_ok=True)
    metadata = parts / 'metadata.json'
    key = dict(size=size, sha256=sha256, chunk_size=chunk_size)
    if metadata.exists() and json.loads(metadata.read_text()) != key:
        raise RuntimeError(f'模型分段缓存不匹配，请清理该模型缓存：{parts}')
    metadata.write_text(json.dumps(key))
    total = (size + chunk_size - 1) // chunk_size

    def part_path(i):
        return parts / f'{i:06d}.chunk'

    def fetch(i):
        start = i * chunk_size
        end = min(start + chunk_size, size) - 1
        path = part_path(i)
        expected = end - start + 1
        if path.is_file() and path.stat().st_size == expected:
            return expected
        partial = path.with_suffix('.partial')
        last_error = None
        for attempt in range(5):
            try:
                joiner = '&' if '?' in url else '?'
                # Distinct cache keys avoid stale redirects bound to other ranges.
                endpoint = url + joiner + f'download=true&range_start={start}&attempt={attempt}'
                req = request.Request(endpoint, headers={'Range': f'bytes={start}-{end}',
                    'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache', 'User-Agent': 'ASMR-Offline/1'})
                with request.urlopen(req, timeout=90) as response, partial.open('wb') as f:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match or tuple(map(int, match.groups())) != (start, end, size):
                        raise RuntimeError('官方服务器未返回所要求的完整分段，停止避免权重损坏。')
                    copied = 0
                    while block := response.read(512 * 1024):
                        f.write(block)
                        copied += len(block)
                if copied != expected:
                    raise RuntimeError(f'分段长度错误：{copied}/{expected}')
                os.replace(partial, path)
                return copied
            except Exception as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f'{target.name} 分段 {i} 下载失败：{last_error}')

    completed = sum(part_path(i).stat().st_size for i in range(total)
                    if part_path(i).is_file() and part_path(i).stat().st_size == min(chunk_size, size - i * chunk_size))
    missing = [i for i in range(total) if not part_path(i).is_file() or
               part_path(i).stat().st_size != min(chunk_size, size - i * chunk_size)]
    LOG.info('%s：断点已保存 %.1f / %.1f MiB，%d 连接下载官方权重。',
             target.name, completed / 2**20, size / 2**20, workers)
    last_report = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, i) for i in missing]
        for future in as_completed(futures):
            completed += future.result()
            if time.monotonic() - last_report >= 10 or completed == size:
                LOG.info('%s：%.1f / %.1f MiB（%.1f%%）', target.name,
                         completed / 2**20, size / 2**20, 100 * completed / size)
                last_report = time.monotonic()
    assembled = target.with_name(target.name + '.assembling')
    h = hashlib.sha256()
    with assembled.open('wb') as destination:
        for i in range(total):
            with part_path(i).open('rb') as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                    destination.write(block)
                    h.update(block)
        destination.flush()
        os.fsync(destination.fileno())
    if assembled.stat().st_size != size or h.hexdigest() != sha256:
        raise RuntimeError(f'官方 SHA256 校验失败：{target.name}，不加载该文件。')
    os.replace(assembled, target)
    # Only our flat cache files; no recursive deletion or traversal of links.
    for path in parts.iterdir():
        if path.is_file() and not path.is_symlink():
            path.unlink()
    parts.rmdir()
    LOG.info('%s 下载完成，官方 SHA256 校验通过。', target.name)
    return target


def download_files(repo, names, folder: Path):
    from huggingface_hub import HfApi, hf_hub_url
    for attempt in range(5):
        try:
            info = HfApi().model_info(repo, files_metadata=True, token=False)
            break
        except Exception as exc:
            if attempt == 4:
                raise
            LOG.warning('官方模型元数据连接中断，重试 %d/5（%s）。', attempt + 1, type(exc).__name__)
            time.sleep(min(2 ** attempt, 8))
    siblings = {file.rfilename: file for file in info.siblings}
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names:
        file = siblings[name]
        target = folder / name
        url = hf_hub_url(repo, name, revision=info.sha)
        if file.lfs and file.size > 32 * 1024 * 1024:
            range_download(url, target, file.size, file.lfs.sha256)
        elif not target.is_file() or target.stat().st_size != file.size:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.part')
            for attempt in range(5):
                try:
                    with request.urlopen(url, timeout=90) as response:
                        data = response.read()
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 8))
            if len(data) != file.size:
                raise RuntimeError(f'官方文件下载不完整：{name}')
            partial.write_bytes(data)
            os.replace(partial, target)
        paths[name] = target
    return paths
