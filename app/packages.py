"""Folder queue operations. No CUDA, model downloads, or audio content edits."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

INBOX_NAME = '待翻译'
OUTBOX_NAME = '已翻译'
SIDECARS = ('.ja.srt', '.srt', '.bi.srt', '.lrc')


@dataclass(frozen=True)
class Package:
    path: Path
    audio: tuple[Path, ...]


def reject_link(path: Path):
    stat = path.lstat()
    if path.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
        raise ValueError(f'不处理符号链接或目录联接：{path}')


def inside(path: Path, parent: Path):
    """Validate the absolute target before any rename or tree traversal."""
    absolute = path.absolute()
    boundary = parent.absolute()
    if not absolute.is_relative_to(boundary) or absolute == boundary:
        raise ValueError(f'路径超出允许目录：{path}')
    cursor = boundary
    if cursor.exists():
        reject_link(cursor)
    for part in absolute.relative_to(boundary).parts:
        cursor /= part
        if os.path.lexists(cursor):
            reject_link(cursor)
    if not absolute.resolve().is_relative_to(boundary.resolve()):
        raise ValueError(f'实际路径超出允许目录：{path}')
    return absolute


def layout(root: Path):
    inbox, outbox = root / INBOX_NAME, root / OUTBOX_NAME
    for directory in (inbox, outbox):
        inside(directory, root)
        directory.mkdir(exist_ok=True)
    return inbox, outbox


def tree_files(folder: Path):
    reject_link(folder)
    pending = [folder]
    while pending:
        current = pending.pop()
        for entry in sorted(current.iterdir(), key=lambda p: p.name.casefold()):
            inside(entry, folder)
            if entry.is_dir():
                pending.append(entry)
            elif entry.is_file():
                yield entry


def check_output_names(audio, folder):
    outputs = {}
    for wav in audio:
        for suffix in SIDECARS:
            key = wav.with_suffix(suffix).relative_to(folder).as_posix().casefold()
            if key in outputs:
                raise ValueError(f'字幕输出名冲突：{outputs[key]} 与 {wav.relative_to(folder)}')
            outputs[key] = wav.relative_to(folder)


def scan(folder: Path):
    files = list(tree_files(folder))  # Also checks cover/text files for links.
    audio = sorted((p for p in files if p.suffix.casefold() == '.wav'),
                   key=lambda p: p.relative_to(folder).as_posix().casefold())
    check_output_names(audio, folder)
    return Package(folder, tuple(audio))


def discover(root: Path):
    inbox, _ = layout(root)
    packages, errors = [], []
    for folder in sorted(inbox.iterdir(), key=lambda p: p.name.casefold()):
        try:
            inside(folder, inbox)
            if not folder.is_dir():
                continue
            package = scan(folder)
            if package.audio:
                packages.append(package)
        except (OSError, ValueError) as e:
            errors.append((folder.name, str(e)))
    return packages, errors


def logical_name(audio: Path, root: Path):
    """Use the same identity in the inbox and outbox, including nested paths."""
    for container in (root / INBOX_NAME, root / OUTBOX_NAME):
        try:
            return audio.absolute().relative_to(container.absolute()).as_posix().casefold()
        except ValueError:
            pass
    try:
        return audio.absolute().relative_to(root.absolute()).as_posix().casefold()
    except ValueError:
        return audio.absolute().as_posix().casefold()


def destination(root: Path, package: Package):
    inbox, outbox = layout(root)
    inside(package.path, inbox)
    if package.path.parent.absolute() != inbox.absolute():
        raise ValueError('只能移动待翻译中的整个顶层作品文件夹。')
    target = inside(outbox / package.path.name, outbox)
    if os.path.lexists(target):
        raise FileExistsError(f'已翻译中已有同名作品「{package.path.name}」，请先改名或移走；不会覆盖。')
    return target


def archive(root: Path, package: Package, complete):
    target = destination(root, package)
    # Discover again: files added while the GPU was working must not get moved
    # into the completed area before they have subtitles.
    fresh = scan(package.path)
    if not fresh.audio or not all(complete(wav) for wav in fresh.audio):
        raise ValueError(f'「{package.path.name}」仍有未完成音轨，继续保留在待翻译。')
    # Inbox/outbox share a checked root on the same volume. Windows rename
    # refuses an existing destination; no recursive copy/delete or overwrites.
    os.rename(package.path, target)
    return target


def import_loose(root: Path, state_path):
    """Group legacy root WAVs and loose inbox WAVs, retaining subtitle sidecars.

    Preflight every source/destination first. Roll back completed renames if a
    later rename fails. Existing files and existing state are never overwritten.
    """
    inbox, _ = layout(root)
    imported = 0
    for source_dir, package_name in ((root, '根目录导入'), (inbox, '散放音频')):
        wavs = sorted((p for p in source_dir.iterdir() if p.suffix.casefold() == '.wav' and p.is_file()),
                      key=lambda p: p.name.casefold())
        if not wavs:
            continue
        check_output_names(wavs, source_dir)
        target_dir = inside(inbox / package_name, inbox)
        if target_dir.exists() and not target_dir.is_dir():
            raise FileExistsError(f'作品目录被文件占用：{target_dir}')
        plan = []
        for wav in wavs:
            inside(wav, source_dir)
            for source in (wav, *(wav.with_suffix(suffix) for suffix in SIDECARS)):
                if not os.path.lexists(source):
                    continue
                inside(source, source_dir)
                target = inside(target_dir / source.name, inbox)
                if os.path.lexists(target):
                    raise FileExistsError(f'导入目标已有同名文件，不会覆盖：{target}')
                plan.append((source, target))
            old_state, new_state = state_path(wav), state_path(target_dir / wav.name)
            if old_state != new_state and old_state.exists():
                inside(old_state, root)
                inside(new_state, root)
                list(tree_files(old_state))
                if os.path.lexists(new_state):
                    raise FileExistsError(f'新旧断点同时存在，不自动覆盖：{new_state}')
                plan.append((old_state, new_state))
        target_dir.mkdir(exist_ok=True)
        moved = []
        try:
            for source, target in plan:
                # Revalidate immediately before mutating the filesystem.
                inside(source, root)
                inside(target, root)
                if os.path.lexists(target):
                    raise FileExistsError(f'目标刚被创建，不会覆盖：{target}')
                os.rename(source, target)
                moved.append((source, target))
        except BaseException:
            for source, target in reversed(moved):
                if os.path.lexists(source):
                    raise RuntimeError(f'导入回滚受阻，请保留两处文件检查：{source}；{target}')
                os.rename(target, source)
            raise
        imported += len(wavs)
    return imported
