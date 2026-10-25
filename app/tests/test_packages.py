from contextlib import ExitStack
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p
import packages as q


class PackageFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        for name, value in [('ROOT', self.root), ('STATE', self.root / '.state')]:
            self.stack.enter_context(patch.object(p, name, value))
        self.inbox, self.outbox = q.layout(self.root)

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def put(self, relative, data=b'unchanged'):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def subtitle(self, wav):
        cues = [p.Cue(0, 1000, 'こんにちは')]
        p.atomic_text(wav.with_suffix('.ja.srt'), p.srt_text(cues), bom=True)
        p.export_subtitles(wav, cues, ['你好'])

    def test_complete_nested_package_moves_with_all_assets_without_gpu(self):
        folder = self.inbox / '作品 A & 100%'
        first = self.put(f'{q.INBOX_NAME}/{folder.name}/CD1/01.wav')
        second = self.put(f'{q.INBOX_NAME}/{folder.name}/CD2/01.wav')
        cover = self.put(f'{q.INBOX_NAME}/{folder.name}/封面/cover.jpg', b'cover bytes')
        self.put(f'{q.INBOX_NAME}/{folder.name}/readme.txt', b'notes')
        (folder / 'empty-folder').mkdir()
        for wav in (first, second):
            self.subtitle(wav)
        original = {file.relative_to(folder): file.read_bytes() for file in q.tree_files(folder)}
        with patch.object(p, 'gpu_info') as gpu, patch.object(p, 'Ollama') as ollama:
            p.coordinate('run', True)
            gpu.assert_not_called()
            ollama.assert_not_called()
        completed = self.outbox / folder.name
        self.assertFalse(folder.exists())
        self.assertEqual({file.relative_to(completed): file.read_bytes() for file in q.tree_files(completed)}, original)
        self.assertTrue((completed / 'empty-folder').is_dir())

    def test_failed_track_keeps_whole_package_then_resumes_only_unfinished(self):
        folder = self.inbox / '作品'
        first = self.put(f'{q.INBOX_NAME}/作品/01.wav')
        second = self.put(f'{q.INBOX_NAME}/作品/sub/02.wav')
        cover = self.put(f'{q.INBOX_NAME}/作品/cover.jpg')
        failed_once = set()
        called = []

        def process(wav, job, offline):
            called.append(wav.name)
            if wav == second and not failed_once:
                failed_once.add(wav)
                raise RuntimeError('simulated ASR failure')
            cues = [p.Cue(0, 1000, 'こんにちは')]
            p.atomic_text(p.japanese_path(wav), p.srt_text(cues), bom=True)
            return cues

        class Translator:
            model = 'test'
            def __init__(self, *args):
                pass
            def generate(self, lines):
                return '\n'.join('你好' for _ in lines)
            def close(self):
                pass

        with patch.object(p, 'gpu_info', return_value={'total': 8188}), \
                patch.object(p, 'ensure_ffmpeg'), patch.object(p, 'ChildJob'), \
                patch.object(p, 'prepare_models'), \
                patch.object(p, 'Ollama', Translator), patch.object(p, 'process_audio', side_effect=process):
            with self.assertLogs(p.LOG, level='ERROR'):
                with self.assertRaisesRegex(RuntimeError, '作品未完成'):
                    p.coordinate('run', True)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(cover.exists())
            self.assertFalse((self.outbox / folder.name).exists())
            self.assertTrue(first.with_suffix('.srt').exists())
            p.coordinate('run', True)
        self.assertEqual(called, ['01.wav', '02.wav', '02.wav'])
        self.assertFalse(folder.exists())
        self.assertTrue((self.outbox / folder.name / 'sub/02.srt').exists())

    def test_existing_completed_package_is_never_overwritten(self):
        wav = self.put(f'{q.INBOX_NAME}/same/01.wav')
        self.subtitle(wav)
        existing = self.put(f'{q.OUTBOX_NAME}/same/keep.txt', b'do not overwrite')
        with patch.object(p, 'gpu_info') as gpu:
            with self.assertLogs(p.LOG, level='ERROR'):
                with self.assertRaises(RuntimeError):
                    p.coordinate('run', True)
            gpu.assert_not_called()
        self.assertTrue(wav.exists())
        self.assertEqual(existing.read_bytes(), b'do not overwrite')

    def test_new_package_has_one_public_chinese_subtitle_and_private_resume_cache(self):
        wav = self.put(f'{q.INBOX_NAME}/新作品/01.wav', b'RIFF\x00\x00\x00\x00WAVE')
        cues = [p.Cue(100, 1100, 'こんにちは。')]

        def worker(job, stage, offline, source, output):
            self.assertEqual(output, p.japanese_path(wav))
            p.atomic_text(output, p.srt_text(cues), bom=True)

        class Translator:
            model = 'test'
            def __init__(self, *args): pass
            def generate(self, lines): return '\n'.join('你好。' for _ in lines)
            def close(self): pass

        with patch.object(p, 'run_worker', side_effect=worker), patch.object(p, 'Ollama', Translator):
            p.process_package(q.scan(wav.parent), None, 'test', True)
        completed = self.outbox/'新作品'
        self.assertEqual({file.name for file in q.tree_files(completed)}, {'01.wav', '01.srt'})
        target = completed/'01.wav'
        self.assertEqual(p.parse_srt(target.with_suffix('.srt')), [p.Cue(100, 1100, '你好。')])
        self.assertEqual(p.parse_srt(p.japanese_path(target)), cues)
        self.assertTrue((p.state_dir(target)/'translation.json').exists())

    def test_added_track_prevents_premature_archive(self):
        first = self.put(f'{q.INBOX_NAME}/work/01.wav')
        self.subtitle(first)
        package = q.scan(first.parent)
        second = self.put(f'{q.INBOX_NAME}/work/02.wav')
        with self.assertRaisesRegex(ValueError, '仍有未完成'):
            q.archive(self.root, package, p.has_subtitle)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_state_is_distinct_for_works_and_discs_but_stable_after_move(self):
        a = self.inbox / 'A/CD1/01.wav'
        b = self.inbox / 'A/CD2/01.wav'
        c = self.inbox / 'B/CD1/01.wav'
        self.assertEqual(len({p.state_dir(a), p.state_dir(b), p.state_dir(c)}), 3)
        self.assertEqual(p.state_dir(a), p.state_dir(self.outbox / 'A/CD1/01.wav'))

    def test_root_import_preserves_bytes_sidecars_file_identity_and_checkpoint(self):
        wav = self.put('legacy.wav', b'original wave bytes')
        self.subtitle(wav)
        wav.with_suffix('.bi.srt').write_bytes(b'legacy bilingual input')
        identity = wav.stat()
        old_state = p.state_dir(wav)
        p.save_json(old_state / 'translation.json', {'keep': 'checkpoint'})
        count = q.import_loose(self.root, p.state_dir)
        target = self.inbox / '根目录导入/legacy.wav'
        self.assertEqual(count, 1)
        self.assertFalse(wav.exists())
        self.assertEqual(target.read_bytes(), b'original wave bytes')
        self.assertEqual((target.stat().st_ino, target.stat().st_size, target.stat().st_mtime_ns),
                         (identity.st_ino, identity.st_size, identity.st_mtime_ns))
        self.assertTrue(target.with_suffix('.bi.srt').exists())
        self.assertEqual(p.read_json(p.state_dir(target) / 'translation.json'), {'keep': 'checkpoint'})
        self.assertFalse(old_state.exists())

    def test_import_conflict_is_detected_before_moving_any_input(self):
        first = self.put('a.wav', b'first')
        second = self.put('b.wav', b'second')
        existing = self.put(f'{q.INBOX_NAME}/根目录导入/b.wav', b'existing')
        with self.assertRaises(FileExistsError):
            q.import_loose(self.root, p.state_dir)
        self.assertEqual(first.read_bytes(), b'first')
        self.assertEqual(second.read_bytes(), b'second')
        self.assertEqual(existing.read_bytes(), b'existing')

    def test_import_rename_failure_rolls_back_prior_moves(self):
        first = self.put('a.wav', b'first')
        second = self.put('b.wav', b'second')
        real_rename = os.rename
        calls = []
        def rename(source, target):
            calls.append(source)
            if len(calls) == 2:
                raise PermissionError('file in use')
            real_rename(source, target)
        with patch.object(q.os, 'rename', side_effect=rename):
            with self.assertRaises(PermissionError):
                q.import_loose(self.root, p.state_dir)
        self.assertEqual(first.read_bytes(), b'first')
        self.assertEqual(second.read_bytes(), b'second')
        self.assertFalse((self.inbox / '根目录导入/a.wav').exists())

    def test_subtitle_output_collision_is_reported(self):
        wav = self.put(f'{q.INBOX_NAME}/work/a.wav')
        other = self.put(f'{q.INBOX_NAME}/work/a.ja.wav')
        with self.assertRaisesRegex(ValueError, '输出名冲突'):
            q.scan(wav.parent)
        self.assertTrue(other.exists())

    def test_legacy_output_collision_prevents_import_before_any_rename(self):
        self.put('a.wav')
        self.put('a.ja.wav')
        with patch.object(q.os, 'rename') as rename:
            with self.assertRaisesRegex(ValueError, '输出名冲突'):
                q.import_loose(self.root, p.state_dir)
            rename.assert_not_called()

    def test_loose_inbox_wav_is_grouped_with_its_subtitles(self):
        wav = self.put(f'{q.INBOX_NAME}/loose.wav')
        self.subtitle(wav)
        wav.with_suffix('.bi.srt').write_bytes(b'legacy bilingual input')
        self.assertEqual(q.import_loose(self.root, p.state_dir), 1)
        target = self.inbox / '散放音频/loose.wav'
        self.assertTrue(target.exists())
        self.assertTrue(target.with_suffix('.bi.srt').exists())
        self.assertFalse(wav.exists())

    def test_empty_or_non_wav_folders_stay_in_inbox(self):
        self.put(f'{q.INBOX_NAME}/cover-only/cover.jpg')
        with patch.object(p, 'gpu_info') as gpu:
            p.coordinate('run', True)
            gpu.assert_not_called()
        self.assertTrue((self.inbox / 'cover-only/cover.jpg').exists())
        self.assertEqual(list(self.outbox.iterdir()), [])

    def test_queue_cannot_escape_using_parent_components(self):
        with self.assertRaises(ValueError):
            q.inside(self.inbox / '..' / 'external', self.inbox)

    def test_model_worker_failure_stops_before_next_track(self):
        self.put(f'{q.INBOX_NAME}/work/01.wav')
        self.put(f'{q.INBOX_NAME}/work/02.wav')
        with patch.object(p, 'gpu_info', return_value={'total': 8188}), \
                patch.object(p, 'ensure_ffmpeg'), patch.object(p, 'ChildJob'), \
                patch.object(p, 'prepare_models'), \
                patch.object(p, 'process_audio', side_effect=p.StageError('CUDA worker failed')) as process:
            with self.assertLogs(p.LOG, level='ERROR'):
                with self.assertRaises(p.StageError):
                    p.coordinate('run', True)
            self.assertEqual(process.call_count, 1)


if __name__ == '__main__':
    unittest.main()
