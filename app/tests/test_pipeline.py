import json
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import wave

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_audio_sniffing_rejects_html_and_accepts_renamed_m4a(self):
        source = self.root / 'renamed.wav'
        source.write_bytes(b'\x00\x00\x00\x1cftypM4A ' + bytes(20))
        self.assertFalse(p.validate_wav(source))
        source.write_bytes(b'<html>download error</html>')
        with self.assertRaises(ValueError):
            p.validate_wav(source)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required for actual M4A decoding test')
    def test_actual_renamed_m4a_decodes_without_modifying_original(self):
        original = self.root / 'original.wav'
        with wave.open(str(original), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(b'\x10\x00' * 32000)
        renamed = self.root / '日文音声.wav'
        exe = shutil.which('ffmpeg')
        subprocess.run([exe, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(original), '-c:a', 'aac', '-f', 'ipod', str(renamed)], check=True)
        before = renamed.read_bytes()
        output = self.root / 'decoded.wav'
        # Use an isolated runtime layout with the actual configured executable.
        runtime = self.root / 'runtime'
        (runtime / 'bin').mkdir(parents=True)
        shutil.copy2(exe, runtime / 'bin/ffmpeg.exe')
        with patch.object(p, 'TEMP', self.root), patch.object(p, 'RUNTIME', runtime):
            p.decode_audio(renamed, output, None)
        self.assertTrue(p.validate_wav(output))
        self.assertEqual(renamed.read_bytes(), before)
        if sf is not None:
            info = sf.info(str(output))
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.samplerate, 16000)
            self.assertLess(abs(info.duration - 2.0), 0.1)

    def test_only_chinese_srt_is_exported_with_unicode_and_long_timestamps(self):
        wav = self.root / '耳语 & 100% [01].wav'
        cues = [p.Cue(1234, 2345, 'こんにちは。'), p.Cue(3661999, 3662500, 'また\n会おう。')]
        ja = wav.with_suffix('.ja.srt')
        p.atomic_text(ja, p.srt_text(cues), bom=True)
        self.assertEqual(p.parse_srt(ja), cues)
        p.export_subtitles(wav, cues, ['你好。', '再见。'])
        zh = p.parse_srt(wav.with_suffix('.srt'))
        self.assertEqual([(x.start, x.end) for x in zh], [(x.start, x.end) for x in cues])
        self.assertEqual(zh[1].text, '再见。')
        self.assertFalse(wav.with_suffix('.bi.srt').exists())
        self.assertFalse(wav.with_suffix('.lrc').exists())
        self.assertEqual(p.parse_srt(ja), cues)  # Existing user inputs are preserved.

    def test_wrong_line_count_does_not_create_final_files(self):
        wav = self.root / 'a.wav'
        with self.assertRaises(ValueError):
            p.export_subtitles(wav, [p.Cue(0, 1000, 'a')], [])
        self.assertFalse(wav.with_suffix('.srt').exists())

    def test_corrupt_timing_is_rejected(self):
        path = self.root / 'bad.srt'
        p.atomic_text(path, '1\n00:00:02,000 --> 00:00:01,000\n文本\n')
        with self.assertRaises(ValueError):
            p.parse_srt(path)

    def test_japanese_only_resumes_and_chinese_skips(self):
        wav = self.root / 'a.wav'
        wav.write_bytes(b'RIFF')
        p.atomic_text(wav.with_suffix('.ja.srt'), p.srt_text([p.Cue(0, 1000, 'あ')]))
        with patch.object(p, 'STATE', self.root / '.state'):
            self.assertFalse(p.has_subtitle(wav))
            p.export_subtitles(wav, [p.Cue(0, 1000, 'あ')], ['啊'])
            self.assertTrue(p.has_subtitle(wav))

    def test_empty_outputs_skip_only_with_matching_source_marker(self):
        wav = self.root / 'empty.wav'
        wav.write_bytes(b'RIFF')
        with patch.object(p, 'STATE', self.root / '.state'):
            p.export_subtitles(wav, [], [])
            self.assertFalse(p.has_subtitle(wav))
            p.save_json(p.state_dir(wav) / 'empty.done.json', p.source_signature(wav))
            self.assertTrue(p.has_subtitle(wav))
            wav.write_bytes(b'RIFFchanged')
            self.assertFalse(p.has_subtitle(wav))

    def test_download_offline_never_opens_network(self):
        with patch.object(p.request, 'urlopen') as network:
            with self.assertRaisesRegex(RuntimeError, '离线'):
                p.download('https://example.invalid/model', self.root / 'missing', offline=True)
            network.assert_not_called()

    def test_cached_download_avoids_network(self):
        target = self.root / 'model.bin'
        target.write_bytes(b'weights')
        with patch.object(p.request, 'urlopen') as network:
            p.download('https://example.invalid/model', target)
            network.assert_not_called()

    def test_temp_vocals_removed_on_asr_failure(self):
        wav = self.root / 'source.wav'
        with wave.open(str(wav), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(b'\x00\x00' * 100)
        original = wav.read_bytes()
        tempdir = self.root / '.temp'
        tempdir.mkdir()
        calls = []

        def worker(job, stage, offline, source=None, output=None):
            calls.append(stage)
            if stage == 'separate':
                vocal = output / 'vocals.wav'
                vocal.write_bytes(original)
                p.save_json(output / 'vocals.json', dict(path=str(vocal)))
            else:
                raise RuntimeError('simulated OOM')

        with patch.object(p, 'TEMP', tempdir), patch.object(p, 'STATE', self.root / '.state'), \
                patch.object(p, 'run_worker', side_effect=worker):
            with self.assertRaisesRegex(RuntimeError, 'simulated OOM'):
                p.legacy_process_audio(wav, None, True)
        self.assertEqual(calls, ['separate', 'whisper'])
        self.assertEqual(list(tempdir.iterdir()), [])
        self.assertEqual(wav.read_bytes(), original)

    def test_wav_change_does_not_silently_reuse_managed_japanese(self):
        wav = self.root / 'source.wav'
        wav.write_bytes(b'RIFF\x00\x00\x00\x00WAVE')
        with patch.object(p, 'STATE', self.root / '.state'):
            p.atomic_text(wav.with_suffix('.ja.srt'), p.srt_text([p.Cue(0, 1000, 'あ')]))
            p.save_json(p.state_dir(wav) / 'japanese.json', dict(name='old'))
            with self.assertRaisesRegex(RuntimeError, '源音频已变更'):
                p.process_audio(wav, None, True)

    def test_quality_default_uses_raw_audio_and_resumes_matching_profile(self):
        import quality_asr
        wav = self.root / 'source.wav'
        wav.write_bytes(b'RIFF\x00\x00\x00\x00WAVE')
        original = wav.read_bytes()
        cues = [p.Cue(100, 1100, '聞こえますか。')]

        def worker(job, stage, offline, source, output):
            self.assertEqual(stage, 'transcribe')
            self.assertEqual(source, wav)
            self.assertTrue(offline)
            p.atomic_text(output, p.srt_text(cues), bom=True)

        with patch.object(p, 'STATE', self.root / '.state'), \
                patch.object(p, 'run_worker', side_effect=worker) as run:
            self.assertEqual(p.process_audio(wav, None, True), cues)
            self.assertEqual(p.process_audio(wav, None, True), cues)
            self.assertEqual(run.call_count, 1)
            state = p.state_dir(wav)
            self.assertFalse(wav.with_suffix('.ja.srt').exists())
            self.assertEqual(p.parse_srt(p.japanese_path(wav)), cues)
            self.assertEqual(p.read_json(state / 'japanese.profile.json')['profile'], quality_asr.PROFILE)
            self.assertEqual(p.read_json(state / 'japanese.json'), p.source_signature(wav))
        self.assertEqual(wav.read_bytes(), original)

    def test_legacy_japanese_moves_into_private_cache_without_changing_hash_or_input(self):
        wav = self.root / 'source.wav'
        wav.write_bytes(b'RIFF\x00\x00\x00\x00WAVE')
        legacy = wav.with_suffix('.ja.srt')
        original = '1\r\n00:00:00,000 --> 00:00:01,000\r\nこんにちは。\r\n'.encode('utf-8')
        legacy.write_bytes(original)
        with patch.object(p, 'STATE', self.root / '.state'), patch.object(p, 'run_worker') as worker:
            cues = p.process_audio(wav, None, True)
            worker.assert_not_called()
            self.assertEqual(p.japanese_path(wav).read_bytes(), original)
            self.assertEqual(p.parse_srt(p.japanese_path(wav)), cues)
        self.assertEqual(legacy.read_bytes(), original)

    def test_rebuild_backs_up_private_japanese_without_colliding_with_chinese(self):
        wav = self.root / 'japanese.wav'
        wav.write_bytes(b'original')
        wav.with_suffix('.srt').write_bytes(b'Chinese')
        with patch.object(p, 'STATE', self.root / '.state'):
            p.atomic_text(p.japanese_path(wav), 'Japanese')
            p.backup_for_rebuild([p.packages.Package(self.root, (wav,))])
            history = next(p.state_dir(wav).glob('subtitle-history-*'))
            self.assertEqual((history/'japanese.srt').read_bytes(), b'Chinese')
            self.assertEqual((history/'cache/japanese.srt').read_bytes(), b'Japanese')
            self.assertFalse(p.japanese_path(wav).exists())
        self.assertEqual(wav.read_bytes(), b'original')

    def test_rebuild_preserves_source_assets_and_backs_up_subtitles_and_checkpoints(self):
        package = self.root / 'work'
        package.mkdir()
        wav = package / '01.wav'
        wav.write_bytes(b'original audio')
        cover = package / 'cover.jpg'
        cover.write_bytes(b'original cover')
        with patch.object(p, 'STATE', self.root / '.state'):
            state = p.state_dir(wav)
            p.atomic_text(wav.with_suffix('.srt'), 'old Chinese')
            p.atomic_text(wav.with_suffix('.ja.srt'), 'old Japanese')
            p.save_json(state / 'asr-v2.json', dict(next_window=3))
            p.save_json(state / 'translation.json', dict(translations=['旧译文']))
            p.backup_for_rebuild([p.packages.Package(package, (wav,))])
            histories = list(state.glob('subtitle-history-*'))
            self.assertEqual(len(histories), 1)
            self.assertEqual({f.name for f in histories[0].iterdir()},
                             {'01.srt', '01.ja.srt', 'asr-v2.json', 'translation.json'})
            self.assertFalse(wav.with_suffix('.srt').exists())
            self.assertFalse((state / 'asr-v2.json').exists())
        self.assertEqual(wav.read_bytes(), b'original audio')
        self.assertEqual(cover.read_bytes(), b'original cover')

    def test_rebuild_partial_move_failure_restores_original_subtitles(self):
        package = self.root / 'work'
        package.mkdir()
        wav = package / '01.wav'
        wav.write_bytes(b'original audio')
        zh, ja = wav.with_suffix('.srt'), wav.with_suffix('.ja.srt')
        zh.write_bytes(b'Chinese')
        ja.write_bytes(b'Japanese')
        rename = Path.rename

        def fail_second(path, target):
            if path == ja:
                raise OSError('simulated locked file')
            return rename(path, target)

        with patch.object(p, 'STATE', self.root / '.state'), \
                patch.object(Path, 'rename', fail_second):
            with self.assertRaisesRegex(OSError, 'simulated locked file'):
                p.backup_for_rebuild([p.packages.Package(package, (wav,))])
        self.assertEqual(zh.read_bytes(), b'Chinese')
        self.assertEqual(ja.read_bytes(), b'Japanese')
        self.assertEqual(wav.read_bytes(), b'original audio')

    @unittest.skipIf(sf is None, 'optional numerical audio dependencies not installed')
    def test_chunked_audio_retains_samples_and_removes_intermediates(self):
        # An identity separator makes dropped/duplicated overlap samples visible.
        rate = 16000
        signal = np.linspace(-0.5, 0.5, rate * 5 + 17, dtype=np.float32)
        wav = self.root / 'long.wav'
        sf.write(str(wav), signal, rate, subtype='FLOAT')
        output = self.root / 'separation'
        output.mkdir()
        calls = []

        class IdentitySeparator:
            def separate(self, source):
                samples, sample_rate = sf.read(source, dtype='float32', always_2d=True)
                calls.append(len(samples))
                target = output / 'chunk-vocals.wav'
                sf.write(str(target), samples, sample_rate, subtype='FLOAT')
                return [str(target)]

        actual_ranges = p.chunk_ranges
        with patch.object(p, 'chunk_ranges', side_effect=lambda n, r: actual_ranges(n, r, seconds=2, overlap=0.25)):
            result = p.separate_long_wav(IdentitySeparator(), wav, output)
        actual, actual_rate = sf.read(str(result), dtype='float32')
        self.assertEqual(actual_rate, rate)
        self.assertEqual(len(calls), 3)
        np.testing.assert_array_equal(actual, signal)
        self.assertEqual(list(output.iterdir()), [result])


class TranslationTest(unittest.TestCase):
    def test_close_ends_owned_tree_even_when_http_unload_fails(self):
        client = object.__new__(p.Ollama)
        process, scope, log = Mock(), Mock(), Mock()
        process.poll.return_value = None
        client.process, client.scope, client.log = process, scope, log
        client.memory_before = None
        client.unload = Mock(side_effect=RuntimeError('connection closed'))
        client.close()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=10)
        scope.close.assert_called_once_with()
        log.close.assert_called_once_with()
        self.assertIsNone(client.process)
        self.assertIsNone(client.scope)
        client.close()  # Cleanup must also be safe after a failed constructor.

    def test_exact_model_selection(self):
        self.assertEqual(p.select_sakura(8188), p.SAKURA14)
        self.assertEqual(p.select_sakura(10240), p.SAKURA14)
        self.assertIn('Q4_K_M', p.SAKURA14)
        self.assertIn('q6k.gguf', p.SAKURA7)

    def test_official_chatml_and_sampling_payload(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA7
        client.api = Mock(return_value=dict(response='你好'))
        self.assertEqual(client.generate(['こんにちは']), '你好')
        endpoint, payload = client.api.call_args.args
        self.assertEqual(endpoint, '/api/generate')
        self.assertTrue(payload['raw'])
        self.assertFalse(payload['stream'])
        self.assertIn('<|im_start|>system\n' + p.SYSTEM + '<|im_end|>', payload['prompt'])
        self.assertIn('将下面的日文文本翻译成中文：こんにちは', payload['prompt'])
        for key, value in [('temperature', 0.1), ('top_p', 0.3), ('frequency_penalty', 0.2)]:
            self.assertEqual(payload['options'][key], value)

    def test_batches_have_five_context_lines_without_duplicate_cues(self):
        sent = []
        class Client:
            model = 'test'
            def generate(self, lines):
                sent.append(lines.copy())
                return '\n'.join('中' + x for x in lines)
        cues = [p.Cue(i * 1000, i * 1000 + 900, str(i)) for i in range(43)]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'translation.json'
            client = Client()
            result = p.translate_all(client, cues, checkpoint, 'hash')
            self.assertEqual(result, ['中' + str(i) for i in range(43)])
            self.assertEqual([len(x) for x in sent], [20, 25, 8])
            self.assertEqual(sent[1][:5], ['15', '16', '17', '18', '19'])
            sent.clear()
            self.assertEqual(p.translate_all(client, cues, checkpoint, 'hash'), result)
            self.assertEqual(sent, [])
            p.translate_all(client, cues, checkpoint, 'changed-japanese')
            self.assertEqual(len(sent), 3)

    def test_misaligned_output_retries_then_splits(self):
        class Client:
            def generate(self, lines):
                if len(lines) > 8:
                    return 'wrong merged line'
                return '\n'.join('中' + x for x in lines)
        with tempfile.TemporaryDirectory() as tmp:
            result = p.translate_range(Client(), list(map(str, range(20))), 0, 20, Path(tmp))
            self.assertEqual(result, ['中' + str(i) for i in range(20)])

    def test_persistent_misalignment_never_pads_or_truncates(self):
        client = Mock()
        client.generate.return_value = '一行\n两行'
        client.generate_focused.return_value = '一行\n两行'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(p.AlignmentError):
                p.translate_range(client, ['ひとつ'], 0, 1, Path(tmp))

    def test_single_target_keeps_context_in_a_separate_completed_turn(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA7
        client.api = Mock(side_effect=[dict(response='感谢到场'), dict(response='谢谢您')])
        self.assertEqual(client.generate_focused('ありがとうございます', ['来てくれて']), '谢谢您')
        payload = client.api.call_args.args[1]
        self.assertIn('<|im_start|>assistant\n感谢到场<|im_end|>', payload['prompt'])
        self.assertTrue(payload['prompt'].endswith('将下面的日文文本翻译成中文：ありがとうございます<|im_end|>\n<|im_start|>assistant\n'))
        for key in ('temperature', 'top_p', 'frequency_penalty'):
            self.assertEqual(payload['options'][key], p.OPTIONS[key])
        self.assertLess(payload['options']['num_predict'], 2048)

    def test_completed_context_is_reused_without_regeneration(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA7
        client.api = Mock(return_value=dict(response='谢谢您'))
        result = client.generate_contextual(['ありがとうございます'], ['来てくれて'], ['谢谢到场'])
        self.assertEqual(result, '谢谢您')
        client.api.assert_called_once()
        prompt = client.api.call_args.args[1]['prompt']
        self.assertIn('<|im_start|>assistant\n谢谢到场<|im_end|>', prompt)

    def test_truncated_response_is_retained_for_diagnosis(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA7
        client.api = Mock(return_value=dict(response='重复内容', done_reason='length'))
        with self.assertRaises(p.OutputLimitError) as caught:
            client.generate(['あ'])
        self.assertEqual(caught.exception.response, '重复内容')
        budget = client.api.call_args.args[1]['options']['num_predict']
        self.assertEqual(budget, 128)

    def test_completed_sub_batches_survive_a_later_failure(self):
        class Client:
            model = 'test'
            def generate(self, lines):
                return 'bad merged output' if '10' in lines else '\n'.join('中' + x for x in lines)
        cues = [p.Cue(i * 1000, i * 1000 + 900, str(i)) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'translation.json'
            with self.assertRaises(p.AlignmentError):
                p.translate_all(Client(), cues, checkpoint, 'hash')
            self.assertEqual(p.read_json(checkpoint)['translations'], ['中' + str(i) for i in range(10)])

    def test_merged_context_uses_focused_translation_without_padding(self):
        client = Mock()
        client.generate.return_value = '合并后的上下文与本句'
        client.generate_focused.return_value = '谢谢您'
        with tempfile.TemporaryDirectory() as tmp:
            result = p.translate_range(client, ['来てくれて', 'ありがとうございます'], 1, 2, Path(tmp))
        self.assertEqual(result, ['谢谢您'])
        client.generate_focused.assert_called_once_with('ありがとうございます', ['来てくれて'])

    def test_cached_ollama_model_does_not_pull(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA7
        client.offline = True
        client.api = Mock(return_value={'models': [{'name': p.SAKURA7.lower()}]})
        client.ensure_model()
        client.api.assert_called_once_with('/api/tags')

    def test_ollama_proxy_dns_error_uses_same_official_gguf_import(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA7
        client.offline = False
        client.api = Mock(return_value={'models': []})
        client.pull_model = Mock(side_effect=RuntimeError('redirect target not allowed: non-public 198.18.0.1'))
        client.import_official_gguf = Mock()
        with self.assertLogs(p.LOG, level='WARNING'):
            client.ensure_model()
        client.import_official_gguf.assert_called_once_with()

    def test_ollama_unrelated_pull_failure_is_not_hidden(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA7
        client.offline = False
        client.api = Mock(return_value={'models': []})
        client.pull_model = Mock(side_effect=RuntimeError('disk full'))
        client.import_official_gguf = Mock()
        with self.assertRaisesRegex(RuntimeError, 'disk full'):
            client.ensure_model()
        client.import_official_gguf.assert_not_called()

    def test_14b_unavailable_http_alias_imports_exact_official_weights(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA14
        client.offline = False
        client.base = 'http://127.0.0.1:12345'
        client.api = Mock(return_value={'models': []})
        client.import_official_gguf = Mock()
        failure = p.error.HTTPError(client.base, 400, 'Bad Request', {},
                                   io.BytesIO(b'{"error":"specified tag is not available"}'))
        with patch.object(p.HTTP_LOCAL, 'open', side_effect=failure), self.assertLogs(p.LOG, level='WARNING'):
            client.ensure_model()
        client.import_official_gguf.assert_called_once_with()

    def test_pull_http_disk_error_does_not_trigger_another_download(self):
        client = object.__new__(p.Ollama)
        client.model = p.SAKURA14
        client.offline = False
        client.base = 'http://127.0.0.1:12345'
        client.api = Mock(return_value={'models': []})
        client.import_official_gguf = Mock()
        failure = p.error.HTTPError(client.base, 500, 'Internal Server Error', {},
                                   io.BytesIO(b'{"error":"disk full"}'))
        with patch.object(p.HTTP_LOCAL, 'open', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                client.ensure_model()
        client.import_official_gguf.assert_not_called()


@unittest.skipUnless(sys.platform == 'win32', 'Windows process lifecycle test')
class WindowsJobTest(unittest.TestCase):
    def test_nested_job_ends_orphaned_runner_before_next_track(self):
        import ctypes
        from ctypes import wintypes
        outer, stage = p.ChildJob(), p.ChildJob()
        process = handle = None
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        try:
            with tempfile.TemporaryDirectory() as folder:
                gate, record = Path(folder)/'go', Path(folder)/'pid'
                script = ('from pathlib import Path; import subprocess,sys,time; '
                          'gate=Path(sys.argv[1]); record=Path(sys.argv[2]); '
                          '\nwhile not gate.exists(): time.sleep(.02)\n'
                          'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"]); '
                          'record.write_text(str(child.pid)); time.sleep(120)')
                process = subprocess.Popen([sys.executable, '-c', script, str(gate), str(record)],
                                           creationflags=p.NO_WINDOW)
                outer.add(process)
                stage.add(process)
                gate.write_text('go')
                deadline = time.monotonic() + 10
                while not record.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail('Test child did not start its runner')
                    time.sleep(.02)
                pid = int(record.read_text())
                handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
                self.assertTrue(handle)
                process.terminate()
                process.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(handle, 0), 258)  # Runner is still alive.
                stage.close()
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
                deadline = time.monotonic() + 2
                while outer.active_processes() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(outer.active_processes(), 0)
        finally:
            stage.close()
            outer.close()
            if process is not None:
                process.wait(timeout=5)
            if handle:
                kernel.CloseHandle(handle)

    def test_owned_child_ends_when_job_handle_closes(self):
        job = p.ChildJob()
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   creationflags=p.NO_WINDOW)
        try:
            job.add(process)
            job.close()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            job.close()


if __name__ == '__main__':
    unittest.main()
