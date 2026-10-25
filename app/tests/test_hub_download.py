import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest

import hub_download as d


class RangeDownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = bytes(range(256)) * 1024
        self.calls = []
        payload, calls = self.payload, self.calls
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', self.headers['Range']).groups())
                calls.append((start, end))
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(payload)}')
                self.send_header('Content-Length', str(end - start + 1))
                self.end_headers()
                self.wfile.write(payload[start:end + 1])
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/model.bin'
        self.sha = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_parallel_assembly_checksum_and_cached_repeat(self):
        target = self.root / 'model.bin'
        d.range_download(self.url, target, len(self.payload), self.sha, workers=3, chunk_size=65536)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(len(self.calls), 4)
        self.assertFalse(target.with_name('model.bin.parts').exists())
        d.range_download(self.url, target, len(self.payload), self.sha, workers=3, chunk_size=65536)
        self.assertEqual(len(self.calls), 4)

    def test_completed_ranges_resume_without_redownload(self):
        target = self.root / 'model.bin'
        parts = self.root / 'model.bin.parts'
        parts.mkdir()
        (parts / 'metadata.json').write_text(json.dumps(dict(size=len(self.payload), sha256=self.sha, chunk_size=65536)))
        (parts / '000000.chunk').write_bytes(self.payload[:65536])
        d.range_download(self.url, target, len(self.payload), self.sha, workers=3, chunk_size=65536)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(len(self.calls), 3)
        self.assertNotIn((0, 65535), self.calls)

    def test_bad_official_checksum_never_publishes_model(self):
        target = self.root / 'model.bin'
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            d.range_download(self.url, target, len(self.payload), '0' * 64, workers=3, chunk_size=65536)
        self.assertFalse(target.exists())
