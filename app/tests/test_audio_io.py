from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None

import audio_io


@unittest.skipIf(sf is None, 'audio dependencies not installed')
class StemWriterTest(unittest.TestCase):
    def test_quiet_stereo_keeps_samples_channels_and_absolute_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'stems'
            separator = SimpleNamespace(output_dir=str(output), sample_rate=44100)
            source = np.array([[0.00001, -0.00002], [0.12345, -0.45678], [0.2, 0.3]], dtype=np.float32)
            audio_io.write_stem(separator, 'quiet.wav', source)
            target = output / 'quiet.wav'
            self.assertTrue(target.is_file())
            data, rate = sf.read(str(target), dtype='float32', always_2d=True)
            self.assertEqual(rate, 44100)
            np.testing.assert_array_equal(data, source)
            self.assertEqual(sf.info(str(target)).frames, 3)
            self.assertEqual(sf.info(str(target)).channels, 2)

    def test_write_errors_are_propagated_and_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            separator = SimpleNamespace(output_dir=tmp, sample_rate=44100)
            with patch.object(sf, 'write', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    audio_io.write_stem(separator, 'quiet.wav', np.ones((10, 2), dtype=np.float32))

    def test_output_cannot_escape_private_temporary_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            separator = SimpleNamespace(output_dir=str(Path(tmp) / 'stems'), sample_rate=44100)
            with self.assertRaisesRegex(ValueError, '越界'):
                audio_io.write_stem(separator, '../outside.wav', np.ones((10, 2), dtype=np.float32))
