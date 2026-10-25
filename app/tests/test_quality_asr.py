from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_asr as q


class RegionsTest(unittest.TestCase):
    def test_cuda_handoff_waits_for_recovery_without_reducing_budget(self):
        torch = Mock()
        torch.cuda.mem_get_info.side_effect = [(5 * 2**30, 8 * 2**30), (6 * 2**30, 8 * 2**30)]
        with patch.object(q.time, 'sleep') as sleep:
            budget, total = q.reserve_cuda_budget(torch, timeout=5)
        self.assertEqual(budget, 6 * 2**30 - 768 * 2**20)
        self.assertEqual(total, 8 * 2**30)
        sleep.assert_called_once_with(1)

    def test_cuda_handoff_timeout_keeps_the_full_model_requirement(self):
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (5 * 2**30, 8 * 2**30)
        with self.assertRaisesRegex(RuntimeError, '未降低模型或精度'):
            q.reserve_cuda_budget(torch, timeout=0)

    def test_short_noise_is_not_promoted_to_speech(self):
        regions = q.mask_regions([False, True, False] + [True] * 6 + [False], 10)
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0][0], 10.06)
        self.assertAlmostEqual(regions[0][1], 10.18)

    def test_windows_cover_speech_without_overlap_or_unbounded_duration(self):
        windows = q.speech_windows([(1, 7), (7.3, 28), (29, 30)], 30)
        self.assertGreaterEqual(len(windows), 3)
        self.assertGreaterEqual(windows[0][0], 0)
        self.assertLessEqual(windows[-1][1], 30)
        for start, end in windows:
            self.assertLessEqual(end - start, 12.001)
            self.assertGreater(end, start)
        for left, right in zip(windows, windows[1:]):
            self.assertLessEqual(left[1], right[0])

    def test_unfinished_adjacent_phrase_is_reconstructed_without_added_words(self):
        original = [dict(start=0, end=1000, text='来てくれて'),
                    dict(start=1200, end=2200, text='ありがとうございます。'),
                    dict(start=2300, end=3000, text='またね。')]
        cues = q.sentence_cues(original)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0], dict(start=0, end=2200, text='来てくれてありがとうございます。'))
        self.assertEqual(''.join(x['text'] for x in original), ''.join(x['text'] for x in cues))

    def test_repetition_and_uncertain_cues_are_not_silently_removed(self):
        rows = [dict(start=0, end=1000, text='だめ。'), dict(start=1200, end=2200, text='だめ。'),
                dict(start=2400, end=3000, text='［聞き取り不明］', review=True)]
        self.assertEqual(q.sentence_cues(rows), rows)
        self.assertIsNone(q.suspicious('だめ、だめ、だめ。'))
        self.assertIsNotNone(q.suspicious('あ' * 90))
        self.assertIsNotNone(q.suspicious('途中の文', exhausted=True))

    def test_long_pause_never_turns_into_one_displayed_sentence(self):
        rows = [dict(start=0, end=1000, text='今日は'), dict(start=7000, end=8000, text='ありがとう')]
        self.assertEqual(q.sentence_cues(rows), rows)

    def test_word_alignment_preserves_text_and_punctuation(self):
        words = [dict(text='来て', start_time=0.1, end_time=0.5),
                 dict(text='くれて', start_time=0.5, end_time=1.0),
                 dict(text='ありがとう', start_time=1.1, end_time=2.0),
                 dict(text='またね', start_time=3, end_time=4)]
        text = '来てくれてありがとう。またね！'
        cues = q.aligned_sentences(text, words, 10, 5)
        self.assertEqual(''.join(c['text'] for c in cues), text)
        self.assertEqual(len(cues), 2)
        self.assertEqual((cues[0]['start'], cues[0]['end']), (10100, 12000))

    def test_incompatible_alignment_is_rejected_instead_of_dropping_words(self):
        with self.assertRaises(ValueError):
            q.aligned_sentences('こんにちは。', [dict(text='別の単語', start_time=0, end_time=1)], 0, 2)


if __name__ == '__main__':
    unittest.main()
