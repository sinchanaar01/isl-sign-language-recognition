import unittest

import numpy as np

from src.isl_recognition.preprocessing import (
    FEATURES_PER_FRAME,
    augment_sequence,
    normalize_frame,
    normalize_sequence,
    signer_independent_split,
)


class PreprocessingTests(unittest.TestCase):
    def frame(self):
        frame = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
        frame[11 * 3:11 * 3 + 3] = [0.4, 0.3, 0.1]
        frame[12 * 3:12 * 3 + 3] = [0.6, 0.3, 0.1]
        frame[99 + 11] = 1
        frame[99 + 12] = 1
        frame[132:135] = [0.6, 0.4, 0.2]
        return frame

    def test_normalization_centers_shoulders_and_scales_width(self):
        normalized = normalize_frame(self.frame())
        pose = normalized[:99].reshape(33, 3)
        self.assertAlmostEqual(float(pose[[11, 12], :2].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(pose[11, :2] - pose[12, :2])), 1.0)

    def test_sequence_resamples_to_fixed_length(self):
        frames = np.stack([self.frame(), self.frame() * 2, self.frame() * 3])
        result = normalize_sequence(frames, 8)
        self.assertEqual(result.shape, (8, FEATURES_PER_FRAME))
        self.assertTrue(np.isfinite(result).all())

    def test_single_frame_is_repeated(self):
        result = normalize_sequence(self.frame()[None, :], 4)
        self.assertEqual(result.shape, (4, FEATURES_PER_FRAME))
        np.testing.assert_allclose(result[0], result[-1])

    def test_invalid_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_sequence(np.empty((0, FEATURES_PER_FRAME)), 4)

    def test_augmentation_is_deterministic_for_seed(self):
        sequence = np.stack([self.frame(), self.frame()])
        first = augment_sequence(sequence, np.random.default_rng(9))
        second = augment_sequence(sequence, np.random.default_rng(9))
        np.testing.assert_array_equal(first, second)

    def test_signers_do_not_cross_splits(self):
        signers = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
        first = signer_independent_split(signers, seed=7)
        second = signer_independent_split(signers, seed=7)
        np.testing.assert_array_equal(first, second)
        by_signer = {}
        for signer, split in zip(signers, first):
            by_signer.setdefault(signer, set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in by_signer.values()))
        self.assertEqual(set(first), {"train", "val", "test"})

    def test_split_requires_three_signers(self):
        with self.assertRaises(ValueError):
            signer_independent_split(["a", "a", "b"])


if __name__ == "__main__":
    unittest.main()
