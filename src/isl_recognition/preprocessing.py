"""Landmark normalization, sequence preparation, augmentation, and data splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

POSE_XYZ = 33 * 3
POSE_VISIBILITY = 33
HAND_XYZ = 21 * 3
FEATURES_PER_FRAME = POSE_XYZ + POSE_VISIBILITY + 2 * HAND_XYZ


def normalize_frame(frame: Sequence[float]) -> np.ndarray:
    """Center landmarks on the shoulders and scale by shoulder width.

    Layout: pose xyz (33), pose visibility (33), left hand xyz (21),
    right hand xyz (21). Visibility is retained unchanged.
    """
    values = np.asarray(frame, dtype=np.float32)
    if values.shape != (FEATURES_PER_FRAME,):
        raise ValueError(
            f"Expected {FEATURES_PER_FRAME} values per frame, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Landmark frame contains NaN or infinite values")

    output = values.copy()
    pose = output[:POSE_XYZ].reshape(33, 3)
    visibility = output[POSE_XYZ:POSE_XYZ + POSE_VISIBILITY]
    hands = output[POSE_XYZ + POSE_VISIBILITY:].reshape(2, 21, 3)

    if visibility[11] <= 0 or visibility[12] <= 0:
        visible_shoulders = pose[[11, 12], :2]
        center = visible_shoulders.mean(axis=0)
        scale = 1.0
    else:
        center = pose[[11, 12], :2].mean(axis=0)
        scale = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
        if scale < 1e-6:
            scale = 1.0

    pose[:, :2] = (pose[:, :2] - center) / scale
    pose[:, 2] /= scale
    hands[:, :, :2] = (hands[:, :, :2] - center) / scale
    hands[:, :, 2] /= scale
    return output


def normalize_sequence(frames: np.ndarray, sequence_length: int) -> np.ndarray:
    """Normalize frames and uniformly resample/pad a clip to fixed length."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != FEATURES_PER_FRAME:
        raise ValueError(
            f"Expected (frames, {FEATURES_PER_FRAME}) landmarks, got {frames.shape}"
        )
    if len(frames) == 0:
        raise ValueError("Cannot normalize an empty clip")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    normalized = np.stack([normalize_frame(frame) for frame in frames])
    if len(normalized) == sequence_length:
        return normalized
    if len(normalized) == 1:
        return np.repeat(normalized, sequence_length, axis=0)

    positions = np.linspace(0, len(normalized) - 1, sequence_length)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(normalized) - 1)
    fraction = (positions - lower)[:, None]
    return normalized[lower] * (1.0 - fraction) + normalized[upper] * fraction


def augment_sequence(
    sequence: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.01,
    hand_dropout_probability: float = 0.1,
) -> np.ndarray:
    """Apply label-preserving landmark noise and occasional hand dropout."""
    values = np.asarray(sequence, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != FEATURES_PER_FRAME:
        raise ValueError(f"Expected sequence shape (T, {FEATURES_PER_FRAME})")
    if noise_std < 0 or not 0 <= hand_dropout_probability <= 1:
        raise ValueError("Invalid augmentation probability or noise standard deviation")
    xyz_indices = np.r_[:POSE_XYZ, POSE_XYZ + POSE_VISIBILITY:FEATURES_PER_FRAME]
    values[:, xyz_indices] += rng.normal(
        0.0, noise_std, size=(len(values), len(xyz_indices))
    ).astype(np.float32)
    hand_start = POSE_XYZ + POSE_VISIBILITY
    for hand in range(2):
        if rng.random() < hand_dropout_probability:
            start = hand_start + hand * HAND_XYZ
            values[:, start:start + HAND_XYZ] = 0
    return values


def signer_independent_split(
    signer_ids: Sequence[str],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> np.ndarray:
    """Assign each signer wholly to train/validation/test, preserving no leakage."""
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1")
    if any(not signer for signer in signer_ids):
        raise ValueError("Every sample must have a non-empty signer ID")

    unique_signers = list(dict.fromkeys(signer_ids))
    if len(unique_signers) < 3:
        raise ValueError("Signer-independent train/val/test requires at least 3 signers")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_signers)
    counts: dict[str, int] = defaultdict(int)
    for signer in signer_ids:
        counts[signer] += 1

    partitions = ("train", "val", "test")
    target_counts = np.cumsum(ratios) * len(signer_ids)
    assignment: dict[str, str] = {}
    assigned = [0, 0, 0]
    for signer in unique_signers:
        partition_index = min(
            range(3),
            key=lambda index: abs(target_counts[index] - (assigned[index] + counts[signer])),
        )
        assignment[signer] = partitions[partition_index]
        assigned[partition_index] += counts[signer]

    # Guarantee all three splits exist even when group sizes are highly uneven.
    absent = [name for name in partitions if name not in assignment.values()]
    for name in absent:
        donor = max(
            (signer for signer in unique_signers if sum(
                1 for current in assignment.values() if current == assignment[signer]
            ) > 1),
            key=lambda signer: counts[signer],
        )
        assignment[donor] = name
    return np.asarray([assignment[signer] for signer in signer_ids])

