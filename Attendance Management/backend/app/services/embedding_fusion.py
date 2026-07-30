"""Quality-weighted embedding fusion, shared by face and person tracks.

A single frame of a small, distant face yields an embedding that is largely
noise — measured on these cameras, an 18-20px face scored 0.09 against its own
enrolled identity. The noise is roughly independent frame to frame, so averaging
N observations of the same person cuts it by about sqrt(N).

Weighting by face size means the close-up frames dominate the distant blurry
ones, which also removes any need to explicitly pick a "best" frame.

Lives in its own module because BOTH FaceTrack (entrance cameras, face-only
pipeline) and PersonTrack (room cameras, body-tracking pipeline) need it. The
room cameras need it MORE — a seated person is visible for minutes, so there are
far more observations to average, and their faces are the smallest.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class EmbeddingFuser:
    """Running quality-weighted mean of one track's face embeddings."""

    __slots__ = ("_sum", "_weight", "observations", "best_quality")

    def __init__(self) -> None:
        self._sum: Optional[np.ndarray] = None
        self._weight: float = 0.0
        self.observations: int = 0
        self.best_quality: float = 0.0

    def add(self, embedding, quality: float) -> None:
        """Fold one observation in. Silently ignores unusable input — a bad
        frame must never break tracking."""
        if embedding is None or quality is None or quality <= 0:
            return
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            return
        weighted = vector * float(quality)
        if self._sum is None or self._sum.shape != vector.shape:
            self._sum = weighted
            self._weight = float(quality)
        else:
            self._sum = self._sum + weighted
            self._weight += float(quality)
        self.observations += 1
        self.best_quality = max(self.best_quality, float(quality))

    def fused(self) -> Optional[np.ndarray]:
        """The re-normalised weighted mean, or None if nothing was added.

        Re-normalisation is required: cosine similarity is taken against unit
        enrolled vectors, and the mean of unit vectors is shorter than one, so
        skipping it depresses every fused score.
        """
        if self._sum is None or self._weight <= 0:
            return None
        mean = self._sum / self._weight
        norm = float(np.linalg.norm(mean))
        return (mean / norm) if norm > 0 else None

    def reset(self) -> None:
        self._sum = None
        self._weight = 0.0
        self.observations = 0
        self.best_quality = 0.0
