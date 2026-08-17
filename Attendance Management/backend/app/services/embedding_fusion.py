"""Quality-weighted embedding fusion, shared by face and person tracks.

A single frame of a small, distant face yields an embedding that is largely
noise — measured on these cameras, an 18-20px face scored 0.09 against its own
enrolled identity. The noise is roughly independent frame to frame, so combining
N observations of the same person cuts it by about sqrt(N).

Lives in its own module because BOTH FaceTrack (entrance cameras, face-only
pipeline) and PersonTrack (room cameras, body-tracking pipeline) need it. The
room cameras need it MORE — a seated person is visible for minutes, so there are
far more observations to combine, and their faces are the smallest.

WHAT CHANGED AND WHY
--------------------
The original implementation kept a running weighted sum with ``quality`` passed
as the face width in pixels. Three defects followed from that, each of which
directly caused wrong or missing names on this deployment:

1. **It averaged everything.** A motion-blurred full profile went into the mean
   with the same standing as a sharp frontal frame, weighted only by size. Since
   a profile embedding points somewhere essentially arbitrary in the 512-d
   space, adding it does not average noise away — it drags the mean toward that
   arbitrary direction. On a camera that mostly sees profiles, the majority of
   observations were of this kind.

2. **It had no outlier rejection, so a tracker ID-switch silently merged two
   people.** When ByteTrack hands one track's box to a different person (routine
   in a doorway where people pass each other), that person's embedding folds
   into the same fused template. The result matches neither of them — or worse,
   matches a third employee. Nothing detected it.

3. **The window was unbounded.** A person seated in view for two hours kept
   accumulating into one running sum, so their fused template was dominated by
   however they looked when they first sat down and could never recover from a
   bad start. Weight from twenty minutes ago should not outvote the frame where
   they finally looked up.

The replacement keeps the same public surface (``add`` / ``fused`` / ``reset`` /
``observations`` / ``best_quality``) so both trackers and their tests keep
working, and adds ``accepted`` / ``rejected`` counters plus ``consensus`` for the
attendance-decision logging.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Keep at most this many observations per track. Enough to average noise down
# by ~3x (sqrt(12)) while staying recent — see defect 3 above. Deliberately a
# deque and not a running sum: bounding the window requires being able to DROP
# the oldest contribution, which a running sum cannot do.
DEFAULT_WINDOW = 12

# An observation whose cosine similarity to the current consensus is below this
# is treated as "a different person landed on this track" and rejected.
#
# MEASURED, not assumed. The first value tried here was 0.35, on the reasoning
# that two views of one person stay well above 0.5. That reasoning is right for
# a well-lit frontal pair and wrong for THESE cameras, where a single small
# ceiling-mounted observation is mostly noise — the code elsewhere records an
# 18-20px face scoring 0.09 against its own identity. Sweeping a genuine noisy
# track (per-observation cosine ~0.42 to the true identity) against a stranger
# intruding on a settled track:
#
#     threshold   genuine track fused score   observations kept   intruder rejected
#     0.35        0.415 -> 0.579                    3 / 15              5 / 5
#     0.25        0.415 -> 0.579                    3 / 15              5 / 5
#     0.20        0.415 -> 0.779                   14 / 15              5 / 5
#     0.15        0.415 -> 0.770                   15 / 15              5 / 5
#
# At 0.35 the check was throwing away 12 of 15 GENUINE observations — destroying
# most of the fusion benefit it sits next to, while catching nothing extra. A
# different identity is near-orthogonal (chance cosine in 512 dimensions has
# standard deviation 1/sqrt(512) = 0.044), so 0.20 is still about 4.5 sigma
# above chance and rejects intruders just as completely.
#
# Re-measure before changing this. It trades directly against recall.
DEFAULT_OUTLIER_MIN_COS = 0.20

# Outlier rejection needs something to compare against. Below this many accepted
# observations the consensus is itself unreliable, so we accept unconditionally
# rather than let the first (possibly bad) frame define who the track is.
_MIN_FOR_OUTLIER_CHECK = 3


class EmbeddingFuser:
    """Bounded, quality-weighted, outlier-rejecting fusion of one track's faces."""

    __slots__ = (
        "_window", "_outlier_min_cos", "_obs",
        "observations", "accepted", "rejected", "best_quality",
        "_cached_fused", "_dirty",
    )

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        outlier_min_cos: float = DEFAULT_OUTLIER_MIN_COS,
    ) -> None:
        self._window = max(1, int(window))
        self._outlier_min_cos = float(outlier_min_cos)
        # (unit_vector, weight) pairs, newest last.
        self._obs: deque = deque(maxlen=self._window)

        self.observations: int = 0   # accepted — kept as the historical meaning
        self.accepted: int = 0
        self.rejected: int = 0
        self.best_quality: float = 0.0

        self._cached_fused: Optional[np.ndarray] = None
        self._dirty: bool = True

    # ── ingestion ───────────────────────────────────────────────────────────
    def add(self, embedding, quality: float) -> bool:
        """Fold one observation in. Returns True when accepted.

        ``quality`` is the FaceQuality soft score in (0, 1] — not a pixel count.
        Callers that still pass pixels get sane behaviour (larger wins) but lose
        the pose/blur weighting, so they should be migrated.

        Silently ignores unusable input: a bad frame must never break tracking.
        """
        if embedding is None or quality is None:
            return False
        try:
            quality = float(quality)
        except (TypeError, ValueError):
            return False
        if quality <= 0:
            return False

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            return False
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return False
        unit = vector / norm

        # ── Outlier rejection ───────────────────────────────────────────────
        # Compare against the CURRENT consensus, not against the newest frame:
        # a single bad frame must not be able to redefine the track's identity.
        if self.accepted >= _MIN_FOR_OUTLIER_CHECK:
            consensus = self._fused_unit()
            if consensus is not None:
                sim = float(np.dot(unit, consensus))
                if sim < self._outlier_min_cos:
                    self.rejected += 1
                    logger.debug(
                        "FUSION outlier rejected sim=%.3f < %.3f (accepted=%d)",
                        sim, self._outlier_min_cos, self.accepted,
                    )
                    return False

        self._obs.append((unit, quality))
        self.accepted += 1
        self.observations = self.accepted
        self.best_quality = max(self.best_quality, quality)
        self._dirty = True
        return True

    # ── output ──────────────────────────────────────────────────────────────
    def fused(self) -> Optional[np.ndarray]:
        """Re-normalised weighted mean of the window, or None if empty.

        Re-normalisation is required: cosine similarity is taken against unit
        enrolled vectors, and the mean of unit vectors is shorter than one, so
        skipping it depresses every fused score.
        """
        return self._fused_unit()

    def _fused_unit(self) -> Optional[np.ndarray]:
        if not self._obs:
            return None
        if not self._dirty and self._cached_fused is not None:
            return self._cached_fused

        total = None
        weight_sum = 0.0
        for unit, weight in self._obs:
            contribution = unit * weight
            total = contribution if total is None else total + contribution
            weight_sum += weight
        if total is None or weight_sum <= 0:
            return None
        mean = total / weight_sum
        norm = float(np.linalg.norm(mean))
        if norm <= 0:
            return None

        self._cached_fused = (mean / norm).astype(np.float32)
        self._dirty = False
        return self._cached_fused

    def consensus(self) -> float:
        """Mean cosine of the window against its own fused vector, in [-1, 1].

        How much the observations AGREE with each other. A track whose frames all
        show the same face converges toward 1.0; a track that quietly merged two
        people sits low even after outlier rejection has done its work (because
        the intruder frames arrived before the consensus was trustworthy).

        This is an independent evidence signal for the attendance gate: a high
        match score computed from a self-contradictory template is exactly the
        situation that produced this system's known mislabelling.
        """
        fused = self._fused_unit()
        if fused is None or not self._obs:
            return 0.0
        return float(np.mean([float(np.dot(unit, fused)) for unit, _ in self._obs]))

    def total_weight(self) -> float:
        """Sum of accepted observation weights — the evidence 'mass' of a track."""
        return float(sum(weight for _, weight in self._obs))

    def reset(self) -> None:
        self._obs.clear()
        self.observations = 0
        self.accepted = 0
        self.rejected = 0
        self.best_quality = 0.0
        self._cached_fused = None
        self._dirty = True
