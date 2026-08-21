"""The single decision point between "we think we recognise someone" and
"write a payroll row".

WHY THIS EXISTS AS ONE MODULE
-----------------------------
The rule was previously implemented three times — the face path in
camera_service, the body-tracking path in camera_service, and again in
hcnetsdk_camera — with three different strengths. The weakest of the three is
what actually governed a given deployment, and nobody could tell which one was
running. Two of the three needed exactly one matched frame to write attendance.

WHAT CHANGED
------------
BEFORE: ``register_identification(emp_id, matched, confirm_frames)`` with
``confirm_frames = CCTV_CONFIRM_FRAMES = 1``. One frame, one match above 0.45
with a 0.10 margin, and an attendance row was written. The code comments record
why it ended up at 1: an earlier attempt at 2 CONSECUTIVE frames meant nobody
was ever marked, because a face at these camera angles is often visible for a
single moment.

That diagnosis was right and the fix was wrong. Consecutive frames is the wrong
axis. A track accumulates evidence over its whole life — the fused template, the
quality of the best observation, and how much the observations agree with each
other — and none of that requires the person to look at the camera twice in a
row.

AFTER: attendance requires ALL of

    1. enough accepted observations in the fused template  (min_observations)
    2. at least one genuinely usable observation           (min_quality)
    3. the track's observations agree with each other      (min_consensus)
    4. the same employee identified consistently           (agreement)
    5. score and margin clear the camera's bar             (threshold, margin)

with ONE documented escape hatch (see ``_strong_single``) so a person whose face
is genuinely visible only once — the real case that killed the previous attempt
— can still be marked when that single look is unambiguous.

On a tracker ID-switch merging two people into one track, the PRIMARY defence is
outlier rejection inside the fuser, not consensus. Measured on synthetic
identities at a realistic within-track similarity of 0.8:

  * intruder arrives mid-track (the common shape) — every one of its frames is
    rejected and the template stays on the original person;
  * two people interleaved from frame one (the hard shape) — outlier rejection
    has no settled consensus to work against, both sets get in, and the template
    really is a blend. It then resembles NEITHER person strongly (0.68 and 0.63,
    against 0.96 for a clean track), so its gallery score is degraded by about a
    third and normally falls below the camera threshold. That is a miss, not a
    confident write on the wrong person — but it is a mitigation, not a fix, and
    it is the main residual weakness in this design.

Consensus (3) is a BACKSTOP for a template whose observations are too scattered
to support any identity claim at all. It is deliberately NOT tuned to catch the
interleaved merge: measured, a merged track reads 0.64 and a genuine noisy track
reads 0.68, and a threshold placed between them would cost more real
recognitions than it saves. Do not raise it without re-measuring both sides.

What consensus adds over score alone: the documented mislabelling on this system
had the WRONG person at 0.77 while correct matches sat at 0.73-0.79, so score
and face size could not separate them. A degenerate template is visible without
reference to the gallery at all.

Every rejection returns a REASON, which is logged and is what makes tuning
possible: "35 attendance decisions blocked by low_quality on camera 4" is
actionable in a way that a silent miss never was.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str                 # "" when allowed; else a machine-greppable code
    path: str = "evidence"      # evidence | strong_single

    observations: int = 0
    agreements: int = 0
    best_quality: float = 0.0
    consensus: float = 0.0
    score: float = 0.0
    margin: float = 0.0

    def as_log_fields(self) -> dict:
        return {
            "obs": self.observations,
            "agree": self.agreements,
            "best_q": round(self.best_quality, 3),
            "consensus": round(self.consensus, 3),
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "path": self.path,
        }


def note_identification(track, employee_id: Optional[int], matched: bool) -> int:
    """Update a track's running agreement count. Returns the count.

    Agreement is 'how many times has this track been identified as THIS
    employee', reset whenever the answer changes. It is deliberately NOT reset
    by a frame that produced no match: at these camera angles a person's face
    disappears constantly, and treating that as evidence against their identity
    is what made the previous consecutive-frame rule unusable.
    """
    if not matched or employee_id is None:
        return int(getattr(track, "confirm_count", 0) or 0)

    if getattr(track, "pending_employee_id", None) == employee_id:
        track.confirm_count = int(track.confirm_count or 0) + 1
    else:
        track.pending_employee_id = employee_id
        track.confirm_count = 1
    return int(track.confirm_count)


def evaluate(
    track,
    *,
    employee_id: Optional[int],
    matched: bool,
    score: float,
    margin: float,
    profile,
) -> GateDecision:
    """Decide whether this track may write an attendance row, and say why not.

    Pure and side-effect free apart from the agreement counter, so it can be
    unit-tested and so a caller can log a rejection without having caused one.
    """
    fuser = getattr(track, "fuser", None)
    observations = int(getattr(fuser, "accepted", 0) or 0) if fuser is not None else 0
    best_quality = float(getattr(fuser, "best_quality", 0.0) or 0.0) if fuser is not None else 0.0
    consensus = float(fuser.consensus()) if fuser is not None else 0.0
    score = float(score or 0.0)
    margin = float(margin or 0.0)

    def decide(allowed: bool, reason: str, path: str = "evidence") -> GateDecision:
        return GateDecision(
            allowed=allowed, reason=reason, path=path,
            observations=observations,
            agreements=int(getattr(track, "confirm_count", 0) or 0),
            best_quality=best_quality, consensus=consensus,
            score=score, margin=margin,
        )

    if getattr(track, "attendance_marked", False):
        return decide(False, "already_marked")

    if not matched or employee_id is None:
        return decide(False, "no_match")

    agreements = note_identification(track, employee_id, matched)

    # Matching bar. recognize_face already applied the camera threshold and the
    # global margin, but the per-camera margin is stricter for IN/OUT cameras and
    # is re-checked here so the attendance path cannot be weakened by a caller
    # that forgot to pass it.
    if score < profile.threshold:
        return decide(False, "below_threshold")
    if margin < profile.margin:
        return decide(False, "below_margin")

    # ── Escape hatch: one unambiguous look ──────────────────────────────────
    if observations < profile.min_observations:
        if _strong_single(observations, best_quality, score, margin, profile):
            return decide(True, "", path="strong_single")
        return decide(False, "insufficient_observations")

    if best_quality < profile.min_quality:
        return decide(False, "low_quality")

    if consensus < profile.min_consensus:
        # The track's own observations disagree with each other — most often a
        # tracker ID-switch that merged two people into one template. A high
        # score computed from that template is not evidence about either of them.
        return decide(False, "inconsistent_track")

    if agreements < _min_agreements(profile):
        return decide(False, "unstable_identity")

    return decide(True, "")


def _min_agreements(profile) -> int:
    """How many times the same employee must be named before we believe it.

    Two, for an attendance camera. Not derived from min_observations: those count
    frames folded into the template, which happens far more often than a
    recognition pass, so tying them together would make the requirement
    accidentally strict. Monitor cameras never reach this code.
    """
    return 2 if profile.marks_attendance else 1


def _strong_single(
    observations: int, best_quality: float, score: float, margin: float, profile
) -> bool:
    """May a single observation mark attendance?

    Only when that one look is unambiguous on every axis at once: a genuinely
    good face (not a 20px profile), a score well clear of the camera's
    threshold, and a margin twice what is normally required — i.e. the runner-up
    is nowhere near.

    This exists because of a real, measured failure mode: at the check-out
    camera a person's face is frequently detected exactly once in their whole
    pass through frame. Without this path they would never be marked, which is
    what caused the previous multi-frame requirement to be abandoned entirely.
    The bar is set so that the frames which caused the known mislabelling —
    upscaled ~30px faces scoring 0.73-0.79 with ordinary margins — do NOT
    qualify.

    Thresholds are derived from the camera profile rather than being new
    columns, so there is one place to tune and no way for the two to drift.
    """
    if observations < 1:
        return False
    return (
        best_quality >= min(0.75, profile.min_quality * 2.5)
        and score >= profile.threshold + 0.10
        and margin >= profile.margin * 2.0
    )
