"""Learn a room's chair inventory from the detector, without letting it wobble.

Why this module exists
----------------------
`ChairZone` says a chair is "configured once and given a stable id", because
re-detecting chairs per frame renumbers them whenever the detector wobbles. That
is still true of the RAW detector: measured against these two rooms it returns
1-9 chairs on camera 59 (about 10 visible) and 3-10 on camera 60 (6-8 visible).
Feeding that straight into occupancy would make the seat count flicker every
sweep and give "chair 3" a different meaning minute to minute.

So this module does not use the detector as the answer. It uses it as EVIDENCE,
and keeps its own inventory that changes slowly and deliberately:

  * a new chair must be seen in several separate sweeps, in the same place,
    before it is admitted -- so one frame's false positive never adds a seat;
  * an existing chair is only forgotten after it has been missing for a long
    run of sweeps in which it was actually VISIBLE. This is the part that
    matters most: a chair with somebody sitting in it is largely hidden from a
    chair detector, so counting those misses would delete exactly the chairs
    that are in use. A miss while occluded is not evidence of absence.
  * ids are handed out once and never reused, so a chair id keeps meaning the
    same seat for the lifetime of the room.

The registry is SEEDED from the hand-verified map in config/geometry.py. Day one
therefore behaves exactly as the configured map did, and detection can only
refine it from there -- rather than starting from nothing and counting up.

This module is deliberately pure: no YOLO, no threads, no I/O beyond an explicit
save/load. All of that lives in the sweeper that drives it, so the decision
logic here can be tested frame-by-frame without a camera.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from app.cctv_v2.config.geometry import (
    ChairZone, RoomGeometry, observed_elsewhere, room_geometry)

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# A detection is the SAME chair as a registered one at or above this IoU. Set
# well below 0.5 because the detector's box on a partly-tucked chair breathes
# quite a lot between sweeps; too high a bar and one chair slowly becomes two.
MATCH_IOU = _env_float("CCTV_CHAIR_MATCH_IOU", 0.35)

# Sweeps a candidate must be seen in before it becomes a real chair. This is the
# whole defence against false positives: a rucksack that momentarily reads as a
# chair does not survive several sweeps in the same place.
PROMOTE_SWEEPS = _env_int("CCTV_CHAIR_PROMOTE_SWEEPS", 4)

# Consecutive sweeps in which a chair was MISSED WHILE VISIBLE before it is
# retired. Deliberately much larger than PROMOTE_SWEEPS: adding a phantom chair
# is a cosmetic error, but deleting a real one silently shrinks the room and
# takes its occupancy history with it.
RETIRE_SWEEPS = _env_int("CCTV_CHAIR_RETIRE_SWEEPS", 12)

# Whether a chair that came from the hand-verified config map may be retired by
# the detector. Off by default, and this is the single most consequential
# default in the module.
#
# Measured on the live cameras: a sweep of camera 59 detects 4-6 chairs out of
# the 8 that are really there. Intermittent misses are harmless -- visible_misses
# resets on any sighting, so a chair seen even a third of the time never
# approaches the bar. But a chair the detector CONSISTENTLY cannot see (tucked
# under a desk, or at an angle it reads badly) would be retired after
# RETIRE_SWEEPS, and it would be a chair a person had measured and confirmed.
#
# So the detector may EXTEND the human's inventory, and may retire what it
# itself added, but it does not get to overrule a human's count. The cost is
# that a seeded chair physically removed from the room has to be taken out of
# config/geometry.py by hand -- a rare, deliberate act, unlike adding one.
RETIRE_SEEDED = os.getenv("CCTV_CHAIR_RETIRE_SEEDED", "0").strip().lower() in (
    "1", "true", "yes", "on")

# A candidate that stops being seen before promotion is dropped this fast. It
# has no history worth preserving.
CANDIDATE_TTL_SWEEPS = _env_int("CCTV_CHAIR_CANDIDATE_TTL", 3)

# How much of a registered chair a person box must cover for the chair to count
# as OCCLUDED this sweep. Low on purpose: the cost of wrongly calling a chair
# occluded is only that we wait longer before retiring it, while the cost of
# wrongly calling it visible is deleting a chair somebody is sitting in.
OCCLUSION_COVERAGE = _env_float("CCTV_CHAIR_OCCLUSION_COVERAGE", 0.10)

# A detection this close to a chair the registry ALREADY holds is treated as a
# second box on that same chair, not as a new one. Deliberately far below
# MATCH_IOU: matching is greedy and one-to-one, so when the detector emits two
# boxes on one chair the second is left unmatched and would otherwise be admitted
# as a neighbour. Observed live -- camera 59 grew a chair "A1" at IoU 0.40 with
# its own R8.
SUPPRESS_IOU = _env_float("CCTV_CHAIR_SUPPRESS_IOU", 0.15)

# A new chair whose box touches the frame edge is refused. Such a box is a
# PARTIAL view: the detector drew what it could see of something leaving the
# picture, so its size and position are both truncated and cannot be trusted --
# and occupancy judged against a truncated seat is equally untrustworthy.
#
# Observed live: both cameras admitted exactly one phantom this way, camera 59
# at y2=0.999 and camera 60 at y2=1.000, each sitting on the bottom edge.
# A real chair that genuinely straddles the edge is better added to the config
# map by hand, where a person can decide what its box should be.
EDGE_MARGIN = _env_float("CCTV_CHAIR_EDGE_MARGIN", 0.004)

# Weight of a new observation when nudging a chair's stored box. Small, so a
# chair that is genuinely moved follows over a few minutes rather than jumping
# on one noisy detection.
EMA_ALPHA = _env_float("CCTV_CHAIR_EMA_ALPHA", 0.20)


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _covered_fraction(target: Box, other: Box) -> float:
    """How much of `target` lies inside `other`, 0..1.

    Coverage, not IoU, because a person box is far larger than a seat: a person
    can hide a chair completely while their IoU with it stays small.
    """
    tx1, ty1, tx2, ty2 = target
    area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    if area <= 0:
        return 0.0
    ox1, oy1, ox2, oy2 = other
    iw = max(0.0, min(tx2, ox2) - max(tx1, ox1))
    ih = max(0.0, min(ty2, oy2) - max(ty1, oy1))
    return (iw * ih) / area


def _centre_inside(box: Box, container: Box) -> bool:
    """Whether box's centre falls within container.

    Catches the case IoU misses: a detection much larger or much smaller than a
    registered chair, sitting right on top of it, can score a low IoU while
    plainly being the same seat.
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]


def _blend(old: Box, new: Box, alpha: float) -> Box:
    return tuple(o + alpha * (n - o) for o, n in zip(old, new))  # type: ignore[return-value]


@dataclass
class TrackedChair:
    """One seat the registry believes in, or is still making its mind up about."""

    chair_id: str
    box: Box
    confirmed: bool = False
    # Sweeps this has been seen in. Only counts up to promotion; afterwards the
    # chair's existence is not in question and the number would just grow.
    sightings: int = 0
    # Consecutive sweeps missed WHILE VISIBLE. Reset by any sighting, and NOT
    # incremented on a sweep where a person was covering the seat.
    visible_misses: int = 0
    # Sweeps since last seen, regardless of visibility -- used only to expire
    # unpromoted candidates.
    sweeps_since_seen: int = 0
    label: Optional[str] = None
    # True for chairs seeded from the hand-verified config map. They start
    # confirmed and are held to the same retirement rule as any other; the flag
    # exists so an operator can tell "someone measured this" from "the detector
    # decided this".
    seeded: bool = False

    def as_zone(self) -> ChairZone:
        return ChairZone(chair_id=self.chair_id, box=self.box, label=self.label)


class ChairRegistry:
    """The chair inventory for ONE camera, learned over many sweeps."""

    def __init__(self, camera_id: int, seed: bool = True):
        self.camera_id = int(camera_id)
        self._base = room_geometry(camera_id)
        self.chairs: dict[str, TrackedChair] = {}
        self.sweeps = 0
        self._next_auto = 1
        if seed:
            self._seed_from_config()

    def _seed_from_config(self) -> None:
        """Start from the configured map so day one matches the measured room."""
        for zone in self._base.chairs:
            self.chairs[zone.chair_id] = TrackedChair(
                chair_id=zone.chair_id,
                box=zone.box,
                confirmed=True,
                sightings=PROMOTE_SWEEPS,
                label=zone.label,
                seeded=True,
            )

    def _new_id(self) -> str:
        """A fresh id that has never been used on this camera.

        Ids are never reused even after a chair is retired: reuse would let an
        old chair's occupancy history appear to belong to a new seat.
        """
        while True:
            candidate = f"A{self._next_auto}"
            self._next_auto += 1
            if candidate not in self.chairs:
                return candidate

    def sweep(
        self,
        detections: Iterable[Box],
        person_boxes: Iterable[Box] = (),
    ) -> None:
        """Fold one detector sweep into the inventory.

        `detections` and `person_boxes` are normalised (0..1) boxes in this
        camera's frame. Person boxes are used ONLY to decide whether a missing
        chair was actually visible this sweep.
        """
        self.sweeps += 1
        dets: list[Box] = [
            tuple(float(v) for v in d) for d in detections if d is not None and len(tuple(d)) == 4
        ]
        people: list[Box] = [
            tuple(float(v) for v in p) for p in person_boxes if p is not None and len(tuple(p)) == 4
        ]

        # Greedy best-IoU matching, strongest pair first, so two chairs close
        # together cannot both latch onto the same detection.
        pairs = sorted(
            (
                (_iou(chair.box, det), cid, di)
                for cid, chair in self.chairs.items()
                for di, det in enumerate(dets)
            ),
            key=lambda t: t[0],
            reverse=True,
        )
        matched_chairs: set[str] = set()
        matched_dets: set[int] = set()
        for score, cid, di in pairs:
            if score < MATCH_IOU:
                break
            if cid in matched_chairs or di in matched_dets:
                continue
            matched_chairs.add(cid)
            matched_dets.add(di)
            chair = self.chairs[cid]
            chair.box = _blend(chair.box, dets[di], EMA_ALPHA)
            chair.sightings = min(chair.sightings + 1, PROMOTE_SWEEPS)
            chair.visible_misses = 0
            chair.sweeps_since_seen = 0
            if not chair.confirmed and chair.sightings >= PROMOTE_SWEEPS:
                chair.confirmed = True
                logger.info(
                    "cctv_v2: camera=%s chair %s confirmed after %s sightings",
                    self.camera_id, chair.chair_id, chair.sightings,
                )

        # Unmatched detections become candidates -- unless they are really
        # something already accounted for.
        for di, det in enumerate(dets):
            if di in matched_dets:
                continue
            if self._touches_frame_edge(det):
                continue
            if self._is_duplicate(det):
                continue
            if self._belongs_to_another_camera(det):
                continue
            chair_id = self._new_id()
            self.chairs[chair_id] = TrackedChair(
                chair_id=chair_id, box=det, sightings=1, label="auto-detected"
            )

        # Unmatched chairs: decide whether this miss is evidence of absence.
        for cid, chair in list(self.chairs.items()):
            if cid in matched_chairs:
                continue
            chair.sweeps_since_seen += 1
            if self._occluded(chair.box, people):
                # Somebody is in front of it. Says nothing about whether the
                # chair is there, so it must not count toward retirement.
                continue
            chair.visible_misses += 1
            if chair.confirmed:
                if chair.seeded and not RETIRE_SEEDED:
                    # A human counted this seat. See RETIRE_SEEDED.
                    continue
                if chair.visible_misses >= RETIRE_SWEEPS:
                    del self.chairs[cid]
                    logger.info(
                        "cctv_v2: camera=%s chair %s retired after %s visible misses",
                        self.camera_id, cid, chair.visible_misses,
                    )
            elif chair.sweeps_since_seen >= CANDIDATE_TTL_SWEEPS:
                del self.chairs[cid]

        self._prune_drifted_duplicates()

    def _is_duplicate(self, det: Box) -> bool:
        """True if this detection is a second box on a chair already held.

        Matching is one-to-one, so a chair the detector draws twice leaves its
        weaker box unmatched. Admitting that box would split one physical seat
        into two and inflate the count -- which is exactly how the room's chair
        total went wrong before.
        """
        return any(
            _iou(chair.box, det) >= SUPPRESS_IOU or _centre_inside(det, chair.box)
            for chair in self.chairs.values()
        )

    @staticmethod
    def _touches_frame_edge(det: Box) -> bool:
        """True for a box clipped by the frame -- a partial, untrustworthy view."""
        x1, y1, x2, y2 = det
        return (
            x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN
            or x2 >= 1.0 - EDGE_MARGIN or y2 >= 1.0 - EDGE_MARGIN
        )

    def _prune_drifted_duplicates(self) -> None:
        """Drop an auto chair that has drifted onto one already held.

        `_is_duplicate` only runs when a candidate is CREATED. Both boxes then
        move under EMA, so two zones that were distinct at birth can converge --
        which is how camera 59 ended up holding an auto chair at IoU 0.235 with
        its own R8, comfortably above the suppression bar it had passed weeks
        earlier at a lower overlap.

        Only ever removes the AUTO side of a pair. A seeded chair was placed by
        a person and is not the registry's to overrule; if a seeded zone is
        wrong, it gets fixed in config/geometry.py.
        """
        chairs = sorted(self.chairs.values(), key=lambda c: c.chair_id)
        for i, a in enumerate(chairs):
            for b in chairs[i + 1:]:
                if a.chair_id not in self.chairs or b.chair_id not in self.chairs:
                    continue
                if _iou(a.box, b.box) < SUPPRESS_IOU:
                    continue
                # Prefer keeping the seeded one; between two autos keep the
                # better-established (more sightings, then older id).
                if a.seeded and b.seeded:
                    continue
                loser = b if (a.seeded or a.sightings >= b.sightings) else a
                if loser.seeded:
                    continue
                del self.chairs[loser.chair_id]
                logger.info(
                    "cctv_v2: camera=%s chair %s dropped -- drifted onto %s",
                    self.camera_id, loser.chair_id,
                    b.chair_id if loser is a else a.chair_id,
                )

    def _belongs_to_another_camera(self, det: Box) -> bool:
        """True if this detection lands on a seat a PEER camera owns.

        Cameras 59 and 60 face each other along one desk and both see the shared
        row. Each chair there is owned by exactly one of them, and the zones the
        other merely observes are listed in OBSERVED_ELSEWHERE precisely so they
        are never counted twice. Detection has to respect that ownership or the
        room total becomes the sum of two overlapping views -- how a 13-chair
        room once looked like 18.

        Seen live: camera 59 admitted a chair at (0.218, 0.710, 0.374, 1.000),
        which is camera 60's seat under camera 59's own L5 observation zone.
        """
        for _zone_id, box in observed_elsewhere(self.camera_id):
            if _iou(box, det) >= SUPPRESS_IOU or _centre_inside(det, box):
                return True
        return False

    @staticmethod
    def _occluded(chair_box: Box, people: list[Box]) -> bool:
        return any(_covered_fraction(chair_box, p) >= OCCLUSION_COVERAGE for p in people)

    def confirmed_zones(self) -> tuple[ChairZone, ...]:
        """The seats the registry is willing to stand behind, in stable order."""
        return tuple(
            c.as_zone()
            for c in sorted(self.chairs.values(), key=lambda c: c.chair_id)
            if c.confirmed
        )

    def geometry(self) -> RoomGeometry:
        """The learned map, keeping the camera's configured occupancy thresholds.

        Only `chairs` is learned. min_overlap and max_drop_below_seat were tuned
        against this camera's projection and have nothing to do with how many
        chairs there are.
        """
        base = self._base
        return RoomGeometry(
            chairs=self.confirmed_zones(),
            min_overlap=base.min_overlap,
            max_drop_below_seat=base.max_drop_below_seat,
            confirm_occupied=base.confirm_occupied,
            confirm_free=base.confirm_free,
        )

    @property
    def confirmed_count(self) -> int:
        return sum(1 for c in self.chairs.values() if c.confirmed)

    @property
    def pending_count(self) -> int:
        return sum(1 for c in self.chairs.values() if not c.confirmed)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "sweeps": self.sweeps,
            "next_auto": self._next_auto,
            "chairs": [
                {
                    "chair_id": c.chair_id, "box": list(c.box), "confirmed": c.confirmed,
                    "sightings": c.sightings, "label": c.label, "seeded": c.seeded,
                }
                for c in sorted(self.chairs.values(), key=lambda c: c.chair_id)
            ],
        }

    def load_dict(self, data: dict) -> None:
        """Restore a saved inventory, replacing whatever is held now.

        Without this the count would relearn from the seed map on every restart,
        so a chair added to the room would be forgotten every time the backend
        bounced -- which is the opposite of automatic.
        """
        chairs = data.get("chairs") or []
        if not chairs:
            return
        restored: dict[str, TrackedChair] = {}
        for row in chairs:
            box = tuple(float(v) for v in (row.get("box") or ()))
            cid = str(row.get("chair_id") or "")
            if len(box) != 4 or not cid:
                continue
            restored[cid] = TrackedChair(
                chair_id=cid, box=box,
                confirmed=bool(row.get("confirmed")),
                sightings=int(row.get("sightings") or 0),
                label=row.get("label"),
                seeded=bool(row.get("seeded")),
            )
        if not restored:
            return
        self.chairs = restored
        self.sweeps = int(data.get("sweeps") or 0)
        self._next_auto = int(data.get("next_auto") or 1)


class ChairRegistryStore:
    """Per-camera registries plus their on-disk copy."""

    def __init__(self, path: Optional[Path] = None):
        self._lock = threading.Lock()
        self._registries: dict[int, ChairRegistry] = {}
        self.path = path

    def get(self, camera_id: int) -> ChairRegistry:
        with self._lock:
            reg = self._registries.get(int(camera_id))
            if reg is None:
                reg = ChairRegistry(int(camera_id))
                self._registries[int(camera_id)] = reg
            return reg

    def cameras(self) -> list[int]:
        with self._lock:
            return sorted(self._registries)

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            payload = {str(cid): reg.to_dict() for cid, reg in self._registries.items()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            # Written to a temp file and moved, so a crash mid-write cannot
            # leave a truncated inventory that fails to parse on next boot.
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            logger.warning("cctv_v2: could not persist chair registry", exc_info=True)

    def load(self) -> None:
        if not self.path or not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("cctv_v2: chair registry file unreadable; using config map",
                           exc_info=True)
            return
        for key, data in (payload or {}).items():
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            self.get(cid).load_dict(data or {})


# One process-wide store, so the sweeper that WRITES the inventory and the
# occupancy path that READS it are looking at the same object. Kept here rather
# than in either of those modules because both need it, and importing one from
# the other would make them mutually dependent.
_STORE: Optional[ChairRegistryStore] = None
_STORE_LOCK = threading.Lock()

# Where the learned inventory is kept between restarts. Under data/ with the
# other runtime state rather than beside the code, so a redeploy cannot
# overwrite what the room has learned.
DEFAULT_STORE_PATH = "data/cctv_v2/chair_registry.json"


def store() -> ChairRegistryStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ChairRegistryStore(
                path=Path(os.getenv("CCTV_CHAIR_REGISTRY_PATH", DEFAULT_STORE_PATH))
            )
            _STORE.load()
        return _STORE


def auto_enabled() -> bool:
    """Whether the inventory is learned from the detector or fixed to config.

    A single switch, because turning it off must fall back to exactly the
    hand-verified map -- not to some half-learned state.
    """
    return os.getenv("CCTV_CHAIR_AUTO", "1").strip().lower() not in ("0", "false", "no", "off")


def room_confirmed_total(camera_id: int, registry_store=None) -> int:
    """Confirmed chairs in this camera's ROOM, each counted once.

    Every chair is owned by exactly one camera, so the room total is the sum
    over the owners -- this camera INCLUDED. `same_room_peers` deliberately
    excludes the caller (`group - {cid}`), and summing only the peers gave each
    camera its neighbour's count as the room total: camera 59 reported owning
    "12 of 9", more chairs than the room it is standing in.
    """
    from app.cctv_v2.config.cameras import same_room_peers

    store_ = registry_store if registry_store is not None else store()
    owners = set(same_room_peers(camera_id)) | {int(camera_id)}
    return sum(store_.get(owner).confirmed_count for owner in sorted(owners))
