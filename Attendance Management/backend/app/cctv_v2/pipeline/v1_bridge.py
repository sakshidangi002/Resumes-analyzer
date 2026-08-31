"""Read V1's live person tracks so V2 can compute occupancy from them.

STRICTLY READ-ONLY. Nothing here writes to V1, mutates its state, or changes its
behaviour in any way. V1 is the running production pipeline and stays untouched.

WHY THIS EXISTS
---------------
The live camera feed is V1's: `/api/cameras/{id}/stream.mjpg` serves frames that
V1's camera_service renders. V2 is not instantiated in the API process at all --
`CCTV_PIPELINE` is `v1`, so its grabber, scheduler and tracker only ever run
inside the standalone validation scripts.

So there were three ways to put chair occupancy on the live view, and two of
them were bad:

    modify V1's renderer            -- V1 must stay untouched
    run V2 alongside V1             -- a second RTSP connection and a second
                                       YOLO pass per camera, purely to draw an
                                       overlay
    reuse what V1 has already done  -- this file

V1 already runs YOLO on camera 59 every cycle and already keeps the resulting
person boxes. Occupancy needs person boxes. There is no reason to compute them
twice, and doing so would cost a whole extra inference stream to display
information the machine has already worked out.

ZERO ADDITIONAL INFERENCE. This reads a list that is already in memory.

WHAT IS DELIBERATELY NOT TAKEN FROM V1
--------------------------------------
V1's tracks carry `employee_id`, `employee_name` and `identity_source`. None of
that crosses this boundary. Chair occupancy answers "is somebody in this seat",
never "who" -- a seat is occupied by a person whether or not anyone knows their
name, and letting identity in here would recreate exactly the coupling that made
V1's counting depend on recognition succeeding.

THE MISMATCH THIS ADAPTER ABSORBS
---------------------------------
V1's PersonTrack and V2's are different types with different fields. V1's has no
`hits` or lifecycle state, because V1 decides elsewhere whether a track is real.
By the time a track appears in `_latest_tracks` V1 has already accepted it, so
it is presented here as CONFIRMED -- V2's occupancy layer counts only CONFIRMED
tracks, and re-deriving confirmation from data V1 does not expose would mean
inventing it.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from app.cctv_v2.pipeline import chair_registry
from app.cctv_v2.pipeline.occupancy import OccupancyRegistry, RoomSnapshot
from app.cctv_v2.pipeline.track import PersonTrack, TrackState

logger = logging.getLogger(__name__)

# One registry for the API process. Occupancy is stateful -- the smoothing that
# stops a chair flickering needs consecutive observations -- so it must persist
# between requests rather than being rebuilt per call.
_registry = OccupancyRegistry()

# THE SMOOTHING COUNTS OBSERVATIONS, AND A POLL IS NOT AN OBSERVATION.
#
# This is the bug that made chair states flicker on a still room. `occupancy_
# snapshot` is called once per HTTP poll -- every 2s from the dashboard, and
# again for every extra viewer -- while V1 re-runs inference only every 4-17s.
# Each poll was advancing the state machine, so the rule "change state only
# after N consecutive observations" was really "after N READS OF THE SAME
# observation", which is no evidence at all.
#
# Measured: at a 2s poll and an 11.7s analysis pass, one pass is read about six
# times, so `confirm_free = 3` was satisfied roughly 6 SECONDS after a single
# noisy pass instead of after three independent ones. That is exactly the
# reported symptom -- a chair changing its mind about a second later while
# nobody moved -- and no threshold could have fixed it, because the thresholds
# were being spent on duplicates.
#
# So an observation is identified by V1's `state.updated_at`, which is stamped
# once per analysis pass under the same lock that publishes the tracks. A poll
# that brings no new observation returns the previous answer verbatim and
# advances nothing.
_last_observed: dict[int, float] = {}
_last_payload: dict[int, dict] = {}
# Two requests for the same camera can arrive together; without this they would
# both advance the state machine for one observation, reintroducing the bug in
# miniature.
_lock = threading.Lock()


def _v1_worker(camera_id: int):
    """V1's worker for a camera, or None. Never raises into a request."""
    try:
        from app.services.camera_service import camera_manager

        with camera_manager._lock:
            return camera_manager._workers.get(int(camera_id))
    except Exception:                                          # noqa: BLE001
        logger.debug("v1_bridge: camera_manager unavailable", exc_info=True)
        return None


def read_v1_tracks(camera_id: int) -> list[PersonTrack]:
    """V1's current person boxes, as V2 track objects. Read-only.

    Returns an empty list if V1 is not running this camera, which reads as "no
    people" -- the same thing an empty room produces. A caller that needs to
    distinguish those uses `v1_is_running`.
    """
    return read_v1_observation(camera_id)[0]


def read_v1_observation(camera_id: int) -> tuple[list[PersonTrack], float]:
    """V1's person boxes AND the stamp identifying which analysis pass made them.

    The stamp is the whole point. Read together under V1's frame lock, the pair
    answers "what did V1 see, and is this the same thing I was told last time" --
    which is what lets the smoothing count observations rather than polls.

    0.0 means V1 published no stamp. Callers treat that as "always new", because
    advancing too often is the milder failure: it degrades to the old behaviour
    rather than freezing the display on a stale answer forever.
    """
    worker = _v1_worker(camera_id)
    if worker is None:
        return [], 0.0

    # VALUES, TAKEN INSIDE THE LOCK. Not references.
    #
    # `_latest_tracks` holds LIVE PersonTrack objects that V1 keeps mutating:
    # the next analysis pass calls `update_box` on the very same objects, and
    # the recognition stage writes identity fields to them for the remainder of
    # the current pass. Copying the LIST under the lock -- which is what this
    # used to do -- copies none of that. It hands out references and then reads
    # their fields after releasing, so a reader could pair pass N's timestamp
    # with pass N+1's boxes.
    #
    # The consequence was bounded but real: the same physical pass could be
    # folded into the occupancy state machine twice, quietly spending one of the
    # confirmations that exist to keep a chair steady.
    #
    # So the four values occupancy needs are read out to plain immutable tuples
    # while the lock is held, together with the stamp. After this block nothing
    # V1 does can alter what this observation says.
    try:
        with worker._frame_lock:
            observed_at = float(
                getattr(getattr(worker, "state", None), "updated_at", 0.0) or 0.0)
            snapshot = tuple(
                (
                    getattr(t, "track_id", None),
                    tuple(getattr(t, "box", ()) or ()),
                    # The DETECTOR's score. `confidence` on a V1 track is the
                    # FACE MATCH score written by bind_identity, so reading it
                    # here published "0.0 confidence" for every unrecognised
                    # person -- which on a room camera is all of them.
                    float(getattr(t, "detection_confidence", 0.0) or 0.0),
                    float(getattr(t, "last_seen", 0.0) or 0.0),
                )
                for t in (getattr(worker, "_latest_tracks", []) or [])
            )
    except Exception:                                          # noqa: BLE001
        logger.debug("v1_bridge: could not read tracks camera=%s",
                     camera_id, exc_info=True)
        return [], 0.0

    now = time.time()
    out: list[PersonTrack] = []
    for tid, box, det_conf, last_seen in snapshot:
        if tid is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(PersonTrack(
            camera_id=int(camera_id),
            track_id=int(tid),
            bbox=(x1, y1, x2, y2),
            confidence=det_conf,
            frame_timestamp=now,
            frame_sequence=0,
            first_seen=last_seen or now,
            last_seen=now,
            hits=1,
            # V1 has already decided this track is real -- see the docstring.
            state=TrackState.CONFIRMED,
        ))
    return out, observed_at


def v1_is_running(camera_id: int) -> bool:
    """Whether V1 has a worker for this camera at all.

    Lets a caller tell "V1 is not running this camera" apart from "the room is
    empty", which look identical in the track list and mean very different
    things to somebody reading a dashboard.
    """
    return _v1_worker(camera_id) is not None


def frame_size(camera_id: int) -> tuple[float, float]:
    """The camera's real frame size, so normalised chair zones land correctly.

    Falls back to the 960x1080 these cameras produce. A wrong size here would
    silently shift every chair zone, so it is read from the live frame when
    one is available rather than assumed.
    """
    worker = _v1_worker(camera_id)
    try:
        with worker._frame_lock:                               # type: ignore[union-attr]
            frame = getattr(worker, "_latest_frame", None)
        if frame is not None and hasattr(frame, "shape"):
            h, w = frame.shape[:2]
            return (float(w), float(h))
    except Exception:                                          # noqa: BLE001
        pass
    return (960.0, 1080.0)


def occupancy_snapshot(camera_id: int) -> Optional[dict]:
    """Current chair occupancy for a room camera, from V1's live tracks.

    Returns None for a camera V1 is not running, so the caller can say so
    rather than reporting an empty room.
    """
    if not v1_is_running(camera_id):
        return None

    cid = int(camera_id)
    with _lock:
        tracks, observed_at = read_v1_observation(cid)

        # Nothing new to fold in. Return the previous answer UNCHANGED rather
        # than running the state machine again on the same evidence -- see the
        # note on _last_observed for why that distinction is the whole fix.
        if observed_at and observed_at == _last_observed.get(cid):
            cached = _last_payload.get(cid)
            if cached is not None:
                out = dict(cached)
                out["observation_age_sec"] = round(time.time() - observed_at, 1)
                # Stated plainly so a caller can tell a fresh answer from a
                # correct-but-old one. The count is only ever as current as the
                # last COMPLETED analysis pass, and on this hardware that is
                # seconds ago.
                out["from_new_observation"] = False
                return out

        room = _registry.get(cid)
        w, h = frame_size(cid)
        room.frame_w, room.frame_h = w, h

        # Adopt whatever the chair sweeper has learned. Done here, under the same
        # lock as the update, so the map cannot change between deciding which
        # seats exist and settling their states.
        #
        # `apply_geometry` refuses an empty map and keeps the state of every seat
        # that survives the change, so a sweeper fault degrades to "the inventory
        # stopped changing" rather than to an empty or reset room.
        if chair_registry.auto_enabled():
            room.apply_geometry(chair_registry.store().get(cid).geometry())
        snap: RoomSnapshot = _registry.update(cid, tracks, time.time())

    data = snap.as_dict()
    visible = {t.track_id for t in tracks}
    for chair in data["chairs"]:
        occ = chair.get("occupant_track_id")
        # A chair stays OCCUPIED through the smoothing window even when its
        # occupant is not in the current frame -- that is the point of the
        # smoothing. The UI still has to be able to tell the two apart, so the
        # distinction is reported rather than left to be inferred.
        chair["track_visible"] = bool(occ is not None and occ in visible)
    # Every tracked person, with their box, so the overlay can show WHAT WAS
    # DETECTED rather than only the conclusion drawn from it. Without this the
    # UI shows chair states and gives no way to see whether they came from the
    # right people -- which is the first question anyone asks of a number like
    # "3 occupied".
    #
    # Normalised, like the chair zones: the feed <img> is object-fit: contain,
    # so pixel coordinates would drift off the picture as soon as it is resized.
    seat_of = {c["occupant_track_id"]: c["id"]
               for c in data["chairs"] if c["occupant_track_id"] is not None}
    data["people"] = [
        {
            "track_id": t.track_id,
            "bbox": [t.bbox[0] / w, t.bbox[1] / h, t.bbox[2] / w, t.bbox[3] / h],
            # HOW SURE ARE WE SOMEBODY IS THERE -- the person detector's score.
            #
            # `confidence` is kept as an alias so nothing that already reads it
            # breaks, but both now carry the DETECTION score. Until this change
            # they carried the FACE MATCH score, so every unrecognised person
            # was published at 0.0 while being detected perfectly well.
            #
            # The face-match score is deliberately NOT offered here. Occupancy
            # answers "is this seat taken", never "who is in it", and letting
            # identity into this payload would rebuild the coupling this module
            # exists to prevent. It is available on /cameras/{id}/status.
            "detection_confidence": round(t.confidence, 3),
            "confidence": round(t.confidence, 3),
            # None means "in no MAPPED seat" -- standing, walking, or in a chair
            # the camera only half sees. It must never be rendered as "not
            # sitting anywhere".
            "chair_id": seat_of.get(t.track_id),
        }
        for t in tracks
    ]
    data["unassigned_people"] = max(0, snap.people_count - snap.occupied_chairs)
    data["source"] = "v1_tracks"     # no additional inference was run
    data["frame_width"] = w
    data["frame_height"] = h
    # Four separate freshness facts, because they answer different questions and
    # collapsing them is how a steady answer gets mistaken for a stuck one:
    #
    #   observation_updated_at  when V1 last completed an analysis pass
    #   observation_age_sec     how old that pass is now
    #   state_updated_at        when a chair last actually CHANGED
    #   from_new_observation    whether THIS response folded in a new pass
    #
    # A room that has been settled for ten minutes has a fresh observation and
    # an old state change, and that is correct, not stale.
    data["observation_updated_at"] = observed_at or None
    data["observation_age_sec"] = (round(time.time() - observed_at, 1)
                                   if observed_at else None)
    data["from_new_observation"] = True

    # Chairs this camera can see but does NOT control, so the overlay can draw
    # them and say whose they are. They are listed separately from `chairs` and
    # carry no state, because they take no part in this camera's occupancy --
    # see OBSERVED_ELSEWHERE in config/geometry.py.
    from app.cctv_v2.config.geometry import (
        observed_elsewhere, observed_owner, room_chair_total)

    data["observed_elsewhere"] = [
        {"id": zid, "zone": list(box), "owned_by_camera": observed_owner(cid)}
        for zid, box in observed_elsewhere(cid)
    ]
    # Physical chairs in the ROOM, each counted once. `chairs_total` above is
    # this camera's share of them; adding the two cameras' shares would
    # double-count the row they both see.
    if chair_registry.auto_enabled():
        # Sum the LEARNED inventories of the cameras that own this room's seats,
        # not the configured map -- otherwise a chair the sweeper found would
        # show up in this camera's count while the room total stayed at the
        # number somebody typed months ago.
        data["room_chairs_total"] = chair_registry.room_confirmed_total(cid)
    else:
        data["room_chairs_total"] = room_chair_total(cid)

    # How the inventory was arrived at, so the UI can distinguish a configured
    # seat count from a learned one -- and show that a newly spotted chair is
    # still gathering evidence rather than being ignored.
    registry = chair_registry.store().get(cid)
    data["chairs_auto"] = chair_registry.auto_enabled()
    data["chairs_pending"] = registry.pending_count
    data["chairs_sweeps"] = registry.sweeps

    _last_observed[cid] = observed_at
    _last_payload[cid] = data
    return data
