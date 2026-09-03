import { useCallback, useEffect, useState } from "react";
import { faceReview, employees as employeesApi, cameras as camerasApi } from "../api/client";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";

/**
 * Teach the cameras what your staff look like TO THEM.
 *
 * Why this screen exists
 * ----------------------
 * An uploaded portrait and a doorway camera's view of the same person are, to a
 * face model, two quite different pictures. Measured on this deployment:
 *
 *     a gate capture vs another gate capture of the same person   0.63 - 0.75
 *     the same gate capture vs that person's uploaded photos      0.32 - 0.54
 *     the match threshold                                         0.45
 *
 * So uploading more portraits does not fix a gate that fails to recognise
 * people — the employee here with the MOST uploaded photos (15) was recognised
 * at the entrance 0 times in a day, while the one with 15 camera-captured
 * enrolments and no uploads at all was recognised 7 times.
 *
 * Every person the cameras could not name is kept here, grouped by appearance.
 * Attributing a group to an employee enrols its best frames, so the gallery
 * finally contains what that camera actually sees.
 */

interface Cluster {
  cluster_id: number;
  size: number;
  first_seen: string | null;
  last_seen: string | null;
  best_quality: number;
  best_near_miss_score: number;
  cameras: (string | number)[];
  sample_face_id: number | null;
  sample_crop: string | null;
  near_miss_employee_id: number | null;
}

interface EmployeeOption {
  id: number;
  employee_code: string;
  full_name: string;
}

const Icons = {
  Faces: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="4" />
      <path d="M9 10h.01M15 10h.01" />
      <path d="M9 15c.9.7 1.9 1 3 1s2.1-.3 3-1" />
    </svg>
  ),
  Refresh: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12a9 9 0 1 1-3-6.7" /><polyline points="21 3 21 9 15 9" />
    </svg>
  ),
};

interface Sighting {
  face_id: number;
  quality_score: number;
  face_px: number;
  captured_at: string | null;
}

/**
 * Every sighting in one cluster, best-quality first.
 *
 * Deliberately NOT a single sample. These are doorway person-crops: measured on
 * this deployment the median is 54x88 px on some days, which holds a face of
 * 15-20 px — nobody identifies a colleague from that. Across seventeen
 * sightings there is usually one frame where the person turned toward the lens
 * or walked close, and the reviewer needs that frame, not the first one.
 *
 * The crops need an auth header, so each is fetched as a blob rather than
 * pointed at with a plain <img src>.
 */
function ClusterFaces({
  clusterId,
  fallbackFaceId,
  onOpen,
}: {
  clusterId: number;
  fallbackFaceId: number | null;
  onOpen: (url: string) => void;
}) {
  const [urls, setUrls] = useState<string[]>([]);
  const [state, setState] = useState<"loading" | "ok" | "empty">("loading");

  useEffect(() => {
    let cancelled = false;
    const made: string[] = [];

    const fetchCrops = async (ids: number[]) => {
      const out: string[] = [];
      for (const id of ids) {
        try {
          const r = await faceReview.crop(id);
          if (cancelled) return out;
          const u = URL.createObjectURL(r.data as Blob);
          made.push(u);
          out.push(u);
        } catch {
          // A missing crop file is normal after retention cleanup — skip it
          // rather than failing the whole group.
        }
      }
      return out;
    };

    (async () => {
      let ids: number[] = [];
      try {
        const r = await faceReview.clusterFaces(clusterId, 8);
        ids = ((r.data?.faces ?? []) as Sighting[]).map((f) => f.face_id);
      } catch {
        // Older backend without the per-cluster endpoint: fall back to the
        // single sample so the screen still works.
        if (fallbackFaceId != null) ids = [fallbackFaceId];
      }
      if (ids.length === 0 && fallbackFaceId != null) ids = [fallbackFaceId];
      const got = await fetchCrops(ids);
      if (cancelled) return;
      setUrls(got);
      setState(got.length ? "ok" : "empty");
    })();

    return () => {
      cancelled = true;
      made.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [clusterId, fallbackFaceId]);

  if (state === "loading")
    return <div className="eds-fr-face eds-fr-face--empty">loading sightings…</div>;
  if (state === "empty")
    return <div className="eds-fr-face eds-fr-face--empty">no crops stored</div>;

  return (
    <div className="eds-fr-strip">
      {urls.map((u, i) => (
        <img
          key={u}
          className={`eds-fr-shot${i === 0 ? " is-best" : ""}`}
          src={u}
          alt={`Sighting ${i + 1}`}
          title="Click to see this capture full size"
          onClick={() => onOpen(u)}
        />
      ))}
    </div>
  );
}

interface CameraOption {
  id: number;
  name: string;
  camera_purpose: string;
}

/**
 * Guided enrolment: a named person stands at a named gate, and the system
 * enrols what it sees.
 *
 * This is the direct route to a working gate, and it sidesteps the two things
 * that make the review queue below slow: you do not have to recognise a 54x88
 * px crop by eye, and you do not have to click through hundreds of clusters.
 * The answer is known before the picture is taken.
 *
 * One capture per call, polled. A 30-second capture inside a single request
 * would block a server thread and show the operator nothing until it finished.
 */
function CaptureEnrolment({
  employees,
  onDone,
}: {
  employees: EmployeeOption[];
  onDone: () => void;
}) {
  const [cams, setCams] = useState<CameraOption[]>([]);
  const [employeeId, setEmployeeId] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [running, setRunning] = useState(false);
  const [captured, setCaptured] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [status, setStatus] = useState("");

  const TARGET = 10;

  useEffect(() => {
    camerasApi
      .list()
      .then((r) =>
        setCams(
          ((r.data as any[]) || [])
            // Only gates. A room camera can never mark attendance, so enrolling
            // against one does nothing for check-in or check-out.
            .filter((c) => ["IN", "OUT"].includes(String(c.camera_purpose)))
            .map((c) => ({ id: c.id, name: c.name, camera_purpose: c.camera_purpose }))
        )
      )
      .catch(() => { });
  }, []);

  // The capture loop. Stops itself at TARGET, on error, or when switched off.
  useEffect(() => {
    if (!running || !employeeId || !cameraId) return;
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      try {
        const r = await faceReview.enrolFromCamera(Number(employeeId), Number(cameraId));
        if (cancelled) return;
        const d = r.data as any;
        setAttempts((a) => a + 1);
        if (d.enrolled) {
          setCaptured((c) => {
            const next = c + 1;
            if (next >= TARGET) {
              setRunning(false);
              setStatus(`Done — ${next} faces enrolled. This gate can now recognise them.`);
              onDone();
            } else {
              setStatus(`Captured ${next} of ${TARGET} (quality ${d.quality}, ${d.face_px}px)`);
            }
            return next;
          });
        } else {
          setStatus(d.detail || d.reason || "Waiting for a usable face…");
        }
      } catch (e: any) {
        if (cancelled) return;
        setRunning(false);
        setStatus(e.response?.data?.detail || "Capture failed.");
      }
    };

    const id = window.setInterval(tick, 1200);
    void tick();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [running, employeeId, cameraId, onDone]);

  const start = () => {
    setCaptured(0);
    setAttempts(0);
    setStatus("Stand facing the camera…");
    setRunning(true);
  };

  const who = employees.find((e) => String(e.id) === employeeId);
  const cam = cams.find((c) => String(c.id) === cameraId);

  return (
    <section className="eds-card eds-cap">
      <div className="eds-card-head">
        <div className="eds-card-titles">
          <h2 className="eds-card-title">Enrol from a gate camera</h2>
          <p className="eds-card-sub">
            The fastest way to make a gate work: pick the person, they stand at
            that gate, and the camera enrols its own view of them.
          </p>
        </div>
      </div>
      <div className="eds-card-body">
        <div className="eds-cap-row">
          <CustomSelect
            className="eds-cselect"
            value={employeeId}
            onChange={setEmployeeId}
            options={[
              { value: "", label: "Which employee?" },
              ...employees.map((e) => ({
                value: String(e.id),
                label: `${e.employee_code} — ${e.full_name}`,
              })),
            ]}
          />
          <CustomSelect
            className="eds-cselect"
            value={cameraId}
            onChange={setCameraId}
            options={[
              { value: "", label: "Which gate?" },
              ...cams.map((c) => ({
                value: String(c.id),
                label: `${c.name} (${c.camera_purpose === "IN" ? "Check-in" : "Check-out"})`,
              })),
            ]}
          />
          {running ? (
            <button type="button" className="eds-action" onClick={() => setRunning(false)}>
              Stop
            </button>
          ) : (
            <button
              type="button"
              className="eds-action eds-action--go"
              onClick={start}
              disabled={!employeeId || !cameraId}
            >
              Start capturing
            </button>
          )}
        </div>

        {(running || captured > 0 || status) && (
          <>
            <div className="eds-cap-bar">
              <div
                className="eds-cap-fill"
                style={{ width: `${Math.min(100, (captured / TARGET) * 100)}%` }}
              />
            </div>
            <p className="eds-cap-status">
              <strong>{captured}</strong> of {TARGET} captured
              {attempts > 0 && <> · {attempts} frame{attempts === 1 ? "" : "s"} checked</>}
              {who && cam && <> · {who.full_name} at {cam.name}</>}
              <br />
              {status}
            </p>
          </>
        )}
      </div>
    </section>
  );
}

export default function FaceReview() {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [employees, setEmployees] = useState<EmployeeOption[]>([]);
  const [picked, setPicked] = useState<Record<number, string>>({});
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [zoom, setZoom] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    faceReview
      .clusters(50)
      .then((r) => setClusters((r.data?.clusters ?? []) as Cluster[]))
      .catch((e) => setError(e.response?.data?.detail || "Could not load the review queue."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    employeesApi
      .list({ status: "Active" })
      .then((r) =>
        setEmployees(
          ((r.data as any[]) || []).map((e) => ({
            id: e.id,
            employee_code: e.employee_code,
            full_name: e.full_name,
          }))
        )
      )
      .catch(() => { });
    load();
  }, [load]);

  const regroup = () => {
    setBusy(-1);
    setError("");
    setSuccess("");
    faceReview
      .recluster()
      .then(() => {
        setSuccess("Regrouped by appearance.");
        load();
      })
      .catch((e) => setError(e.response?.data?.detail || "Could not regroup."))
      .finally(() => setBusy(null));
  };

  const assign = (c: Cluster) => {
    const employeeId = Number(picked[c.cluster_id]);
    if (!employeeId) {
      setError("Choose which employee this is first.");
      return;
    }
    const who = employees.find((e) => e.id === employeeId);
    setBusy(c.cluster_id);
    setError("");
    setSuccess("");
    // Cap at the cluster size: asking to enrol 5 from a group of 2 is not an
    // error, but the message should say what actually happened.
    faceReview
      .assign(c.cluster_id, employeeId, Math.min(5, Math.max(1, c.size)))
      .catch((e) => {
        // 409 = the cluster does not look like this employee. Do not force it
        // silently: a wrong enrolment is invisible afterwards and turns that
        // employee's gallery into a magnet for strangers. Ask.
        if (e.response?.status === 409) {
          const detail = e.response?.data?.detail || "This may not be the same person.";
          if (!window.confirm(`${detail}

Enrol anyway?`)) throw e;
          return faceReview.assign(
            c.cluster_id, employeeId, Math.min(5, Math.max(1, c.size)), true
          );
        }
        throw e;
      })
      .then((r) => {
        const added = (r.data as any)?.enrolled ?? (r.data as any)?.added;
        setSuccess(
          `Enrolled ${added ?? "the best"} face(s) for ${who?.full_name ?? "employee"} ` +
          `from camera ${c.cameras.join(", ")}. They should start being recognised there.`
        );
        setClusters((prev) => prev.filter((x) => x.cluster_id !== c.cluster_id));
      })
      .catch((e) => setError(e.response?.data?.detail || "Could not enrol this group."))
      .finally(() => setBusy(null));
  };

  const ignore = (c: Cluster) => {
    setBusy(c.cluster_id);
    setError("");
    faceReview
      .ignore(c.cluster_id)
      .then(() => setClusters((prev) => prev.filter((x) => x.cluster_id !== c.cluster_id)))
      .catch((e) => setError(e.response?.data?.detail || "Could not dismiss this group."))
      .finally(() => setBusy(null));
  };

  const when = (iso: string | null) =>
    iso ? new Date(iso).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : "—";

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Unrecognised Faces</h1>
          <p className="eds-subtitle">
            People the cameras saw but could not name. Attributing a group enrols
            it — this is how an employee gets recognised at a gate.
          </p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        {success && <div className="alert alert-success">{success}</div>}
        {error && <div className="alert alert-error">{error}</div>}

        <div className="note-panel">
          <strong>Uploading more portraits will not fix a gate.</strong> A doorway
          camera sees a small face from above; an uploaded photo is a frontal
          portrait, and a face model scores those two around 0.4 against each
          other — below the 0.45 needed to match. Two frames from the same camera
          score 0.63–0.75. Enrol people from the camera that has to recognise
          them, roughly 10–15 frames each.
        </div>

        <CaptureEnrolment employees={employees} onDone={load} />

        <div className="eds-controls" style={{ alignItems: "flex-end" }}>
          <div className="eds-controls-end">
            <button type="button" className="eds-action eds-action--go"
                    onClick={regroup} disabled={busy !== null}>
              <Icons.Refresh />
              {busy === -1 ? "Regrouping…" : "Regroup by appearance"}
            </button>
          </div>
        </div>

        {loading ? (
          <SectionLoader rows={4} />
        ) : clusters.length === 0 ? (
          <section className="eds-card">
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.Faces /></span>
              <span>
                Nothing waiting for review. If the cameras are still reporting
                Unknown people, press &quot;Regroup by appearance&quot; — sightings are
                only grouped when asked.
              </span>
            </div>
          </section>
        ) : (
          <div className="eds-fr-grid">
            {clusters.map((c) => {
              const suggestion = c.near_miss_employee_id
                ? employees.find((e) => e.id === c.near_miss_employee_id)
                : undefined;
              return (
                <section key={c.cluster_id} className="eds-card eds-fr-card">
                  <ClusterFaces
                    clusterId={c.cluster_id}
                    fallbackFaceId={c.sample_face_id}
                    onOpen={setZoom}
                  />
                  <div className="eds-fr-body">
                    <div className="eds-fr-head">
                      <span className="eds-fr-size">{c.size}</span>
                      <span className="eds-fr-sub">
                        sighting{c.size === 1 ? "" : "s"} · camera {c.cameras.join(", ") || "?"}
                      </span>
                    </div>
                    <p className="eds-fr-meta">
                      {when(c.first_seen)} → {when(c.last_seen)}
                      <br />
                      best quality {c.best_quality.toFixed(2)}
                      {c.best_near_miss_score > 0 && (
                        <> · closest match {c.best_near_miss_score.toFixed(2)}</>
                      )}
                    </p>
                    {suggestion && (
                      <p className="eds-fr-hint">
                        Nearly matched <strong>{suggestion.full_name}</strong> — check
                        the picture before accepting that.
                      </p>
                    )}
                    <CustomSelect
                      className="eds-cselect"
                      value={picked[c.cluster_id] ?? ""}
                      onChange={(v) => setPicked((p) => ({ ...p, [c.cluster_id]: v }))}
                      options={[
                        { value: "", label: "Who is this?" },
                        ...employees.map((e) => ({
                          value: String(e.id),
                          label: `${e.employee_code} — ${e.full_name}`,
                        })),
                      ]}
                    />
                    <div className="eds-fr-actions">
                      <button type="button" className="eds-action eds-action--go"
                              onClick={() => assign(c)}
                              disabled={busy !== null || !picked[c.cluster_id]}>
                        {busy === c.cluster_id ? "Enrolling…" : "Enrol as this person"}
                      </button>
                      <button type="button" className="eds-action"
                              onClick={() => ignore(c)} disabled={busy !== null}
                              title="Not an employee — a visitor or a false detection">
                        Not staff
                      </button>
                    </div>
                  </div>
                </section>
              );
            })}
          </div>
        )}
      </div>

      {zoom && (
        <div
          className="eds-fr-zoom"
          onClick={() => setZoom(null)}
          role="dialog"
          aria-label="Capture, full size"
        >
          <img src={zoom} alt="Unrecognised person, full size" />
          <span className="eds-fr-zoom-cap">
            This is exactly what the camera captured. Click anywhere to close.
          </span>
        </div>
      )}
    </div>
  );
}
