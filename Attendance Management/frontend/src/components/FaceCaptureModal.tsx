import { useEffect, useRef, useState } from "react";

/**
 * Guided multi-angle face capture. Walks the user through several head poses,
 * grabs one webcam frame per pose, and returns them as JPEG Files. Capturing
 * multiple angles is what makes the multi-embedding recognition accurate.
 */
const STEPS = [
  { key: "front", label: "Look straight at the camera", hint: "Face centered, eyes forward, good light" },
  { key: "left", label: "Turn your head slightly LEFT", hint: "About 15–20°" },
  { key: "right", label: "Turn your head slightly RIGHT", hint: "About 15–20°" },
  { key: "up", label: "Lift your chin slightly UP", hint: "Small tilt only" },
  { key: "normal", label: "Look straight again", hint: "Neutral expression" },
];

export default function FaceCaptureModal({
  onDone,
  onClose,
}: {
  onDone: (files: File[]) => void;
  onClose: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [step, setStep] = useState(0);
  const [captured, setCaptured] = useState<File[]>([]);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let active = true;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false })
      .then((stream) => {
        if (!active) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setReady(true);
      })
      .catch(() => setError("Cannot access the webcam. Allow camera permission, or use file upload instead."));
    return () => {
      active = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const stopCam = () => streamRef.current?.getTracks().forEach((t) => t.stop());

  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `face_${STEPS[step].key}.jpg`, { type: "image/jpeg" });
        const next = [...captured, file];
        setCaptured(next);
        if (step + 1 < STEPS.length) {
          setStep(step + 1);
        } else {
          stopCam();
          onDone(next);
        }
      },
      "image/jpeg",
      0.92,
    );
  };

  const cancel = () => {
    stopCam();
    onClose();
  };

  return (
    <div className="modal-backdrop" onClick={cancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 560 }}>
        <h3 style={{ marginTop: 0 }}>Guided Face Capture</h3>
        <p style={{ marginTop: 0, fontSize: "0.85rem", opacity: 0.7 }}>
          Capture {STEPS.length} angles for accurate recognition. Keep your whole face in view.
        </p>

        {error ? (
          <div style={{ color: "#f87171", padding: "1rem 0" }}>{error}</div>
        ) : (
          <>
            <div style={{ position: "relative", background: "#000", borderRadius: 10, overflow: "hidden" }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: "100%", display: "block", transform: "scaleX(-1)" }}
              />
              <div
                style={{
                  position: "absolute", top: 10, left: 10, right: 10,
                  background: "rgba(0,0,0,0.6)", color: "#fff", padding: "0.5rem 0.75rem",
                  borderRadius: 8, fontSize: "0.9rem", fontWeight: 700,
                }}
              >
                Step {step + 1}/{STEPS.length}: {STEPS[step].label}
                <div style={{ fontSize: "0.78rem", fontWeight: 400, opacity: 0.85 }}>{STEPS[step].hint}</div>
              </div>
            </div>

            {/* progress dots */}
            <div style={{ display: "flex", gap: 6, justifyContent: "center", margin: "0.9rem 0" }}>
              {STEPS.map((s, i) => (
                <span
                  key={s.key}
                  style={{
                    width: 10, height: 10, borderRadius: "50%",
                    background: i < captured.length ? "#22c55e" : i === step ? "#7aa2ff" : "rgba(255,255,255,0.2)",
                  }}
                />
              ))}
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.6rem" }}>
              <button type="button" className="btn btn-cancel-alt" onClick={cancel}>Cancel</button>
              <button type="button" className="btn btn-primary" onClick={capture} disabled={!ready}>
                {ready ? `📷 Capture (${captured.length}/${STEPS.length})` : "Starting camera…"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
