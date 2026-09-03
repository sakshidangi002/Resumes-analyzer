"""Export AdaFace ir_50 to ONNX, then PROVE the vectors are unchanged.

Why the proof matters more than the export
------------------------------------------
Every enrolled face in the database is an AdaFace vector, and a match is a
cosine similarity between a live vector and those. If the ONNX build produced
even slightly different vectors, every enrolment would silently drift and
recognition would degrade in a way that looks like "the cameras got worse" —
the exact failure this deployment already lived through once when the gallery
was rebuilt from portraits.

So this script exports, then compares the two implementations on real face
crops, and reports the worst disagreement it can find. Nothing is wired in
here; that only happens if the numbers say it is safe.
"""
import glob
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(r"c:\sakshi folder\application\Resume analyzer"
               r"\Attendance Management\backend")
SOURCE = BACKEND / "third_party" / "AdaFace"
WEIGHTS = SOURCE / "pretrained" / "adaface_ir50_webface4m.ckpt"
OUT = BACKEND / "models" / "adaface_ir50_webface4m.onnx"

os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402
import net    # noqa: E402

print("loading the PyTorch model …")
model = net.build_model("ir_50")
ckpt = torch.load(str(WEIGHTS), map_location="cpu")
state = ckpt.get("state_dict", ckpt)
state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
model.load_state_dict(state)
model.eval()


class EmbeddingOnly(torch.nn.Module):
    """AdaFace returns (feature, quality_norm); ONNX only needs the feature.

    Exporting the tuple works but leaves a second output that every caller has
    to remember to ignore — and forgetting would silently feed a norm into the
    gallery.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)[0]


if not OUT.exists():
    print(f"exporting -> {OUT}")
    torch.onnx.export(
        EmbeddingOnly(model),
        torch.randn(1, 3, 112, 112),
        str(OUT),
        input_names=["input"],
        output_names=["embedding"],
        # Batch stays dynamic so several faces from one frame could be embedded
        # in a single call later; the pipeline does one at a time today.
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        # Legacy TorchScript exporter. torch 2.12 defaults to the dynamo path,
        # which needs onnxscript — a new dependency in a production venv, added
        # to export a model once. The TorchScript path handles a plain CNN
        # perfectly well and the equivalence check below is what decides
        # whether the result is usable, not which exporter produced it.
        dynamo=False,
    )
    print(f"   {OUT.stat().st_size / 1e6:.1f} MB")
else:
    print(f"already exported: {OUT}")

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])


def torch_embed(batch):
    with torch.inference_mode():
        feat = model(torch.from_numpy(batch))[0]
    return feat.detach().float().cpu().numpy()


def onnx_embed(batch):
    return sess.run(["embedding"], {"input": batch})[0]


def norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1)


# ── Real faces, not noise ──────────────────────────────────────────────────
# A random tensor exercises the graph but not the activations a real face
# produces; a difference that only shows on real input would be missed.
crops = []
for p in sorted(glob.glob(str(BACKEND / "data" / "unknown_faces" / "**" / "*.jpg"),
                          recursive=True))[:60]:
    import cv2
    im = cv2.imread(p)
    if im is None:
        continue
    im = cv2.resize(im, (112, 112))
    t = ((im.astype(np.float32) / 255.0) - 0.5) / 0.5      # same as adaface_service
    # Contiguous: AdaFace uses .view() internally and rejects a transposed view.
    crops.append(np.ascontiguousarray(t.transpose(2, 0, 1)))

if len(crops) < 5:
    print("not enough real crops; falling back to random input")
    crops = [np.random.randn(3, 112, 112).astype(np.float32) for _ in range(20)]

X = np.ascontiguousarray(np.stack(crops).astype(np.float32))
print(f"\ncomparing on {len(X)} real face crops …")

T = norm(torch_embed(X))
O = norm(onnx_embed(X))

cos = np.sum(T * O, axis=1)
absdiff = np.abs(T - O)

print()
print("EQUIVALENCE — normalised vectors, PyTorch vs ONNX")
print(f"  cosine similarity   min {cos.min():.8f}   mean {cos.mean():.8f}")
print(f"  max abs difference  {absdiff.max():.2e}")
print(f"  vectors below 0.9999 cosine: {int((cos < 0.9999).sum())} of {len(cos)}")

# The number that actually matters: does a match against the gallery change?
worst = float(1.0 - cos.min())
print(f"\n  worst possible score shift on a gallery match: {worst:.2e}")
print(f"  (the match margin bar is 0.18, the threshold 0.45)")

# ── Speed ──────────────────────────────────────────────────────────────────
one = X[:1]
for name, fn in (("PyTorch eager", torch_embed), ("ONNX Runtime", onnx_embed)):
    for _ in range(3):
        fn(one)
    ts = []
    for _ in range(12):
        t0 = time.perf_counter()
        fn(one)
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"\n{name:<16} {statistics.median(ts):>7.1f} ms/face   "
          f"(min {min(ts):.1f}, max {max(ts):.1f})")
