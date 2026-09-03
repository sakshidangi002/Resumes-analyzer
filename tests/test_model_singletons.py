"""Lazy AI-model singletons must load exactly once under concurrency.

`if _model is None: _model = load()` is not atomic. These loaders run inside the
Resume API's 4-worker executor, so a burst of requests arriving before a model
is warm put several threads inside that branch at once and each loaded its own
copy of a multi-GB transformer. Measured before the fix: 12 threads produced 12
loads.
"""
import sys
import threading
import time
import types

import pytest

# backend.main pulls in pdfplumber / python-docx for resume text extraction,
# which CI deliberately omits (see .github/workflows/ci.yml). Skip visibly rather than
# failing collection; CI runs with -rs so this is reported.
try:
    from backend import main as resume_main
except ImportError as exc:  # pragma: no cover - depends on the environment
    pytest.skip(
        f"requires the resume extraction stack ({exc}); "
        "install requirements.txt to run",
        allow_module_level=True,
    )


def test_embedding_model_loads_once_under_concurrent_first_use(monkeypatch):
    load_count = 0
    count_lock = threading.Lock()

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            nonlocal load_count
            time.sleep(0.2)          # widen the race window
            with count_lock:
                load_count += 1

    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)

    monkeypatch.setattr(resume_main, "_embedding_model", None, raising=False)

    threads_count = 12
    barrier = threading.Barrier(threads_count)
    results = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait()               # all threads enter together
        model = resume_main.get_embedding_model()
        with results_lock:
            results.append(model)

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_count == 1, f"model loaded {load_count} times — locking regressed"
    assert len({id(r) for r in results}) == 1, "threads received different instances"
