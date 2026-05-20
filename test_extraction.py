import argparse
import json
import os
import sys


def _load_main_module():
    # Keep the test local and lightweight: import the app module only when run.
    import main

    return main


def main_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local resume PDF extraction quality check."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=os.environ.get("SAMPLE_PDF", ""),
        help="Path to a sample resume PDF.",
    )
    args = parser.parse_args()

    if not args.pdf:
        print("Usage: python test_extraction.py <resume.pdf>")
        return 2
    if not os.path.exists(args.pdf):
        print(f"File not found: {args.pdf}")
        return 2

    main_mod = _load_main_module()

    # Show the quality gate inputs and the final text chosen by the extraction layer.
    text, meta = main_mod.run_fallback_extraction(args.pdf)
    normalized_text, norm_meta = main_mod.normalize_resume_text(text)
    quality = main_mod.score_extraction_quality(text)
    normalized_quality = main_mod.score_extraction_quality(normalized_text)

    print(json.dumps(
        {
            "file": os.path.abspath(args.pdf),
            "extraction": meta,
            "raw_quality": quality,
            "normalized_quality": normalized_quality,
            "normalization_actions": norm_meta.get("actions", []),
            "text_preview": text[:1000],
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
