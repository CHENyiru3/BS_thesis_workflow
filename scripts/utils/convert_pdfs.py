#!/usr/bin/env python3
"""
Convert local PDFs to Markdown using `magic-pdf`.
"""
import os
import subprocess
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
WORKFLOW_ROOT = SCRIPT_ROOT.parent.parent
THESIS_ROOT = WORKFLOW_ROOT.parent

INPUT_DIR = Path(os.environ.get("PDF_INPUT_DIR", str(THESIS_ROOT / "Ref" / "Ref_collection")))
OUTPUT_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", str(THESIS_ROOT / "Ref" / "Markitdown")))


def convert_one(pdf_path: Path, output_dir: Path, idx: int, total: int) -> str:
    output_path = output_dir / f"{pdf_path.stem}.md"
    if output_path.exists():
        print(f"[{idx}/{total}] Skip {pdf_path.name} (already converted)")
        return "skipped"

    print(f"[{idx}/{total}] Converting {pdf_path.name}")
    cmd = ["uv", "run", "magic-pdf", "-p", str(pdf_path), "-o", str(output_dir)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {pdf_path.name}")
        return "failed"
    except Exception as exc:  # pragma: no cover - safety fallback for local CLI issues
        print(f"  Error: {pdf_path.name} - {exc}")
        return "failed"

    if result.returncode == 0:
        print(f"  Done: {pdf_path.name}")
        return "ok"

    print(f"  Failed: {pdf_path.name}")
    if result.stderr:
        print(f"  stderr: {result.stderr[:200]}")
    return "failed"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    status_counts = {"ok": 0, "failed": 0, "skipped": 0}
    total = len(pdf_files)
    for idx, pdf_path in enumerate(pdf_files, start=1):
        status = convert_one(pdf_path, OUTPUT_DIR, idx, total)
        status_counts[status] += 1

    print("\nConversion summary")
    print(f"  successful: {status_counts['ok']}")
    print(f"  failed: {status_counts['failed']}")
    print(f"  skipped: {status_counts['skipped']}")
    print(f"  total: {total}")


if __name__ == "__main__":
    main()
