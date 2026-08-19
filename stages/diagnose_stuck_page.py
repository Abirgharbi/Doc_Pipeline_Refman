"""
diagnose_stuck_page.py — Identify exactly which page(s) in a PDF cause Docling
to hang, by converting ONE page at a time inside a killable subprocess with a
short timeout.

Usage (from the folder containing doc_pipeline/):
    python -u diagnose_stuck_page.py "doc_pipeline\\refman\\rm0399-....pdf" --start 1 --end 50

This will print, for every page in [start, end]:
    page N: OK in X.Xs
or
    page N: STUCK (killed after Ts)

Once you know which page(s) are stuck, we can decide whether to skip them,
pre-process them differently, or investigate what's on them (e.g. open just
that page in a PDF viewer).
"""
from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import sys
import tempfile
import time

# Make sure doc_pipeline's dependencies (docling etc.) are importable when this
# script is run from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _convert_one_page_worker(result_queue: "multiprocessing.Queue", pdf_path: str, page_num_1based: int) -> None:
    """
    Runs in a child process. Extracts page `page_num_1based` (1-indexed) from
    the source PDF into a standalone 1-page PDF, then runs Docling on it.
    Pushes ("ok", elapsed) or ("error", message) onto the queue.
    """
    t0 = time.monotonic()
    try:
        from pypdf import PdfReader, PdfWriter
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num_1based - 1])

        buf = io.BytesIO()
        writer.write(buf)
        page_bytes = buf.getvalue()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(page_bytes)
            tmp_path = tmp.name

        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = 1.0
            pipeline_options.do_ocr = False

            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
            result = converter.convert(tmp_path)
            _ = result.document.export_to_markdown()  # force full materialisation
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        elapsed = time.monotonic() - t0
        result_queue.put(("ok", elapsed))

    except BaseException as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        result_queue.put(("error", f"{type(exc).__name__}: {exc} (after {elapsed:.1f}s)"))


def test_page(pdf_path: str, page_num: int, timeout: int) -> tuple[str, float]:
    """Returns (status, elapsed) where status is 'ok', 'error', or 'stuck'."""
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_convert_one_page_worker,
        args=(result_queue, pdf_path, page_num),
        daemon=True,
    )
    t0 = time.monotonic()
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.kill()
        process.join(timeout=10)
        return "stuck", time.monotonic() - t0

    try:
        status, payload = result_queue.get(timeout=5)
    except Exception:
        return "crashed", time.monotonic() - t0

    if status == "ok":
        return "ok", payload
    return "error", time.monotonic() - t0


def main():
    parser = argparse.ArgumentParser(description="Find which PDF page makes Docling hang.")
    parser.add_argument("pdf_path", help="Path to the PDF file to diagnose")
    parser.add_argument("--start", type=int, default=1, help="First page to test (1-indexed)")
    parser.add_argument("--end", type=int, default=50, help="Last page to test (inclusive)")
    parser.add_argument("--timeout", type=int, default=60, help="Seconds before a page is declared stuck")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf_path)
    if not os.path.exists(pdf_path):
        print(f"ERROR: file not found: {pdf_path}")
        sys.exit(1)

    print(f"Diagnosing '{pdf_path}'")
    print(f"Testing pages {args.start} to {args.end}, timeout={args.timeout}s/page\n", flush=True)

    stuck_pages: list[int] = []

    for page_num in range(args.start, args.end + 1):
        print(f"page {page_num}: testing...", end=" ", flush=True)
        status, elapsed = test_page(pdf_path, page_num, args.timeout)

        if status == "ok":
            print(f"OK in {elapsed:.1f}s")
        elif status == "stuck":
            print(f"STUCK (killed after {elapsed:.1f}s)")
            stuck_pages.append(page_num)
        elif status == "error":
            print(f"ERROR ({elapsed})")
        else:
            print(f"CRASHED (no result, after {elapsed:.1f}s)")

    print("\n--- Summary ---")
    if stuck_pages:
        print(f"Stuck page(s): {stuck_pages}")
    else:
        print("No stuck pages found in this range.")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for Windows + multiprocessing
    main()