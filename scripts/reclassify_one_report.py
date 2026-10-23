#!/usr/bin/env python3
"""Re-run research-report classification for one report.

Uses DeepSeek, then the OpenRouter fallback inside deepseek_classifier.
``--force-ocr`` runs MinerU on the PDF before classification.

Examples:
    python scripts/reclassify_one_report.py --text data/reports/42/ocr.md
    python scripts/reclassify_one_report.py --report-id 42
    python scripts/reclassify_one_report.py --pdf data/reports/42/report.pdf --force-ocr

Report lookup uses REPORTS_DIR (default data/reports):
    <dir>/<id>.md, <dir>/<id>.pdf, or files under <dir>/<id>/.
MinerU is invoked as ``$MINERU_BIN -p <pdf> -o <outdir>`` (default bin: mineru).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_classifier import ProviderError, classify_report  # noqa: E402

logger = logging.getLogger("reclassify_one_report")

_REPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ReportInputError(Exception):
    """The requested report path or id cannot be classified."""


@dataclass
class LocatedReport:
    report_id: str
    pdf: Path | None
    text_path: Path | None
    title: str


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        located = locate_report(
            report_id=args.report_id,
            pdf=args.pdf,
            text=args.text,
            title=args.title,
            reports_dir=args.reports_dir or _reports_dir(),
        )
        text_path, text = prepare_text(
            located,
            force_ocr=args.force_ocr,
            mineru_bin=args.mineru_bin or os.environ.get("MINERU_BIN", "mineru"),
        )
        result = classify_report(text, title=located.title)
    except ReportInputError as exc:
        logger.error("%s", exc)
        return 2
    except ProviderError as exc:
        logger.error("classification failed: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("classification failed")
        return 1

    payload = {
        "provider": result.provider,
        "model": result.model,
        "report_id": located.report_id,
        "source": str(text_path),
        "classification": result.data,
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    output = args.output
    if output is None:
        output = text_path.with_suffix(".classification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(raw, encoding="utf-8")
    logger.info(
        "classified report %s via %s model %s -> %s",
        located.report_id or text_path.name,
        result.provider,
        result.model,
        output,
    )
    sys.stdout.write(raw)
    return 0


def locate_report(
    *,
    report_id: str,
    pdf: Path | None,
    text: Path | None,
    title: str,
    reports_dir: Path,
) -> LocatedReport:
    if report_id and not _REPORT_ID_RE.fullmatch(report_id):
        raise ReportInputError(f"invalid report id: {report_id}")

    found_pdf = pdf
    found_text = text
    if report_id and found_pdf is None and found_text is None:
        found_pdf, found_text = _find_under_reports_dir(reports_dir, report_id)
        if found_pdf is None and found_text is None:
            raise ReportInputError(
                f"no report files for id {report_id} under {reports_dir}"
            )

    if found_pdf is not None:
        found_pdf = found_pdf.expanduser().resolve()
        if not found_pdf.is_file():
            raise ReportInputError(f"PDF not found: {found_pdf}")
    if found_text is not None:
        found_text = found_text.expanduser().resolve()
        if not found_text.is_file():
            raise ReportInputError(f"text not found: {found_text}")

    if found_pdf is None and found_text is None:
        raise ReportInputError("pass --report-id, --pdf, or --text")

    if found_text is None and found_pdf is not None:
        found_text = _sibling_text(found_pdf)

    label = title.strip()
    if not label:
        source = found_text or found_pdf
        label = source.stem if source is not None else report_id
    return LocatedReport(report_id, found_pdf, found_text, label)


def prepare_text(
    located: LocatedReport,
    *,
    force_ocr: bool,
    mineru_bin: str,
) -> tuple[Path, str]:
    if force_ocr:
        if located.pdf is None:
            raise ReportInputError("--force-ocr needs a PDF (--pdf or a report id with a PDF)")
        out_dir = located.pdf.parent / "mineru" / located.pdf.stem
        text_path = run_mineru(located.pdf, out_dir, mineru_bin)
        return text_path, _read_text(text_path)
    if located.text_path is None:
        raise ReportInputError(
            "no OCR text found; pass --text or re-run with --force-ocr to run MinerU"
        )
    return located.text_path, _read_text(located.text_path)


def run_mineru(pdf: Path, out_dir: Path, mineru_bin: str) -> Path:
    """Run MinerU and return the markdown file it wrote."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [mineru_bin, "-p", str(pdf), "-o", str(out_dir)]
    logger.info("MinerU re-OCR starting: %s", " ".join(command))
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ReportInputError(
            f"MinerU executable not found: {mineru_bin}. Install MinerU or set MINERU_BIN."
        ) from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise ReportInputError(f"MinerU failed (exit {completed.returncode}): {tail}")
    markdown = [path for path in out_dir.rglob("*.md") if path.is_file()]
    if not markdown:
        raise ReportInputError(f"MinerU wrote no markdown under {out_dir}")
    markdown.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    logger.info("MinerU re-OCR wrote %s", markdown[0])
    return markdown[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-classify one research report. "
            "DeepSeek is tried first; HTTP 402 / insufficient balance and similar "
            "hard failures fall back to OpenRouter."
        )
    )
    parser.add_argument("--report-id", default="", help="Report id under REPORTS_DIR")
    parser.add_argument("--pdf", type=Path, help="PDF to classify (OCR text or --force-ocr)")
    parser.add_argument("--text", type=Path, help="Existing OCR markdown or plain text")
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Re-run MinerU on the PDF before classification",
    )
    parser.add_argument("--title", default="", help="Report title passed to the classifier")
    parser.add_argument("--output", type=Path, help="Classification JSON path")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Directory of reports (default: REPORTS_DIR or data/reports)",
    )
    parser.add_argument(
        "--mineru-bin",
        default="",
        help="MinerU executable (default: MINERU_BIN or mineru)",
    )
    return parser


def _reports_dir() -> Path:
    return Path(os.environ.get("REPORTS_DIR", "data/reports")).expanduser()


def _find_under_reports_dir(reports_dir: Path, report_id: str) -> tuple[Path | None, Path | None]:
    root = reports_dir.expanduser()
    direct_text = _first_existing(
        [root / f"{report_id}.md", root / f"{report_id}.txt"]
    )
    direct_pdf = _first_existing([root / f"{report_id}.pdf"])
    folder = root / report_id
    folder_text = None
    folder_pdf = None
    if folder.is_dir():
        texts = [
            path
            for pattern in ("*.md", "*.txt")
            for path in folder.rglob(pattern)
            if path.is_file() and "mineru" not in path.parts
        ]
        pdfs = [path for path in folder.rglob("*.pdf") if path.is_file()]
        texts.sort(key=lambda path: (len(path.parts), str(path)))
        pdfs.sort(key=lambda path: str(path))
        folder_text = texts[0] if texts else None
        folder_pdf = pdfs[0] if pdfs else None
    return folder_pdf or direct_pdf, folder_text or direct_text


def _sibling_text(pdf: Path) -> Path | None:
    candidates = [pdf.with_suffix(suffix) for suffix in (".md", ".txt")]
    parent = pdf.parent
    candidates.extend(
        [
            parent / f"{pdf.stem}.md",
            parent / f"{pdf.stem}.txt",
        ]
    )
    return _first_existing(candidates)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ReportInputError(f"text file is empty: {path}")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
