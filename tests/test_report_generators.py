"""Regression tests for the one-page evaluation report generators.

These guard the user-facing constraints: the PDF must be exactly one page,
the DOCX must contain no ``<w:hyperlink>`` elements, and neither artefact
may contain timestamps, dates, or contact metadata.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "report" / "AI_Personal_Assistant_Benchmark_Report.pdf"
DOCX_PATH = ROOT / "report" / "AI_Personal_Assistant_Benchmark_Report.docx"

FORBIDDEN_PATTERNS = [
    r"\bgenerated on\b",
    r"\bas of\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\bUTC\b",
    r"\bIST\b",
    r"\bollive\b",
    r"work@",
]


@pytest.fixture(scope="module")
def pdf_reader() -> PdfReader:
    if not PDF_PATH.exists():
        pytest.skip("PDF not generated yet; run python -m report.generate_pdf")
    return PdfReader(str(PDF_PATH))


@pytest.fixture(scope="module")
def docx_xml() -> str:
    if not DOCX_PATH.exists():
        pytest.skip("DOCX not generated yet; run python scripts/generate_eval_docx.py")
    with zipfile.ZipFile(DOCX_PATH) as zf:
        return zf.read("word/document.xml").decode("utf-8")


def test_pdf_is_exactly_one_page(pdf_reader: PdfReader) -> None:
    assert len(pdf_reader.pages) == 1


def test_docx_has_no_hyperlink_elements(docx_xml: str) -> None:
    assert "<w:hyperlink" not in docx_xml


def test_pdf_has_no_forbidden_text(pdf_reader: PdfReader) -> None:
    text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages).lower()
    for pat in FORBIDDEN_PATTERNS:
        assert not re.search(pat, text, flags=re.IGNORECASE), (
            f"PDF contains forbidden pattern: {pat}"
        )


def test_docx_has_no_forbidden_text(docx_xml: str) -> None:
    body = docx_xml.lower()
    for pat in FORBIDDEN_PATTERNS:
        assert not re.search(pat, body, flags=re.IGNORECASE), (
            f"DOCX contains forbidden pattern: {pat}"
        )


def test_docx_has_url_footer(docx_xml: str) -> None:
    assert "github.com/G26karthik/Dual-AI-Assistant-s-Benchmark" in docx_xml
    assert "huggingface.co/spaces/LuciferMrng/dual-ai-assistant-benchmark-oss" in docx_xml
