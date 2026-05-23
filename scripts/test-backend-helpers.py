#!/usr/bin/env python3
"""Focused regression tests for backend helper behavior."""

import unittest
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend import ocr_server


MULTI_BOX_TOKEN_TEXT = (
    "Before\n"
    "<|ref|>Important Safeguards<|/ref|>"
    "<|det|>[[78, 321, 465, 349], [532, 134, 888, 172]]<|/det|>\n"
    "After"
)


class MlxOutputNormalizationTests(unittest.TestCase):
    def test_document_output_strips_grounding_tokens_with_multiple_boxes(self):
        cleaned, raw_tokens = ocr_server.normalize_mlx_output("document", MULTI_BOX_TOKEN_TEXT)

        self.assertEqual(raw_tokens, MULTI_BOX_TOKEN_TEXT)
        self.assertEqual(cleaned, "Before\n\nAfter")
        self.assertNotIn("<|ref|>", cleaned)
        self.assertNotIn("<|det|>", cleaned)

    def test_ocr_output_extracts_ref_text_from_multiple_box_grounding_token(self):
        cleaned, raw_tokens = ocr_server.normalize_mlx_output("ocr", MULTI_BOX_TOKEN_TEXT)

        self.assertEqual(raw_tokens, MULTI_BOX_TOKEN_TEXT)
        self.assertEqual(cleaned, "Important Safeguards")


class SearchablePdfTests(unittest.TestCase):
    def test_extract_searchable_pdf_page_texts_handles_document_markdown(self):
        result_text = (
            "## Page 1\n\nCover text\n\n"
            "## Page 2\n\nRecipe text\n\n"
            "## Page 3\n\nIndex text"
        )

        self.assertEqual(
            ocr_server.extract_searchable_pdf_page_texts(result_text, page_labels=[1, 2, 3]),
            ["Cover text", "Recipe text", "Index text"],
        )

    def test_extract_searchable_pdf_page_texts_handles_free_ocr_markers(self):
        result_text = (
            "--- Page 1 ---\nCover text\n\n"
            "--- Page 2 ---\nRecipe text"
        )

        self.assertEqual(
            ocr_server.extract_searchable_pdf_page_texts(result_text, page_labels=[1, 2]),
            ["Cover text", "Recipe text"],
        )

    def test_create_searchable_pdf_adds_extractable_text_layer(self):
        from pypdf import PdfReader
        from reportlab.pdfgen import canvas

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_pdf = Path(tmp_dir) / "source.pdf"
            output_pdf = Path(tmp_dir) / "searchable.pdf"

            c = canvas.Canvas(str(source_pdf), pagesize=(200, 200))
            c.drawString(40, 100, "Scanned-looking source")
            c.showPage()
            c.save()

            ocr_server.create_searchable_pdf(
                str(source_pdf),
                ["Aviation cocktail recipe with gin and lemon"],
                str(output_pdf),
            )

            reader = PdfReader(str(output_pdf))
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
            self.assertIn("Aviation cocktail recipe", extracted)


if __name__ == "__main__":
    unittest.main()
