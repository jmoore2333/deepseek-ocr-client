#!/usr/bin/env python3
"""Focused regression tests for backend helper behavior."""

import unittest
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
from xml.etree import ElementTree as ET

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
    @unittest.skipUnless(ocr_server.find_tesseract_executable(), "Tesseract is not installed")
    def test_create_searchable_pdf_adds_positioned_text_layer(self):
        from PIL import Image, ImageDraw, ImageFont
        from pypdf import PdfReader

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_pdf = Path(tmp_dir) / "source.pdf"
            output_pdf = Path(tmp_dir) / "searchable.pdf"

            image = Image.new("RGB", (600, 300), "white")
            draw = ImageDraw.Draw(image)
            font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
            font = ImageFont.truetype(str(font_path), 48) if font_path.exists() else ImageFont.load_default()
            draw.text((330, 205), "MOSCOW", fill="black", font=font)
            image.save(source_pdf, "PDF", resolution=150)

            ocr_server.create_searchable_pdf(str(source_pdf), str(output_pdf), dpi=200, psm=6)

            reader = PdfReader(str(output_pdf))
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
            self.assertIn("MOSCOW", extracted.upper())

            pdftotext = shutil.which("pdftotext")
            if not pdftotext:
                return

            bbox_path = Path(tmp_dir) / "bbox.html"
            subprocess.run(
                [pdftotext, "-bbox", str(output_pdf), str(bbox_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            root = ET.parse(bbox_path).getroot()
            namespace = {"x": "http://www.w3.org/1999/xhtml"}
            words = root.findall(".//x:word", namespace)
            moscow = next(word for word in words if "MOSCOW" in "".join(word.itertext()).upper())
            page = root.find(".//x:page", namespace)
            page_width = float(page.attrib["width"])
            page_height = float(page.attrib["height"])

            self.assertGreater(float(moscow.attrib["xMin"]), page_width * 0.45)
            self.assertGreater(float(moscow.attrib["yMin"]), page_height * 0.55)


if __name__ == "__main__":
    unittest.main()
