#!/usr/bin/env python3
"""Focused regression tests for backend helper behavior."""

import unittest
from pathlib import Path
import sys

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


if __name__ == "__main__":
    unittest.main()
