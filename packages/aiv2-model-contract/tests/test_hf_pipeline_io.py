"""Tests for aiv2_model_contract.hf_pipeline_io."""

from __future__ import annotations

import unittest

from aiv2_model_contract.hf_pipeline_io import (
    hf_pipeline_tag_ids,
    load_hf_pipeline_snapshot,
    resolve_pipeline_tag_io,
    soft_parse_pipeline_tag_io,
)
from aiv2_model_contract.llm_taxonomy import pipeline_defaults


class TestHfPipelineIo(unittest.TestCase):
    def test_snapshot_loaded(self) -> None:
        snap = load_hf_pipeline_snapshot()
        self.assertGreaterEqual(len(snap.get("pipeline_tag_ids") or []), 50)
        self.assertIn("image-text-to-image", hf_pipeline_tag_ids())

    def test_curated_image_text_to_image(self) -> None:
        io = resolve_pipeline_tag_io("image-text-to-image")
        self.assertEqual(io["output_modalities"], ["image"])
        self.assertIn("image", io["input_modalities"])
        self.assertEqual(io["api"], "image")
        self.assertEqual(io["api_protocol"], "openai_compat_images")
        self.assertNotIn("dispatch_key", io)
        self.assertFalse(io["needs_review"])
        self.assertTrue(io["known_in_hf_catalog"])

    def test_unknown_tag_not_forced_to_chat(self) -> None:
        io = resolve_pipeline_tag_io("totally-made-up-xyz")
        self.assertNotEqual(io.get("api_protocol"), "openai_compat_chat")
        self.assertTrue(io["needs_review"])
        self.assertFalse(io["known_in_hf_catalog"])

    def test_soft_parse_x_to_y(self) -> None:
        soft = soft_parse_pipeline_tag_io("audio-text-to-text")
        self.assertIn("audio", soft["input_modalities"])
        self.assertIn("text", soft["input_modalities"])
        self.assertEqual(soft["output_modalities"], ["text"])

    def test_pipeline_defaults_no_chat_fallback(self) -> None:
        d = pipeline_defaults("graph-ml")
        # Must not silently become text-generation chat
        self.assertNotEqual(d.get("pipeline_tag"), "text-generation")
        if not d.get("input_modalities") and not d.get("output_modalities"):
            self.assertTrue(d.get("needs_review"))

    def test_legacy_text_generation_still_works(self) -> None:
        d = pipeline_defaults("text-generation")
        self.assertEqual(d["output_modalities"], ["text"])
        self.assertEqual(d["api"], "chat")


if __name__ == "__main__":
    unittest.main()
