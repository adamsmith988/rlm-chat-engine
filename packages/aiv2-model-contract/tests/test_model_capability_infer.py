"""Tests for aiv2_model_contract.model_capability_infer."""

import unittest

from aiv2_model_contract.model_capability_infer import (
    infer_io_profile,
    merge_io_candidates,
    modalities_from_tensor_names,
    profile_from_capabilities,
)


class TestModelCapabilityInfer(unittest.TestCase):
    def test_vlm_from_capabilities(self) -> None:
        prof = profile_from_capabilities(["chat", "vlm"])
        self.assertIn("image", prof["input_modalities"])
        self.assertIn("text", prof["output_modalities"])

    def test_embed_task(self) -> None:
        prof = profile_from_capabilities(["embed"], task="embed")
        self.assertEqual(prof["output_modalities"], ["embedding"])
        self.assertNotIn("dispatch_key", prof)

    def test_embed_from_capabilities(self) -> None:
        prof = profile_from_capabilities(["embed"])
        self.assertEqual(prof["output_modalities"], ["embedding"])
        self.assertEqual(prof["api"], "embed")

    def test_sidecar_override_wins(self) -> None:
        out = infer_io_profile(
            sidecar={"io_profile": {"input_modalities": ["text"], "output_modalities": ["text"], "api": "chat"}},
            extractor_signals=[
                {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "api": "chat",
                    "confidence": 0.5,
                    "source": "low",
                }
            ],
        )
        self.assertEqual(out["input_modalities"], ["text"])

    def test_tensor_name_modalities(self) -> None:
        inp, out = modalities_from_tensor_names(
            {
                "inputs": [{"name": "input_ids"}],
                "outputs": [{"name": "embedding"}],
            }
        )
        self.assertIn("text", inp)
        self.assertIn("embedding", out)

    def test_ocr_triton_tensors_are_image_to_text_not_embedding(self) -> None:
        """PaddleOCR-style opaque tensors must stay unknown — never invent embedding."""
        inp, out = modalities_from_tensor_names(
            {
                "inputs": [{"name": "x", "shape": [-1, 3, -1, -1]}],
                "outputs": [{"name": "softmax_2.tmp_0", "shape": [-1, 97]}],
            }
        )
        self.assertEqual(inp, [])
        self.assertEqual(out, [])

        # Named embedder tensors still resolve.
        inp2, out2 = modalities_from_tensor_names(
            {
                "inputs": [{"name": "pixel_values"}],
                "outputs": [{"name": "embedding"}],
            }
        )
        self.assertEqual(inp2, ["image"])
        self.assertEqual(out2, ["embedding"])

    def test_ocr_capability_beats_empty_tensor_output(self) -> None:
        out = infer_io_profile(
            extractor_signals=[
                {
                    "input_modalities": [],
                    "output_modalities": [],
                    "api": "infer",
                    "confidence": 0.2,
                    "source": "triton:config.pbtxt",
                    "tensor_io": {"inputs": [{"name": "x"}], "outputs": [{"name": "683"}]},
                }
            ],
            detected={"capabilities": ["ocr"], "suggested_task": "ocr"},
        )
        # Capability may suggest image→text, but must stay needs_review for the operator.
        self.assertTrue(out.get("needs_review"))
        self.assertEqual(out["input_modalities"], ["image"])
        self.assertEqual(out["output_modalities"], ["text"])

    def test_unknown_tensors_stay_empty_without_capability(self) -> None:
        out = infer_io_profile(
            extractor_signals=[
                {
                    "input_modalities": [],
                    "output_modalities": [],
                    "confidence": 0.2,
                    "source": "triton:config.pbtxt",
                    "tensor_io": {"inputs": [{"name": "x"}], "outputs": [{"name": "out"}]},
                }
            ],
        )
        self.assertEqual(out["input_modalities"], [])
        self.assertEqual(out["output_modalities"], [])
        self.assertTrue(out.get("needs_review"))

    def test_image_gen_capability_outputs_image(self) -> None:
        prof = profile_from_capabilities(["image-gen"])
        self.assertEqual(prof["output_modalities"], ["image"])
        self.assertEqual(prof["api"], "image")

    def test_pipeline_tag_image_text_to_image(self) -> None:
        out = infer_io_profile(
            detected={"pipeline_tag": "image-text-to-image", "capabilities": ["image-gen"]},
        )
        self.assertIn("image", out["output_modalities"])
        self.assertNotEqual(out["output_modalities"], ["text"])

    def test_gguf_architecture_is_strong_source(self) -> None:
        out = infer_io_profile(
            extractor_signals=[
                {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "api": "chat",
                    "confidence": 0.95,
                    "source": "gguf:architecture:qwen3",
                }
            ],
        )
        self.assertEqual(out["input_modalities"], ["text"])
        self.assertEqual(out["output_modalities"], ["text"])
        self.assertNotIn("dispatch_key", out)
        self.assertFalse(out.get("needs_review"))
        self.assertTrue(any(s.startswith("gguf:architecture:") for s in out.get("sources") or []))

    def test_gguf_filename_hint_still_needs_review(self) -> None:
        out = infer_io_profile(
            extractor_signals=[
                {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "api": "chat",
                    "confidence": 0.7,
                    "source": "gguf:filename:vlm",
                }
            ],
        )
        self.assertTrue(out.get("needs_review"))
        self.assertEqual(out["input_modalities"], ["text", "image"])


if __name__ == "__main__":
    unittest.main()
