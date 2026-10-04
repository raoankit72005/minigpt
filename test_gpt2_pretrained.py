"""Offline parity tests; no pretrained download required."""
import unittest
import torch
from transformers import GPT2Config as HFConfig, GPT2LMHeadModel
from gpt2_pretrained import GPT2, config_from_hf, load_hf_weights


class WeightMappingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        cfg = HFConfig(vocab_size=37, n_positions=16, n_embd=24, n_head=4,
                       n_layer=2, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
        cfg._attn_implementation = "eager"
        self.reference = GPT2LMHeadModel(cfg).eval()
        self.model = load_hf_weights(GPT2(config_from_hf(cfg)), self.reference.state_dict())

    def test_logits_match_reference(self):
        for length in (1, 7, 16):
            ids = torch.randint(37, (2, length))
            with torch.inference_mode():
                torch.testing.assert_close(self.model(ids), self.reference(ids).logits,
                                           rtol=1e-4, atol=1e-5)
        self.assertIs(self.model.output.weight, self.model.token_embedding.weight)

    def test_causality(self):
        ids = torch.randint(37, (1, 10))
        changed = ids.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % 37
        with torch.inference_mode():
            torch.testing.assert_close(self.model(ids)[:, :5], self.model(changed)[:, :5])

    def test_invalid_shape_rejected(self):
        state = dict(self.reference.state_dict())
        state["transformer.h.0.attn.c_attn.weight"] = torch.zeros(1, 1)
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            load_hf_weights(self.model, state)

    def test_generation_crops_context(self):
        ids = torch.randint(37, (1, 16))
        result = self.model.generate(ids, max_new_tokens=3, top_k=1, eos_id=-1)
        self.assertEqual(result.shape, (1, 19))
        self.assertTrue(torch.equal(ids, result[:, :16]))


if __name__ == "__main__":
    unittest.main()
