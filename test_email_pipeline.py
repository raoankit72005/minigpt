import json
import unittest
import torch
from finetune_email import clean_and_split, collate, encode_record, format_prompt


class TokenizerStub:
    eos_token_id = 999
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class EmailPipelineTests(unittest.TestCase):
    def test_response_only_labels_and_padding(self):
        tokenizer = TokenizerStub()
        row = {"instruction": "Write a meeting email", "response": "Subject: Meeting\nHello"}
        record = encode_record(row, tokenizer, 200)
        prompt_length = len(format_prompt(row["instruction"]))
        self.assertEqual(record["labels"][:prompt_length], [-100] * prompt_length)
        self.assertEqual(record["labels"][prompt_length:], record["input_ids"][prompt_length:])
        short = {"input_ids": [1, 2], "labels": [-100, 2]}
        batch = collate([record, short], tokenizer.eos_token_id)
        self.assertTrue(torch.all(batch["labels"][1, 2:] == -100))
        self.assertTrue(torch.all(batch["attention_mask"][1, 2:] == 0))
        self.assertIsNone(encode_record(row, tokenizer, 10))

    def test_purpose_groups_disjoint_and_deduplicated(self):
        rows = []
        for purpose in range(20):
            for tone in ("friendly", "formal"):
                rows.append({"id": len(rows), "instruction": f"Write a {tone} email regarding purpose {purpose}, specifically today.",
                             "output": json.dumps({"subject": f"Purpose {purpose}",
                                                   "body": f"Dear recipient,\\n{tone} " + f"body {purpose} " * 15})})
        rows.append(rows[0])
        splits = clean_and_split(rows)
        groups = {name: {r["group"] for r in values} for name, values in splits.items()}
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertEqual(sum(map(len, splits.values())), 40)
        self.assertTrue(all("\\n" not in r["response"] for v in splits.values() for r in v))


if __name__ == "__main__":
    unittest.main()
