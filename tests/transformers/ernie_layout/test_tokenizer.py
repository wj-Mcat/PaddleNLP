# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from paddlenlp.transformers.ernie_layout.tokenizer import ErnieLayoutTokenizer
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin


class ErnieLayoutEnglishTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ErnieLayoutTokenizer
    space_between_special_tokens = True

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        return ErnieLayoutTokenizer.from_pretrained("ernie-layoutx-base-uncased", **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "This is a test"
        output_text = "This is a test"
        return input_text, output_text

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "[CLS]"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [475, 98, 6, 4, 264])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                "▁I",
                "▁was",
                "▁b",
                "or",
                "n",
                "▁in",
                "▁",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                "▁and",
                "▁this",
                "▁is",
                "▁f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids, [17, 53, 13, 28, 937, 40, 932, 3, 999, 993, 993, 993, 954, 33, 120, 98, 21, 82, 940, 3, 952]
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                "▁I",
                "▁was",
                "▁b",
                "or",
                "n",
                "▁in",
                "▁",
                "[UNK]",
                "2",
                "0",
                "0",
                "0",
                ",",
                "▁and",
                "▁this",
                "▁is",
                "▁f",
                "al",
                "s",
                "[UNK]",
                ".",
            ],
        )

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["▁T", "est"], ["▁", "\xad"], ["▁t", "est"]]
        )

    def test_sequence_builders(self):
        tokenizer = self.get_tokenizer()

        text = tokenizer.encode("sequence builders", return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=None, add_special_tokens=False)[
            "input_ids"
        ]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [
            tokenizer.sep_token_id,
            tokenizer.sep_token_id,
        ] + text_2 + [tokenizer.sep_token_id]

    def test_add_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                vocab_size = len(tokenizer)
                self.assertEqual(tokenizer.add_tokens(""), 0)
                self.assertEqual(tokenizer.add_tokens("testoken"), 1)
                self.assertEqual(tokenizer.add_tokens(["testoken1", "testtoken2"]), 2)
                self.assertEqual(len(tokenizer), vocab_size + 3)

                self.assertEqual(tokenizer.add_special_tokens({}), 0)
                self.assertRaises(
                    AssertionError, tokenizer.add_special_tokens, {"additional_special_tokens": "<testtoken1>"}
                )
                self.assertEqual(tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken2>"]}), 1)
                self.assertEqual(
                    tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken3>", "<testtoken4>"]}), 2
                )
                self.assertIn("<testtoken3>", tokenizer.special_tokens_map["additional_special_tokens"])
                self.assertIsInstance(tokenizer.special_tokens_map["additional_special_tokens"], list)
                self.assertGreaterEqual(len(tokenizer.special_tokens_map["additional_special_tokens"]), 2)

                self.assertEqual(len(tokenizer), vocab_size + 6)

    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode(
                    "aaaaa bbbbbb low cccccccccdddddddd l", return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

    def test_token_type_ids(self):
        self.skipTest("Ernie-Layout model doesn't have token_type embedding. so skip this test")
