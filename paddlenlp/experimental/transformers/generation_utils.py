# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import json
from typing import Callable, List, Optional, Union

import paddle
import paddle.nn.functional as F
from paddlenlp_ops import (
    get_token_penalty_multi_scores,
    get_token_penalty_multi_scores_v2,
    save_output,
    save_with_output,
    set_stop_value_multi_ends,
    set_stop_value_multi_ends_v2,
    set_value_by_flags_and_idx,
    set_value_by_flags_and_idx_v2,
    update_inputs,
)

from paddlenlp.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
)

__all__ = ["GenerationInferenceModel", "GenerationBlockInferenceModel"]


class ForcedDecodingEOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the last generated token to be the selected `forced_eos_token`.

    Args:
        max_length (int): The maximum length of the sequence to be generated.
        forced_eos_token_id (int): The id of the token to be generated as the last token.
    """

    def __init__(self, max_decoding_step: int, forced_eos_token_id: Union[int, List[int]]):
        self.max_decoding_step = max_decoding_step
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores, decoding_step):
        if decoding_step == self.max_decoding_step:
            scores[:] = paddle.finfo(scores.dtype).min
            scores[:, self.forced_eos_token_id] = 0
        return scores


class GenerationInferenceModel(GenerationMixin):
    @classmethod
    def get_cache_kvs_shape(cls, max_batch_size: int = None, max_length: int = None) -> list[list[int]]:
        raise NotImplementedError

    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())

        cache_kvs_shapes = self.get_cache_kvs_shape(self.config, max_length=config.get("max_length", None))
        export_precache = config.get("export_precache", False)
        if export_precache:
            precache_input_spec = [
                paddle.static.InputSpec(shape=[2, None, None, None, None], dtype=dtype, name=f"pre_caches_{i}")
                for i in range(len(cache_kvs_shapes))
            ]
        else:
            precache_input_spec = None

        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype=dtype, name="attention_mask"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),  # position_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_pos"),  # tgt_pos
            paddle.static.InputSpec(
                shape=[None, 1, 1, None], dtype=dtype, name="tgt_generation_mask"
            ),  # tgt_generation_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
            [
                paddle.static.InputSpec(
                    shape=shape,
                    dtype=dtype,
                    name="cache_kvs_{}".format(i),
                )
                for i, shape in enumerate(cache_kvs_shapes)
            ],  # cache_kvs
            None,  # inputs_embeds
            config.get("logits_processors", None),
            precache_input_spec,
        ]
        # use "==" to distingusih between chatglm and chatglm_v2.
        if self.config["model_type"] and "chatglm" == self.config.model_type.lower():
            input_spec[2] = paddle.static.InputSpec(
                shape=[None, None, None], dtype="int64", name="position_ids"
            )  # position_ids
            input_spec[16] = paddle.static.InputSpec(shape=[None, 2, 1], dtype="int64", name="tgt_pos")  # tgt_pos
        elif self.config["model_type"] and "gpt" in self.config.model_type:
            input_spec[2] = paddle.static.InputSpec(shape=[None], dtype="int64", name="position_ids")  # position_ids
        model = paddle.jit.to_static(self.generate, input_spec=input_spec)
        paddle.jit.save(
            model, output_path, skip_prune_program=True
        )  # Note(Zhengzekang): If we prune program it may cause some inference error.

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        inputs_embeds=None,
        logits_processors=None,
        pre_caches=None,
        **model_kwargs,
    ):
        model_kwargs["position_ids"] = position_ids
        model_kwargs["attention_mask"] = attention_mask

        model_kwargs["seq_len_encoder"] = seq_len_encoder
        model_kwargs["seq_len_decoder"] = seq_len_decoder
        model_kwargs["tgt_ids"] = tgt_ids
        model_kwargs["tgt_generation_mask"] = tgt_generation_mask
        model_kwargs["tgt_pos"] = tgt_pos
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["logits_processors"] = logits_processors or LogitsProcessorList()
        model_kwargs["pre_caches"] = pre_caches

        ret = self.sample(
            input_ids,
            eos_token_id,
            top_p=top_p,
            cache_kvs=cache_kvs,
            temperature=temperature,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        return ret

    def update_model_kwargs_for_generation(self, cache, just_decoder, next_tokens, eos_token_id, model_kwargs):
        if cache is None:
            model_kwargs["step_idx"] = paddle.where(
                model_kwargs["seq_len_encoder"] == 0,
                model_kwargs["step_idx"],
                model_kwargs["step_idx"] + 1,
            )
        else:
            model_kwargs["step_idx"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["step_idx"],
                model_kwargs["step_idx"] + 1,
            )
        length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
        model_kwargs["stop_flags"] = paddle.logical_or(model_kwargs["stop_flags"], length_cond)
        if cache is None:
            next_tokens = paddle.where(just_decoder, paddle.full_like(next_tokens, -1), next_tokens)
        next_tokens, model_kwargs["stop_flags"] = set_stop_value_multi_ends(
            next_tokens, model_kwargs["stop_flags"], eos_token_id, 2
        )  # multi ends

        if cache is None:
            # encoder's generation
            model_kwargs["tgt_ids"] = paddle.where(just_decoder, model_kwargs["tgt_ids"], next_tokens)
            if self.config["position_encoding_2d"] and self.config.position_encoding_2d is True:
                tgt_pos = model_kwargs["tgt_pos"]
                new_position_id = tgt_pos[:, 0, :].clone()
                new_block_id = tgt_pos[:, 1, :].clone()
                new_block_id = new_block_id + 1

                model_kwargs["tgt_pos"] = paddle.concat(
                    [new_position_id.unsqueeze(1), new_block_id.unsqueeze(1)], axis=1
                )
            else:
                model_kwargs["tgt_pos"] = paddle.where(
                    just_decoder, model_kwargs["tgt_pos"], model_kwargs["tgt_pos"] + 1
                )
            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"],
            )
        else:
            model_kwargs["tgt_ids"] = next_tokens
            if self.config["position_encoding_2d"] and self.config.position_encoding_2d is True:
                tgt_pos = model_kwargs["tgt_pos"]
                new_position_id = tgt_pos[:, 0, :].clone()
                new_block_id = tgt_pos[:, 1, :].clone()
                new_block_id = new_block_id + 1

                model_kwargs["tgt_pos"] = paddle.concat(
                    [new_position_id.unsqueeze(1), new_block_id.unsqueeze(1)], axis=1
                )
            else:
                model_kwargs["tgt_pos"] = paddle.where(
                    model_kwargs["stop_flags"],
                    model_kwargs["tgt_pos"],
                    model_kwargs["tgt_pos"] + 1,
                )

            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"] + 1,
            )

            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"],
            )

        model_kwargs["next_tokens"] = next_tokens
        return model_kwargs

    def sample(
        self,
        input_ids=None,
        eos_token_id=None,
        cache_kvs=[],
        top_p=None,
        temperature=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        step_idx_ori = paddle.full(shape=[1], dtype="int64", fill_value=1)
        batch_idx = paddle.full(shape=[1], dtype="int32", fill_value=-1)

        # fake temp next_tokens
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        next_tokens = paddle.full(shape=[batch, 1], dtype="int32", fill_value=0)

        # let inputs_embeds enter into model_kwargs.
        # because the code below directly use the model_kwargs as a parameter without using inputs_embeds.
        model_kwargs["inputs_embeds"] = inputs_embeds
        model_kwargs["all_input_ids"] = input_ids
        logits_processors = model_kwargs.pop("logits_processors")

        def _forward_(**args):
            # cache_kvs is never empty because it is passed as a parameter in def sample.
            model_inputs = self.prepare_inputs_for_generation(input_ids, cache_kvs, **args)
            return self(**model_inputs)

        def _post_process_(outputs, top_p, temperature, step_idx_ori, model_kwargs):
            cache = model_kwargs.get("cache", None)
            just_decoder = model_kwargs["seq_len_encoder"] == 0
            if cache is None:  # first decoder
                step_idx = paddle.where(
                    just_decoder,
                    paddle.full_like(model_kwargs["step_idx"], -1),
                    model_kwargs["step_idx"],
                )  # not update when continue decode
            else:
                step_idx = model_kwargs["step_idx"]
            model_kwargs["stop_flags"] = set_value_by_flags_and_idx(
                model_kwargs["pre_ids"],
                model_kwargs["tgt_ids"],
                step_idx,
                model_kwargs["stop_flags"],
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            logits = paddle.cast(logits, paddle.float32)
            logits = logits_processors(model_kwargs["all_input_ids"], logits, decoding_step=step_idx_ori)

            logits = get_token_penalty_multi_scores(
                model_kwargs["pre_ids"],
                logits,
                model_kwargs["penalty_score"],
                model_kwargs["frequency_score"],
                model_kwargs["presence_score"],
                step_idx,
                model_kwargs["min_dec_len"],
                eos_token_id,
            )
            # sample
            probs = F.softmax(logits)

            # compute next_tokens, use paddle.tensor.top_p_sampling
            logits = logits / temperature

            _, next_tokens = paddle.tensor.top_p_sampling(probs, top_p)

            if self.config.tensor_parallel_degree > 1:
                paddle.distributed.broadcast(next_tokens, 0)

            model_kwargs = self.update_model_kwargs_for_generation(
                cache, just_decoder, next_tokens, eos_token_id, model_kwargs
            )
            next_tokens = model_kwargs["next_tokens"]

            if model_kwargs["all_input_ids"] is None:
                model_kwargs["all_input_ids"] = next_tokens
            else:
                model_kwargs["all_input_ids"] = paddle.concat([model_kwargs["all_input_ids"], next_tokens], axis=1)

            save_with_output(
                next_tokens,
                batch_idx,
                step_idx_ori,
                "real_time_save.temp_ids",
                self.config.tensor_parallel_rank,
            )

            return next_tokens, model_kwargs

        # encoder
        outputs = _forward_(**model_kwargs)
        # first decoder
        next_tokens, model_kwargs = _post_process_(
            outputs,
            top_p,
            temperature,
            step_idx_ori,
            model_kwargs,
        )
        step_idx_ori += 1

        # gives it a value, means we will entered into decoder phase.
        model_kwargs["cache"] = 0

        # decoder
        while paddle.less_than(
            paddle.sum(paddle.cast(model_kwargs["stop_flags"], "int64")),
            model_kwargs["stop_nums"],
        ):
            next_tokens, model_kwargs = _post_process_(
                _forward_(**model_kwargs),
                top_p,
                temperature,
                step_idx_ori,
                model_kwargs,
            )
            step_idx_ori += 1

        return (
            next_tokens,
            model_kwargs["step_idx"],
            paddle.cast(model_kwargs["stop_flags"], "int32"),
            model_kwargs["seq_len_decoder"],
            model_kwargs["tgt_pos"],
        )


class GenerationBlockInferenceModel(GenerationMixin):
    @classmethod
    def get_cache_kvs_shape(cls, max_batch_size: int = None, max_length: int = None) -> list[list[int]]:
        raise NotImplementedError

    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())
        cachekv_dtype = dtype

        cache_kvs_shapes = self.get_cache_kvs_shape(
            self.config, max_batch_size=config.get("max_batch_size", -1), max_length=config.get("max_length", None)
        )
        export_precache = config.get("export_precache", False)
        if export_precache:
            precache_kv_spec = [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype=dtype, name=f"pre_caches_{i}")
                for i in range(len(cache_kvs_shapes))
            ]
        else:
            precache_kv_spec = None
        use_cachekv_int8 = config.get("use_cachekv_int8", "None")

        if use_cachekv_int8 == "static" or use_cachekv_int8 == "dynamic":
            cachekv_dtype = "uint8"

        if use_cachekv_int8 == "dynamic":
            cache_k_quant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="k_quant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]

            cache_v_quant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="v_quant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]

            cache_k_dequant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="k_dequant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]
            cache_v_dequant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="v_dequant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]
        else:
            cache_k_quant_scales = None
            cache_v_quant_scales = None
            cache_k_dequant_scales = None
            cache_v_dequant_scales = None

        caches = []
        for i in range(len(cache_kvs_shapes) // 2):
            caches.append(
                paddle.static.InputSpec(
                    shape=cache_kvs_shapes[2 * i], dtype=cachekv_dtype, name="key_caches_{}".format(i)
                )
            )
            caches.append(
                paddle.static.InputSpec(
                    shape=cache_kvs_shapes[2 * i + 1], dtype=cachekv_dtype, name="value_caches_{}".format(i)
                )
            )
        if export_precache:
            src_mask_spec = paddle.static.InputSpec(shape=[None, 1, None, None], dtype=dtype, name="src_mask")
        else:
            src_mask_spec = None

        # bloom model needs src_mask and tgt_mask!
        if "bloom" in self.config.architectures[0].lower():
            src_mask_spec = paddle.static.InputSpec(shape=[None, None, None, None], dtype=dtype, name="src_mask")
            tgt_mask_spec = paddle.static.InputSpec(shape=[None, None, 1, None], dtype=dtype, name="tgt_mask")
        else:
            tgt_mask_spec = None

        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            src_mask_spec,  # src_mask
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="next_tokens"),  # next_tokens
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="is_block_step"),  # is_block_step
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_this_time"),  # seq_lens_this_time
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_encoder"),  # seq_lens_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_decoder"),  # seq_lens_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(
                shape=[2, None, self.config.max_seq_len, None, None], dtype="float32", name="rope_emb"
            ),  # rope_emb
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_dec_len
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_dec_len
            paddle.static.InputSpec(shape=[1, 1], dtype="int64", name="stop_nums"),  # stop_nums
            paddle.static.InputSpec(shape=[None], dtype="int64", name="bad_tokens"),  # bad_tokens
            paddle.static.InputSpec(shape=[1, 1], dtype="bool", name="not_need_stop"),  # not_need_stop
            paddle.static.InputSpec(shape=[None, None], dtype="int32", name="block_tables"),  # block_tables
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            precache_kv_spec,
            caches,  # cache_kvs
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            tgt_mask_spec,
        ]
        model = paddle.jit.to_static(self.generate, input_spec=input_spec)
        paddle.jit.save(
            model, output_path, skip_prune_program=True
        )  # Note(Zhengzekang): If we prune program it may cause some inference error.

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        src_mask=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        next_tokens=None,
        is_block_step=None,
        seq_lens_this_time=None,  # update
        seq_lens_encoder=None,  # update
        seq_lens_decoder=None,  # update
        step_idx=None,
        stop_flags=None,
        rope_emb=None,
        min_length=None,
        max_length=None,
        stop_nums=None,
        bad_tokens=None,
        not_need_stop=None,
        block_tables=None,
        pre_ids=None,
        pre_caches=None,
        cache_kvs=[],
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
        tgt_mask=None,
        **model_kwargs,
    ):

        model_kwargs["input_ids"] = input_ids
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["seq_lens_this_time"] = seq_lens_this_time
        model_kwargs["seq_lens_encoder"] = seq_lens_encoder
        model_kwargs["seq_lens_decoder"] = seq_lens_decoder
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["rope_emb"] = rope_emb
        model_kwargs["bad_tokens"] = bad_tokens
        model_kwargs["block_tables"] = block_tables
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["not_need_stop"] = not_need_stop
        model_kwargs["caches"] = cache_kvs
        model_kwargs["k_quant_scales"] = k_quant_scales
        model_kwargs["v_quant_scales"] = v_quant_scales
        model_kwargs["k_dequant_scales"] = k_dequant_scales
        model_kwargs["v_dequant_scales"] = v_dequant_scales
        model_kwargs["pre_caches"] = pre_caches
        model_kwargs["next_tokens"] = next_tokens
        model_kwargs["is_block_step"] = is_block_step
        model_kwargs["src_mask"] = src_mask
        model_kwargs["tgt_mask"] = tgt_mask

        ret = self.sample(
            eos_token_id,
            top_k=0,
            top_p=top_p,
            temperature=temperature,
            **model_kwargs,
        )
        return ret

    def sample(
        self,
        eos_token_id,
        top_k,
        top_p,
        penalty_score,
        frequency_score,
        presence_score,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs
    ):
        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(**args)
            return self(**model_inputs)

        def _post_process_(
            outputs,
            top_k,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        ):
            step_idx = model_kwargs["step_idx"]
            set_value_by_flags_and_idx_v2(
                model_kwargs["pre_ids"],
                model_kwargs["input_ids"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                step_idx,
                model_kwargs["stop_flags"],
            )

            logits = paddle.cast(outputs, paddle.float32)

            # pre-process distribution
            logits = get_token_penalty_multi_scores_v2(
                model_kwargs["pre_ids"],
                logits,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                model_kwargs["bad_tokens"],
                step_idx,
                model_kwargs["min_dec_len"],
                eos_token_id,
            )

            # sample
            probs = F.softmax(logits)
            # _, next_tokens = top_p_sampling(probs, top_p, -1)
            _, next_tokens = paddle.topk(probs, 1, -1)

            if self.config.tensor_parallel_degree > 1:
                paddle.distributed.broadcast(next_tokens, 0)

            step_idx = paddle.where(model_kwargs["stop_flags"], model_kwargs["step_idx"], model_kwargs["step_idx"] + 1)
            paddle.assign(step_idx, model_kwargs["step_idx"])
            length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
            stop_flags = paddle.logical_or(model_kwargs["stop_flags"], length_cond)
            set_stop_value_multi_ends_v2(
                next_tokens, stop_flags, model_kwargs["seq_lens_this_time"], eos_token_id, model_kwargs["next_tokens"]
            )  # multi ends
            paddle.assign(stop_flags, model_kwargs["stop_flags"])
            # update inputs
            update_inputs(
                model_kwargs["stop_flags"],
                model_kwargs["not_need_stop"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                model_kwargs["input_ids"],
                model_kwargs["stop_nums"],
                next_tokens,
                model_kwargs["is_block_step"],
            )
            save_output(next_tokens, model_kwargs["not_need_stop"], self.config.tensor_parallel_rank)
            return next_tokens

        # encoder
        outputs = _forward_(**model_kwargs)  # [bs, 1, dim_embed]
        # first decoder
        next_tokens = _post_process_(
            outputs,
            top_k,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        )

        return next_tokens


class PrefixConstrainedNoBeamLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, paddle.Tensor], List[int]]):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor) -> paddle.Tensor:
        mask = paddle.full_like(scores, paddle.finfo(scores.dtype).min)
        prefix_allowed_tokens = self._prefix_allowed_tokens_fn(0, input_ids[0])
        mask[:, prefix_allowed_tokens] = 0

        return scores + mask


class GenerationStep:
    tokenizer = None
    generation_config = None

    def __init__(self, generation_config: Optional[GenerationConfig] = None, skip: bool = False):
        self.generation_config = generation_config
        self.skip = skip

    def forward(self, model):
        raise NotImplementedError

    def __call__(self, model, *args, **kwargs):
        return self.forward(model, *args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **model_inputs):
        batch_size, seq_length = input_ids.shape
        if past_key_values is None:
            if attention_mask is None:
                attention_mask = paddle.ones_like(input_ids)
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))
        else:
            past_key_values_length = past_key_values[0][0].shape[1]
            if attention_mask is None:
                attention_mask = paddle.ones([input_ids.shape[0], past_key_values_length + input_ids.shape[1]])
            position_ids = paddle.to_tensor(
                [[i for i in range(past_key_values_length, past_key_values_length + seq_length)]] * batch_size,
                dtype="int64",
            )
        model_inputs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
            }
        )
        return model_inputs

    def clip_past_key_values(self, past_key_values):
        """clip past key values for next generation"""
        # tuple -> list: will not malloc new memory for past_key_values
        past_key_values = [list(item) for item in past_key_values]
        for kv_index in range(len(past_key_values)):
            for layer_index in range(len(past_key_values[kv_index])):
                length = past_key_values[kv_index][layer_index].shape[1]
                past_key_values[kv_index][layer_index] = past_key_values[kv_index][layer_index][:, : length - 1, :, :]
        return past_key_values

    def get_eos_token_id(self, custom_eos_token_id: Optional[int, list[int], list[list[int]]] = None):
        def to_two_dim_eos_token_ids(ids):
            if isinstance(ids, int):
                return [[ids]]
            if isinstance(ids, list) and isinstance(ids[0], int):
                return [ids]
            return ids

        eos_token_id = to_two_dim_eos_token_ids(self.generation_config.eos_token_id)
        if custom_eos_token_id is not None:
            eos_token_id.extend(custom_eos_token_id)
        return eos_token_id

    def sample(self, model, input_ids, logits_processors=None, custom_eos_token_id=None, **model_inputs):
        if self.generation_config is None:
            self.generation_config = model.generation_config

        eos_token_id = self.get_eos_token_id(custom_eos_token_id)
        cur_len = 0
        unfinished_flag = paddle.full([input_ids.shape[0], 1], True, dtype="bool")
        from paddlenlp.generation.utils import get_unfinished_flag

        generate_end = False

        result_ids = input_ids
        while True:
            model_inputs["input_ids"] = input_ids
            model_inputs = model.prepare_inputs_for_generation(**model_inputs)
            # NOTE: to decrease ref-count and clear outdate cache in-time
            outputs = model(**model_inputs)

            logits = outputs[0]
            model_inputs["past_key_values"] = outputs[-1]

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            if logits_processors is not None:
                logits = logits_processors(result_ids, logits)

            # sample
            logits = logits / self.generation_config.temperature
            probs = F.softmax(logits)
            _, input_ids = paddle.tensor.top_p_sampling(
                probs, paddle.to_tensor([self.generation_config.top_p], dtype=logits.dtype)
            )

            cur_len += 1
            result_ids = paddle.concat([result_ids, input_ids], axis=1)
            sentence = self.tokenizer.decode(result_ids.tolist()[0])
            print("sentence ->", sentence)

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)
                if not paddle.any(unfinished_flag):
                    generate_end = True

            if generate_end:
                break
            model_inputs = model.update_model_kwargs_for_generation(outputs, model_inputs)
        model_inputs["past_key_values"] = self.clip_past_key_values(model_inputs["past_key_values"])
        return result_ids, model_inputs["past_key_values"]


class Choices(GenerationStep):
    def __init__(self, values: List[Union[str, int]], tokenizer, **kwargs):
        self.input_ids = [paddle.to_tensor([self.tokenizer(str(value))["input_ids"]]) for value in values]
        super().__init__(**kwargs)

    def forward(self, model, input_ids, **model_inputs):
        min_probs, min_past_key_values = 1000000, None
        result_ids = None
        start_input_ids = input_ids
        model_inputs.pop("attention_mask", None)
        for input_ids in self.input_ids:
            input_ids = paddle.concat([start_input_ids, input_ids], axis=1)
            local_model_inputs = self.prepare_inputs_for_generation(input_ids, **model_inputs)
            result = model(**local_model_inputs)
            mean_probs = paddle.nn.functional.softmax(result[0]).squeeze(0)
            probs = paddle.index_sample(mean_probs, input_ids.squeeze(0).unsqueeze(-1)).mean()

            if min_probs > probs:
                min_probs = probs
                min_past_key_values = result[-1]
                result_ids = input_ids
        min_past_key_values = self.clip_past_key_values(min_past_key_values)
        return result_ids, min_past_key_values


class String(GenerationStep):
    def __init__(self, eos_token: str | list[str], **kwargs):
        if isinstance(eos_token, str):
            eos_token = [eos_token]
        eos_token = [str(i) for i in eos_token]
        super().__init__(**kwargs)
        self.eos_token_id = self.tokenizer(eos_token, add_special_tokens=False)["input_ids"]

    def forward(self, model, input_ids, **model_inputs):
        result_ids, past_key_values = self.sample(
            model, input_ids, custom_eos_token_id=self.eos_token_id, **model_inputs
        )
        past_key_values = self.clip_past_key_values(past_key_values)
        return result_ids, past_key_values


class Constant(GenerationStep):
    def __init__(self, value: Union[str, int, float], **kwargs):
        self.inputs = self.tokenizer(str(value), return_tensors="pd")

    def prepare_inputs_for_generation(self, **kwargs):
        if kwargs["input_ids"] is None:
            kwargs.update(self.inputs)
        else:
            for key, value in self.inputs.items():
                if key not in kwargs:
                    continue
                kwargs[key] = paddle.concat([kwargs[key], value], -1)
        return super().prepare_inputs_for_generation(**kwargs)

    def forward(self, model, **model_inputs):
        result = model(**model_inputs)
        past_key_values = self.clip_past_key_values(result[1])
        return model_inputs["input_ids"], past_key_values


class Json(GenerationStep):
    def __init__(
        self,
        schema: dict,
        top_p: float = 0.1,
        temperature: float = 1.0,
        eos_token_id: int | list[int] | list[list[int]] = None,
        dtype: Optional[str] = None,
        **kwargs
    ):
        from lmformatenforcer import JsonSchemaParser

        from paddlenlp.generation.format_enforce import build_prefix_allowed_tokens_fn
        from paddlenlp.generation.logits_process import LogitsProcessorList

        parser = JsonSchemaParser(schema)
        prefix_function = build_prefix_allowed_tokens_fn(self.tokenizer, parser)

        self.logits_processors = LogitsProcessorList([PrefixConstrainedNoBeamLogitsProcessor(prefix_function)])
        self.top_p = paddle.to_tensor([top_p], dtype=dtype)
        self.temperature = temperature
        if eos_token_id is None:
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.eos_token_id = eos_token_id
        super().__init__(**kwargs)

    def forward(self, model, input_ids, **model_inputs):
        return self.sample(model, input_ids=input_ids, logits_processors=self.logits_processors, **model_inputs)


def compile_json_schema_steps(schema: str):
    if isinstance(schema, str):
        schema = json.loads(schema)

    # plain json structure
    steps = []
    count, index = len(schema["properties"]), 0
    for property_name, property in schema["properties"].items():
        if index == 0:
            steps.append("{")

        steps.append(f'"{property_name}":"')
        if property["type"] == "string":
            if index < count - 1:
                steps.append(String(eos_token=['","']))
            else:
                steps.append(String(eos_token=['"}']))
        index += 1

    # join the string steps
    joined_steps = []
    for index, step in enumerate(steps):
        if index == 0:
            joined_steps.append(step)
        elif index > 0 and isinstance(step, str) and isinstance(joined_steps[-1], str):
            joined_steps[-1] = joined_steps[-1] + step
        else:
            joined_steps.append(step)
    return joined_steps


def steps_generate(model, tokenizer, steps):
    past_key_values = None
    result_ids = paddle.to_tensor([[0]], dtype=paddle.int64)
    input_ids = None
    # last_input_ids = None
    for index, step in enumerate(steps):
        if isinstance(step, (int, str, float)):
            step = Constant(step)

        model_inputs = step.prepare_inputs_for_generation(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=True
        )

        next_token_ids, past_key_values = step(model, **model_inputs)
        print("generated result ->", tokenizer.decode(next_token_ids.tolist()[0]))
        if index == len(steps) - 1:
            result_ids = paddle.concat([result_ids, next_token_ids], axis=1)
        else:
            result_ids = paddle.concat([result_ids, next_token_ids[:, :-1]], axis=1)
            input_ids = next_token_ids[:, -1:]

        print("current result ->", tokenizer.decode(result_ids.tolist()[0]))
        # last_input_ids = next_token_ids

    return result_ids[:, 1:]


def set_tokenizer(tokenizer):
    GenerationStep.tokenizer = tokenizer
