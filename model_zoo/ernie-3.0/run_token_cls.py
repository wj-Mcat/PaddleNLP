# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
from datasets import load_dataset, load_metric
from utils import DataArguments, ModelArguments, load_config, token_convert_example

import paddlenlp
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import (
    ErnieForTokenClassification,
    ErnieTokenizer,
    PretrainedTokenizer,
)
from paddlenlp.utils.log import logger


@dataclass
class DataCollatorWithDyanmicMaxLength:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        tokenizer (`paddlenlp.transformers.PretrainedTokenizer`):
            The tokenizer used for encoding the data.
    """

    tokenizer: PretrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pd"
    return_attention_mask: Optional[bool] = None
    label_list: Optional[int] = None

    # [32, 64, 128]
    dynamic_max_length: Optional[list[int]] = None

    def get_dynamic_max_length(self, examples):
        if not self.dynamic_max_length:
            return self.max_length

        if "sentence1" in examples[0]:
            lengths = list(
                map(
                    lambda example: len(
                        self.tokenizer.tokenize(example["sentence1"]) + self.tokenizer.tokenize(example["sentence2"])
                    )
                    + 3,
                    examples,
                )
            )

        else:
            lengths = list(map(lambda example: len(self.tokenizer.tokenize(example["sentence"])) + 2, examples))
        max_length = min(max(lengths), self.max_length)
        lengths = [length for length in self.dynamic_max_length if max_length < length]
        if not lengths:
            return self.max_length
        return lengths[0]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = self.get_dynamic_max_length(examples)
        examples = list(
            map(
                lambda example: token_convert_example(
                    example,
                    tokenizer=self.tokenizer,
                    label_list=self.label_list,
                    max_seq_length=max_length,
                ),
                examples,
            )
        )
        features = {}
        for key in examples[0].keys():
            features[key] = np.array([example[key] for example in examples])

        if "label" in features:
            features["labels"] = features["label"]
            del features["label"]
        if "label_ids" in features:
            features["labels"] = features["label_ids"]
            del features["label_ids"]
        return features


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Log model and data config
    model_args, data_args, training_args = load_config(
        model_args.config, "TokenClassification", data_args.dataset, model_args, data_args, training_args
    )

    # Print model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    data_args.dataset = data_args.dataset.strip()
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.dataset)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    raw_datasets = load_dataset(data_args.dataset)
    label_list = raw_datasets["train"].features["ner_tags"].feature.names
    data_args.label_list = label_list
    data_args.ignore_label = -100
    data_args.no_entity_id = 0
    print(label_list)

    num_classes = len(label_list)

    # Define tokenizer, model, loss function.
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)
    model = ErnieForTokenClassification.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)

    class criterion(nn.Layer):
        def __init__(self):
            super(criterion, self).__init__()
            self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=data_args.ignore_label)

        def forward(self, *args, **kwargs):
            return paddle.mean(self.loss_fn(*args, **kwargs))

    loss_fct = criterion()

    # Define data collector
    data_collator = DataCollatorWithDyanmicMaxLength(
        tokenizer=tokenizer,
        label_pad_token_id=data_args.ignore_label,
        padding="max_length",
        max_length=data_args.max_seq_length,
        dynamic_max_length=data_args.dynamic_max_length,
    )

    # Dataset pre-process
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
    if training_args.do_eval:
        # The msra_ner dataset do not have the dev dataset, use the test dataset for the evaluation
        eval_dataset = raw_datasets["test"]
    if training_args.do_predict:
        test_dataset = raw_datasets["test"]

    # Define the metrics of tasks.
    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        criterion=loss_fct,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
    if training_args.do_predict:
        test_ret = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_ret.metrics)
        tokens_label = test_ret.predictions.argmax(axis=-1)
        tokens_label = tokens_label.tolist()
        value = []
        for idx, token_label in enumerate(tokens_label):
            label_name = ""
            items = []
            input_data = tokenizer.convert_ids_to_tokens(test_dataset[idx]["input_ids"])[1:-1]
            input_len = len(input_data)
            words = ""
            tag = " "
            start = 0
            for i, label in enumerate(token_label[1 : input_len + 1]):
                label_name = data_args.label_list[label]
                if label_name == "O" or label_name.startswith("B-"):
                    if len(words):
                        items.append({"pos": [start, i], "entity": words, "label": tag})

                    if label_name.startswith("B-"):
                        tag = label_name.split("-")[1]
                    else:
                        tag = label_name
                    start = i
                    words = input_data[i]
                else:
                    words += input_data[i]
            if len(words) > 0:
                items.append({"pos": [start, i], "entity": words, "label": tag})
            value.append(items)

        out_dict = {"value": value, "tokens_label": tokens_label}
        out_file = open(os.path.join(training_args.output_dir, "test_results.json"), "w")
        json.dump(out_dict, out_file, ensure_ascii=True)

    # Export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ]
        model_args.export_model_dir = os.path.join(model_args.export_model_dir, data_args.dataset, "export")
        paddlenlp.transformers.export_model(
            model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir
        )


if __name__ == "__main__":
    main()
