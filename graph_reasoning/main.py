# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""
import json
import logging
import math
import os
import sys

sys.path.append('.')
from typing import Dict, Optional

import dgl
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import stanza
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed, Trainer,
)
from transformers.trainer_utils import is_main_process

from graph_reasoning import Config
from graph_reasoning.build_model import RobertaGraphReasoning
from AdaLoGN.utils.data_utils import MyRobertaTokenizer
from graph_reasoning.parseargs import ModelArguments, DataTrainingArguments
from graph_reasoning.process_data import ReasoningGraphDataset

logger = logging.getLogger(__name__)


def main(config_path):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    if config_path is None:
        config_path = "configs/control_dataset.json"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(config_path)

    Config.rgcn_relation_nums = model_args.rgcn_relation_nums
    Config.model_args = model_args

    model_class = {"LogiGraph": RobertaGraphReasoning, "GraphReasoning": RobertaGraphReasoning}

    dataset_class = {"LogiGraph": ReasoningGraphDataset, "GraphReasoning": ReasoningGraphDataset}
    nlp = stanza.Pipeline(lang='en', dir='../stanza_resources', processors='tokenize,mwt,pos,sentiment,lemma,depparse',
                          use_gpu=False, download_method=None)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
    dgl.seed(training_args.seed)
    Config.seed = training_args.seed
    Config.extension_threshold = model_args.extension_threshold

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=3,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer_class = MyRobertaTokenizer if model_args.model_type == 'Roberta' else AutoTokenizer

    logger.info(f'model type: {model_args.model_type}')
    logger.info(f'tokenizer class: {tokenizer_class.__name__}')

    tokenizer = tokenizer_class.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': [Config.NODE_SEP_TOKEN]})
    Config.tokenizer = tokenizer
    logger.info(f'roberta class: {model_class}')

    def model_init():
        return RobertaGraphReasoning.from_pretrained(
            model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir, )

    logger.info(f'model type: {Config.model_type}')

    # Get datasets
    train_dataset = (
        ReasoningGraphDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode="train",
            nlp=nlp
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        ReasoningGraphDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode="dev",
            nlp=nlp
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        ReasoningGraphDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode="test",
            nlp=nlp
        )
        if training_args.do_eval
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        _preds = p.predictions
        # print(f'pred: {_preds}')
        # _preds[_preds != _preds] = 0
        dataset_dir = data_args.data_dir

        preds = np.argmax(p.predictions, axis=1)

        torch.cuda.empty_cache()
        if Config.eval_test:
            dev_len = 500 if dataset_dir == 'ReclorDataset' else (len(preds) // 2)
            # dev_acc = accuracy_score(p.label_ids[:dev_len], preds[:dev_len])

            res = {}

            if len(set(p.label_ids)) > 2:
                full_label_f1 = f1_score(y_true=preds, y_pred=p.label_ids, average=None)
                average_f1 = np.mean(full_label_f1)
                acc = accuracy_score(y_true=preds, y_pred=p.label_ids)
                res['acc'] = acc
                for label, score in zip(eval_dataset.labels, full_label_f1.tolist()):
                    res['{}_f1'.format(label)] = score
                res['average_f1'] = average_f1
            else:
                full_label_f1 = f1_score(y_true=preds, y_pred=p.label_ids, pos_label=0)
                acc = accuracy_score(y_true=preds, y_pred=p.label_ids)
                res['acc'] = acc
                res['f1'] = full_label_f1

            if not os.path.exists(os.path.join('results', training_args.run_name)):
                logger.info(f'mkdir {os.path.join("results", training_args.run_name)}')
                os.mkdir(os.path.join('results', training_args.run_name))

            count = 0
            save_path = f'test_{acc}_{count}.npy'
            while os.path.exists(os.path.join('results', training_args.run_name, save_path)):
                count += 1
                save_path = f'test_{acc}_{count}.npy'
            np.save(os.path.join('results', training_args.run_name, save_path), preds[dev_len:])

            return res
        else:
            return {"acc": accuracy_score(p.label_ids, preds)}

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=None,
    )

    # Training
    last_checkpoint = None
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(training_args.output_dir) and training_args.resume_from_checkpoint:
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            dirs = os.listdir(training_args.output_dir)
            checkpoints = []
            for d in dirs:
                if d.startswith('checkpoint'):
                    checkpoints.append(d)
            if checkpoints:
                checkpoint = os.path.join(training_args.output_dir, sorted(checkpoints)[-1])
                trainer.train(resume_from_checkpoint=checkpoint)
            else:
                trainer.train(
                    resume_from_checkpoint=checkpoint,
                    model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
                )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    if training_args.do_eval:
        model_path = os.path.join(training_args.output_dir, 'pytorch_model.bin')
        if not training_args.do_train and os.path.exists(model_path):
            logger.info("load best model from {}".format(model_path))
            trainer.model.load_state_dict(torch.load(model_path))
        logger.info("*** Evaluate ***")

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        test_result = trainer.evaluate(eval_dataset=test_dataset)
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in eval_result.items():
                logger.info("eval  %s = %s", key, value)
                writer.write("eval %s = %s\n" % (key, value))
            logger.info("***** Test results *****")
            for key, value in test_result.items():
                logger.info("test  %s = %s", key, value)
                writer.write("test %s = %s\n" % (key, value))


if __name__ == "__main__":
    main(config_path=sys.argv[1])
