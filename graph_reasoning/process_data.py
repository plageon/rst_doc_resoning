import json
import os
from dataclasses import dataclass
from typing import List, Optional

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from graph_reasoning import Config
from graph_reasoning.graph_utils import construct_reasoning_graph
from AdaLoGN.models import logger


@dataclass(frozen=True)
class ReasoningSample:
    example_id: str
    context_origin: str
    endings_origin: List[str]
    context: List[List[str]]
    endings: List[List[str]]
    graphs: List[dgl.graph]
    edge_types: List[List[int]]
    edge_norms: List[List[int]]
    graph_node_nums: List[int]
    label: Optional[str]
    nodes_num: List[List[int]]
    base_nodes_ids: Optional[List[List[int]]]
    exten_nodes_ids: Optional[List[List[List[int]]]]
    exten_edges_ids: Optional[List[List[List[int]]]]


@dataclass(frozen=True)
class ReasoningGraphInputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids_origin: List[List[int]]
    attention_mask_origin: Optional[List[List[int]]]
    token_type_ids_origin: Optional[List[List[int]]]
    input_ids: Optional[List[List[int]]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    graphs: List[List[torch.tensor]]
    graph_node_nums: List[int]
    edge_types: List[List[int]]
    edge_norms: List[List[float]]
    label: Optional[int]

    question_interval: Optional[List[List[int]]]
    node_intervals: Optional[List[List[List[int]]]]
    node_intervals_len: Optional[List[int]]
    context_interval: Optional[List[List[int]]]
    answer_interval: Optional[List[List[int]]]

    nodes_num: List[List[int]]

    base_nodes_ids: Optional[List[List[int]]]
    exten_nodes_ids: Optional[List[List[List[int]]]]
    exten_edges_ids: Optional[List[List[List[int]]]]

    @staticmethod
    def get_split_intervals(input_ids: List[List[int]], tokenizer, node_interval_padding_len):
        node_sep_id = tokenizer.convert_tokens_to_ids(Config.NODE_SEP_TOKEN)
        sep_id = tokenizer.convert_tokens_to_ids(Config.SEP_TOKEN)
        seq_locs = [np.where((np.array(input_id) == sep_id))[0].tolist() for input_id in input_ids]
        assert sum(list(map(len, seq_locs))) == 2 * len(input_ids) and len(seq_locs) == len(input_ids)
        node_intervals = []
        node_sep_locs = [sorted([0] + np.where((np.array(input_id) == node_sep_id))[0].tolist() + seq_locs[index]) for
                         index, input_id in enumerate(input_ids)]
        node_intervals_num = []
        for index, node_sep_loc in enumerate(node_sep_locs):
            node_interval = []
            for i in range(len(node_sep_loc) - 1):
                node_interval.append([node_sep_loc[i] + 1, node_sep_loc[i + 1]])
            node_interval = list(filter(lambda x: x[1] - x[0] > 2, node_interval))
            node_intervals_num.append(len(node_interval))

            assert len(node_interval) < node_interval_padding_len
            while len(node_interval) < node_interval_padding_len:
                node_interval.append([Config.node_intervals_padding_id, Config.node_intervals_padding_id])
            node_intervals.append(node_interval)
        return None, node_intervals, node_intervals_num, None, None


class ReasoningGraphProcessor:
    """Processor for the RACE data set."""

    def __init__(self, data_dir, nlp):
        self.data_dir = data_dir
        self.EDUs = self.load_EDUS(data_dir)
        self.new_not_sentence_map = json.load(
            open('{}/negative_sentences_map.json'.format(data_dir), 'r', encoding='utf-8'))
        self.nlp = nlp

    def load_data(self, dataset_dir):
        train_dataset = json.load(open(f'{dataset_dir}/train.json', 'r', encoding='utf-8'))
        train_dataset = dict([(data['id_string'], data) for data in train_dataset])

        val_dataset = json.load(open(f'{dataset_dir}/val.json', 'r', encoding='utf-8'))
        val_dataset = dict([(data['id_string'], data) for data in val_dataset])

        test_dataset = json.load(open(f'{dataset_dir}/test.json', 'r', encoding='utf-8'))
        test_dataset = dict([(data['id_string'], data) for data in test_dataset])
        #  subsentences_with_logic = load_all_subsentences_with_logic()

        return train_dataset, val_dataset, test_dataset

    def load_EDUS(self, dataset_dir):
        EDUs = json.load(
            open('{}/{}_EDUs.json'.format(dataset_dir, dataset_dir.replace('data/', '').replace('/', '_')), 'r',
                 encoding='utf-8'))
        EDUs_tmp = {}
        for sentence in EDUs:
            EDUs_tmp[sentence.strip()] = EDUs[sentence]
        EDUs = EDUs_tmp
        return EDUs

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        if 'anli' in data_dir:
            return self._create_examples(f'{data_dir}/train.jsonl', "train")
        elif 'NLI' in data_dir:
            return self._create_examples(f'{data_dir}/train.jsonl', "train")
        elif 'control' in data_dir:
            return self._create_examples(f'{data_dir}/train.jsonl', "train")
        else:
            return self._create_examples(f'{data_dir}/train.txt', "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        if 'anli' in data_dir:
            return self._create_examples(f'{data_dir}/dev.jsonl', "dev")
        elif 'NLI' in data_dir:
            return self._create_examples(f'{data_dir}/dev.jsonl', "dev")
        elif 'control' in data_dir:
            return self._create_examples(f'{data_dir}/dev.jsonl', "dev")
        else:
            return self._create_examples(f'{data_dir}/dev.txt', "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        if 'anli' in data_dir:
            return self._create_examples(f'{data_dir}/test.jsonl', "test")
        elif 'NLI' in data_dir:
            return self._create_examples(f'{data_dir}/test.jsonl', "test")
        elif 'control' in data_dir:
            return self._create_examples(f'{data_dir}/test.jsonl', "test")
        else:
            return self._create_examples(f'{data_dir}/test.txt', "test")

    def get_labels(self) -> List:
        """See base class."""
        if 'anli' in self.data_dir or 'control' in self.data_dir:
            Config.label_num = 3
            return ['e', 'n', 'c']
        elif 'logiqa' in self.data_dir:
            Config.label_num = 2
            return ['entailed', 'not entailed']
        elif 'NLI' in self.data_dir:
            Config.label_num = 2
            return ['entailment', 'non_entailment']
        else:
            return []

    def _read_json(self, input_dir):
        with open(input_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines]

    def _create_examples(self, data_file, type):
        """Creates examples for the training and dev sets."""
        max_edge_num = 0
        max_node_num = 0
        datas = self._read_json(data_file)
        examples = []

        for d in tqdm(datas, total=len(datas), desc=f'preparing {type} data...'):
            label = d['label'] if 'label' in d else 0

            id_string = d['uid'] if 'anli' in self.data_dir else d['id']
            premise = d['context'] if 'anli' in self.data_dir else d['premise']
            hypothesis = d['hypothesis']

            context, endings, graphs, node_sentences_a, node_sentences_b, relations, edge_norms, base_node_ids, \
                cont_exten_node_ids, trans_exten_edge_ids = construct_reasoning_graph(id_string, premise, hypothesis,
                                                                                      self.EDUs,
                                                                                      self.new_not_sentence_map,
                                                                                      return_base_nodes=True,
                                                                                      dataset_dir=self.data_dir,
                                                                                      nlp=self.nlp)
            exten_node_ids = None
            if base_node_ids is not None:
                assert len(base_node_ids) <= Config.node_interval_padding_len, "len(base_node_ids) = {}".format(len(base_node_ids))
                base_node_ids += [Config.extension_padding_value] * (Config.node_interval_padding_len - len(base_node_ids))

            if cont_exten_node_ids is not None:
                exten_node_ids = cont_exten_node_ids
                for inner_index in range(len(exten_node_ids)):
                    assert max(exten_node_ids[inner_index]) < len(node_sentences_a) + len(
                        node_sentences_b), f'id: {id_string}'
                    exten_node_ids[inner_index] += [Config.extension_padding_value] * (
                            4 - len(exten_node_ids[inner_index]))
                assert len(exten_node_ids) <= Config.extension_padding_len, f'exten node len: {len(exten_node_ids)}'
                exten_node_ids += [[Config.extension_padding_value] * 4] * (
                        Config.extension_padding_len - len(exten_node_ids))

                exten_edge_ids = trans_exten_edge_ids
                for j in range(len(exten_edge_ids)):
                    assert len(exten_edge_ids[j]) == 3
                if os.path.exists('print_extension_len'):
                    print(len(exten_node_ids))
                assert len(exten_node_ids) <= Config.extension_padding_len, f'exten edge len: {len(exten_edge_ids)}'
                exten_edge_ids += [[Config.extension_padding_value] * 3] * (
                        Config.extension_padding_len - len(exten_edge_ids))

            max_edge_num = max(max_edge_num, len(relations))
            max_node_num = max(max_node_num, graphs.num_nodes())
            example = ReasoningSample(
                example_id=id_string,
                context_origin=context,
                endings_origin=endings,
                context=node_sentences_a,
                endings=node_sentences_b,
                graphs=graphs,
                edge_types=relations,
                edge_norms=edge_norms,
                graph_node_nums=graphs.num_nodes(),
                label=label,
                nodes_num=[[len(node_sentences_a), len(node_sentences_b)]],
                base_nodes_ids=base_node_ids,
                exten_nodes_ids=exten_node_ids,
                exten_edges_ids=exten_edge_ids,
            )
            examples.append(example)

        logger.info(f'max edge num: {max_edge_num}')
        Config.max_edge_num = max(max_edge_num, Config.max_edge_num)
        Config.node_interval_padding_len = max(max_node_num, Config.node_interval_padding_len)
        return examples


class ReasoningGraphDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[ReasoningGraphInputFeatures]

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode='train',
            nlp=None
    ):
        processor = ReasoningGraphProcessor(data_dir, nlp)
        self.labels = processor.get_labels()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(max_seq_length),
                "AdaLoGN_Reclor",
            ),
        )
        self.tokenizer = tokenizer

        if '160' in data_dir:
            Config.node_interval_padding_len = 71
        elif '500' in data_dir:
            Config.node_interval_padding_len = 86
        elif 'Binary' in data_dir:
            Config.node_interval_padding_len = 144
        elif 'control' in data_dir:
            Config.node_interval_padding_len = 160
        logger.info(f'looking for cached file {cached_features_file}')

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        # lock_path = cached_features_file + ".lock"
        # with FileLock(lock_path):

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            label_list = processor.get_labels()
            if mode == 'dev':
                examples = processor.get_dev_examples(data_dir)
            elif mode == 'test':
                examples = processor.get_test_examples(data_dir)
            elif mode == 'train':
                examples = processor.get_train_examples(data_dir)
            elif mode == 'dev_and_test':
                examples = processor.get_dev_examples(data_dir) + processor.get_test_examples(data_dir)
            else:
                raise NotImplementedError
            logger.info("Training examples: %s", len(examples))
            self.features = self.convert_examples_to_features_graph_with_origin_rgcn(
                examples,
                label_list,
                max_seq_length,
                tokenizer, mode='train',
            )
            # save_new_not_sentence_map()
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> ReasoningGraphInputFeatures:
        return self.features[i]

    def convert_examples_to_features_graph_with_origin_rgcn(
            self,
            examples: List[ReasoningSample],
            label_list: List[str],
            max_length: int,
            tokenizer: PreTrainedTokenizer,
            mode='train'
    ) -> List[ReasoningGraphInputFeatures]:
        """
        Loads a data file into a list of `InputFeatures`
        """

        label_map = {label: i for i, label in enumerate(label_list)}

        logger.info(f'label map: {label_map}')
        trun_count = 0
        total_count = 1
        features = []
        t = tqdm(enumerate(examples), desc=f"convert examples to features")
        for (ex_index, example) in t:
            if ex_index % 2000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            attention_mask, attention_mask_origin, input_ids, input_ids_origin, label, token_type_ids, token_type_ids_origin, _trun_count, _total_count = self.tokenizer_encode(
                example, label_map, max_length, tokenizer)

            trun_count += _trun_count
            total_count += _total_count
            t.set_description(
                f'convert examples to features, trun count: {trun_count}, total_count: {total_count}, trun ratio: {trun_count / total_count}')

            edges = [example.graphs.edges()[0].numpy().tolist(), example.graphs.edges()[1].numpy().tolist()]

            edge_types = example.edge_types
            edge_norms = example.edge_norms

            assert len(edges[0]) == len(edges[1]) == len(edge_types) == len(edge_norms)
            edges[0] = edges[0] + [-1] * (Config.max_edge_num - len(edges[0]))
            edges[1] = edges[1] + [-1] * (Config.max_edge_num - len(edges[1]))
            edge_types = edge_types + [-1] * (Config.max_edge_num - len(edge_types))
            edge_norms = edge_norms + [-1] * (Config.max_edge_num - len(edge_norms))

            question_interval, node_intervals, node_intervals_num, context_interval, answer_interval = ReasoningGraphInputFeatures.get_split_intervals(
                input_ids, tokenizer, Config.node_interval_padding_len)

            new_feature = ReasoningGraphInputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                graphs=edges,
                graph_node_nums=example.graph_node_nums,
                label=label,
                input_ids_origin=input_ids_origin,
                attention_mask_origin=attention_mask_origin,
                token_type_ids_origin=token_type_ids_origin,
                edge_types=edge_types,
                edge_norms=edge_norms,
                question_interval=question_interval,
                node_intervals=node_intervals,
                node_intervals_len=node_intervals_num,
                nodes_num=example.nodes_num,
                context_interval=context_interval,
                answer_interval=answer_interval,
                base_nodes_ids=example.base_nodes_ids,
                exten_nodes_ids=example.exten_nodes_ids,
                exten_edges_ids=example.exten_edges_ids
            )
            features.append(new_feature)

        for f in features[:2]:
            logger.info("*** Example ***")
            logger.info("feature: %s" % f)

        return features

    def tokenizer_encode(self, example, label_map, max_length, tokenizer):
        choices_inputs = []
        choices_inputs_origin = []
        truncated_count = 0
        total_count = 0
        context = example.context
        context_origin = example.context_origin
        ending_origin = example.endings_origin

        assert isinstance(context_origin, str)
        assert isinstance(ending_origin, str)
        assert isinstance(context, list)
        assert isinstance(example.endings, list)

        inputs_origin = tokenizer(
            context_origin,
            ending_origin,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            # return_overflowing_tokens=True,
        )

        # [CLS] node_a_1 [N_SEP] node_a_2 [N_SEP] ... [N_SEP] node_a_n [SEP] node_b_1 [N_SEP] ... [N_SEP] node_b_n [SEP]
        node_sep_token = ' {} '.format(Config.NODE_SEP_TOKEN)
        text_a = node_sep_token.join(context)

        text_b = node_sep_token.join(example.endings)

        inputs = tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            # return_overflowing_tokens=True,
        )
        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens (swag task is ok). "
                "If you are training ARC and RACE and you are poping question + options,"
                "you need to try to use a bigger max seq length!"
            )
            truncated_count += 1
        total_count += 1

        choices_inputs.append(inputs)
        choices_inputs_origin.append(inputs_origin)

        label = label_map[example.label]
        input_ids = [x["input_ids"][0] if Config.model_type == 'Bert' else x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"][0] if Config.model_type == 'Bert' else x["attention_mask"] for x in
             choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"][0] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )
        input_ids_origin = [x["input_ids"][0] if Config.model_type == 'Bert' else x["input_ids"] for x in
                            choices_inputs_origin]
        attention_mask_origin = (
            [x["attention_mask"][0] if Config.model_type == 'Bert' else x["attention_mask"] for x in
             choices_inputs_origin] if "attention_mask" in choices_inputs_origin[
                0] else None)
        token_type_ids_origin = (
            [x["token_type_ids"][0] for x in choices_inputs_origin] if "token_type_ids" in choices_inputs_origin[
                0] else None)
        return attention_mask, attention_mask_origin, input_ids, input_ids_origin, label, token_type_ids, token_type_ids_origin, truncated_count, total_count
