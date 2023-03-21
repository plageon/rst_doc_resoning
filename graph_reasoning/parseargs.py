from dataclasses import dataclass, field
from typing import Optional

from AdaLoGN.utils.data_utils import processors


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='/home/xli/bert_model_en',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    gnn_layers_num: int = field(default=-1)
    model_type: str = field(default='Bert')
    base_num: int = field(default=6)
    rgcn_relation_nums: int = field(default=6)
    dropout: float = field(default=0.1)
    results_output_dir: str = field(default='results')
    pooling_type: str = field(default='none')
    label_smoothing: bool = field(default=True)
    extension_threshold: float = field(default=0.6)
    label_smoothing_factor2: float = field(default=0.25)
    eval_only: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )