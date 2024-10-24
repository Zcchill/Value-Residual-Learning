from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, MODEL_FOR_CAUSAL_LM_MAPPING
import json

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    modeling_name: Optional[str] = field(
        default=None,
        metadata={"help": "choose modeling_gpt2 python file"},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "choose which kind of attention to use"},
    )
    
    def __post_init__(self):
        if self.config_overrides is not None and self.model_name_or_path is not None:
            raise ValueError(
                "--config_overrides can't be used in combination with --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    domain_weight_train: Optional[str] = field(
        default=None, metadata={"help": "The weight of each domain for training."}
    )
    domain_weight_eval: Optional[str] = field(
        default=None, metadata={"help": "The weight of each domain for evaluation."}
    )
    dataset_dir: str = field(
        default='.', metadata={"help": "Path to the dataset directory."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_token_length: int = field(
        default=1024,
        metadata={
            "help": (
                "Input sequence length after tokenization. "
            )
        },
    )
    do_padding: bool = field(
        default=False, metadata={"help": "Pad the inputs."}
    )
    use_data_cache: bool = field(
        default=False, metadata={"help": "If use cached data when SFT."}
    )


    def __post_init__(self):
        self.domain_weight_train = json.loads(self.domain_weight_train)
        self.domain_weight_eval = json.loads(self.domain_weight_eval)


@dataclass
class FullTrainingArguments(TrainingArguments):
    lr_end: float = field(
            default=1e-2,
            metadata={"help": "The final learning rate of the learning rate scheduler."},
    )
    lr_scheduler_name: str = field(
        default=None, metadata={"help": "Custom LR scheduler name (linear_warmup_exponential, linear_warmup_cosine)"}
    )
    num_warmup_stop_steps: float = field(
            default=None,
            metadata={"help": "The decay of Lr stops"},
    )        