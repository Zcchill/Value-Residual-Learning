import sys
import os

import logging
from pathlib import Path
import os
import sys

import datasets
import torch
from itertools import islice

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.trainer_callback import TrainerState
from transformers.trainer import TRAINER_STATE_NAME

from training_args import ModelArguments, DataTrainingArguments, FullTrainingArguments
import dataloaders as data_utils
from trainer import UpdatableTrainer

check_min_version("4.27.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/training_args.py or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FullTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load different model architecture
    if model_args.modeling_name == "modeling_gpt2_baseline":
        from modeling.modeling_gpt2_baseline import GPT2LMHeadModel
    elif model_args.modeling_name == "modeling_llama_baseline":
        from modeling.modeling_llama_baseline import LlamaForCausalLM
    elif model_args.modeling_name == "modeling_llama_resformer":
        from modeling.modeling_llama_resformer import LlamaForCausalLM
    elif model_args.modeling_name == "modeling_llama_svformer":
        from modeling.modeling_llama_svformer import LlamaForCausalLM
    elif model_args.modeling_name == "modeling_llama_NeuTRENO_lambda04":
        from modeling.modeling_llama_NeuTRENO_lambda04 import LlamaForCausalLM
    elif model_args.modeling_name == "modeling_llama_NeuTRENO_resformer":
        from modeling.modeling_llama_NeuTRENO_resformer import LlamaForCausalLM
    else:
        raise ValueError(f"Unknown modeling name: {model_args.modeling_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary.
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            state = TrainerState.load_from_json(str(Path(last_checkpoint) / TRAINER_STATE_NAME))
            global_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
            num_skip_examples = state.global_step * global_batch_size
            logger.info(f"Skipping {num_skip_examples} examples")

    # Set seed before initializing model. Default to be 42.
    set_seed(training_args.seed)

    # Load config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
    }
    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        config._attn_implementation = model_args.attn_implementation
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.info(f"Original config: {config}")
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
        config._attn_implementation = model_args.attn_implementation

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": True,
    }
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load model
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            torch_dtype=None,
            low_cpu_mem_usage=False,
        )
    else:
        if "gpt" in model_args.modeling_name:
            model = GPT2LMHeadModel(config)
        elif "llama" in model_args.modeling_name:
            model = LlamaForCausalLM(config)
        else:
            raise ValueError(f"{model_args.modeling_name} is not supported.")
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    logger.info(f"The model: {model}")    
    logger.info(f"len(tokenizer): {len(tokenizer)}")
    
    # Preprocessing the datasets
    if training_args.do_train:
        train_dataset = data_utils.get_preprocessed_mixed_dataset(
                preprocessed_dir=data_args.dataset_dir,
                dataset_name=data_args.dataset_name,
                domain_weight=data_args.domain_weight_train,
                cache_dir=model_args.cache_dir,
                split='train',
                max_samples=data_args.max_train_samples,
                no_interleave=False,
                seed=training_args.seed,
                shuffle=True,
                keep_in_memory=False)
        data_iterator = iter(train_dataset)
    if training_args.do_eval:
        eval_dataset = dict()
        for domain, weight in data_args.domain_weight_eval.items():
            eval_dataset[domain] = data_utils.get_preprocessed_mixed_dataset(
                    preprocessed_dir=data_args.dataset_dir,
                    dataset_name=data_args.dataset_name,
                    domain_weight={domain: weight},
                    cache_dir=model_args.cache_dir,
                    split='validation',
                    max_samples=data_args.max_eval_samples,
                    no_interleave=True,
                    keep_in_memory=False)
        data_iterator = iter(eval_dataset[domain])
    sample = list(islice(data_iterator, 2))
    logger.info(f"Dataset slice: {sample}")

    # Empty cache
    torch.cuda.empty_cache()
    training_args.ddp_find_unused_parameters = False

    # Initialize Trainer
    trainer = UpdatableTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_utils.get_data_collator(tokenizer, do_padding=data_args.do_padding, max_length=data_args.max_token_length),
    )
    if training_args.load_best_model_at_end:
        trainer.add_callback(EarlyStoppingCallback)

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        metrics["train_samples"] = data_args.max_train_samples if data_args.max_train_samples is not None else None
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, eval_dataset in eval_dataset.items():
                dataset_metrics = trainer.evaluate(
                    eval_dataset=eval_dataset,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
                trainer.log_metrics(eval_dataset_name, dataset_metrics)
                trainer.save_metrics(eval_dataset_name, dataset_metrics)
        else:
            metrics = trainer.evaluate()
            metrics["eval_samples"] = data_args.max_eval_samples if data_args.max_eval_samples is not None else None
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        

if __name__ == "__main__":
    main()
