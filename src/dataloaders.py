from pathlib import Path
import random
import numpy as np
import torch
from datasets import IterableDataset
from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
from datasets.utils.logging import get_logger
import logging
from datasets import load_from_disk

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

class UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
        RandomlyCyclingMultiSourcesExamplesIterable):

    def __init__(self, ex_iterables, generator, probabilities=None, stopping_strategy="all_exhausted"):
        '''
        probabilities: vector of static probabilities over training
        stopping_strategy: `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped
                            as soon as every samples of every dataset has been added at least once.
        '''
        super().__init__(ex_iterables, generator, stopping_strategy=stopping_strategy)
        self.probabilities = probabilities

    @staticmethod
    def _iter_random_indices(rng, num_sources, p, random_batch_size=8096):
        while True:
            yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=p))

    def _give_indice_iterator(self):
        rng = deepcopy(self.generator)
        logger.info(f'self.probabilities is {self.probabilities}')
        return self._iter_random_indices(rng, len(self.ex_iterables), p=self.probabilities)

    def shard_data_sources(self, worker_id, num_workers):
        return self

    @property
    def n_shards(self):
        return 1

    def shuffle_data_sources(self, seed):
        """Shuffle the data sources of each wrapped examples iterable."""
        self.ex_iterables = [ex_iterable.shuffle_data_sources(seed) for ex_iterable in self.ex_iterables]
        return self


def data_generator_interleave(datasets, probabilities=None, seed=None):
    iterable_datasets = []
    for dataset in datasets.values():
        if not isinstance(dataset, IterableDataset):
            iterable_datasets.append(dataset.to_iterable_dataset())
        else:
            iterable_datasets.append(dataset)

    ex_iterables = [d._ex_iterable for d in iterable_datasets]

    generator = np.random.default_rng(seed)
    ex_iterable = UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables, generator=generator, probabilities=probabilities)

    return IterableDataset(ex_iterable=ex_iterable)

def domain_data_gen(shards, domain_name, max_samples_domain, loop=True, seed=111, shuffle=False, keep_in_memory=False):
    def get_shard_ds(shard_dir, seed, shuffle):
        # TODO hack:
        if keep_in_memory:
            curr_keep_in_memory = (hash(str(shard_dir)) % 2 == 0)
        else:
            curr_keep_in_memory = False
        shard = load_from_disk(dataset_path=str(shard_dir), keep_in_memory=curr_keep_in_memory)
        if shuffle:
            shard = shard.shuffle(seed=seed)
        return shard

    if loop:
        # during training phase, some examples may be sampled more than once due to our "stopping_strategy="all_exhausted" in multi-domain sampling phase, change the seed to avoid redirect to the beginning of the dataset
        while True:
            logger.info(f"Start {domain_name} domain data loading.")
            idx = 0
            for shard_dir in shards:
                shard = get_shard_ds(shard_dir, seed, shuffle)
                for ex in shard:
                    yield ex
                    idx += 1
                    if idx%10000 == 0:
                        logger.info(f'The model has trained on {idx} eamples from {domain_name}.')
                    if max_samples_domain is not None and idx >= max_samples_domain:
                        logger.info(f'The final dataset size is {max_samples_domain} for {domain_name}.')
                        return
                seed += 1
    else:
        idx = 0
        for shard_dir in shards:
            shard = get_shard_ds(shard_dir, seed, shuffle)
            for ex in shard:
                yield ex
                idx += 1                    
                if max_samples_domain is not None and idx >= max_samples_domain:
                    logger.info(f'The final dataset size is {max_samples_domain} for {domain_name}.')
                    return
            seed += 1


def get_perdomain_datasets(
        preprocessed_dir,
        cache_dir=None,
        split=None,
        seed=42,
        domain_weight=None,
        max_samples=None,
        shuffle=False,
        keep_in_memory=False):
    '''
    Returns a dictionary from domain name to IterableDataset.
    '''
    all_ds = {}
    for domain in domain_weight.keys():
        domain_dir = preprocessed_dir / domain

        if (domain_dir / 'dataset_info.json').exists():
            curr_shards = [domain_dir]
        else:
            curr_shards = list(domain_dir.iterdir())
            random.Random(seed).shuffle(curr_shards)

        if max_samples is not None:
            max_samples_domain = max_samples * domain_weight[domain]
        else:
            max_samples_domain = None

        ds = IterableDataset.from_generator(
                domain_data_gen,
                gen_kwargs={'shards': curr_shards,
                            'domain_name': domain,
                            'max_samples_domain': max_samples_domain,
                            'loop': (split == 'train'),
                            'seed': seed,
                            'shuffle': shuffle,
                            'keep_in_memory': keep_in_memory}
                )
        all_ds[domain] = ds
        seed += 1
    return all_ds

def get_preprocessed_mixed_dataset(
        preprocessed_dir,
        dataset_name='pile',
        domain_weight=None,
        cache_dir=None,
        split='train',
        max_samples=None,
        seed=42,
        no_interleave=True,
        shuffle=False,
        keep_in_memory=False):
    '''preprocessed_dir: has the following format
               first level: domain directories
               second level: shards for each domain. number of shards per domain should be the same.
       cache_dir: cache directory for arrow files (if needed)
       split: train or validation
       seed: int (controls ordering of data shards)
       no_interleave: don't interleave the domains - just iterate through the data in order
       shuffle: on-the-fly shuffle with a buffer size 100k
    '''
    # select split
    preprocessed_dir = Path(preprocessed_dir)
    if split is not None and (preprocessed_dir / split).exists():
        preprocessed_dir = preprocessed_dir / split
    else:
        logger.warn("No split used or split directory not found: using same data for all splits.")
    
    # load data
    try:
        all_ds = get_perdomain_datasets(
            preprocessed_dir,
            cache_dir=cache_dir,
            split=split,
            seed=seed,
            domain_weight=domain_weight,
            max_samples=max_samples,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory)
    except Exception:
        raise ValueError(f"dataset_name {dataset_name} not implemented.")

    # Combine these datasets
    def data_generator_no_interleave(shards):
        for domain, shard in shards.items():
            for ex in shard:
                yield ex

    if no_interleave or len(domain_weight)==1:
        logger.info('Here dataset will be organied in a no_interleave way')
        ds = IterableDataset.from_generator(data_generator_no_interleave, gen_kwargs={'shards': all_ds})
    else:
        ds = data_generator_interleave(
                all_ds,
                probabilities=list(domain_weight.values()),
                seed=seed)

    return ds

def get_data_collator(tokenizer, return_tensors='pt', do_padding=False, max_length=1024):
    def data_collator(features):
        # pad data if it haven't been done during preprocess
        if not do_padding:
            batch = {
                    k: torch.tensor([f[k] for f in features])
                    for k in features[0].keys() if k in ['input_ids', 'attention_mask', 'labels']
                    }
        else:
            batch = tokenizer.pad(features, return_tensors=return_tensors, pad_to_multiple_of=max_length)
        
        # set "input_ids"
        batch['input_ids'] = batch['input_ids'].long()
        
        # set "attention_mask"
        if 'attention_mask' not in batch:
            batch['attention_mask'] = torch.ones_like(batch['input_ids']).long()
        else:
            batch['attention_mask'] = batch['attention_mask'].long()

        # set "label"
        if 'labels' not in batch:
            labels = batch['input_ids'].clone()
            batch["labels"] = labels
        if tokenizer.pad_token_id is not None:
            batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100

        return batch
    return data_collator