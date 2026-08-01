"""Deterministic batch samplers with an epoch-local resume offset."""

import torch


class ResumableRandomBatchSampler:
    """Deterministic shuffled batches that can start at an epoch batch offset."""

    def __init__(self, dataset_size, batch_size, seed):
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.start_batch = 0
        if self.dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def total_batches(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch, start_batch=0):
        start_batch = int(start_batch)
        if start_batch < 0 or start_batch > self.total_batches:
            raise ValueError(
                f"start_batch={start_batch} is outside [0, {self.total_batches}]"
            )
        self.epoch = int(epoch)
        self.start_batch = start_batch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            self.dataset_size, generator=generator
        ).tolist()
        for offset in range(
            self.start_batch * self.batch_size,
            self.dataset_size,
            self.batch_size,
        ):
            yield indices[offset:offset + self.batch_size]

    def __len__(self):
        return self.total_batches - self.start_batch


class ResumableDistributedBatchSampler:
    """Batch a DistributedSampler while skipping completed batches by index."""

    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.start_batch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def total_batches(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch, start_batch=0):
        start_batch = int(start_batch)
        if start_batch < 0 or start_batch > self.total_batches:
            raise ValueError(
                f"start_batch={start_batch} is outside [0, {self.total_batches}]"
            )
        self.sampler.set_epoch(int(epoch))
        self.start_batch = start_batch

    def __iter__(self):
        batch = []
        batch_index = 0
        for sample_index in self.sampler:
            batch.append(sample_index)
            if len(batch) != self.batch_size:
                continue
            if batch_index >= self.start_batch:
                yield batch
            batch = []
            batch_index += 1
        if batch and batch_index >= self.start_batch:
            yield batch

    def __len__(self):
        return self.total_batches - self.start_batch
