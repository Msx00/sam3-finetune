from torch.utils.data.distributed import DistributedSampler

from models.resumable_batch_sampler import (
    ResumableDistributedBatchSampler,
    ResumableRandomBatchSampler,
)


def test_random_sampler_resume_matches_full_epoch_suffix():
    sampler = ResumableRandomBatchSampler(dataset_size=11, batch_size=3, seed=7)
    sampler.set_epoch(4)
    full_epoch = list(sampler)

    sampler.set_epoch(4, start_batch=2)
    assert list(sampler) == full_epoch[2:]
    assert len(sampler) == len(full_epoch) - 2


def test_each_distributed_rank_resumes_its_own_epoch_suffix():
    dataset = list(range(14))
    for rank in (0, 1):
        distributed = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=rank,
            shuffle=True,
            seed=19,
        )
        sampler = ResumableDistributedBatchSampler(distributed, batch_size=2)
        sampler.set_epoch(3)
        full_epoch = list(sampler)

        sampler.set_epoch(3, start_batch=2)
        assert list(sampler) == full_epoch[2:]
        assert len(sampler) == len(full_epoch) - 2
