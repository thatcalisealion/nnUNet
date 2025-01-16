import os
from typing import Optional

import numpy as np
import torch

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetElasticDataSampler(object):
    """
    A data sampler based on PyTorch's :class:`torch.utils.data.ElasticDatasetSampler` to handle distributed training.
    It loads a subset of data from the original dataset that is exclusive to the current process.
    """
    def __init__(self, dataset: nnUNetDataset, num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 start_index: int = 0):
        if start_index >= len(dataset):
            raise ValueError(f"Start index {start_index} should be less than dataset size {len(dataset)}")

        self._dataset = dataset
        if num_replicas is None:
            self.num_replicas = int(os.environ['WORLD_SIZE'])
        if rank is None:
            self.rank = int(os.environ['RANK'])
        self.start_index = start_index
        self.epoch = 0
        self.num_samples = int(np.ceil(float(len(self._dataset) - self.start_index) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def get_dataset_bounds(self) -> list[int]:
        # Shuffle data deterministically based on current epoch
        gen = torch.Generator()
        gen.manual_seed(self.epoch)
        next_indices = (
            torch.randperm(len(self._dataset) - self.start_index, generator=gen).add(self.start_index).tolist()
        )

        # Add extra samples to make it evenly divisible
        next_indices += next_indices[: (self.total_size - len(next_indices))]
        assert len(next_indices) == self.total_size

        # Pick a subsample
        selected_indices = next_indices[self.rank: self.total_size: self.num_replicas]
        assert len(selected_indices) == self.num_samples
        return selected_indices

    def set_epoch(self, epoch):
        """
        If shuffle in data is required, ensure this method is called at the beginning of each epoch to guarantee
        data is randomized. Otherwise, each epoch will have the same ordering.
        """
        self.epoch = epoch

    def __len__(self):
        return self.num_samples
