from typing import Union, Tuple, List

import numpy as np

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.elastic_data_sampler import nnUNetElasticDataSampler
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetElasticDataLoader2D(nnUNetDataLoader2D):
    def __init__(self, data: nnUNetDataset, batch_size: int, patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray], label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False, transforms=None, num_replicas=None, rank=None,
                 start_index=0):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent,
                         sampling_probabilities, pad_sides, probabilistic_oversampling, transforms)
        self._dataset_sampler = nnUNetElasticDataSampler(data, num_replicas, rank, start_index)
        selected_indices = self._dataset_sampler.get_dataset_bounds()
        self.indices = [key for idx, key in enumerate(self._data.keys()) if idx in selected_indices]

    def get_indices(self):
        selected_indices = self._dataset_sampler.get_dataset_bounds()
        self.indices = [key for idx, key in enumerate(self._data.keys()) if idx in selected_indices]
        return super().get_indices()

    def set_epoch(self, epoch):
        self._dataset_sampler.set_epoch(epoch)

    def __len__(self):
        return self._dataset_sampler.__len__()
