"""Keep native LeRobot row/image behavior; query numeric windows without decoding images."""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hf_transform_to_torch


class NumericQueryLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, **kwargs):
        self._initializing_numeric_indices = True
        super().__init__(*args, **kwargs)
        self._initializing_numeric_indices = False
        self.hf_dataset.set_transform(hf_transform_to_torch)

    def load_hf_dataset(self):
        loaded = super().load_hf_dataset()
        if getattr(self, "_initializing_numeric_indices", False):
            loaded.reset_format()
        return loaded

    def _query_hf_dataset(self, query_indices):
        plain = self.hf_dataset.with_format(None)
        result = {}
        for key, indices in query_indices.items():
            if key in self.meta.video_keys:
                continue
            if self._absolute_to_relative_idx is not None:
                indices = [self._absolute_to_relative_idx[index] for index in indices]
            result[key] = torch.as_tensor(plain.select_columns([key])[indices][key])
        return result
