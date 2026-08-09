from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from .config import DataConfig, ModelConfig


class DeterministicFakeImages(Dataset[tuple[Tensor, int]]):
    """Small offline dataset used only for pipeline tests."""

    def __init__(self, size: int, image_size: int, channels: int, num_classes: int, seed: int) -> None:
        self.size = size
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.seed = seed
        self.targets = [index % num_classes for index in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand(
            self.channels,
            self.image_size,
            self.image_size,
            generator=generator,
        ) * 0.4 - 0.2
        label = self.targets[index]
        stripe = (label * 3) % self.image_size
        image[label % self.channels, :, stripe : stripe + 2] += 0.8
        return image.clamp(-1, 1), label


class EpochKeyDataset(Dataset[Any]):
    """Allow a normal dataset to receive `(local_index, epoch)` sampler keys."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: int | tuple[int, int]) -> Any:
        local_index = key[0] if isinstance(key, tuple) else key
        return self.dataset[local_index]


def _stateless_flip(seed: int, epoch: int, sample_index: int) -> bool:
    """SplitMix64 bit used for deterministic augmentation without worker RNG state."""

    mask = (1 << 64) - 1
    value = (seed ^ (epoch * 0x9E3779B97F4A7C15) ^ sample_index) & mask
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return bool(value & 1)


class CIFARTrainingSubset(Dataset[tuple[Tensor, int]]):
    def __init__(self, dataset: Dataset[Any], indices: Sequence[int], flip_seed: int) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)
        self.flip_seed = flip_seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, key: int | tuple[int, int]) -> tuple[Tensor, int]:
        if isinstance(key, tuple):
            local_index, epoch = key
        else:
            local_index, epoch = key, 0
        original_index = self.indices[local_index]
        image, label = self.dataset[original_index]
        from torchvision.transforms import functional as vision_functional

        if _stateless_flip(self.flip_seed, epoch, original_index):
            image = vision_functional.hflip(image)
        tensor = vision_functional.to_tensor(image)
        tensor = vision_functional.normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return tensor, int(label)


class CIFARValidationSubset(Dataset[tuple[Tensor, int]]):
    def __init__(self, dataset: Dataset[Any], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[self.indices[local_index]]
        from torchvision.transforms import functional as vision_functional

        tensor = vision_functional.to_tensor(image)
        tensor = vision_functional.normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return tensor, int(label)


class InfiniteDeterministicBatchSampler(Sampler[list[tuple[int, int]]]):
    """Infinite, epoch-shuffled batches addressable by consumed microbatch count.

    Prefetch may advance the iterator internally, but resume reconstructs the next
    batch from the count that the trainer actually consumed, not sampler state.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        consumed_batches: int = 0,
    ) -> None:
        if dataset_size < batch_size or batch_size <= 0 or consumed_batches < 0:
            raise ValueError("invalid deterministic batch sampler settings")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.consumed_batches = consumed_batches
        self.batches_per_epoch = dataset_size // batch_size

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        epoch, first_batch = divmod(self.consumed_batches, self.batches_per_epoch)
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.dataset_size, generator=generator).tolist()
            usable = order[: self.batches_per_epoch * self.batch_size]
            for batch_index in range(first_batch, self.batches_per_epoch):
                start = batch_index * self.batch_size
                indices = usable[start : start + self.batch_size]
                yield [(index, epoch) for index in indices]
            epoch += 1
            first_batch = 0

    def __len__(self) -> int:
        return 2**31


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset[Any]
    validation: Dataset[Any]
    metadata: dict[str, Any]


def stratified_split_indices(
    targets: Sequence[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    grouped: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        grouped.setdefault(int(target), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for target in sorted(grouped):
        indices = grouped[target]
        order = torch.randperm(len(indices), generator=generator).tolist()
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[position] for position in order[:validation_count])
        train_indices.extend(indices[position] for position in order[validation_count:])
    train_indices.sort()
    validation_indices.sort()
    return train_indices, validation_indices


def _indices_hash(train_indices: Sequence[int], validation_indices: Sequence[int]) -> str:
    payload = json.dumps(
        {"train": list(train_indices), "validation": list(validation_indices)},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_datasets(data: DataConfig, model: ModelConfig, data_seed: int) -> DatasetBundle:
    if data.dataset == "fake":
        full = DeterministicFakeImages(
            data.fake_size,
            model.image_size,
            model.in_channels,
            model.num_classes,
            data_seed,
        )
        train_indices, validation_indices = stratified_split_indices(
            full.targets, data.validation_fraction, data.split_seed
        )
        preprocessing = "deterministic_fake_v1_range_-1_1"
        return DatasetBundle(
            train=EpochKeyDataset(Subset(full, train_indices)),
            validation=Subset(full, validation_indices),
            metadata={
                "dataset": "fake",
                "train_size": len(train_indices),
                "validation_size": len(validation_indices),
                "split_hash": _indices_hash(train_indices, validation_indices),
                "preprocessing": preprocessing,
                "preprocessing_hash": hashlib.sha256(preprocessing.encode()).hexdigest(),
            },
        )

    if data.dataset == "cifar10":
        if model.image_size != 32 or model.in_channels != 3 or model.num_classes != 10:
            raise ValueError("CIFAR-10 requires image_size=32, in_channels=3, num_classes=10")
        from torchvision import datasets

        train_base = datasets.CIFAR10(
            root=data.root, train=True, transform=None, download=data.download
        )
        validation_base = datasets.CIFAR10(
            root=data.root, train=True, transform=None, download=data.download
        )
        train_indices, validation_indices = stratified_split_indices(
            train_base.targets, data.validation_fraction, data.split_seed
        )
        preprocessing = (
            "cifar10_train:stateless_hflip_v1,to_tensor,normalize_0.5;"
            "val:to_tensor,normalize_0.5"
        )
        return DatasetBundle(
            train=CIFARTrainingSubset(train_base, train_indices, data_seed),
            validation=CIFARValidationSubset(validation_base, validation_indices),
            metadata={
                "dataset": "cifar10",
                "root": data.root,
                "train_size": len(train_indices),
                "validation_size": len(validation_indices),
                "split_hash": _indices_hash(train_indices, validation_indices),
                "preprocessing": preprocessing,
                "preprocessing_hash": hashlib.sha256(preprocessing.encode()).hexdigest(),
            },
        )
    raise ValueError(f"unsupported dataset: {data.dataset}")


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_loaders(
    bundle: DatasetBundle,
    data: DataConfig,
    batch_size: int,
    data_seed: int,
    *,
    pin_memory: bool,
    consumed_train_batches: int = 0,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    common = {
        "num_workers": data.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": data.num_workers > 0,
    }
    if len(bundle.train) < batch_size:
        raise ValueError("training dataset is smaller than one full batch")
    batch_sampler = InfiniteDeterministicBatchSampler(
        len(bundle.train), batch_size, data_seed, consumed_train_batches
    )
    train = DataLoader(
        bundle.train,
        batch_sampler=batch_sampler,
        **common,
    )
    validation = DataLoader(
        bundle.validation,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train, validation
