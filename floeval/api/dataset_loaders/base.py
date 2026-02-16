from abc import ABC, abstractmethod

from floeval.config.schemas.io.dataset import PartialSample, Sample


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @classmethod
    @abstractmethod
    def to_samples(cls, *args, **kwargs) -> list[Sample]:
        """Abstractmethod to convert raw data into a list of Sample objects.

        Returns:
            list[Sample]: A list of Sample objects representing the dataset.
        """
        ...

    @classmethod
    @abstractmethod
    def to_partial_samples(cls, *args, **kwargs) -> list[PartialSample]:
        """Abstractmethod to convert raw data into a list of PartialSample objects.

        PartialSample allows empty llm_response field (for PartialDataset creation)

        Returns:
            list[PartialSample]: A list of PartialSample objects with llm_response field empty.
        """
        ...
