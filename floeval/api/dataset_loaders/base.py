from abc import ABC, abstractmethod

from floeval.config.schemas.io.dataset import Sample


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders. Defines the interface and common functionality for loading datasets.
    """

    @staticmethod
    @abstractmethod
    def to_samples(*args, **kwargs) -> list[Sample]:
        """Abstractmethod to convert raw data into a list of Sample objects. Must be implemented by subclasses.

        Returns:
            list[Sample]: A list of Sample objects representing the dataset.
        """
        ...
