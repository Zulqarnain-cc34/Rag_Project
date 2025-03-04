import logging
from abc import ABC, abstractmethod
from typing import Union, List

class BaseExtractor(ABC):
    """
    Abstract base class for file extractors.
    """

    @staticmethod
    @abstractmethod
    def extract(file_path: str) -> Union[str, List[str]]:
        """
        Extract content from a file.
        Must be implemented by subclasses.
        """
        pass
