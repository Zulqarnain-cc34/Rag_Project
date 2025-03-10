import os
import logging
from typing import Union, List, Dict
from .markdown_extractor import MarkdownExtractor
from .pptx_extractor import PPTXExtractor
from .base_extractor import BaseExtractor

class FileProcessor:
    """
    Processes files within a root folder and extracts text content
    using appropriate extractor classes based on file extension.
    """

    def __init__(self, root_folder: str):
        self.root_folder = root_folder
        self.extractors: Dict[str, BaseExtractor] = {
            ".md": MarkdownExtractor,
            ".pptx": PPTXExtractor,
        }

    def process_files(self) -> Dict[str, Union[str, List[str]]]:
        """
        Recursively walk the root folder, extract content from supported files,
        and return a dictionary mapping file paths to their extracted content.
        """
        extracted_text: Dict[str, Union[str, List[str]]] = {}
        for dirpath, _, filenames in os.walk(self.root_folder):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                current_file = os.path.join(dirpath, filename)
                extractor_class = self.extractors.get(ext)
                if extractor_class:
                    logging.info(f"Processing file: {current_file}")
                    extracted_text[current_file] = extractor_class.extract(current_file)
                else:
                    logging.debug(f"Skipping unsupported file type: {current_file}")
        return extracted_text
