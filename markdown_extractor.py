import logging
from base_extractor import BaseExtractor

class MarkdownExtractor(BaseExtractor):
    """Extracts text from Markdown (.md) files."""

    @staticmethod
    def extract(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logging.info(f"Successfully extracted Markdown file: {file_path}")
            return content
        except Exception as e:
            logging.error(f"Error reading Markdown file {file_path}: {e}")
            return ""

