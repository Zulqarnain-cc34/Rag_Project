from extractor import FileProcessor
from typing import Union, List, Dict

def display_extracted_text(extracted: Dict[str, Union[str, List[str]]]) -> None:
    """
    Display the extracted content in a readable format.
    For PPTX files, slide-wise content is printed.
    """
    separator = "=" * 40
    for file_path, content in extracted.items():
        print(separator)
        print(f"File: {file_path}")
        if isinstance(content, list):
            for idx, slide_text in enumerate(content, start=1):
                print(f"\n--- Slide {idx} ---")
                print(slide_text)
        else:
            print(content)
        print(separator)

def main():
    root_folder = './unprocessed_docs'
    processor = FileProcessor(root_folder)
    extracted = processor.process_files()
    display_extracted_text(extracted)

if __name__ == '__main__':
    main()
