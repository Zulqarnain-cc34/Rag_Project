from processor.file_processor import FileProcessor
from strategies.adaptive_retrieval import AdaptiveRAG

if __name__ == "__main__":
    root_folder = './data/unprocessed_docs'
    processor = FileProcessor(root_folder)
    texts = processor.process_files()
    combined_texts_list = []
    
    for file_path, content in texts.items():
        # If the file is a PPTX (content is a list of slides)
        if isinstance(content, list):
            combined = ""
            for idx, slide_text in enumerate(content, start=1):
                combined += f"Slide {idx}:\n{slide_text}\n\n"
            combined_texts_list.append(combined)
        else:
            # For markdown or other file types where content is already a single string
            combined_texts_list.append(content)

    rag_system = AdaptiveRAG(combined_texts_list)

    queries = [
        "Tell me about the Cold Bore Technology project we did"
    ]
    #
    for query in queries:
        print(f"Query: {query}")
        result = rag_system.answer(query)
        print(f"Answer: {result}")
