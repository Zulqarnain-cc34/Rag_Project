from processor.file_processor import FileProcessor
from strategies.adaptive_retrieval import AdaptiveRAG
import os 

# Function to save the combined texts into separate files
def save_combined_texts_to_files(combined_texts_list, output_folder='./data/processed_docs'):
    # Ensure the output folder exists, create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through the combined texts list and save each text into a separate file
    for idx, text in enumerate(combined_texts_list, start=1):
        file_path = os.path.join(output_folder, f"combined_text_{idx}.txt")  # Construct file path with an index
        with open(file_path, 'w') as file:  # Open the file in write mode
            file.write(text)  # Write the text content into the file
        print(f"Saved: {file_path}")  # Print confirmation of the saved file

# Function to load combined text files from a given folder
def load_combined_texts_from_files(input_folder='./data/processed_docs'):
    combined_texts_list = []  # Initialize an empty list to store texts
    
    # Loop through each file in the specified folder
    for file_name in os.listdir(input_folder):  
        # Check if the file is a text file
        if file_name.endswith(".txt"):  file_path = os.path.join(input_folder, file_name)  # Get the full file path
            with open(file_path, 'r') as file:  # Open the file in read mode
                combined_texts_list.append(file.read())  # Append the file content to the list
            print(f"Loaded: {file_path}")  # Print confirmation of the loaded file
    
    return combined_texts_list  # Return the list of loaded texts

# Main entry point of the script
if __name__ == "__main__":
    root_folder = './data/unprocessed_docs'  # Define the folder with unprocessed documents
    processor = FileProcessor(root_folder)  # Create an instance of the FileProcessor to process files
    texts = processor.process_files()  # Process files and get the content in a dictionary format
    combined_texts_list = []  # Initialize an empty list to store the combined text content

    # Loop through each file in the processed files dictionary
    for file_path, content in texts.items():
        # If the content is a list (which means the file is a PowerPoint presentation)
        if isinstance(content, list):
            combined = ""  # Initialize an empty string to hold combined slide content
            # Loop through each slide in the PowerPoint presentation
            for idx, slide_text in enumerate(content, start=1):
                combined += f"Slide {idx}:\n{slide_text}\n\n"  # Combine the text with slide number and content
            combined_texts_list.append(combined)  # Append the combined content for this PPTX file
        else:
            # For other types of files (e.g., markdown), append the content directly
            combined_texts_list.append(content)

    save_combined_texts_to_files(combined_texts_list)  # Save the combined texts to files in the specified folder
