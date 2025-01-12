# pdf_extractor.py

import PyPDF2
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    text = ""
    
    # Open the PDF file in read-binary mode
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        
        # Iterate through all pages
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    
    return text

if __name__ == "__main__":
    # Path to your PDF
    pdf_path = os.path.join("data", "scientific_papers", "FEPmadeSimple.pdf")
    
    # Extract text
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Print or store the text
    print(extracted_text)
    
    # (Optional) If you want to save the text to a .txt file:
    # output_text_path = "FEPmadeSimple.txt"
    # with open(output_text_path, "w", encoding="utf-8") as f:
    #     f.write(extracted_text)
