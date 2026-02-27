"""
PDF to Text Converter for The Consciousness AI project

This script handles:
1. Conversion of research papers from PDF to text format
2. Extraction of key information and insights
3. Organization of extracted content
4. Integration with project documentation

Dependencies:
- PyPDF2 for PDF processing
- nltk for text processing
- models/memory/emotional_memory_core.py for storage
"""

import PyPDF2
import sys
import logging

def convert_pdf_to_text(pdf_path: str, output_path: str) -> None:
    """Convert PDF file to plain text"""
    try:
        # Initialize PDF reader
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        
        # Extract text from all pages
        text_content = []
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
            
        # Write extracted text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_content))
            
    except Exception as e:
        logging.error(f"Failed to convert PDF: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    convert_pdf_to_text("acm_iit_gnw_explanation.pdf", "acm_iit_gnw_explanation.txt")