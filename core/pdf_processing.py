from PyPDF2 import PdfReader

def get_text_from_pdf(pdf_docs):
    """
    Extracts text from the uploaded PDF documents.

    - Reads each page of the uploaded PDF documents and extracts the text.
    
    Args:
        pdf_docs (list): The list of uploaded PDF documents.

    Returns:
        str: The raw text extracted from the PDF documents.
    """
    raw_text = ""
    try:
        for pdf_doc in pdf_docs:
            pdf_reader = PdfReader(pdf_doc)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")
    
    return raw_text

def get_text_chunks(raw_text):
    """
    Splits the raw text into manageable chunks for processing.

    - Uses a character-based text splitter to break the raw text into chunks.
    
    Args:
        raw_text (str): The raw text extracted from the PDF documents.

    Returns:
        list: A list of text chunks.
    """
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(raw_text)
    except Exception as e:
        raise ValueError(f"Error splitting text into chunks: {e}")
    
    return text_chunks
