import os
from typing import List, Dict
from pypdf import PdfReader
import docx

class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str) -> List[str]:
        """
        Extract text from PDF files
        
        Args:
            file_path (str): Path to the PDF file
        
        Returns:
            List[str]: Extracted text chunks
        """
        try:
            reader = PdfReader(file_path)
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text())
            return texts
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []

    @staticmethod
    def load_docx(file_path: str) -> List[str]:
        """
        Extract text from Word documents
        
        Args:
            file_path (str): Path to the DOCX file
        
        Returns:
            List[str]: Extracted text chunks
        """
        try:
            doc = docx.Document(file_path)
            texts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text]
            return texts
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
            return []

    @staticmethod
    def load_txt(file_path: str) -> List[str]:
        """
        Extract text from plain text files
        
        Args:
            file_path (str): Path to the TXT file
        
        Returns:
            List[str]: Extracted text chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return [file.read()]
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return []

    @classmethod
    def load_document(cls, file_path: str) -> List[str]:
        """
        Load document based on file extension
        
        Args:
            file_path (str): Path to the document
        
        Returns:
            List[str]: Extracted text chunks
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            '.pdf': cls.load_pdf,
            '.docx': cls.load_docx,
            '.doc': cls.load_docx,
            '.txt': cls.load_txt
        }
        
        loader = loaders.get(file_extension)
        if loader:
            return loader(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return []

    @staticmethod
    def text_splitter(texts: List[str], chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into manageable chunks
        
        Args:
            texts (List[str]): Input texts
            chunk_size (int): Size of each text chunk
            overlap (int): Number of characters to overlap between chunks
        
        Returns:
            List[str]: Text chunks
        """
        chunks = []
        for text in texts:
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])
        return chunks