import os
import faiss
import numpy as np
from typing import List, Dict, Any

from document_loader import DocumentLoader
from embedding_service import EmbeddingService
from model_interface import MemoryForgeModel

class RAGPipeline:
    def __init__(
        self, 
        documents_dir: str = 'data/documents',
        chunk_size: int = 500,
        top_k: int = 3
    ):
        """
        Initialize RAG Pipeline
        
        Args:
            documents_dir (str): Directory containing documents
            chunk_size (int): Text chunk size for processing
            top_k (int): Number of top context chunks to retrieve
        """
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.top_k = top_k
        
        # Initialize services
        self.document_loader = DocumentLoader()
        self.embedding_service = EmbeddingService()
        self.model = MemoryForgeModel()
        
        # Document and embedding storage
        self.document_texts = []
        self.document_embeddings = None
        self.faiss_index = None
        
    def load_documents(self) -> List[str]:
        """
        Load all documents from specified directory
        
        Returns:
            List of processed text chunks
        """
        all_texts = []
        
        for filename in os.listdir(self.documents_dir):
            filepath = os.path.join(self.documents_dir, filename)
            
            # Load document texts
            texts = self.document_loader.load_document(filepath)
            
            # Split into chunks
            chunks = self.document_loader.text_splitter(
                texts, 
                chunk_size=self.chunk_size
            )
            
            all_texts.extend(chunks)
        
        self.document_texts = all_texts
        return all_texts
    
    def create_index(self):
        """
        Create embeddings and FAISS index for semantic search
        """
        if not self.document_texts:
            self.load_documents()
        
        # Generate embeddings
        self.document_embeddings = self.embedding_service.generate_embeddings(
            self.document_texts
        )
        
        # Create FAISS index
        self.faiss_index = self.embedding_service.create_faiss_index(
            self.document_embeddings
        )
    
    def retrieve_context(self, query: str) -> List[str]:
        """
        Retrieve most relevant document chunks
        
        Args:
            query (str): User's query
        
        Returns:
            List of most relevant context chunks
        """
        if self.faiss_index is None:
            self.create_index()
        
        # Semantic search
        results = self.embedding_service.semantic_search(
            query, 
            self.faiss_index, 
            self.document_texts, 
            top_k=self.top_k
        )
        
        return results
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate AI-powered response with retrieved context
        
        Args:
            query (str): User's query
        
        Returns:
            Dict containing response and context
        """
        # Retrieve relevant context
        context_chunks = self.retrieve_context(query)
        context = " ".join(context_chunks)
        
        # Generate response
        response = self.model.generate_response(
            query, 
            context=context
        )
        
        return {
            "response": response,
            "context": context_chunks,
            "query": query
        }

# Example Usage
if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    
    # Load documents
    rag_pipeline.load_documents()
    
    # Create search index
    rag_pipeline.create_index()
    
    # Example query
    query = "What are the key concepts in machine learning?"
    result = rag_pipeline.generate_response(query)
    
    print("Query:", result['query'])
    print("Response:", result['response'])
    print("Context Chunks:", result['context'])