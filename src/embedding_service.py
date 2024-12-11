import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

class EmbeddingService:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedding model and tokenizer
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on model output
        
        Args:
            model_output: Transformer model output
            attention_mask: Attention mask tensor
        
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, texts):
        """
        Generate embeddings for given texts
        
        Args:
            texts (List[str]): List of text chunks
        
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])

        # Tokenize texts
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()

    def create_faiss_index(self, embeddings):
        """
        Create FAISS index for efficient similarity search
        
        Args:
            embeddings (np.ndarray): Array of embeddings
        
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def semantic_search(self, query, index, texts, top_k=5):
        """
        Perform semantic search
        
        Args:
            query (str): Search query
            index (faiss.Index): FAISS index
            texts (List[str]): Original texts
            top_k (int): Number of top results
        
        Returns:
            List of top matching texts
        """
        query_embedding = self.generate_embeddings([query])
        distances, indices = index.search(query_embedding, top_k)
        
        return [texts[i] for i in indices[0]]

# Usage example
if __name__ == "__main__":
    embedding_service = EmbeddingService()
    
    # Example texts
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Python is a popular programming language for data science"
    ]
    
    # Generate embeddings
    embeddings = embedding_service.generate_embeddings(texts)
    
    # Create FAISS index
    index = embedding_service.create_faiss_index(embeddings)
    
    # Semantic search
    query = "AI and machine learning"
    results = embedding_service.semantic_search(query, index, texts)
    print("Search Results:", results)