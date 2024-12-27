# Personal Knowledge Management Assistant

A conversational AI tool that helps you manage, search, and derive insights from your personal documents using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

## Features

- **Document Processing**
  - Support for multiple file formats (PDF, TXT, DOCX)
  - Automatic text chunking and metadata extraction
  - Document tagging and categorization

- **Intelligent Search & Retrieval**
  - Semantic search using vector embeddings
  - Context-aware document retrieval
  - Source citation in responses

- **AI-Powered Analysis**
  - Document summarization
  - Insight extraction
  - Conversational interface for document queries
  - Context retention across conversations

## Technology Stack

- **Backend**
  - Python 3.9+
  - LangChain for RAG pipeline
  - OpenAI/Anthropic API for LLM integration
  - Hugging Face Transformers for embeddings
  - ChromaDB/Pinecone for vector storage

- **Document Processing**
  - PyPDF2 for PDF processing
  - python-docx for DOCX files
  - Beautiful Soup for HTML parsing

- **Frontend**
  - Streamlit/FastAPI
  - React (optional for advanced UI)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/personal-knowledge-assistant.git
cd personal-knowledge-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Project Structure

```
personal-knowledge-assistant/
├── app/
│   ├── core/              # Core application logic
│   │   ├── document.py    # Document processing
│   │   ├── embedding.py   # Embedding generation
│   │   ├── rag.py        # RAG pipeline
│   │   └── llm.py        # LLM integration
│   ├── api/              # API endpoints
│   ├── database/         # Database models
│   └── utils/            # Utility functions
├── frontend/            # Frontend application
├── tests/              # Test suite
├── docs/              # Documentation
└── scripts/          # Utility scripts
```

## Usage

1. Start the application:
```bash
python run.py
```

2. Access the web interface at `http://localhost:8000`

3. Upload documents and start interacting with your knowledge base

## Configuration

Key configuration options in `.env`:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
VECTOR_DB_TYPE=chroma
VECTOR_DB_PATH=./data/vectorstore
MODEL_NAME=gpt-4
CHUNK_SIZE=512
```

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Future Enhancements

- User authentication and multi-user support
- Advanced document categorization using ML
- Custom embedding model training
- API integrations for external knowledge sources
- Enhanced visualization of document relationships
- Bulk document processing and analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
