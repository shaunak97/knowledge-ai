import streamlit as st
import os
import tempfile

from src.rag_pipeline import RAGPipeline

class MemoryForgeApp:
    def __init__(self):
        """
        Initialize Streamlit application for MemoryForge
        """
        self.rag_pipeline = RAGPipeline()
        
    def document_upload_section(self):
        """
        Streamlit section for document upload
        """
        st.sidebar.header("📤 Document Upload")
        uploaded_files = st.sidebar.file_uploader(
            "Upload Documents", 
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx']
        )
        
        if uploaded_files:
            document_dir = 'data/documents'
            os.makedirs(document_dir, exist_ok=True)
            
            for file in uploaded_files:
                # Save uploaded file
                temp_path = os.path.join(document_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
            
            st.sidebar.success(f"Uploaded {len(uploaded_files)} documents")
            
            # Rebuild index after upload
            self.rag_pipeline.load_documents()
            self.rag_pipeline.create_index()
    
    def chat_interface(self):
        """
        Main chat interface for querying documents
        """
        st.title("🧠 MemoryForge: AI Knowledge Assistant")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # User input
        user_query = st.text_input("Ask a question about your documents")
        
        if user_query:
            # Generate response
            result = self.rag_pipeline.generate_response(user_query)
            
            # Store in chat history
            st.session_state.chat_history.append({
                'query': user_query,
                'response': result['response']
            })
        
        # Display chat history
        for chat in st.session_state.chat_history:
            st.chat_message("human").write(chat['query'])
            st.chat_message("ai").write(chat['response'])
    
    def run(self):
        """
        Run Streamlit application
        """
        self.document_upload_section()
        self.chat_interface()

def main():
    app = MemoryForgeApp()
    app.run()

if __name__ == "__main__":
    main()