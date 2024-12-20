import streamlit as st
import os
from rag_pipeline import RAGPipeline

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

    def show_context(self, context_chunks):
        """
        Display context chunks in an expander
        """
        with st.expander("View Source Context"):
            for i, chunk in enumerate(context_chunks, 1):
                st.markdown(f"**Source {i}:**")
                st.write(chunk)
    
    def chat_interface(self):
        """
        Main chat interface for querying documents
        """
        st.title("🧠 MemoryForge: AI Knowledge Assistant")
        
        # Initialize chat history
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
                'response': result['response'],
                'context': result.get('context', [])
            })
        
        # Display chat history with context
        for chat in st.session_state.chat_history:
            st.chat_message("human").write(chat['query'])
            with st.chat_message("ai"):
                st.write(chat['response'])
                if chat.get('context'):
                    self.show_context(chat['context'])
    
    def settings_section(self):
        """
        Streamlit section for app settings
        """
        st.sidebar.header("⚙️ Settings")
        with st.sidebar.expander("App Settings"):
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
            
            if st.button("Clear Document Cache"):
                document_dir = 'data/documents'
                if os.path.exists(document_dir):
                    for file in os.listdir(document_dir):
                        os.remove(os.path.join(document_dir, file))
                st.success("Document cache cleared!")
    
    def run(self):
        """
        Run Streamlit application
        """
        # Add app description
        st.sidebar.markdown(""" 
        # About
        MemoryForge is an AI-powered knowledge assistant that helps you:
        - 📚 Process and understand documents
        - 🤖 Generate intelligent responses
        """)
        
        # Run all sections
        self.document_upload_section()
        self.settings_section()
        self.chat_interface()

def main():
    # Set Streamlit page config
    st.set_page_config(
        page_title="MemoryForge",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize and run app
    app = MemoryForgeApp()
    app.run()

if __name__ == "__main__":
    main()
