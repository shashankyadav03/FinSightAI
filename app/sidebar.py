import streamlit as st
from core.pdf_processing import get_text_from_pdf, get_text_chunks
from core.vector_store import get_vector_store
from core.conversation_chain import get_conversation_chain

def setup_sidebar(chat_placeholder):
    """
    Sets up the sidebar for the Streamlit app.

    - Provides options to upload documents, process them, and start a new chat session.
    - Initializes the conversation chain after processing the documents.

    Args:
        chat_placeholder: A Streamlit placeholder for displaying messages in the chat interface.
    """
    with st.sidebar:
        if st.button("Start New Chat"):
            st.session_state.conversation_chain = None
            st.session_state.chat_history = []
            st.success("New chat started! Please upload a document to begin.")
            chat_placeholder.empty()
        
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your financial documents", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                try:
                    with st.spinner("Processing..."):
                        raw_text = get_text_from_pdf(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        st.success("Processing complete!")
                        st.session_state.conversation_chain = get_conversation_chain(vector_store)
                        chat_placeholder.empty()
                        st.write("🤖: Processing is complete! You can now start asking your questions.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
            else:
                st.warning("Please upload at least one document before processing.")
        st.write("🤖: OR")
        if st.button("Process Sample Document"):
            pdf_docs = ["data/FinSightAI_Report.pdf"]
            sample_text = get_text_from_pdf(pdf_docs)
            text_chunks = get_text_chunks(sample_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation_chain = get_conversation_chain(vector_store)
            st.success("Sample document processed successfully!")
            chat_placeholder.empty()
            st.write("🤖: Sample document processed successfully! You can now start asking your questions.")

        
