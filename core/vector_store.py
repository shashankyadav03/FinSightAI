import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_vector_store(text_chunks):
    """
    Creates a vector store from text chunks using embeddings.

    - Converts text chunks into vector embeddings and stores them using FAISS.

    Args:
        text_chunks (list): A list of text chunks.

    Returns:
        FAISS: A FAISS vector store object containing the text embeddings.
    """
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(embedding=embeddings, texts=text_chunks)
        return vector_store
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        return None
