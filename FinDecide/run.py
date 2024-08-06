import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from transformers import LLaMATokenizer, LLaMAForCausalLM
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain_openai import ChatOpenAI

# Load the fine-tuned model and tokenizer
def initialize_llama_model():
    model_path = "./finetuned-llama-7b"
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    return model, tokenizer

def get_text_from_pdf(pdf_docs):
    raw_text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(embedding=embeddings, texts=text_chunks)
    return vector_store

def get_conversation_chain(vector_store):
    # Initialize the LLaMA model
    model, tokenizer = initialize_llama_model()

    # Optionally, you can use GPT-4 or other models
    llm = ChatOpenAI(model_name="gpt-4")
    # llm = HuggingFaceHub(repo_id="facebook/llama-7b", model_kwargs={"max_length": 512, "temperature": 0.7})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def handle_user_query(user_query):
    # Check if conversation chain is initialized
    if st.session_state.conversation_chain is None:
        st.write("🤖: Please upload your document to continue!")
        return

    response = st.session_state.conversation_chain.invoke({'question': user_query})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # User message (right-aligned)
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <strong>👤: {message.content}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Bot message (left-aligned)
            st.markdown(
                f"""
                <div style="text-align: left;">
                    <strong>🤖: {message.content}</strong>
                </div>
                """,
                unsafe_allow_html=True
        )

        # Add a "Verify" button next to the bot's message
        if st.button(f"Verify Response {i//2 + 1}"):
            with st.spinner("Verifying response..."):
                verify_response(message.content)

def verify_response(bot_message):
    st.write("Verifying the following response:")
    st.write(f"🤖: {bot_message}")

    # Verification logic (real-time data check)
    # Placeholder: Implement your actual verification logic here
    # Example: Compare the bot's message with real-time financial data fetched from an API

    st.success("Verification complete!")
    st.radio("Is the response correct?", ["Yes", "No"])

    # Add a button to continue the conversation
    if st.button("Continue"):
        # Clear all chat messages
        st.session_state.chat_history = []
        st.write("🤖: Great! What else would you like to know?")

def main():
    # Load environment variables
    load_dotenv()

    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Financial Recommendations System", page_icon="🤖")

    # Initialize session state variables
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Header of the app
    st.header("Financial Recommendations AI Agent 🤖")

    # Placeholder for initial greeting
    chat_placeholder = st.empty()
    
    # Show greeting message if the conversation chain has not been initialized
    if not st.session_state.conversation_chain:
        chat_placeholder.write("🤖: Please upload your document to continue!")
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents here", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Get text from PDF
                    raw_text = get_text_from_pdf(pdf_docs)
                    
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector embeddings
                    vector_store = get_vector_store(text_chunks)
                    st.success("Processing complete!")

                    # Create conversation chain
                    st.session_state.conversation_chain = get_conversation_chain(vector_store)

                    # Clear the initial greeting and prompt user to start conversation
                    chat_placeholder.empty()
                    st.write("🤖: Processing is complete! You can now start asking your questions.")
            else:
                st.warning("Please upload at least one document before processing.")

    # Chat input and handling
    user_query = st.text_input("Chat with the chatbot below:")
    if user_query:
        handle_user_query(user_query)

if __name__ == '__main__':
    main()
