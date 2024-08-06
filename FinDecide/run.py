import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain_openai import ChatOpenAI
import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# Fix for OpenMP initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Load the fine-tuned model and tokenizer
# def initialize_llama_model():
#     model_path = "./finetuned-llama-7b"
#     model = LlamaForCausalLM.from_pretrained(model_path)
#     tokenizer = LlamaTokenizer.from_pretrained(model_path)
#     return model, tokenizer

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
    # Create a cache directory for the same text chunks
    if not os.path.exists('utilities/vectorstores'):
        os.makedirs('utilities/vectorstores')

    # Check if the vector store already exists
    vector_store_path = 'utilities/vectorstores/db_faiss'
    if os.path.exists(vector_store_path):
        return FAISS.load_local(vector_store_path)
    

    # Initialize the OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(embedding=embeddings, texts=text_chunks)
    vector_store.save_local(vector_store_path)
    return vector_store

custom_prompt_template = """
You're tasked with providing a helpful response based on the given context and question.
Accuracy is paramount, so if you're uncertain, it's best to acknowledge that rather than providing potentially incorrect information.

Context: {context}
Question: {question}


Please craft a clear and informative response that directly addresses the question.
Aim for accuracy and relevance, keeping the user's needs in mind.
Response:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    qa_chain = create_retrieval_chain(
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           combine_docs_chain = combine_docs_chain
                                           )
    return qa_chain

def get_conversation_chain(vector_store):
    # Initialize the LLM (e.g., GPT-4)
    db = vector_store
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def handle_user_query(user_query):
    if 'conversation_chain' not in st.session_state or st.session_state.conversation_chain is None:
        st.write("🤖: Please upload your document to continue!")
        return
    print(user_query)
    print(type(user_query))
    # KeyError: "Input to PromptTemplate is missing variables {'question'}.  Expected: ['context', 'question'] Received: ['input', 'context']"
    input = {
        'context': "",
        'question': user_query
    }
    
    # Pass 'question' instead of 'input' to match the expected variable name
    response = st.session_state.conversation_chain.invoke({'input': input})
    print(response)
    system_message = response['result']
    print(system_message)

    # Update the chat history
    st.session_state.chat_history.append({"role": "system", "content": system_message})

    # Display the updated conversation
    display_chat_history()

    # Add the option for verification
    st.button("Verify the Response", on_click=verify_response, args=(system_message,))


def verify_response(bot_message):
    if st.session_state.conversation_chain is None:
        st.write("🤖: The conversation chain is not initialized!")
        return
    
    # Prepare input for LLM and RAG
    verification_input = {
        'question': f"Verify the following strategy: {bot_message}",
        'chat_history': st.session_state.chat_history  # Pass the chat history for context
    }

    # Invoke the conversation chain to perform the verification
    response = st.session_state.conversation_chain.invoke(verification_input)
    
    # Extract the LLM's response and update chat history
    verification_message = response['answer']  # Assuming 'answer' key holds the LLM's response
    st.session_state.chat_history.append({"role": "system", "content": verification_message})

    # Display the updated conversation with the verification result
    display_chat_history()

    # Optionally, you could also ask for user feedback here
    st.radio("Is the verification result helpful?", ["Yes", "No"])


def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <strong>👤: {message['content']}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="text-align: left;">
                    <strong>🤖: {message['content']}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

def main():
    load_dotenv()

    st.set_page_config(page_title="Financial Recommendations System", page_icon="🤖")

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.header("Financial Recommendations AI Agent 🤖")

    chat_placeholder = st.empty()
    
    if not st.session_state.conversation_chain:
        chat_placeholder.write("🤖: Please upload your document to continue!")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents here", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_text_from_pdf(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.success("Processing complete!")
                    st.session_state.conversation_chain = get_conversation_chain(vector_store)
                    chat_placeholder.empty()
                    st.write("🤖: Processing is complete! You can now start asking your questions.")
            else:
                st.warning("Please upload at least one document before processing.")

    user_query = st.text_input("Chat with the chatbot below:")
    if user_query:
        handle_user_query(user_query)

if __name__ == '__main__':
    main()