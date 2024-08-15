import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain_openai import ChatOpenAI
import os
from langchain_huggingface import HuggingFaceEmbeddings
from services.news_verification import run_news_api
from services.inference import run_openai_api
import logging

logging.basicConfig(level=logging.INFO)

# Fix for OpenMP initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Load the fine-tuned model and tokenizer
# def initialize_llama_model():
#     model_path = "./finetuned-llama-7b"
#     model = LlamaForCausalLM.from_pretrained(model_path)
#     tokenizer = LlamaTokenizer.from_pretrained(model_path)
#     return model, tokenizer

base_prompt = """
You are a financial advisor chatbot. Your task is to help users with investment strategies. Depending on the user's inputs, you should suggest relevant strategies that align with their financial goals and risk tolerance. Your responses should be varied and adapted to the specific needs of the user.

You should:
- Ask relevant questions to understand the user's financial situation if not given yet.
- Provide detailed advice based on the user’s preferences, recent market trends, and best practices in investment.

Keep the conversation interactive and focused on helping the user make informed investment decisions. Maximum 2 points per response.
"""


def get_text_from_pdf(pdf_docs):
    """
    Extracts text from the uploaded PDF documents.

    Args:
        pdf_docs (list): The list of uploaded PDF documents.

    Returns:
        str: The raw text extracted from the PDF documents.
    """
    raw_text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def get_text_chunks(raw_text):
    """
    Splits the raw text into chunks suitable for processing.

    Args:
        raw_text (str): The raw text extracted from the PDF documents.

    Returns:
        list: The list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks


def get_vector_store(text_chunks):
    try:
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
        #                                model_kwargs={'device': 'auto'})
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(embedding=embeddings, texts=text_chunks)
        return vector_store
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        return None


def get_conversation_chain(vector_store):
    llm = ChatOpenAI(model='gpt-4o-mini',max_tokens=200)
    # llm =  HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B",model_kwargs={"max_length": 512, "temperature": 0.7})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vector_store.as_retriever(), memory = memory)
    return conversation_chain

def parse_chat_history(chat_history):
    """
    Parses the chat history into a single string suitable for use as a prompt.
    
    Args:
        chat_history (list): The list of chat messages with roles and content.
    
    Returns:
        str: The formatted conversation history as a string.
    """
    parsed_history = ""
    for message in chat_history:
        role = "User" if message['role'] == "user" else "System"
        parsed_history += f"{role}: {message['content']}\n"
    return parsed_history

def handle_user_query(user_query):
    """
    Handles the user's query, sends it to the conversation chain, and updates the chat history.
    
    Args:
        user_query (str): The user's query.
    """
    # Ensure necessary session state variables are initialized
    if 'conversation_chain' not in st.session_state or st.session_state.conversation_chain is None:
        st.write("🤖: Please upload your document to continue!")
        return

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    try:
        # Send the user query to the conversation chain and get the response
        response = st.session_state.conversation_chain.invoke({'question': user_query})
        
        # Safely extract the LLM's response, with a fallback message
        system_message = response.get('answer', "I'm sorry, I didn't understand that.")
        logging.info(f"System Message: {system_message}")

        # Check if the response indicates uncertainty
        if system_message.lower() in ["i don't know.", "i'm not sure.", "i didn't understand that."] or len(system_message) < 20:
            # Parse chat history into a formatted string
            parsed_history = parse_chat_history(st.session_state.chat_history)
            new_system_message = f"You are a financial advisor chatbot. Your task is to help users with investment strategies. Give maximum 2 points per response."
            new_prompt = f"Focus on current prompt: {user_query}\n The chat history is as follows just take this as context: {parsed_history}"
            # Use the base prompt if there's no chat history, otherwise continue the conversation
            if not parsed_history:
                system_message = run_openai_api(user_query, base_prompt)
            else:
                system_message = run_openai_api(new_prompt, new_system_message)

        # Update chat history with the user query and system response
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "system", "content": system_message})

        # Display the updated chat history
        display_chat_history()

        # Add an option for verification
        st.button("Verify the Response", on_click=verify_response, args=(system_message,))

    except KeyError as ke:
        logging.error(f"KeyError: Missing expected key - {ke}")
        st.error(f"A system error occurred: {ke}. Please try again.")
    except Exception as e:
        logging.exception("An unexpected error occurred")
        st.error(f"An error occurred while processing your query: {e}")


def get_verified_data(bot_message):
    """
    Get the verified data from the trusted sources.
    
    Returns:
        output : str, list
    """
    try:
        # Get asset word from llm_response.py
        asset_sector = run_openai_api(bot_message,"Find one investment sector from the given details. Just give the sector name.")
        print(f"Asset Sector: {asset_sector}")
        # Get news data from get_news.py
        news_df = run_news_api(asset_sector)
        print(f"News Data: {news_df}")
        # Update the bot_message with news data as context using llm_response.py
        prompt_content = f"Current investment strategy: {bot_message}, News: {news_df['title'].tolist()}"
        print(f"Prompt Content: {prompt_content}")
        output = run_openai_api(prompt_content,"Update the investment strategy based on the news articles.")
        print(f"Output: {output}")
        return output, news_df, asset_sector
    except Exception as e:
        st.error(f"An error occurred while verifying the response: {e}")


def verify_response(bot_message):
    """
    Verifies the response provided by the LLM and displays the unique sources and news titles.
    
    Args:
        bot_message (str): The message from the bot to verify.
    """
    try:
        with st.sidebar:
            st.subheader("Verification")

            # Assuming get_verified_data returns a tuple (output, news_df)
            output, news_df, asset_sector = get_verified_data(bot_message)

            # Extract unique source names
            source_dicts = news_df['source'].tolist()
            unique_sources = sorted({source['name'] for source in source_dicts})

            # Extract titles
            titles = news_df['title'].tolist()

            # Clear chat history
            st.session_state.chat_history = []
            st.empty()  # Clear the main area
            st.session_state.chat_history.append({"role": "user", "content": "Please verify the strategy."})

            st.session_state.chat_history.append({"role": "system", "content": "Current investment strategy: " + f"- {bot_message}"})
            #Display the asset sector
            st.write("🤖: **Asset Sector:**")
            st.write(f"- {asset_sector}")


            # Display the unique sources
            
            st.write("🤖: **Sources for the news articles:**")
            for source in unique_sources:
                st.write(f"- {source}")

            # Display the news titles
            st.write("🤖: **News for this sector:**")
            for title in titles:
                st.write(f"- {title}")

            # Save the output in chat history
            st.session_state.chat_history.append({"role": "system", "content": "Updated investment strategy: " + f"- {output}"})

    except Exception as e:
        st.error(f"An error occurred during verification: {e}")


def display_chat_history():
    """
    Displays the chat history in the Streamlit app.
    """
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <strong>👤: {message['content']}</strong>
                </div>
                <br>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="text-align: left;">
                    <strong>🤖: {message['content']}</strong>
                </div>
                <br>
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

    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = None
    else:
        pdf_docs = st.session_state.pdf_docs

    st.header("Financial Recommendations AI Agent 🤖")
    st.subheader("Asset Allocation and Investment Strategy")

    

    chat_placeholder = st.empty()
    
    if not st.session_state.conversation_chain:
        chat_placeholder.write("🤖: Please upload your document to continue!")
    
    with st.sidebar:
        # Button to start a new chat session
        if st.button("Start New Chat"):
            st.session_state.conversation_chain = None
            st.session_state.chat_history = []
            st.session_state.pdf_docs = None
            st.success("New chat started! Please upload a document to begin.")
        
        st.subheader("**Your Documents**")
        st.write(" ")
        st.session_state.pdf_docs = st.file_uploader("Upload your documents here", accept_multiple_files=True)
        st.write("OR")
        if st.button("Use Sample PDF"):
            pdf_file_location = "data/FinSightAI_Report.pdf"
            st.session_state.pdf_docs = [pdf_file_location]
            st.write("Sample PDF : FinSightAI_Report.pdf")

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
        
        if st.session_state.pdf_docs:
            st.write("RAG has been initiated")
        

    user_query = st.text_input("Chat with the chatbot below:")
    if user_query:
        print("User Query: ", user_query)
        print("Chat History: ", st.session_state.chat_history)
        if st.session_state.chat_history and st.session_state.chat_history[0]['content'] == "Please verify the strategy.":
            print("Verified response")
            display_chat_history()
        else:
            handle_user_query(user_query)

if __name__ == '__main__':
    main()