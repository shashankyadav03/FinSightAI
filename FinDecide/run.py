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
    if st.session_state.conversation_chain is None:
        st.write("🤖: Please upload your document to continue!")
        return

    chat_history = st.session_state.chat_history

    if not chat_history:
        # Initial question by the user
        if "diversify" in user_query.lower():
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            system_message = "Absolutely! To start, could you tell me more about your risk tolerance and investment horizon? For example, are you comfortable with higher risks for potentially higher returns, or would you prefer a more conservative approach?"
            st.session_state.chat_history.append({"role": "system", "content": system_message})
    else:
        last_message = chat_history[-1]["content"]
        if "risk tolerance" in last_message:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            system_message = "Great! Based on your moderate risk tolerance and a 10-year investment horizon, a balanced asset allocation strategy could be a good fit. Typically, this might involve:\n- 40% in bonds to provide stability and income.\n- 40% in equities, including both blue-chip stocks for steady growth and some growth stocks for higher returns.\n- 20% in alternative investments like real estate or commodities to diversify risk.\n\nWould you like to see a more detailed breakdown, or should I go ahead and verify this strategy against recent market conditions?"
            st.session_state.chat_history.append({"role": "system", "content": system_message})
        elif "verify this strategy" in last_message:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.spinner("Verifying the strategy..."):
                verify_response()
        elif "revised allocation" in last_message:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            system_message = "Sure! Here’s the summary of your tailored asset allocation strategy:\n- 35% Bonds: Primarily in government and corporate bonds to maintain stability.\n- 45% Equities: Split between blue-chip stocks in tech and healthcare sectors for growth, with a small allocation to growth stocks for potential higher returns.\n- 20% Alternative Investments: Including real estate and commodities like gold to diversify and protect against market volatility.\n\nThis strategy is designed to balance growth with moderate risk, taking into account the latest market conditions."
            st.session_state.chat_history.append({"role": "system", "content": system_message})

    display_chat_history()

def verify_response():
    # Simulate verification logic
    bot_message = "Here’s what I found:\n- Bond Market: Recent news indicates that rising interest rates might affect bond yields negatively. You might want to consider reducing the bond allocation slightly to 35% and shifting the extra 5% to equities or other stable income-generating investments.\n- Equity Market: There’s positive sentiment around blue-chip stocks, particularly in the tech and healthcare sectors. Growth stocks, however, are facing some volatility due to economic uncertainties.\n- Alternative Investments: Commodities like gold are seeing increased demand, which could be a safe bet as part of your alternative investment allocation.\n\nBased on this, I’d recommend a revised allocation:\n- 35% in bonds\n- 45% in equities, with a focus on blue-chip tech and healthcare stocks\n- 20% in alternative investments, including some exposure to commodities like gold."
    
    st.session_state.chat_history.append({"role": "system", "content": bot_message})
    display_chat_history()

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