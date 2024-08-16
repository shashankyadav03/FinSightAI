import streamlit as st
from services.dotenv_loader import load_environment
from app.sidebar import setup_sidebar
from app.chat import handle_user_query
from app.display import display_chat_history

def main():
    """
    Main function to run the Streamlit app.
    
    - Sets up the environment variables.
    - Initializes session state variables.
    - Sets up the Streamlit page configuration.
    - Handles user interactions via the chat interface and sidebar.
    """
    load_environment()

    # Set up Streamlit page configurations
    st.set_page_config(page_title="Financial Recommendations System", page_icon="🤖")

    # Initialize session state variables if they do not exist
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Header and Subheader for the application
    st.header("Financial Recommendations AI Agent 🤖")
    st.subheader("Asset Allocation and Investment Strategy")

    # Placeholder for the chat interface
    chat_placeholder = st.empty()

    # If no conversation chain is available, prompt the user to upload a document
    if not st.session_state.conversation_chain:
        chat_placeholder.write("🤖: Please upload your document to continue!")

    # Setup the sidebar for document upload and processing
    setup_sidebar(chat_placeholder)

    # Handle user input through the text input box
    user_query = st.text_input("Chat with the chatbot below:")
    if user_query:
        if st.session_state.chat_history and st.session_state.chat_history[0]['content'] == "Please verify the strategy.":
            display_chat_history()
        else:
            handle_user_query(user_query)

if __name__ == '__main__':
    main()
