import streamlit as st
import logging
from core.conversation_chain import get_conversation_chain
from core.verification import verify_response
from app.display import display_chat_history

base_prompt = """
You are a financial advisor chatbot. Your task is to help users with investment strategies...
"""

def handle_user_query(user_query):
    """
    Handles the user query and generates a response using the conversation chain.

    - Checks if the conversation chain is initialized.
    - Sends the user query to the conversation chain and retrieves the response.
    - Handles cases where the LLM is uncertain or provides an insufficient response.
    - Updates and displays the chat history.
    
    Args:
        user_query (str): The user's input/query.
    """
    if 'conversation_chain' not in st.session_state or st.session_state.conversation_chain is None:
        st.write("🤖: Please upload your document to continue!")
        return

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    try:
        # Get response from the conversation chain
        response = st.session_state.conversation_chain.invoke({'question': user_query})
        system_message = response.get('answer', "I'm sorry, I didn't understand that.")
        logging.info(f"System Message: {system_message}")

        # Handle cases where the model's response is uncertain or insufficient
        if system_message.lower() in ["i don't know.", "i'm not sure.", "i didn't understand that."] or len(system_message) < 20:
            parsed_history = parse_chat_history(st.session_state.chat_history)
            new_prompt = f"Focus on current prompt: {user_query}\n The chat history is as follows just take this as context: {parsed_history}"
            if not parsed_history:
                system_message = run_openai_api(user_query, base_prompt)
            else:
                system_message = run_openai_api(new_prompt, "Give maximum 2 points per response.")

        # Update the chat history with user query and system response
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "system", "content": system_message})

        # Display the chat history
        display_chat_history()

        # Provide an option for the user to verify the response
        st.button("Verify the Response", on_click=verify_response, args=(system_message,))

    except KeyError as ke:
        logging.error(f"KeyError: Missing expected key - {ke}")
        st.error(f"A system error occurred: {ke}. Please try again.")
    except Exception as e:
        logging.exception("An unexpected error occurred")
        st.error(f"An error occurred while processing your query: {e}")
