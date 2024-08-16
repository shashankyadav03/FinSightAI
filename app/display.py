import streamlit as st
from core.verification import verify_response

def display_chat_history():
    """
    Displays the chat history in the Streamlit app with the latest messages on top.

    - Iterates through the chat history stored in session state in reverse order.
    - Displays user messages aligned to the right and system messages to the left.
    - Adds a "Verify the Response" button directly after each system message.
    """
    for i, message in enumerate(reversed(st.session_state.chat_history)):
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
            if i == 0:
                st.button("Verify the Response", on_click=verify_response, args=(message['content'],))

    
