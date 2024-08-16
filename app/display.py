import streamlit as st

def display_chat_history():
    """
    Displays the chat history in the Streamlit app.

    - Iterates through the chat history stored in session state.
    - Displays user messages aligned to the right and system messages to the left.
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
