import streamlit as st
from services.news_verification import run_news_api
from services.inference import run_inference

def get_verified_data(bot_message):
    """
    Verifies the LLM-generated response by comparing it against real-time news.

    - Extracts an investment sector from the bot message.
    - Retrieves news articles related to the extracted sector.
    - Updates the investment strategy based on the news articles.

    Args:
        bot_message (str): The investment strategy provided by the LLM.

    Returns:
        tuple: Updated investment strategy, DataFrame of news articles, and the extracted sector.
    """
    try:
        asset_sector = run_inference(bot_message, "Find one investment sector from the given details. Just give the sector name.")
        news_df = run_news_api(asset_sector)
        prompt_content = f"Current investment strategy: {bot_message}, News: {news_df['title'].tolist()}"
        output = run_inference(prompt_content, "Update the investment strategy based on the news articles. Give only 3 points.")
        return output, news_df, asset_sector
    except Exception as e:
        raise ValueError(f"An error occurred while verifying the response: {e}")

def verify_response(bot_message):
    """
    Verifies and displays the response in the Streamlit sidebar.

    - Retrieves and displays unique sources and news titles related to the investment strategy.
    - Updates the chat history with the verified strategy.

    Args:
        bot_message (str): The message from the bot to verify.
    """
    try:
        with st.sidebar:
            st.subheader("Verification")

            output, news_df, asset_sector = get_verified_data(bot_message)

            source_dicts = news_df['source'].tolist()
            unique_sources = sorted({source['name'] for source in source_dicts})
            titles = news_df['title'].tolist()

            st.session_state.chat_history = []
            st.empty()
            st.session_state.chat_history.append({"role": "user", "content": "Please verify the strategy."})

            st.session_state.chat_history.append({"role": "system", "content": "Current investment strategy: " + f"- {bot_message}"})
            st.write("🤖: **Asset Sector:**")
            st.write(f"- {asset_sector}")

            st.write("🤖: **Sources for the news articles:**")
            for source in unique_sources:
                st.write(f"- {source}")

            st.write("🤖: **News for this sector:**")
            for title in titles:
                st.write(f"- {title}")

            st.session_state.chat_history.append({"role": "system", "content": "Updated investment strategy: " + f"- {output}"})

            st.markdown("🤖: Please start new chat")

    except Exception as e:
        st.error(f"An error occurred during verification: {e}")
