from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

def get_conversation_chain(vector_store):
    """
    Initializes the conversation chain for handling user queries.

    - Sets up a conversational retrieval chain using the provided vector store.
    - Utilizes a memory buffer to maintain conversation context.

    Args:
        vector_store (FAISS): The vector store containing text embeddings.

    Returns:
        ConversationalRetrievalChain: An initialized conversation chain for querying.
    """
    try:
        llm = ChatOpenAI(model='gpt-4o-mini', max_tokens=200)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        raise ValueError(f"Error initializing the conversation chain: {e}")
