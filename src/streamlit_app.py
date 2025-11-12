import streamlit as st

from MultRAG import MultiModalRAG, RAGResponse
from config import config

# Page configuration
st.set_page_config(
    page_title=config.streamlit.PAGE_TITLE,
    page_icon=config.streamlit.PAGE_ICON,
    layout=config.streamlit.LAYOUT,
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_rag_system():
    """Load RAG system with caching"""
    try:
        return MultiModalRAG()
    except FileNotFoundError as e:
        st.error(f"{str(e)}")
        st.info("Please run `python indexing.py` first to create the vectorstore.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        st.stop()


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_system" not in st.session_state:
        with st.spinner("Loading RAG system..."):
            st.session_state.rag_system = load_rag_system()


def display_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content)


def display_sources(response: RAGResponse):
    """Display sources articles and images"""

    # Display articles
    if response.articles:
        with st.expander(f"📚 Related Articles ({len(response.articles)})", expanded=False):
            for i, article in enumerate(response.articles, 1):
                st.markdown(f"**{i}. [{article['title']}]({article['url']})**")
                if article.get('snippet'):
                    st.caption(article['snippet'])
                st.divider()

    # Display images
    if response.images:
        with st.expander(f"🖼️ Related Images ({len(response.images)})", expanded=True):
            # Use columns for better image layout
            cols = st.columns(min(len(response.images), 2))

            for i, img in enumerate(response.images):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    st.image(
                        img['image_url'],
                        caption=f"From: {img['article_title']}",
                        use_container_width=True
                    )
                    st.caption(f"{img['description'][:150]}...")
                    st.markdown(f"[View Article]({img['article_url']})")
                    st.divider()

    if not response.articles and not response.images:
        st.info("📭 No additional sources to display.")


def display_chat_history():
    """Display all messages in chat history"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            display_message("user", message["content"])
        else:
            display_message("assistant", message["content"])
            if "response" in message:
                display_sources(message["response"])


def handle_query(query: str):
    """Process user query and display response"""
    st.session_state.messages.append({"role": "user", "content": query})
    display_message("user", query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            response = st.session_state.rag_system.query(query)

        st.markdown(response.answer)

        display_sources(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "response": response
    })

    if len(st.session_state.messages) > config.streamlit.MAX_CHAT_HISTORY:
        st.session_state.messages = st.session_state.messages[-config.streamlit.MAX_CHAT_HISTORY:]


def render_sidebar():
    """Render sidebar with information and controls."""
    with st.sidebar:
        st.title("🤖 MultiRAG Assistant")
        st.markdown("---")

        # System information
        st.subheader("ℹ️ System Info")
        st.info(f"""
        **Model:** {config.rag.LLM_MODEL}  
        **Embedding:** {config.embedding.MODEL_NAME.split('/')[-1]}  
        **Articles Retrieved:** {config.rag.TOP_K_ARTICLES}  
        **Images Retrieved:** {config.rag.TOP_K_IMAGES}
        """)

        # Example queries
        st.markdown("---")
        st.subheader("💡 Example Queries")

        example_queries = [
            "What are reasoning models?",
            "How does differential privacy work?",
            "What are the latest AI developments?",
            "Show me performance comparisons of AI models",
            "What is MiniMax-M2?"
        ]

        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.example_query = query

        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Statistics
        st.markdown("---")
        st.subheader("Statistics")
        st.metric("Messages", len(st.session_state.messages)/2)

        # About section
        st.markdown("---")
        st.caption("""
        **About MultiRAG**  
        Multi-modal RAG system for querying DeepLearning.AI articles.
        Retrieves relevant text and images to answer your questions.

        **Note:** Sources are only shown when the model can confidently answer based on the knowledge base.
        """)


def main():
    initialize_session_state()

    render_sidebar()

    st.title("🤖 MultiRAG - AI Article Assistant")
    st.markdown("Ask questions about AI topics from DeepLearning.AI articles")
    st.markdown("---")

    display_chat_history()

    if hasattr(st.session_state, 'example_query'):
        query = st.session_state.example_query
        delattr(st.session_state, 'example_query')
        handle_query(query)
        st.rerun()

    if prompt := st.chat_input("Ask a question about AI..."):
        handle_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()