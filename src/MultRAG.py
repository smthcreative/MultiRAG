from typing import Dict, List, Optional
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from config import config


@dataclass
class RAGResponse:
    """Structured response from RAG system."""
    answer: str
    articles: List[Dict]
    images: List[Dict]
    total_sources: int
    can_answer: bool  # whether there is retrieved data


class MultiModalRAG:
    """
    Multi-modal RAG system for querying articles and images
    """

    def __init__(self):
        """Initialize RAG components"""
        self._validate_config()
        # print("Configuration validated")
        self.embedding_model = self._load_embeddings()
        # print("Embedding model loaded")
        self.vectorstore = self._load_vectorstore()
        # print("Vectorstore loaded")
        self.llm = self._initialize_llm()
        # print("LLM initialized")

    def _validate_config(self):
        """Validate required configuration"""
        if not config.paths.VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"Vectorstore not found at {config.paths.VECTORSTORE_DIR}. "
                "Please run indexing.py first."
            )
        config.validate()

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load embedding model."""
        print(f"Loading embedding model: {config.embedding.MODEL_NAME}")
        return HuggingFaceEmbeddings(
            model_name=config.embedding.MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _load_vectorstore(self) -> FAISS:
        """Load FAISS vectorstore."""
        print(f"Loading vectorstore from: {config.paths.VECTORSTORE_DIR}")
        return FAISS.load_local(
            str(config.paths.VECTORSTORE_DIR),
            # "D:\\Work\\MultiRAG\\vectorstore",
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize LLM"""
        return ChatGoogleGenerativeAI(
            model=config.rag.LLM_MODEL,
            temperature=config.rag.LLM_TEMPERATURE,
            max_output_tokens=config.rag.LLM_MAX_TOKENS
        )

    def _create_enhanced_prompt(self, question: str, articles: List[Dict], images: List[Dict]) -> str:
        """Create enhanced prompt with article and image context."""
        template =""" 
**Role and Goal:**
You are an expert AI assistant answering questions about technical AI topics from DeepLearning.AI articles. 
If the context does not contain enough information to answer the question, clearly state that you cannot answer more detailed based on the given information.

**Constraints:**
- Do NOT include any links, URLs, markdown links, or direct references to article titles in your main answer. These will be handled separately.
- Do NOT make up information. Only use the provided context.
- Only answer questions related to the articles or provided context.
- Do NOT repeat the user's question unless necessary for clarity.
- Stay neutral and factual. Do not express opinions or speculate beyond the content.

**Context from Articles:**
{articles_context}

**Retrieved Image Context:**
{images_context}

**Rules for Image References:**
- You may only acknowledge the image if the article context directly refers to or explains its content.
- If the article context is empty or does NOT cover the topic of the user’s question, do NOT mention the image.
- Do NOT attempt to interpret or describe the image if the context does not support it.

**User Question:**
{question}

**Task:**
Provide a detailed, accurate answer using only the information above.
If the articles do not provide enough information, explicitly state that you cannot answer based on the given context.

"""

        # Format articles context
        articles_context = ""
        if articles:
            for i, article in enumerate(articles, 1):
                articles_context += f"\n[Article {i}] Title: {article['title']}\n"
                articles_context += f"URL: {article['url']}\n"
                articles_context += f"Content: {article['snippet']}\n"
        else:
            articles_context = "No relevant articles found."

        # Format images context
        images_context = ""
        if images:
            for i, img in enumerate(images, 1):
                images_context += f"\n[Image {i}] From article: {img['article_title']}\n"
                images_context += f"Description: {img['description']}\n"
        else:
            images_context = "No relevant images found."

        return template.format(
            articles_context=articles_context,
            images_context=images_context,
            question=question
        )

    def _retrieve_articles(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant article documents"""
        if k is None:
            k = config.rag.TOP_K_ARTICLES

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter={
                    "type": "article_text"}
        )


        threshold = config.rag.RELEVANCE_THRESHOLD
        return [doc for doc, score in results if score >= 0.3]

    def _retrieve_images(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant images for query"""
        if k is None:
            k = config.rag.TOP_K_IMAGES

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter={"type": "image_description"}
        )

        return [doc for doc, score in results if score >= 0.3]


    # def _retrieve_most_similar_img_from_article(self, query: str, articles) -> List[Document]:
    #     """Retrieve relevant images for query."""
    #     # if k is None:
    #     #     k = config.rag.TOP_K_IMAGES
    #     for article in articles:
    #         # print(article['url'])
    #         results = self.vectorstore.similarity_search_with_score(
    #             "",
    #             k=10,
    #             filter={"type": "image_description",
    #                     "article_title": article['title']
    #                     }
    #         )

        # return [doc for doc, score in results]

    def _deduplicate_articles(self, documents: List[Document]) -> List[Dict]:
        """Deduplicate and organize article sources."""
        articles_map = {}

        for doc in documents:
            metadata = doc.metadata
            if metadata.get("type") != "article_text":
                continue

            url = metadata.get("article_url")
            if url and url not in articles_map:
                articles_map[url] = {
                    "title": metadata.get("article_title"),
                    "url": url,
                    "snippet": doc.page_content[:500] + "..."
                }

        return list(articles_map.values())

    def _format_images(self, image_docs: List[Document]) -> List[Dict]:
        """Format image documents for response"""
        images = []
        seen = set()

        for doc in image_docs:
            meta = doc.metadata
            image_url = meta.get("image_url")

            if image_url in seen:
                continue
            seen.add(image_url)

            images.append({
                "article_title": meta.get("article_title"),
                "article_url": meta.get("article_url"),
                "image_url": image_url,
                "description": doc.page_content
            })

        return images

    def query(self, question: str) -> RAGResponse:
        """
        Execute RAG query and return structured response.

        Args:
            question: User's question

        Returns:
            RAGResponse with answer, articles, and images
        """
        can_answer = True
        # Retrieve articles and images
        article_docs = self._retrieve_articles(question)
        image_docs = self._retrieve_images(question)

        # Process and deduplicate sources
        articles = self._deduplicate_articles(article_docs)
        images = self._format_images(image_docs)

        if len(article_docs) + len(image_docs) == 0:
            article_docs = []
            image_docs = []
            can_answer = False

        # elif len(image_docs) == 0:
        #     image_docs = self._retrieve_most_similar_img_from_article(question, articles)


        enhanced_prompt = self._create_enhanced_prompt(question, articles, images)

        response = self.llm.invoke(enhanced_prompt)
        answer = response.content.strip()

        return RAGResponse(
            answer=answer,
            articles=articles,
            images=images,
            total_sources=len(article_docs) + len(image_docs),
            can_answer=can_answer
        )




def main():
    print("Initializing MultiModal RAG System...\n")
    rag = MultiModalRAG()
    print("System ready!\n")


    queries = [
        'Which AI tools were used by participants at the Buildathon?'
    ]

    for i, query in enumerate(queries, 1):
        print(f"QUERY {i}/{len(queries)}: {query}")

        response = rag.query(query)
        print(response)


if __name__ == "__main__":
    main()