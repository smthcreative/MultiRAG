import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config

# conf
INPUT_JSON = config.paths.MERGED_ARTICLES
# INDEX_DIR = "D:\\Work\\MultiRAG\\vectorstore"
INDEX_DIR = config.paths.VECTORSTORE_DIR

EMBED_MODEL = config.embedding.MODEL_NAME

CHUNK_SIZE = config.chunking.TEXT_CHUNK_SIZE
CHUNK_OVERLAP = config.chunking.TEXT_CHUNK_OVERLAP


CHUNK_IMAGE_DESCRIPTIONS = config.chunking.CHUNK_IMAGE_DESCRIPTIONS
IMAGE_CHUNK_SIZE = config.chunking.IMAGE_CHUNK_SIZE
IMAGE_CHUNK_OVERLAP = config.chunking.IMAGE_CHUNK_OVERLAP


def load_articles(path: Path) -> List[Dict]:
    """Load articles from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_text_documents(article: Dict, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Create document chunks from article text
    """
    docs = []
    title = article.get("title", "Untitled")
    url = article.get("url", "")
    text = article.get("text", "")

    if not text or not text.strip():
        return docs

    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "type": "article_text",
                    "article_title": title,
                    "article_url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": "text"
                }
            )
        )

    return docs


def create_image_documents(article: Dict, image_splitter: RecursiveCharacterTextSplitter = None) -> List[Document]:
    """
    Create documents from image descriptions.
    Each image description becomes a searchable document
    """
    docs = []
    title = article.get("title", "Untitled")
    url = article.get("url", "")
    images = article.get("images", [])

    for idx, img in enumerate(images):
        description = img.get("description", "")
        image_url = img.get("image_url", "")

        if not description or description.strip() in ["", "<---image--->"]:
            continue

        if CHUNK_IMAGE_DESCRIPTIONS and image_splitter and len(description) > IMAGE_CHUNK_SIZE * 1.5:
            desc_chunks = image_splitter.split_text(description)

            for chunk_idx, desc_chunk in enumerate(desc_chunks):
                docs.append(
                    Document(
                        page_content=desc_chunk,
                        metadata={
                            "type": "image_description",
                            "article_title": title,
                            "article_url": url,
                            "image_url": image_url,
                            "image_index": idx,
                            "total_images": len(images),
                            "chunk_index": chunk_idx,
                            "total_chunks": len(desc_chunks),
                            "content_type": "image"
                        }
                    )
                )
        else:
            docs.append(
                Document(
                    page_content=description,
                    metadata={
                        "type": "image_description",
                        "article_title": title,
                        "article_url": url,
                        "image_url": image_url,
                        "image_index": idx,
                        "total_images": len(images),
                        "content_type": "image"
                    }
                )
            )

    return docs


def prepare_documents(articles: List[Dict]) -> List[Document]:
    """
    Convert articles with embedded images into LangChain Documents.
    Creates separate documents for text chunks and image descriptions.
    """
    # text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # separators=["\n\n", "\n", ". ", " ", ""]
    )

    # in case if chunking is True
    image_splitter = None
    if CHUNK_IMAGE_DESCRIPTIONS:
        image_splitter = RecursiveCharacterTextSplitter(
            chunk_size=IMAGE_CHUNK_SIZE,
            chunk_overlap=IMAGE_CHUNK_OVERLAP,
            # separators=[". ", ", ", " ", ""]
        )

    all_docs = []

    print("\n Processing articles...")
    for article in tqdm(articles, desc="Creating documents"):
        text_docs = create_text_documents(article, text_splitter)
        all_docs.extend(text_docs)

        image_docs = create_image_documents(article, image_splitter)
        all_docs.extend(image_docs)

    return all_docs


def build_and_save_vectorstore(docs: List[Document]):
    """
    Create embeddings and save FAISS index
    """
    print(f"\n Creating embeddings with model: {EMBED_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(" Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embedding_model)

    print(" Saving index to disk...")
    vectorstore.save_local(str(INDEX_DIR))
    print(f" Saved vectorstore to {INDEX_DIR}")

    return vectorstore


# def print_statistics(docs: List[Document]):
#     """Print useful statistics about the indexed documents"""
#     text_docs = [d for d in docs if d.metadata.get("type") == "article_text"]
#     image_docs = [d for d in docs if d.metadata.get("type") == "image_description"]
#
#     unique_articles = len(set(d.metadata.get("article_url") for d in docs))
#
#     print("\n" + "=" * 60)
#     print("📊 INDEXING STATISTICS")
#     print("=" * 60)
#     print(f"Total documents indexed: {len(docs)}")
#     print(f"  ├─ Text chunks: {len(text_docs)}")
#     print(f"  └─ Image descriptions: {len(image_docs)}")
#     print(f"Unique articles: {unique_articles}")
#     print(f"Average text chunks per article: {len(text_docs) / unique_articles:.1f}")
#     print(f"Average images per article: {len(image_docs) / unique_articles:.1f}")
#     print("=" * 60 + "\n")


def main(file=INPUT_JSON):
    print("Starting Multi-Modal Indexing Pipeline\n")

    if not file.exists():
        print(f"ERROR: Input file not found: {file}")
        exit(1)

    # Load articles
    print(f"Loading articles from: {file}")
    articles = load_articles(file)
    print(f"Loaded {len(articles)} articles")

    documents = prepare_documents(articles)

    if not documents:
        print("No documents to index! Check your input data.")
        exit(1)

    # Print statistics
    # print_statistics(documents)

    # Build and save vectorstore
    build_and_save_vectorstore(documents)

    print("\nIndexing complete! Your vectorstore is ready for RAG queries.")


# ==================== MAIN ====================
if __name__ == "__main__":
    main()