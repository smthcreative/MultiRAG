from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PathConfig:
    """File and directory paths."""
    # Data paths
    RAW_DATA_DIR: Path = Path("../data/raw")
    PROCESSED_DATA_DIR: Path = Path("../data/processed")

    # Input files
    RAW_ARTICLES: Path = RAW_DATA_DIR / "articles.json"
    IMAGE_DESCRIPTIONS: Path = PROCESSED_DATA_DIR / "image_descriptions.json"
    MERGED_ARTICLES: Path = PROCESSED_DATA_DIR / "articles_with_image_descriptions.json"

    # Vectorstore
    VECTORSTORE_DIR: Path = Path("vectorstore")

    # Evaluation
    # EVAL_DIR: Path = Path("../evaluation")
    # EVAL_RESULTS: Path = EVAL_DIR / "results.json"

    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR,
                     self.VECTORSTORE_DIR]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ScrapingConfig:
    """Web scraping configuration."""
    BASE_URL: str = "https://www.deeplearning.ai/the-batch/"
    NUM_PAGES: int = 3
    REQUEST_DELAY: float = 1.5
    HEADERS: dict = None

    def __post_init__(self):
        if self.HEADERS is None:
            self.HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass
class ImageDescriptionConfig:
    """Image description generation configuration."""
    # MODEL_NAME: str = "gemini-2.0-flash"
    MODEL_NAME: str = "gemini-2.0-flash-lite"
    FREE_TIER_RPM: int = 10
    REQUEST_DELAY: float = None

    def __post_init__(self):
        if self.REQUEST_DELAY is None:
            self.REQUEST_DELAY = (60 / self.FREE_TIER_RPM) + 0.5


@dataclass
class EmbeddingConfig:
    """Embedding model"""
    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L12-v2"



@dataclass
class ChunkingConfig:
    """Text chunking configuration"""
    # Text chunking
    TEXT_CHUNK_SIZE: int = 512
    TEXT_CHUNK_OVERLAP: int = 128

    # Image description chunking
    CHUNK_IMAGE_DESCRIPTIONS: bool = True
    IMAGE_CHUNK_SIZE: int = 128
    IMAGE_CHUNK_OVERLAP: int = 24


@dataclass
class RAGConfig:
    """RAG system configuration."""
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    # Retrieval parameters
    TOP_K_ARTICLES: int = 5
    TOP_K_IMAGES: int = 1
    RELEVANCE_THRESHOLD: float = 0.4


@dataclass
class StreamlitConfig:
    """Streamlit UI configuration."""
    PAGE_TITLE: str = "MultiRAG - AI Article Assistant"
    PAGE_ICON: str = "🤖"
    LAYOUT: str = "wide"
    MAX_CHAT_HISTORY: int = 50


class Config:
    """Main configuration class aggregating all configs"""

    def __init__(self):
        self.paths = PathConfig()
        self.scraping = ScrapingConfig()
        self.image_description = ImageDescriptionConfig()
        self.embedding = EmbeddingConfig()
        self.chunking = ChunkingConfig()
        self.rag = RAGConfig()
        self.streamlit = StreamlitConfig()


        self.google_api_key = os.getenv("GOOGLE_API_KEY")

    def validate(self) -> bool:
        """Validate API key"""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return True


config = Config()