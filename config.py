"""
config.py - Central configuration for the Financial Intelligence Agent.
All settings are loaded from environment variables for production safety.
"""

import os
from dataclasses import dataclass, field


@dataclass
class GeminiConfig:
    """Google Gemini — used for LLM (chat) only. Embeddings are local."""
    api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    chat_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    )


@dataclass
class TavilyConfig:
    """Tavily web search API — get a free key at tavily.com."""
    api_key: str = field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))
    max_results: int = 5


@dataclass
class VectorStoreConfig:
    """ChromaDB settings."""
    persist_directory: str = field(
        default_factory=lambda: os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    collection_name: str = "apple_10k_sections"


@dataclass
class ChunkingConfig:
    """Controls how section text is split into chunks."""
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_length: int = 100


@dataclass
class AgentConfig:
    """LangGraph agent behaviour."""
    max_retries: int = 2
    max_web_results: int = 5
    relevance_threshold: float = 0.7
    top_k_retrieval: int = 6


@dataclass
class PathConfig:
    """Filesystem paths — must match the upstream ingestion pipeline."""
    sections_dir: str = field(
        default_factory=lambda: os.environ.get("SECTIONS_DIR", "./sections")
    )
    processed_dir: str = field(
        default_factory=lambda: os.environ.get("PROCESSED_DIR", "./processed")
    )


# ---------------------------------------------------------------------------
# Singleton instances — import these everywhere
# ---------------------------------------------------------------------------

gemini_cfg = GeminiConfig()
tavily_cfg = TavilyConfig()
vector_cfg = VectorStoreConfig()
chunk_cfg  = ChunkingConfig()
agent_cfg  = AgentConfig()
path_cfg   = PathConfig()


def validate_config() -> list[str]:
    """Return a list of missing required environment variables."""
    missing = []
    if not gemini_cfg.api_key:
        missing.append("GEMINI_API_KEY")
    if not tavily_cfg.api_key:
        missing.append("TAVILY_API_KEY")
    return missing