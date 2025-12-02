"""Pydantic models for configuration validation."""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class QueryExtractionConfig(BaseModel):
    """Configuration for query extraction from user messages."""
    max_tokens: int = Field(100, ge=1, le=1000, description="Maximum tokens for query extraction")
    prompt_template: str = Field(..., description="Prompt template for query extraction")


class RAGConfig(BaseModel):
    """Configuration for Retrieval-Augmented Generation."""
    enabled: bool = Field(True, description="Enable/disable RAG feature")
    n_results: int = Field(3, ge=1, le=20, description="Number of documents to retrieve")
    relevance_threshold: float = Field(1.0, ge=0.0, le=2.0, description="Distance threshold for filtering")
    collection_name: str = Field("documents", description="Default collection name")
    query_extraction: QueryExtractionConfig


class LLMTimeouts(BaseModel):
    """Timeout configuration for LLM operations."""
    streaming: float = Field(120.0, gt=0, description="Timeout for streaming requests")
    completion: float = Field(30.0, gt=0, description="Timeout for non-streaming completions")
    connection: float = Field(10.0, gt=0, description="Connection timeout")
    read: float = Field(60.0, gt=0, description="Read timeout")
    write: float = Field(60.0, gt=0, description="Write timeout")


class LLMCompletion(BaseModel):
    """LLM completion parameters."""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(50, ge=0, description="Top-k sampling parameter")


class LLMConfig(BaseModel):
    """Configuration for LLM service."""
    default_model: str = Field("llama3.2:1b", description="Default model to use")
    enforce_default_model: bool = Field(True, description="Force use of default model")
    default_embedding_model: Optional[str] = Field(None, description="Default embedding model")
    timeouts: LLMTimeouts
    completion: LLMCompletion


class AcademicSearchConfig(BaseModel):
    """Configuration for academic search (arXiv)."""
    enabled: bool = Field(False, description="Enable/disable academic search")
    max_results: int = Field(5, ge=1, le=50, description="Maximum papers to retrieve")
    search_fields: List[str] = Field(
        ["title", "abstract", "category"],
        description="Fields to search in"
    )
    sort_by: Literal["relevance", "date"] = Field("relevance", description="Sort order")


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    max_chars: int = Field(1500, ge=100, le=10000, description="Maximum characters per chunk")
    overlap: int = Field(0, ge=0, le=500, description="Overlap between chunks")


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing."""
    chunking: ChunkingConfig
    supported_formats: List[str] = Field([".pdf", ".txt"], description="Supported file formats")
    batch_size: int = Field(50, ge=1, le=500, description="Batch size for ingestion")


class VectorDBQueryConfig(BaseModel):
    """Configuration for vector database queries."""
    default_n_results: int = Field(5, ge=1, le=100, description="Default number of results")
    max_n_results: int = Field(100, description="Maximum allowed results")
    min_n_results: int = Field(1, description="Minimum allowed results")


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    default_collection: str = Field("documents", description="Default collection name")
    query: VectorDBQueryConfig
    timeout: int = Field(60, gt=0, description="HTTP timeout for DB operations")


class PromptsConfig(BaseModel):
    """Configuration for prompts and templates."""
    system_message: str = Field(..., description="Default system message for LLM")
    context_template: str = Field(..., description="Template for formatting RAG context")
    academic_context_header: str = Field(..., description="Header for academic search results")


class ServicesConfig(BaseModel):
    """Configuration for service URLs."""
    llm_url: str = Field(..., description="LLM service URL")
    vectordb_url: str = Field(..., description="Vector database URL")
    ollama_url: str = Field(..., description="Ollama service URL")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Log level")
    format: str = Field(..., description="Log format string")


class AppConfig(BaseModel):
    """Complete application configuration."""
    rag: RAGConfig
    llm: LLMConfig
    academic_search: AcademicSearchConfig
    document_processing: DocumentProcessingConfig
    vectordb: VectorDBConfig
    prompts: PromptsConfig
    services: ServicesConfig
    logging: LoggingConfig

    class Config:
        """Pydantic configuration."""
        validate_assignment = True  # Validate on attribute assignment
        extra = "forbid"  # Forbid extra fields
