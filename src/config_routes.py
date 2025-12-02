"""Configuration API endpoints for runtime configuration management."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from .auth import verify_frontend_api_key
from .config.config_manager import get_config, get_config_manager

router = APIRouter(prefix="/v1/config", tags=["configuration"])


# Request models for updates
class RAGUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    n_results: Optional[int] = Field(None, ge=1, le=20)
    relevance_threshold: Optional[float] = Field(None, ge=0.0, le=2.0)
    collection_name: Optional[str] = None


class QueryExtractionUpdateRequest(BaseModel):
    max_tokens: Optional[int] = Field(None, ge=1, le=1000)


class AcademicSearchUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    max_results: Optional[int] = Field(None, ge=1, le=50)


class ChunkingUpdateRequest(BaseModel):
    max_chars: Optional[int] = Field(None, ge=100, le=10000)
    overlap: Optional[int] = Field(None, ge=0, le=500)


class DocumentProcessingUpdateRequest(BaseModel):
    batch_size: Optional[int] = Field(None, ge=1, le=500)


class PromptsUpdateRequest(BaseModel):
    system_message: Optional[str] = Field(None, min_length=1, max_length=5000)


# Endpoints
@router.get("")
async def get_full_config(_: str = Depends(verify_frontend_api_key)):
    """Get complete configuration."""
    config = get_config()
    return {
        "rag": config.rag.dict(),
        "llm": config.llm.dict(),
        "academic_search": config.academic_search.dict(),
        "document_processing": config.document_processing.dict(),
        "vectordb": config.vectordb.dict(),
        "prompts": config.prompts.dict(),
        "services": config.services.dict(),
        "logging": config.logging.dict()
    }


@router.get("/rag")
async def get_rag_config(_: str = Depends(verify_frontend_api_key)):
    """Get RAG configuration."""
    config = get_config()
    return config.rag.dict()


@router.patch("/rag")
async def update_rag_config(updates: RAGUpdateRequest, _: str = Depends(verify_frontend_api_key)):
    """Update RAG configuration."""
    config = get_config()
    
    if updates.enabled is not None:
        config.rag.enabled = updates.enabled
    if updates.n_results is not None:
        config.rag.n_results = updates.n_results
    if updates.relevance_threshold is not None:
        config.rag.relevance_threshold = updates.relevance_threshold
    if updates.collection_name is not None:
        config.rag.collection_name = updates.collection_name
    
    return {"status": "ok", "rag_config": config.rag.dict()}


@router.post("/rag/toggle")
async def toggle_rag(enabled: bool, _: str = Depends(verify_frontend_api_key)):
    """Toggle RAG on/off."""
    config = get_config()
    config.rag.enabled = enabled
    return {"status": "ok", "enabled": enabled}


@router.patch("/rag/query_extraction")
async def update_query_extraction(updates: QueryExtractionUpdateRequest, _: str = Depends(verify_frontend_api_key)):
    """Update query extraction configuration."""
    config = get_config()
    
    if updates.max_tokens is not None:
        config.rag.query_extraction.max_tokens = updates.max_tokens
    
    return {"status": "ok", "query_extraction_config": config.rag.query_extraction.dict()}


@router.get("/academic_search")
async def get_academic_search_config(_: str = Depends(verify_frontend_api_key)):
    """Get academic search configuration."""
    config = get_config()
    return config.academic_search.dict()


@router.patch("/academic_search")
async def update_academic_search_config(updates: AcademicSearchUpdateRequest, _: str = Depends(verify_frontend_api_key)):
    """Update academic search configuration."""
    config = get_config()
    
    if updates.enabled is not None:
        config.academic_search.enabled = updates.enabled
    if updates.max_results is not None:
        config.academic_search.max_results = updates.max_results
    
    return {"status": "ok", "academic_search_config": config.academic_search.dict()}


@router.post("/academic_search/toggle")
async def toggle_academic_search(enabled: bool, _: str = Depends(verify_frontend_api_key)):
    """Toggle academic search on/off."""
    config = get_config()
    config.academic_search.enabled = enabled
    return {"status": "ok", "enabled": enabled}


@router.get("/document_processing")
async def get_document_processing_config(_: str = Depends(verify_frontend_api_key)):
    """Get document processing configuration."""
    config = get_config()
    return config.document_processing.dict()


@router.patch("/document_processing/chunking")
async def update_chunking_config(updates: ChunkingUpdateRequest, _: str = Depends(verify_frontend_api_key)):
    """Update document chunking configuration."""
    config = get_config()
    
    if updates.max_chars is not None:
        config.document_processing.chunking.max_chars = updates.max_chars
    if updates.overlap is not None:
        config.document_processing.chunking.overlap = updates.overlap
    
    return {"status": "ok", "chunking_config": config.document_processing.chunking.dict()}


@router.patch("/document_processing/batch_size")
async def update_batch_size(updates: DocumentProcessingUpdateRequest, _: str = Depends(verify_frontend_api_key)):
    """Update document processing batch size."""
    config = get_config()
    
    if updates.batch_size is not None:
        config.document_processing.batch_size = updates.batch_size
    
    return {"status": "ok", "batch_size": config.document_processing.batch_size}


@router.get("/prompts")
async def get_prompts_config(_: str = Depends(verify_frontend_api_key)):
    """Get prompts configuration."""
    config = get_config()
    return config.prompts.dict()


@router.patch("/prompts")
async def update_prompts_config(updates: PromptsUpdateRequest, _: str = Depends(verify_frontend_api_key)):
    """Update prompts configuration."""
    config = get_config()
    
    if updates.system_message is not None:
        config.prompts.system_message = updates.system_message
    
    return {"status": "ok", "prompts_config": config.prompts.dict()}


@router.get("/llm")
async def get_llm_config(_: str = Depends(verify_frontend_api_key)):
    """Get LLM configuration."""
    config = get_config()
    return config.llm.dict()


@router.post("/reload")
async def reload_config(_: str = Depends(verify_frontend_api_key)):
    """Reload configuration from YAML file."""
    try:
        get_config_manager().reload_config()
        return {"status": "ok", "message": "Configuration reloaded from file"}
    except Exception as e:
        raise HTTPException(500, f"Failed to reload config: {str(e)}")
