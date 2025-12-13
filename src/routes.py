from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from .auth import verify_frontend_api_key
from .llm_client import llm_client
from .mcp_client.academic_search.arxiv_search import arxiv_mcp
from .mcp_client.gdrive.drive_fetcher import fetch_and_extract
from .vectordb_client import VectorDBClient
from .text_ingest_utils import chunk_text, extract_text_with_ocr_fallback
from .config.config_manager import get_config
import re
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

vectordb_client = VectorDBClient()

# Initialize config
config = get_config()


async def extract_search_query(prompt: str) -> str:
    """
    Use LLM to extract a clean search query from conversational prompts.
    Removes filler words and focuses on the core information need.
    """
    logger.info(f"[QUERY EXTRACTION] Starting extraction for prompt: {prompt[:100]}...")
    
    # Use query template from config
    extraction_prompt = config.rag.query_extraction.query_template.format(prompt=prompt)
    
    try:
        query = await llm_client.complete(extraction_prompt, max_tokens=config.rag.query_extraction.max_tokens)
        if query:
            logger.info(f"[QUERY EXTRACTION] ✓ Refined query: '{query}'")
            return query
        else:
            logger.warning(f"[QUERY EXTRACTION] ⚠ Empty response, using original prompt")
            return prompt
    except Exception as e:
        logger.error(f"[QUERY EXTRACTION] ✗ Failed: {e}")
        return prompt  # Fallback to original


async def expand_prompt_with_rag(
    content: str,
    collection_name: str = "documents",
    n_results: int = 3,
    relevance_threshold: float = 0.7
) -> str:
    """
    Expand prompt using RAG with query refinement.
    
    Process:
    1. Extract clean search query from conversational prompt
    2. Query vector database with refined query
    3. Filter by relevance threshold
    4. Format and inject context into prompt
    """
    logger.info(f"[RAG] ========== Starting RAG Enhancement ==========")
    logger.info(f"[RAG] Collection: {collection_name}, Max results: {n_results}, Threshold: {relevance_threshold}")
    
    try:
        # Step 1: Refine the query
        logger.info(f"[RAG] Step 1: Extracting search query from user message...")
        search_query = await extract_search_query(content)
        logger.info(f"[RAG] Original prompt: '{content[:100]}{'...' if len(content) > 100 else ''}'")
        logger.info(f"[RAG] Refined query: '{search_query}'")
        
        # Step 2: Query vector DB with refined query
        logger.info(f"[RAG] Step 2: Querying vector database...")
        results = vectordb_client.query(
            collection_name=collection_name,
            query_text=search_query,  # Use refined query
            n_results=n_results * 2  # Get extra, then filter
        )
        logger.info(f"[RAG] ✓ Retrieved {len(results.get('results', []))} documents from vector DB")
        
        # Step 3: Filter by relevance
        logger.info(f"[RAG] Step 3: Filtering by relevance threshold {relevance_threshold}...")
        relevant_docs = []
        for result in results.get('results', []):
            distance = result.get('distance', 1.0)
            doc_id = result.get('id', 'unknown')
            # Lower distance = more similar (closer to 0 is better)
            logger.info(f"[RAG]   Doc '{doc_id}' - distance: {distance:.3f}")
            if distance < relevance_threshold:
                relevant_docs.append(result)
                logger.info(f"[RAG]     ✓ Included (distance {distance:.3f} < {relevance_threshold})")
            else:
                logger.info(f"[RAG]     ✗ Filtered (distance {distance:.3f} >= {relevance_threshold})")
        
        logger.info(f"[RAG] ✓ Kept {len(relevant_docs)}/{len(results.get('results', []))} relevant documents")
        
        if not relevant_docs:
            logger.warning(f"[RAG] ⚠ No relevant documents found, using original prompt")
            return content
        
        # Step 4: Format context
        logger.info(f"[RAG] Step 4: Formatting context from top {n_results} documents...")
        context_parts = ["=== Retrieved Context ==="]
        for i, doc in enumerate(relevant_docs[:n_results], 1):
            doc_text = doc['document'][:500]  # Limit size
            metadata = doc.get('metadata', {})
            source = metadata.get('filename', metadata.get('source', 'Unknown'))
            
            context_parts.append(f"\n[Document {i}] (Source: {source})")
            context_parts.append(doc_text)
            logger.info(f"[RAG]   - Document {i}: {source} ({len(doc['document'])} chars)")
        
        context = "\n".join(context_parts)
        
        # Step 5: Combine context with original prompt
        logger.info(f"[RAG] Step 5: Combining context with user query...")
        expanded = config.prompts.context_template.format(context=context, content=content)
        
        logger.info(f"[RAG] ✓ Successfully expanded prompt (final length: {len(expanded)} chars)")
        logger.info(f"[RAG] ========== RAG Enhancement Complete ==========")
        return expanded
        
    except Exception as e:
        logger.error(f"[RAG] ✗ Error during expansion: {e}", exc_info=True)
        return content  # Fallback to original on error


# Pydantic models for request/response validation
class Message(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


# Chat request model, used to validate incoming requests are in expected format
class ChatRequest(BaseModel):
    """Request model for chat completion."""
    messages: List[Message] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(None, description="Optional model name")
    stream: Optional[bool] = Field(True, description="Whether to stream the response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class LocalFolderIngestRequest(BaseModel):
    folder_path: str = Field(..., description="Local path inside the container/host to scan for documents")
    collection_name: str = Field("documents", description="Target collection name in vector DB")
    max_docs: Optional[int] = Field(
        None,
        description="Optional limit on number of chunks/documents to send (for safety during testing)",
    )

class UploadResult(BaseModel):
    status: str
    collection_name: str
    ingested: int
# Create router
router = APIRouter(
    tags=["LLM Operations"]
)


@router.post("/v1/chat")
async def chat(request: ChatRequest, req: Request, _: str = Depends(verify_frontend_api_key)):
    """
    Middleware in the downstream direction
    - Forwards chat request directly to the LLM service
    - Streaming is done in llm_client.stream_chat_request function
    - Response moves back upstream through same nodes as the downstream request
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"[CHAT REQUEST] New chat request received")
    
    # Convert Pydantic models to dicts for the LLM client (tokens which are more readable by LLM)
    # This converts: [Message(role="user", content="Hello!")]
    # Into: [{"role": "user", "content": "Hello!"}]
    messages = [msg.dict() for msg in request.messages]
    
    # Get feature flags from config
    enable_rag = config.rag.enabled
    enable_academic_search = config.academic_search.enabled
    
    logger.info(f"[CHAT REQUEST] Feature flags - RAG: {enable_rag}, Academic Search: {enable_academic_search}")
    
    # RAG Enhancement (runs first if enabled)
    if enable_rag and messages:
        logger.info(f"[CHAT REQUEST] RAG is enabled, processing...")
        last_message = messages[-1]
        if last_message.get('role') == 'user':
            content = last_message.get('content', '')
            # Expand with RAG using config values
            enhanced_content = await expand_prompt_with_rag(
                content=content,
                collection_name=config.rag.collection_name,
                n_results=config.rag.n_results,
                relevance_threshold=config.rag.relevance_threshold
            )
            messages[-1]['content'] = enhanced_content
    
    # Academic Search (can run after RAG if both enabled)
    if enable_academic_search:
        # Check for 'academic_search' keyword in the last user message
        if messages:
            last_message = messages[-1]
            if last_message.get('role') == 'user':
                content = last_message.get('content', '')
                
                # Check if 'academic_search' keyword is present
                if 'academic_search' in content.lower():
                    # Extract query after 'academic_search'
                    # Pattern: academic_search: query or academic_search query
                    pattern = r'academic_search[:\s]+(.+?)(?:\n|$)'
                    match = re.search(pattern, content, re.IGNORECASE)
                    
                    if match:
                        search_query = match.group(1).strip()
                    else:
                        # If no explicit query, use the rest of the message
                        search_query = re.sub(r'academic_search', '', content, flags=re.IGNORECASE).strip()
                    
                    # Perform arXiv search
                    search_results = arxiv_mcp.search(query=search_query, max_results=config.academic_search.max_results)
                    
                    # Format results as context
                    context = arxiv_mcp.format_results_for_context(search_results)
                    
                    # Inject context into the user's message
                    # Add context before the user's original query
                    enhanced_content = f"{context}\n\nUser Query: {content}"
                    messages[-1]['content'] = enhanced_content
    
    
    # Stream from LLM service
    logger.info(f"[CHAT REQUEST] Streaming to LLM service...")
    logger.info(f"{'='*80}\n")
    return StreamingResponse(
        llm_client.stream_chat_request(
            messages=messages,
            model=request.model,
            **(request.metadata or {})
        ),
        media_type="application/x-ndjson"
    )



 
#@router.post("/v1/local-ingest", tags=["LLM Operations"])
#async def ingest_local_folder(
#    req: LocalFolderIngestRequest,
#    _: str = Depends(verify_frontend_api_key),
#):
#    try:
#        documents = build_documents_from_folder(req.folder_path, req.collection_name)
#    except FileNotFoundError as e:
#        raise HTTPException(status_code=400, detail=str(e))
#
#    if not documents:
#        return {"status": "ok", "ingested": 0, "message": "No supported files found (pdf/txt)."}
#
#    if req.max_docs is not None:
#        documents = documents[: req.max_docs]
#
#    BATCH_SIZE = 50
#    total_ingested = 0
#
#    for i in range(0, len(documents), BATCH_SIZE):
#        batch = documents[i : i + BATCH_SIZE]
#        vectordb_client.ingest(req.collection_name, batch)
#        total_ingested += len(batch)
#
#    return {
#        "status": "ok",
#        "collection_name": req.collection_name,
#        "ingested": total_ingested,
#    }


@router.post("/v1/upload-docs", response_model=UploadResult, tags=["LLM Operations"])
async def upload_docs(
    files: List[UploadFile] = File(..., description="One or more documents to ingest"),
    collection_name: str = Form("documents"),
    user_id: str = Form("anonymous"),
    _: str = Depends(verify_frontend_api_key),
):
    """
    Upload multiple files (PDF/TXT etc) and ingest them into the vector DB.

    Frontend can drag & drop a folder and send all files in one request.
    Backend just sees a list of files.
    """
    documents: List[Dict[str, Any]] = []
    doc_index = 0

    for uploaded in files:
        filename = uploaded.filename or "unnamed"
        suffix = Path(filename).suffix.lower()

        # Read file contents into memory
        content = await uploaded.read()

        # Write to a temp file so we can reuse extract_text()
        tmp_path = Path("/tmp") / f"upload_{doc_index}{suffix}"
        tmp_path.write_bytes(content)

        try:
            full_text = extract_text_with_ocr_fallback(tmp_path)
        except ValueError as e:
            print(f"Skipping {filename}: {e}")
            continue
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        chunks = chunk_text(full_text, max_chars=config.document_processing.chunking.max_chars, overlap=config.document_processing.chunking.overlap)

        for idx, chunk in enumerate(chunks):
            doc_id = f"{user_id}:{filename}:chunk-{idx}"
            documents.append(
                {
                    "id": doc_id,
                    "text": chunk,
                    "metadata": {
                        "user_id": user_id,
                        "source": "upload",
                        "filename": filename,
                        "chunk_index": idx,
                    },
                }
            )

        doc_index += 1

    if not documents:
        return UploadResult(
            status="ok",
            collection_name=collection_name,
            ingested=0,
        )

    # Ingest in batches to avoid huge payloads
    BATCH_SIZE = config.document_processing.batch_size
    total_ingested = 0
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        vectordb_client.ingest(collection_name, batch)
        total_ingested += len(batch)

    return UploadResult(
        status="ok",
        collection_name=collection_name,
        ingested=total_ingested,
    )

class DriveFetchRequest(BaseModel):
    access_token: str
    collection_name: Optional[str] = Field("documents")
    max_files: Optional[int] = Field(100)
    q: Optional[str] = None
    output_dir: Optional[str] = None
    user_id: Optional[str] = Field("drive_import")
    ingest: Optional[bool] = Field(True)

@router.post("/v1/fetch-and-ingest/drive", tags=["LLM Operations"])
async def fetch_and_ingest_drive(
    body: DriveFetchRequest,
    _: str = Depends(verify_frontend_api_key),
):
    """
    Fetch files from Google Drive and ingest them into the vector DB after
    running full extraction/cleaning/chunking using text_ingest_utils.
    - Frontend must call this endpoint (authenticated with frontend API key)
    - Main service downloads Drive files, runs extract_text_with_ocr_fallback,
      chunks them, and then calls vectordb_client.ingest(...) to store chunks.
    """
    collection_name = body.collection_name or "documents"
    user_id = body.user_id or "drive_import"

    output_dir, docs = fetch_and_extract(body.access_token, max_files=body.max_files, q=body.q, output_dir=body.output_dir)

    if not docs:
        return {"output_dir": output_dir, "ingested": 0, "files": []}

    documents: List[Dict[str, Any]] = []
    for d in docs:
        file_path = d.get("path")
        filename = Path(file_path).name if file_path else d.get("metadata", {}).get("name", "unknown")
        try:
            # Use the robust extraction/ocr/clean path on the actual file
            cleaned_text = extract_text_with_ocr_fallback(Path(file_path))
        except ValueError as e:
            logger.warning(f"Skipping {filename}: {e}")
            continue
        except Exception as e:
            logger.exception(f"Error extracting {filename}: {e}")
            continue

        # Chunk the cleaned text
        chunks = chunk_text(cleaned_text, max_chars=config.document_processing.chunking.max_chars)

        for idx, chunk in enumerate(chunks):
            doc_id = f"{user_id}:{filename}:chunk-{idx}"
            documents.append(
                {
                    "id": doc_id,
                    "text": chunk,
                    "metadata": {
                        "user_id": user_id,
                        "source": "drive",
                        "filename": filename,
                        "original_id": d.get("id"),
                        "chunk_index": idx,
                    },
                }
            )

    if not documents:
        return {"output_dir": output_dir, "ingested": 0, "files": [{"id": d.get("id"), "path": d.get("path")} for d in docs]}

    # Batch ingest using the existing vectordb client
    BATCH_SIZE = config.document_processing.batch_size
    total_ingested = 0
    try:
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            vectordb_client.ingest(collection_name, batch)
            total_ingested += len(batch)
    except Exception as e:
        logger.exception("Failed to ingest to vectordb")
        raise HTTPException(status_code=500, detail=f"Failed to ingest to vector DB: {str(e)}")

    return {
        "output_dir": output_dir,
        "ingested": total_ingested,
        "files": [{"id": d.get("id"), "path": d.get("path")} for d in docs],
    }