import httpx
import json
import logging
from typing import Dict, Any, Optional, AsyncIterator
from fastapi import HTTPException, status
from .config import settings

logger = logging.getLogger(__name__)

# Simple LLM client interface, holds functions to interact with LLM service
class LLMServiceClient:
    
    def __init__(self):
        self.llm_base_url = settings.llm_service_url
        self.llm_api_key = settings.llm_service_api_key
        self.timeout = 120.0  # > 2 minutes response time -> timeout for LLM operation
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication for LLM service."""
        return {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
    
    async def stream_chat_request(
        self,
        messages: list[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        - The chat prompt lands at this endpoint first, comming from the frontend
        - This functions then starts a stream, streaming the prompt to the LLM service
        
        - Simple authorization happens here, with the simple API key we implement
        - Route to the next service in the stream is also configured here
        """
        logger.info(f"[LLM CLIENT] Starting streaming request")
        logger.info(f"[LLM CLIENT] Model: {model or 'default'}, Messages: {len(messages)}")
        
        payload = {
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        if model is not None:
            payload["model"] = model
        
        try:
            logger.info(f"[LLM CLIENT] Connecting to LLM service at {self.llm_base_url}/v1/chat")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.llm_base_url}/v1/chat",
                    json=payload,
                    headers=self._get_headers()
                ) as response:
                    
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"[LLM CLIENT] ✗ LLM service error ({response.status_code}): {error_text.decode()}")
                        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM service error")
                    
                    logger.info(f"[LLM CLIENT] ✓ Connection established, streaming response...")
                    # Stream NDJSON lines directly from LLM service
                    chunk_count = 0
                    async for line in response.aiter_lines():
                        if line.strip():  # Only yield non-empty lines
                            chunk_count += 1
                            yield line.encode("utf-8") + b"\n"
                    
                    logger.info(f"[LLM CLIENT] ✓ Streaming complete ({chunk_count} chunks sent)")
                    
        except httpx.TimeoutException:
            logger.error(f"[LLM CLIENT] ✗ Request timed out after {self.timeout}s")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM service request timed out"
            )
        except httpx.RequestError as e:
            logger.error(f"[LLM CLIENT] ✗ Connection failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to LLM service: {str(e)}"
            )
    
    async def complete(self, prompt: str, model: Optional[str] = None, *, max_tokens: int) -> str:
        """
        Non-streaming completion for quick LLM calls (e.g., query extraction).
        
        Args:
            prompt: The prompt to send to LLM
            model: Optional model override
            max_tokens: Maximum tokens to generate (REQUIRED - caller must specify)
        
        Returns:
            Just the text content of the LLM response.
        """
        logger.info(f"[LLM CLIENT] Complete request - max_tokens: {max_tokens}")
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens
        }
        
        if model is not None:
            payload["model"] = model
        
        try:
            logger.info(f"[LLM CLIENT] Sending non-streaming request to {self.llm_base_url}/v1/chat")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.llm_base_url}/v1/chat",
                    json=payload,
                    headers=self._get_headers()
                )
                
                if response.status_code != 200:
                    logger.error(f"[LLM CLIENT] ✗ LLM service error ({response.status_code}): {response.text}")
                    return ""  # Return empty on error
                
                data = response.json()
                logger.info(f"[LLM CLIENT] Response data keys: {list(data.keys())}")
                
                # Extract content from response
                if "choices" in data and len(data["choices"]) > 0:
                    result = data["choices"][0].get("message", {}).get("content", "").strip()
                    logger.info(f"[LLM CLIENT] ✓ Received response ({len(result)} chars): '{result[:100]}...'")
                    return result
                else:
                    logger.warning(f"[LLM CLIENT] ⚠ No 'choices' in response or empty choices")
                    return ""
                
        except Exception as e:
            logger.error(f"[LLM CLIENT] ✗ Error in LLM completion: {e}")
            return ""  # Fail gracefully

# Global client instance
llm_client = LLMServiceClient()
