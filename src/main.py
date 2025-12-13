from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .app_settings import settings
import uvicorn
import logging

# Custom formatter for compact logs with newlines after tags
class CompactFormatter(logging.Formatter):
    def format(self, record):
        # Extract tag like [RAG] or [LLM CLIENT] from message
        msg = record.getMessage()
        if msg.startswith('[') and ']' in msg:
            tag_end = msg.index(']') + 1
            tag = msg[:tag_end]
            rest = msg[tag_end:].lstrip()
            record.msg = f"{tag}\n  {rest}"
        return super().format(record)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(CompactFormatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

app = FastAPI(
    title="System Logic Service API",
    description="Middleware service between Next.js frontend and LLM Service",
    version="1.0.0"
)

# Configure CORS - Allow all origins as this is just a prototype
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {request.method} {request.url.path} - Status: {response.status_code}")
    return response


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "service": "System Logic Service API",
        "version": "1.0.0",
        "status": "running"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "system-logic-service"
    }


# Import and include routers
from .routes import router as api_router
from .config_routes import router as config_router

app.include_router(api_router)
app.include_router(config_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
