"""
Updated startup event for main.py with LLM RAG integration
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import logging
import threading
import uuid
from datetime import datetime
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Make INFO logs (vector-store state, retrieval counts, errors) visible in the
# terminal. uvicorn's default config suppresses app-level INFO lines, which
# made "empty vector store" / retrieval failures look like silent empty answers.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Your existing imports
from rag.pipeline import RAGPipeline
from rag.llm_rag import LLMRAGPipeline, LLMConfig, RAGAnswer

logger = logging.getLogger(__name__)

# Global variables to store the RAG components
retriever = None
llm_rag_pipeline = None

# In-memory registry of bulk-upload jobs (job_id -> job state). Lives only as
# long as the process runs (lost on restart); guarded by _bulk_jobs_lock.
bulk_jobs: Dict[str, Any] = {}
_bulk_jobs_lock = threading.Lock()

# Configuration
# Paths are anchored to the project root (src/api/main.py -> parents[2]) so the
# app works regardless of the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Load .env from the project root (gitignored; see .env.example). Existing
# shell-exported env vars win, since load_dotenv won't override by default.
load_dotenv(PROJECT_ROOT / ".env")
UPLOAD_DIRECTORY = str(PROJECT_ROOT / "data" / "uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".md", ".rtf", ".csv", ".json"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Ensure upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True, mode=0o755)

# Pydantic models for request/response
class TextUploadRequest(BaseModel):
    content: str
    filename: str = "text_content.txt"
    metadata: Optional[Dict[str, Any]] = None
    chunking_strategy: str = "recursive"

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_path: Optional[str] = None
    chunks_created: Optional[int] = None
    document_ids: Optional[List[str]] = None
    processing_status: str
    error: Optional[str] = None

class FileInfo(BaseModel):
    filename: str
    size_bytes: int
    size_mb: float
    created: str
    modified: str
    extension: str

class SystemStatus(BaseModel):
    upload_directory: str
    allowed_extensions: List[str]
    max_file_size_mb: int
    uploaded_files_count: int
    rag_system: Dict[str, Any]
    llm_system: Dict[str, Any]

class QuestionRequest(BaseModel):
    question: str
    k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ClearVectorStoreRequest(BaseModel):
    # Defaults to False so a missing field / {} is rejected by the handler with
    # 400 (not pydantic's 422). Guarded: only {"confirm": true} passes.
    confirm: bool = False

class ClearVectorStoreResponse(BaseModel):
    status: str          # "success"
    message: str         # "Vector store cleared"
    chunks_removed: int
    remaining_chunks: int
    timestamp: str

class VectorStoreRow(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}
    embedding_length: int = 0
    embedding: Optional[List[float]] = None   # populated only when include_embeddings=true

class VectorStoreRowsResponse(BaseModel):
    total_chunks: int
    offset: int
    limit: int
    rows: List[VectorStoreRow]

class BulkFileResult(BaseModel):
    filename: str
    status: str                # pending | success | failed | skipped
    chunks_created: int = 0
    error: Optional[str] = None

class BulkUploadResponse(BaseModel):
    job_id: str
    status: str                # queued
    files_accepted: int
    files_rejected: int
    message: str

class FolderIngestRequest(BaseModel):
    # Which server-side folder to scan. `folder` is resolved relative to the
    # project root unless absolute (it must still resolve inside the project).
    # All fields optional so a bare `POST /api/ingest/folder` ingests documents/.
    folder: str = "documents"
    recursive: bool = True
    chunking_strategy: str = "recursive"

class BulkJobResponse(BaseModel):
    job_id: str
    status: str                # queued | running | completed
    total_files: int
    succeeded: int
    failed: int
    skipped: int
    total_chunks: int
    files: List[BulkFileResult]
    created_at: str
    finished_at: Optional[str] = None


def _sse(event: str, data: Any) -> str:
    """Format one SSE event block. data must be JSON-serializable."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    global retriever, llm_rag_pipeline
    logger.info("Initializing services...")
    
    try:
        # Set up directories and environment variables
        setup_environment()
        
        # Initialize RAG pipeline with error handling
        retriever = initialize_rag_pipeline()
        
        if retriever:
            # Log system information
            system_info = retriever.get_system_info()
            logger.info(f"RAG Pipeline initialized: {system_info}")
            
            # Initialize LLM RAG Pipeline
            llm_rag_pipeline = initialize_llm_rag_pipeline(retriever)
            
            if llm_rag_pipeline:
                logger.info("LLM RAG Pipeline initialized successfully")
            else:
                logger.warning("LLM RAG Pipeline initialization failed - Q&A features will be unavailable")
            
            # Optionally load existing documents or perform health check
            doc_count = retriever.get_document_count()
            logger.info(f"Vector store contains {doc_count} documents")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        # Don't raise the exception - let the app start but with limited functionality
        retriever = None
        llm_rag_pipeline = None
        logger.warning("Application starting with limited functionality")
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    # Add any cleanup code here if needed


def setup_environment():
    """Setup environment variables and directories"""
    
    # Create necessary directories
    directories = [
        str(PROJECT_ROOT / "data"),
        str(PROJECT_ROOT / "data" / "chromadb"),
        str(PROJECT_ROOT / "data" / "documents"),
        str(PROJECT_ROOT / "logs")
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True, mode=0o755)
            logger.info(f"Directory ready: {directory}")
        except Exception as e:
            logger.warning(f"Could not create directory {directory}: {e}")
    
    # Set environment variables for model caching. Derive the cache dir from the
    # same VECTOR_STORE_PATH the store uses, so the model cache lives at
    # <persist_dir>/model_cache — the location vector_store.py defaults to.
    persist_dir = os.getenv("VECTOR_STORE_PATH", str(PROJECT_ROOT / "data" / "chromadb"))
    cache_dir = os.path.join(persist_dir, "model_cache")
    os.makedirs(cache_dir, exist_ok=True, mode=0o755)

    os.environ.setdefault('TRANSFORMERS_CACHE', cache_dir)
    os.environ.setdefault('HF_HOME', cache_dir)
    
    logger.info("Environment setup completed")


def initialize_rag_pipeline() -> RAGPipeline:
    """Initialize the RAG pipeline with error handling"""
    
    try:
        # RAG pipeline configuration
        rag_config = {
            "chunk_size": int(os.getenv("CHUNK_SIZE", "200")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "persist_directory": os.getenv("VECTOR_STORE_PATH", str(PROJECT_ROOT / "data" / "chromadb")),
            "collection_name": os.getenv("COLLECTION_NAME", "business_documents")
        }
        
        logger.info(f"Initializing RAG pipeline with config: {rag_config}")
        
        # Create the RAG pipeline
        rag_pipeline = RAGPipeline(**rag_config)
        
        logger.info("RAG pipeline initialized successfully")
        return rag_pipeline
        
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {str(e)}")
        
        # Try with minimal configuration as fallback
        try:
            logger.info("Attempting fallback RAG pipeline initialization...")
            
            fallback_config = {
                "chunk_size": 200,
                "chunk_overlap": 50,
                "embedding_model": "all-MiniLM-L6-v2",
                "persist_directory": os.path.join(os.path.expanduser("~"), ".rag_data"),
                "collection_name": "fallback_documents"
            }
            
            rag_pipeline = RAGPipeline(**fallback_config)
            logger.info("Fallback RAG pipeline initialized successfully")
            return rag_pipeline
            
        except Exception as fallback_error:
            logger.error(f"Fallback RAG pipeline also failed: {str(fallback_error)}")
            return None


def initialize_llm_rag_pipeline(rag_pipeline: RAGPipeline) -> Optional[LLMRAGPipeline]:
    """Initialize the LLM RAG pipeline for question answering"""
    
    try:
        # Provider-agnostic LLM configuration from env vars (see llm_rag.LLMConfig):
        # provider is chosen via LLM_PROVIDER. LLMConfig raises if the chosen
        # provider's key is missing, so Q&A stays gracefully unavailable (503)
        # until a .env with a real key is provided.
        logger.info("Initializing LLM configuration...")

        # Create LLM config
        llm_config = LLMConfig()
        
        # Custom system prompt for business documents
        system_prompt = """You are a helpful AI assistant that answers questions based on business documents and company information.

Guidelines:
- Use ONLY the information from the provided context to answer questions
- If the answer cannot be found in the context, clearly state "I don't have enough information to answer this question based on the available documents"
- Be precise and professional in your responses
- When referencing specific information, mention the source document if available
- Keep answers concise but comprehensive
- If you find conflicting information, mention this and present both perspectives

Format your response as:
1. Direct answer to the question
2. Supporting details from the context
3. Source references (document names, sections, etc.) if available"""
        
        # Create LLM RAG pipeline
        llm_rag = LLMRAGPipeline(
            retriever=rag_pipeline,
            llm_config=llm_config,
            system_prompt=system_prompt
        )
        
        logger.info("LLM RAG pipeline initialized successfully")
        return llm_rag
        
    except Exception as e:
        logger.error(f"Error initializing LLM RAG pipeline: {str(e)}")
        logger.warning("Q&A functionality will be unavailable")
        return None


# Create the FastAPI app with lifespan
app = FastAPI(
    title="RAG API with Q&A",
    description="Document processing, retrieval, and question answering API",
    version="1.0.0",
    lifespan=lifespan
)


# Your existing startup event (if you still need it for other initialization)
@app.on_event("startup")
async def startup_event():
    """Legacy startup event - use lifespan instead for new code"""
    pass


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global retriever, llm_rag_pipeline
    
    status = {
        "status": "healthy",
        "rag_pipeline": "available" if retriever else "unavailable",
        "llm_pipeline": "available" if llm_rag_pipeline else "unavailable",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if retriever:
        try:
            system_info = retriever.get_system_info()
            status["rag_info"] = system_info
        except Exception as e:
            status["rag_error"] = str(e)
    
    if llm_rag_pipeline:
        try:
            status["llm_info"] = {
                "provider": llm_rag_pipeline.llm_config.provider,
                "model": llm_rag_pipeline.llm_config.model,
                "base_url": llm_rag_pipeline.llm_config.base_url,
                "temperature": llm_rag_pipeline.llm_config.temperature,
                "max_tokens": llm_rag_pipeline.llm_config.max_tokens
            }
        except Exception as e:
            status["llm_error"] = str(e)
    
    return status


# System info endpoint
@app.get("/system/info")
async def get_system_info():
    """Get system information"""
    global retriever, llm_rag_pipeline
    
    info = {}
    
    if retriever:
        try:
            info["rag_system"] = retriever.get_system_info()
        except Exception as e:
            info["rag_error"] = str(e)
    else:
        info["rag_system"] = "not_available"
    
    if llm_rag_pipeline:
        info["llm_system"] = {
            "available": True,
            "provider": llm_rag_pipeline.llm_config.provider,
            "model": llm_rag_pipeline.llm_config.model,
            "base_url": llm_rag_pipeline.llm_config.base_url,
            "temperature": llm_rag_pipeline.llm_config.temperature,
            "max_tokens": llm_rag_pipeline.llm_config.max_tokens,
        }
    else:
        info["llm_system"] = {"available": False, "error": "LLM pipeline not initialized"}
    
    return info


# Utility functions to get the pipeline instances
def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance"""
    global retriever
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    return retriever


def get_llm_rag_pipeline() -> LLMRAGPipeline:
    """Get the global LLM RAG pipeline instance"""
    global llm_rag_pipeline
    if not llm_rag_pipeline:
        raise HTTPException(status_code=503, detail="LLM RAG pipeline is not initialized")
    return llm_rag_pipeline


# NEW: Question Answering Endpoints
@app.post("/api/qa/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an AI-generated answer based on your documents
    
    - **question**: The question to ask
    - **k**: Number of relevant chunks to retrieve (default: 5)
    - **filter_metadata**: Optional filters for document search
    - **temperature**: LLM temperature override (0.0-1.0)
    - **max_tokens**: Maximum tokens in response
    """
    
    try:
        # Get LLM RAG pipeline
        llm_rag = get_llm_rag_pipeline()
        
        # Prepare LLM overrides
        llm_overrides = {}
        if request.temperature is not None:
            llm_overrides['temperature'] = request.temperature
        if request.max_tokens is not None:
            llm_overrides['max_tokens'] = request.max_tokens
        
        # Log the exact request — a stray filter_metadata silently returning
        # 0 hits has been the root cause of "not enough information" answers.
        logger.info(
            "Q&A request: question=%r k=%s filter_metadata=%r temperature=%s max_tokens=%s",
            request.question, request.k, request.filter_metadata,
            request.temperature, request.max_tokens
        )

        # Get answer from LLM RAG pipeline
        answer: RAGAnswer = llm_rag.answer(
            query=request.question,
            k=request.k,
            filter_metadata=request.filter_metadata,
            llm_overrides=llm_overrides if llm_overrides else None
        )
        
        # Prepare metadata
        metadata = {
            "model": answer.model,
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "total_tokens": answer.total_tokens,
            "sources_count": len(answer.sources),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return QuestionResponse(
            question=answer.query,
            answer=answer.answer,
            sources=answer.sources,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question '{request.question}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.get("/api/qa/ask")
async def ask_question_get(
    question: str = Query(..., description="The question to ask"),
    k: int = Query(5, description="Number of relevant chunks to retrieve"),
    temperature: Optional[float] = Query(None, description="LLM temperature (0.0-1.0)"),
    max_tokens: Optional[int] = Query(None, description="Maximum tokens in response")
):
    """
    Ask a question via GET request (for simple integrations)
    """
    
    request = QuestionRequest(
        question=question,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return await ask_question(request)


@app.post("/api/qa/ask/stream", response_model=None)
async def ask_question_stream(request: QuestionRequest):
    """
    Stream an answer via Server-Sent Events.

    Emits `sources` (first), then repeated `token` deltas, then `done` with usage
    metadata, or `error` if generation fails mid-stream. Same body as /api/qa/ask.
    """
    llm_rag = get_llm_rag_pipeline()  # 503 before streaming if unavailable

    llm_overrides = {}
    if request.temperature is not None:
        llm_overrides['temperature'] = request.temperature
    if request.max_tokens is not None:
        llm_overrides['max_tokens'] = request.max_tokens

    logger.info(
        "Q&A stream request: question=%r k=%s filter_metadata=%r temperature=%s max_tokens=%s",
        request.question, request.k, request.filter_metadata,
        request.temperature, request.max_tokens,
    )

    # Retrieval + prompt building happen HERE (sync, ~ms) so failures return a
    # normal JSON HTTPException before streaming.
    try:
        messages, sources = llm_rag.build_messages(
            request.question, k=request.k, filter_metadata=request.filter_metadata
        )
    except Exception as e:
        logger.error("Retrieval failed for stream request '%s': %s", request.question, e)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    async def event_gen():
        yield _sse("sources", {"question": request.question, "sources": sources})
        async for event, payload in llm_rag.answer_stream(
            messages, sources, llm_overrides=llm_overrides or None
        ):
            yield _sse(event, payload)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/search")
async def search_documents(
    query: str = Form(...),
    k: int = Form(5),
    with_scores: bool = Form(False),
    filter_metadata: Optional[str] = Form(None)  # JSON string
):
    """
    Search for relevant documents (without LLM generation)
    
    - **query**: Search query
    - **k**: Number of documents to return
    - **with_scores**: Whether to include similarity scores
    - **filter_metadata**: Optional metadata filter (as JSON string)
    """
    
    try:
        pipeline = get_rag_pipeline()
        
        # Parse filter metadata if provided
        filter_dict = None
        if filter_metadata:
            try:
                filter_dict = json.loads(filter_metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in filter_metadata")
        
        # Perform search
        if with_scores:
            results = pipeline.search_with_scores(query, k, filter_dict)
            # Convert to serializable format with scores
            serializable_results = []
            for doc, score in results:
                serializable_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
        else:
            results = pipeline.search(query, k, filter_dict)
            # Convert Document objects to serializable format
            serializable_results = []
            for doc in results:
                serializable_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score")
                })
        
        return {
            "query": query,
            "results": serializable_results,
            "count": len(serializable_results),
            "with_scores": with_scores
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Update system status to include LLM info
@app.get("/api/upload/status", response_model=SystemStatus)
async def get_upload_status():
    """Get upload system status and configuration"""
    
    try:
        # Get RAG pipeline info
        rag_pipeline = get_rag_pipeline()
        rag_system_info = rag_pipeline.get_system_info()
        
        # Get LLM system info
        llm_system_info = {"available": False}
        try:
            llm_rag = get_llm_rag_pipeline()
            llm_system_info = {
                "available": True,
                "provider": llm_rag.llm_config.provider,
                "model": llm_rag.llm_config.model,
                "base_url": llm_rag.llm_config.base_url,
                "temperature": llm_rag.llm_config.temperature,
                "max_tokens": llm_rag.llm_config.max_tokens
            }
        except:
            llm_system_info = {"available": False, "error": "LLM pipeline not available"}
        
        # Get upload directory info
        upload_files = list(Path(UPLOAD_DIRECTORY).glob("*"))
        
        return SystemStatus(
            upload_directory=UPLOAD_DIRECTORY,
            allowed_extensions=list(ALLOWED_EXTENSIONS),
            max_file_size_mb=MAX_FILE_SIZE // (1024 * 1024),
            uploaded_files_count=len(upload_files),
            rag_system=rag_system_info,
            llm_system=llm_system_info
        )
        
    except Exception as e:
        logger.error(f"Error getting upload status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    return True


async def save_uploaded_file(file: UploadFile) -> Path:
    """Save uploaded file to disk and return the path"""
    
    # Generate a unique filename. The random token guards against two uploads
    # sharing the same second and the same original name — a bulk request with
    # duplicate filenames would otherwise silently overwrite the first file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid.uuid4().hex[:8]
    safe_filename = f"{timestamp}_{token}_{file.filename}"
    file_path = Path(UPLOAD_DIRECTORY) / safe_filename
    
    # Save file with size checking
    total_size = 0
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(8192):  # Read in chunks
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                # Clean up partial file
                os.remove(file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                )
            buffer.write(chunk)
    
    logger.info(f"Saved uploaded file: {file_path} ({total_size} bytes)")
    return file_path


def process_document_background(file_paths: List[Path], rag_pipeline: RAGPipeline, chunking_strategy: str = "recursive"):
    """Background task to process documents"""
    try:
        logger.info(f"Starting background processing of {len(file_paths)} files")
        
        # Process the documents
        chunk_count = rag_pipeline.process_documents(file_paths, chunking_strategy)
        
        logger.info(f"Successfully processed {len(file_paths)} files: {chunk_count} chunks created")
        
    except Exception as e:
        logger.error(f"Error processing documents in background: {str(e)}")


# Keep all your existing upload endpoints unchanged...
@app.post("/api/upload/single", response_model=UploadResponse)
async def upload_single_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunking_strategy: str = Form("recursive"),
    process_immediately: bool = Form(True)
):
    """Upload a single document (same as before)"""
    
    try:
        validate_file(file)
        file_path = await save_uploaded_file(file)
        rag_pipeline = get_rag_pipeline()
        
        if process_immediately:
            try:
                chunk_count = rag_pipeline.process_documents([file_path], chunking_strategy)
                
                return UploadResponse(
                    message="Document uploaded and processed successfully",
                    filename=file.filename,
                    file_path=str(file_path),
                    chunks_created=chunk_count,
                    processing_status="completed"
                )
                
            except Exception as process_error:
                logger.error(f"Error processing document immediately: {str(process_error)}")
                
                return UploadResponse(
                    message="Document uploaded but processing failed",
                    filename=file.filename,
                    file_path=str(file_path),
                    processing_status="failed",
                    error=str(process_error)
                )
        else:
            background_tasks.add_task(
                process_document_background, 
                [file_path], 
                rag_pipeline, 
                chunking_strategy
            )
            
            return UploadResponse(
                message="Document uploaded successfully, processing in background",
                filename=file.filename,
                file_path=str(file_path),
                processing_status="queued"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")



# Health check for Q&A system
@app.get("/api/qa/health")
async def qa_health_check():
    """Health check for Q&A system"""
    
    global llm_rag_pipeline
    
    if not llm_rag_pipeline:
        return {
            "status": "unavailable",
            "error": "LLM RAG pipeline not initialized",
            "rag_pipeline": "available" if retriever else "unavailable"
        }
    
    try:
        # Test with a simple query
        test_answer = llm_rag_pipeline.answer("test", k=1)
        
        return {
            "status": "healthy",
            "rag_pipeline": "available",
            "llm_pipeline": "available",
            "provider": llm_rag_pipeline.llm_config.provider,
            "model": llm_rag_pipeline.llm_config.model,
            "base_url": llm_rag_pipeline.llm_config.base_url,
            "test_successful": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "rag_pipeline": "available" if retriever else "unavailable",
            "llm_pipeline": "error",
            "error": str(e)
        }


# --- Vector store admin ---
def _check_clear_guard(request: ClearVectorStoreRequest) -> None:
    """Guard against accidental wipes: a stray Swagger click (empty body or {})
    must never empty the store. Only an explicit {"confirm": true} passes."""
    if request.confirm is not True:
        raise HTTPException(
            status_code=400,
            detail='Clearing the vector store requires body {"confirm": true}',
        )


@app.post("/api/vector-store/clear", response_model=ClearVectorStoreResponse)
async def clear_vector_store(request: ClearVectorStoreRequest):
    """Empty the Chroma collection (chunk text + metadata + embedding vectors).

    Clear-only: does NOT delete data/chromadb files (model_cache survives),
    does NOT touch data/uploads, and does NOT re-ingest.
    """
    pipeline = get_rag_pipeline()      # 503 if RAG pipeline is not initialized
    _check_clear_guard(request)        # 400 unless confirm == true

    try:
        before = pipeline.get_document_count()
        ok = pipeline.clear_vector_store()
        after = pipeline.get_document_count()

        if not ok:
            raise HTTPException(status_code=500, detail="Vector store clear failed")
        if after != 0:
            logger.error("Vector store clear left %d chunks behind", after)
            raise HTTPException(
                status_code=500,
                detail=f"Vector store not empty after clear ({after} chunks remain)",
            )

        logger.info("Vector store cleared: removed %d chunks, %d remaining",
                    before - after, after)
        return ClearVectorStoreResponse(
            status="success",
            message="Vector store cleared",
            chunks_removed=before - after,
            remaining_chunks=after,
            timestamp=datetime.utcnow().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error clearing vector store: %s", e)
        raise HTTPException(status_code=500, detail=f"Vector store clear failed: {e}")


@app.get("/api/vector-store/rows", response_model=VectorStoreRowsResponse)
async def list_vector_store_rows(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
    include_embeddings: bool = Query(False),
):
    """Page over the stored chunks — the vector-store equivalent of SELECT *.

    Every collection entry is a chunk: an id, the chunk text, its metadata, and
    its embedding vector. Returns rows in store order, paginated with
    limit/offset; pass include_embeddings=true to fetch the full float vector
    (dimension is always reported via embedding_length).
    """
    pipeline = get_rag_pipeline()      # 503 if RAG pipeline is not initialized
    try:
        data = pipeline.list_chunks(
            limit=limit, offset=offset, include_embeddings=include_embeddings)
        total = pipeline.get_document_count()
    except Exception as e:
        logger.error("Error reading vector store rows: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to read vector store: {e}")

    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    # Chroma returns embeddings as a NumPy array, never use `or []` on it (a
    # truth-test on a multi-element array raises "truth value is ambiguous").
    raw_embeddings = data.get("embeddings")
    embeddings = raw_embeddings if raw_embeddings is not None else []

    rows = []
    for i, chunk_id in enumerate(ids):
        text = documents[i] if i < len(documents) and documents[i] else ""
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        emb = embeddings[i] if i < len(embeddings) else None
        emb_list = None
        if emb is not None:
            # Normalize numpy row -> plain Python list of floats for pydantic.
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        rows.append(VectorStoreRow(
            id=chunk_id,
            text=text,
            metadata=meta,
            embedding_length=len(emb_list) if emb_list is not None else 0,
            embedding=emb_list,
        ))

    return VectorStoreRowsResponse(
        total_chunks=total,
        offset=offset,
        limit=limit,
        rows=rows,
    )


# --- Bulk upload (many documents in one request) ---
MAX_BULK_JOBS = 20


def _run_bulk_job(job_id: str, rag_pipeline: RAGPipeline) -> None:
    """Process one bulk-upload job in a background thread.

    Each file goes through the existing read -> chunk -> embed -> add path as its
    own process_documents() call, so a failure is isolated to that file (earlier
    files stay stored) and peak memory is bounded to one file's chunks at a time.
    """
    with _bulk_jobs_lock:
        job = bulk_jobs.get(job_id)
        if job is None:
            logger.error("Bulk job %s not found", job_id)
            return
        job["status"] = "running"
        results = job["results"]
        paths = job["paths"]
        strategy = job["chunking_strategy"]

    for result, path in zip(results, paths):
        if path is None:
            continue  # skipped during upload (validation / save failed)
        filename = result["filename"]
        try:
            chunk_count = rag_pipeline.process_documents([path], strategy)
        except Exception as e:
            logger.error("Bulk job %s: failed to process %s: %s", job_id, filename, e)
            with _bulk_jobs_lock:
                result.update(status="failed", error=str(e))
                job["failed"] += 1
        else:
            logger.info("Bulk job %s: processed %s (%d chunks)", job_id, filename, chunk_count)
            with _bulk_jobs_lock:
                result.update(status="success", chunks_created=chunk_count)
                job["succeeded"] += 1
                job["total_chunks"] += chunk_count

    with _bulk_jobs_lock:
        job["status"] = "completed"
        job["finished_at"] = datetime.utcnow().isoformat()
    logger.info("Bulk job %s completed: %d ok, %d failed, %d chunks total",
                job_id, job["succeeded"], job["failed"], job["total_chunks"])


@app.post("/api/upload/bulk", response_model=BulkUploadResponse)
async def upload_bulk_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    chunking_strategy: str = Form("recursive"),
):
    """Upload many documents in one request.

    Saves every file to disk immediately, then processes them one at a time in a
    background thread (the request does not block on embedding). Returns a job_id
    to poll via GET /api/upload/bulk/{job_id}. Files that fail validation or size
    checks are reported as skipped and the rest still process.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    rag_pipeline = get_rag_pipeline()      # 503 if RAG pipeline is not initialized

    results = []
    paths = []
    rejected = 0
    for file in files:
        filename = file.filename or "unnamed"
        try:
            validate_file(file)
            path = await save_uploaded_file(file)
        except HTTPException as e:
            rejected += 1
            results.append({
                "filename": filename,
                "status": "skipped",
                "chunks_created": 0,
                "error": e.detail,
            })
            paths.append(None)
            continue
        results.append({
            "filename": filename,
            "status": "pending",
            "chunks_created": 0,
            "error": None,
        })
        paths.append(path)

    accepted = len(results) - rejected
    if accepted == 0:
        first_error = next((r["error"] for r in results if r["error"]), "unknown")
        raise HTTPException(
            status_code=400,
            detail=f"No files could be accepted for processing. First error: {first_error}",
        )

    job_id = uuid.uuid4().hex[:12]
    with _bulk_jobs_lock:
        bulk_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "chunking_strategy": chunking_strategy,
            "results": results,
            "paths": paths,
            "succeeded": 0,
            "failed": 0,
            "total_chunks": 0,
            "created_at": datetime.utcnow().isoformat(),
            "finished_at": None,
        }
        # Keep only the most recent jobs so the in-memory registry can't grow forever.
        while len(bulk_jobs) > MAX_BULK_JOBS:
            bulk_jobs.pop(next(iter(bulk_jobs)))

    background_tasks.add_task(_run_bulk_job, job_id, rag_pipeline)
    logger.info("Bulk upload queued: job=%s accepted=%d rejected=%d",
                job_id, accepted, rejected)
    return BulkUploadResponse(
        job_id=job_id,
        status="queued",
        files_accepted=accepted,
        files_rejected=rejected,
        message=(
            f"Bulk upload queued: {accepted} file(s) accepted, {rejected} rejected. "
            f"Poll GET /api/upload/bulk/{job_id}"
        ),
    )


# Both poll paths serve the same in-memory job registry: upload-bulk jobs and
# folder-ingest jobs live together in `bulk_jobs`, so one handler backs both.
@app.get("/api/upload/bulk/{job_id}", response_model=BulkJobResponse)
@app.get("/api/ingest/jobs/{job_id}", response_model=BulkJobResponse)
async def get_bulk_job_status(job_id: str):
    """Return the current status of an ingest job (bulk upload or folder ingest)."""
    with _bulk_jobs_lock:
        job = bulk_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Ingest job {job_id} not found")
        # Snapshot under the lock so a concurrent worker can't leave a half-written row.
        snapshot = {
            "job_id": job["id"],
            "status": job["status"],
            "results": [dict(r) for r in job["results"]],
            "succeeded": job["succeeded"],
            "failed": job["failed"],
            "total_chunks": job["total_chunks"],
            "created_at": job["created_at"],
            "finished_at": job["finished_at"],
        }

    return BulkJobResponse(
        job_id=snapshot["job_id"],
        status=snapshot["status"],
        total_files=len(snapshot["results"]),
        succeeded=snapshot["succeeded"],
        failed=snapshot["failed"],
        skipped=sum(1 for r in snapshot["results"] if r["status"] == "skipped"),
        total_chunks=snapshot["total_chunks"],
        files=[BulkFileResult(**r) for r in snapshot["results"]],
        created_at=snapshot["created_at"],
        finished_at=snapshot["finished_at"],
    )


# --- Folder ingest (documents already on the server, no HTTP upload) ---

# Top-level project dirs that must never be scanned as "documents". data/ holds
# the Chroma store + embedding model cache — those .json files are *not* source
# documents and ingesting them would poison the store. uploads_data/ is the
# upload spool (re-ingesting it would duplicate every uploaded file).
_INGEST_FORBIDDEN_ROOTS = {"data", ".git", ".venv", "src", "tests", "uploads_data"}


def _resolve_ingest_folder(folder: str) -> Path:
    """Resolve a user-supplied folder to an absolute path we may scan.

    Keeps reads confined to the project tree (in Docker that is /app — the only
    host folders visible are the bind mounts, of which documents/ is the intended
    one). Raises HTTPException(400) for anything outside it or that looks like an
    internal directory.
    """
    raw = Path(folder).expanduser()
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    try:
        raw = raw.resolve()
        rel = raw.relative_to(PROJECT_ROOT)
    except (OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"Folder must be inside the project root (got: {folder})",
        )
    if not raw.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Folder does not exist or is not a directory: {raw}",
        )
    if not rel.parts or rel.parts[0] in _INGEST_FORBIDDEN_ROOTS:
        raise HTTPException(
            status_code=400,
            detail="Please point at a folder of source documents, e.g. 'documents' "
                   f"(not '{raw.name}').",
        )
    return raw


@app.post("/api/ingest/folder", response_model=BulkUploadResponse)
async def ingest_folder(
    background_tasks: BackgroundTasks,
    request: FolderIngestRequest = FolderIngestRequest(),
):
    """Scan a server-side folder and ingest every supported document in it.

    Unlike the bulk-upload endpoint nothing is transferred over HTTP: files are
    read straight off disk (e.g. the documents/ folder, which Docker bind-mounts
    into the container). Registers the same background job as /api/upload/bulk
    (files processed one at a time, per-file failure isolation) and is polled the
    same way: GET /api/ingest/jobs/{job_id}.
    """
    rag_pipeline = get_rag_pipeline()      # 503 if RAG pipeline is not initialized
    folder = _resolve_ingest_folder(request.folder)

    iterator = folder.rglob("*") if request.recursive else folder.iterdir()
    found = sorted(p for p in iterator if p.is_file())

    results: List[Dict[str, Any]] = []
    paths: List[Optional[Path]] = []
    rejected = 0
    for path in found:
        display = str(path.relative_to(folder))     # keeps subfolder structure readable
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            rejected += 1
            results.append({
                "filename": display,
                "status": "skipped",
                "chunks_created": 0,
                "error": (f"File type {path.suffix.lower() or '(none)'} not supported. "
                          f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"),
            })
            paths.append(None)
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            rejected += 1
            results.append({
                "filename": display,
                "status": "skipped",
                "chunks_created": 0,
                "error": f"File too large (> {MAX_FILE_SIZE // (1024 * 1024)}MB)",
            })
            paths.append(None)
            continue
        results.append({
            "filename": display,
            "status": "pending",
            "chunks_created": 0,
            "error": None,
        })
        paths.append(path)

    accepted = len(results) - rejected
    if accepted == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No supported documents found in {folder}. "
                   f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    job_id = uuid.uuid4().hex[:12]
    with _bulk_jobs_lock:
        bulk_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "chunking_strategy": request.chunking_strategy,
            "results": results,
            "paths": paths,
            "succeeded": 0,
            "failed": 0,
            "total_chunks": 0,
            "created_at": datetime.utcnow().isoformat(),
            "finished_at": None,
        }
        # Keep only the most recent jobs so the in-memory registry can't grow forever.
        while len(bulk_jobs) > MAX_BULK_JOBS:
            bulk_jobs.pop(next(iter(bulk_jobs)))

    background_tasks.add_task(_run_bulk_job, job_id, rag_pipeline)
    logger.info("Folder ingest queued: folder=%s job=%s accepted=%d skipped=%d",
                folder, job_id, accepted, rejected)
    return BulkUploadResponse(
        job_id=job_id,
        status="queued",
        files_accepted=accepted,
        files_rejected=rejected,
        message=(
            f"Folder ingest queued: {accepted} file(s) from {folder}, "
            f"{rejected} skipped. Poll GET /api/ingest/jobs/{job_id}"
        ),
    )