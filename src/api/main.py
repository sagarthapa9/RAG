"""
Updated startup event for main.py with LLM RAG integration
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
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
    
    # Set environment variables for model caching
    cache_dir = os.path.join("./data", "model_cache")
    os.makedirs(cache_dir, exist_ok=True, mode=0o755)
    
    os.environ.setdefault('TRANSFORMERS_CACHE', cache_dir)
    os.environ.setdefault('HF_HOME', cache_dir)
    
    logger.info("Environment setup completed")


def initialize_rag_pipeline() -> RAGPipeline:
    """Initialize the RAG pipeline with error handling"""
    
    try:
        # RAG pipeline configuration
        rag_config = {
            "chunk_size": int(os.getenv("CHUNK_SIZE", "512")),
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
                "chunk_size": 512,
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
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
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