"""
Updated EmbeddingGenerator with lazy loading and error handling
"""

import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import os

logger = logging.getLogger(__name__)

# Model cache directories.
# Previously hardcoded to /app/cache/... which crashed on Windows at import time
# (and hardcoded absolute paths break any non-Docker deployment). Now we:
#   - respect an already-set env var (HF_HOME / TRANSFORMERS_CACHE), else
#   - default to a project-local path under ./data/model_cache, and
#   - create dirs defensively (fall back to a temp dir if unwritable).
# Setup is deferred to EmbeddingGenerator.__init__ so importing this module
# has no filesystem side effects.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "data" / "model_cache"


def _resolve_cache_dir(subdir: str) -> str:
    """Return a writable cache dir for `subdir`, preferring an existing env var."""
    import tempfile

    env_var = "HF_HOME" if subdir == "huggingface" else "TRANSFORMERS_CACHE"
    if env_var in os.environ and os.environ[env_var]:
        return os.environ[env_var]

    for candidate in (
        DEFAULT_MODEL_CACHE / subdir,
        Path(tempfile.gettempdir()) / "rag_model_cache" / subdir,
    ):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)
        except OSError:
            continue
    # Last resort; model loading will fail loudly later rather than on import.
    return str(DEFAULT_MODEL_CACHE / subdir)

class EmbeddingGenerator:
    """
    Embedding generator with lazy loading and fallback options.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 fallback_models: List[str] = None,
                 max_retries: int = 3):
        """
        Initialize embedding generator with lazy loading.
        
        Args:
            model_name: Primary model name to use
            fallback_models: List of fallback models to try
            max_retries: Maximum retry attempts for model loading
        """
        self.model_name = model_name
        self.fallback_models = fallback_models or ['all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L6-v2']
        self.max_retries = max_retries
        self._model = None
        self._model_loaded = False
        self._load_attempted = False
        self._configure_cache()

        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")

    @staticmethod
    def _configure_cache() -> None:
        """Point HF cache env vars at a writable local dir (no-op if already set)."""
        os.environ.setdefault("TRANSFORMERS_CACHE", _resolve_cache_dir("transformers"))
        os.environ.setdefault("HF_HOME", _resolve_cache_dir("huggingface"))
    
    def _load_model(self) -> bool:
        """
        Load the sentence transformer model with fallback options.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._load_attempted:
            return self._model is not None
        
        self._load_attempted = True
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers package not available")
            return False
        
        # List of models to try (primary + fallbacks)
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        
        for model_name in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Attempting to load model: {model_name} (attempt {attempt + 1}/{self.max_retries})")
                    
                    # Try loading the model
                    self._model = SentenceTransformer(model_name)
                    
                    # Test the model with a simple encoding
                    test_embedding = self._model.encode(["test"], show_progress_bar=False)
                    logger.info(f"Successfully loaded model: {model_name}")
                    logger.info(f"Model embedding dimension: {test_embedding.shape[1]}")
                    
                    self._model_loaded = True
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} (attempt {attempt + 1}): {str(e)}")
                    if attempt < self.max_retries - 1:
                        import time
                        time.sleep(1)  # Brief delay before retry
        
        logger.error("All model loading attempts failed")
        return False
    
    @property
    def model(self):
        """Get the model, loading it if necessary."""
        if not self._model_loaded and not self._load_model():
            raise RuntimeError("Failed to load any embedding model")
        return self._model
    
    @property
    def is_available(self) -> bool:
        """Check if the model is available without triggering a load."""
        if self._model_loaded:
            return True
        if not self._load_attempted:
            return self._load_model()
        return self._model is not None
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
            
        Raises:
            RuntimeError: If model cannot be loaded
        """
        if not self.is_available:
            raise RuntimeError("Embedding model is not available")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension or None if model not available
        """
        if not self.is_available:
            return None
        
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            return test_embedding.shape[1]
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {str(e)}")
            return None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "configured_model": self.model_name,
            "fallback_models": self.fallback_models,
            "model_loaded": self._model_loaded,
            "load_attempted": self._load_attempted,
        }
        
        if self._model_loaded and self._model:
            try:
                info.update({
                    "loaded_model_name": getattr(self._model, 'model_name', 'unknown'),
                    "embedding_dimension": self.get_embedding_dimension(),
                    "max_seq_length": getattr(self._model, 'max_seq_length', 'unknown')
                })
            except Exception as e:
                logger.warning(f"Error getting model info: {str(e)}")
        
        return info


# Fallback embedding generator for testing without models
class DummyEmbeddingGenerator:
    """
    Dummy embedding generator for testing when models are not available.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        logger.warning("Using DummyEmbeddingGenerator - embeddings will be random!")
    
    @property
    def is_available(self) -> bool:
        return True
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate random embeddings for testing."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate random normalized embeddings
        embeddings = np.random.randn(len(texts), self.embedding_dim)
        # Normalize to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        return {
            "configured_model": "dummy",
            "embedding_dimension": self.embedding_dim,
            "model_loaded": True,
            "note": "This is a dummy generator for testing only"
        }


def create_embedding_generator(
    model_name: str = 'all-MiniLM-L6-v2',
    allow_dummy: bool = False,
    **kwargs
) -> Union[EmbeddingGenerator, DummyEmbeddingGenerator]:
    """
    Factory function to create an embedding generator.
    
    Args:
        model_name: Model name to use
        allow_dummy: Whether to fall back to dummy generator if models fail
        **kwargs: Additional arguments for EmbeddingGenerator
        
    Returns:
        EmbeddingGenerator instance (real or dummy)
    """
    try:
        generator = EmbeddingGenerator(model_name, **kwargs)
        
        # Try to load the model
        if generator.is_available:
            logger.info(f"Successfully created EmbeddingGenerator with model: {model_name}")
            return generator
        else:
            raise RuntimeError("Model loading failed")
            
    except Exception as e:
        logger.error(f"Failed to create EmbeddingGenerator: {str(e)}")
        
        if allow_dummy:
            logger.warning("Falling back to DummyEmbeddingGenerator")
            return DummyEmbeddingGenerator()
        else:
            raise