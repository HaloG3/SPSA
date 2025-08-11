import logging
from typing import List, Union, Optional
from abc import ABC, abstractmethod
import numpy as np

from config.settings import settings
from utils.cache import get_cache_manager

logger = logging.getLogger(__name__)

class EmbeddingService(ABC):
    """Abstract base class for embedding services"""
    
    @abstractmethod
    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name for caching"""
        pass

class SentenceTransformerEmbeddingService(EmbeddingService):
    """Sentence Transformers embedding service (free, local)"""
    def __init__(self, model_name: str = None, batch_size: int = 32):
        self.model_name = model_name or settings.SENTENCE_TRANSFORMER_MODEL
        self.cache_manager = get_cache_manager()
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"SentenceTransformer model loaded: {self.model_name}")
        except ImportError:
            raise ImportError("sentence-transformers library required. Install with: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")

    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using SentenceTransformer (batch-optimized)"""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        all_embeddings = [None] * len(texts)
        # Check cache for each text
        cached_embeddings = []
        texts_to_encode = []
        text_indices = []
        for i, txt in enumerate(texts):
            cached_embedding = self.cache_manager.get_cached_embedding(txt, self.model_name)
            if cached_embedding is not None:
                all_embeddings[i] = cached_embedding
            else:
                texts_to_encode.append(txt)
                text_indices.append(i)
        # Batch encode uncached texts
        if texts_to_encode:
            try:
                for batch_start in range(0, len(texts_to_encode), self.batch_size):
                    batch = texts_to_encode[batch_start:batch_start + self.batch_size]
                    batch_embeddings = self.model.encode(batch, convert_to_tensor=False)
                    batch_embeddings = [emb.tolist() for emb in batch_embeddings]
                    for txt, emb in zip(batch, batch_embeddings):
                        idx = text_indices[texts_to_encode.index(txt)]
                        all_embeddings[idx] = emb
                        self.cache_manager.cache_embedding(txt, emb, self.model_name)
            except Exception as e:
                logger.error(f"Error encoding texts: {e}")
                raise
        return all_embeddings[0] if is_single else all_embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Get model name for caching"""
        return self.model_name

class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service"""
    def __init__(self, api_key: str = None, model: str = None, batch_size: int = 32):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_EMBEDDING_MODEL
        self.cache_manager = get_cache_manager()
        self.batch_size = batch_size
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI embedding service initialized with model: {self.model}")
        except ImportError:
            raise ImportError("openai library required. Install with: pip install openai")

    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using OpenAI (batch-optimized)"""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        all_embeddings = [None] * len(texts)
        # Check cache for each text
        cached_embeddings = []
        texts_to_encode = []
        text_indices = []
        for i, txt in enumerate(texts):
            cached_embedding = self.cache_manager.get_cached_embedding(txt, self.model)
            if cached_embedding is not None:
                all_embeddings[i] = cached_embedding
            else:
                texts_to_encode.append(txt)
                text_indices.append(i)
        # Batch encode uncached texts
        if texts_to_encode:
            try:
                for batch_start in range(0, len(texts_to_encode), self.batch_size):
                    batch = texts_to_encode[batch_start:batch_start + self.batch_size]
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        encoding_format="float"
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    for txt, emb in zip(batch, batch_embeddings):
                        idx = text_indices[texts_to_encode.index(txt)]
                        all_embeddings[idx] = emb
                        self.cache_manager.cache_embedding(txt, emb, self.model)
            except Exception as e:
                logger.error(f"Error with OpenAI embedding API: {e}")
                raise
        return all_embeddings[0] if is_single else all_embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension for OpenAI models"""
        dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)
    
    def get_model_name(self) -> str:
        """Get model name for caching"""
        return self.model

class HuggingFaceEmbeddingService(EmbeddingService):
    """HuggingFace embedding service"""
    def __init__(self, model_name: str = None, batch_size: int = 32):
        self.model_name = model_name or settings.HUGGINGFACE_MODEL
        self.cache_manager = get_cache_manager()
        self.batch_size = batch_size
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            logger.info(f"HuggingFace model loaded: {self.model_name}")
        except ImportError:
            raise ImportError("transformers and torch libraries required. Install with: pip install transformers torch")

    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using HuggingFace model (batch-optimized)"""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        all_embeddings = [None] * len(texts)
        # Check cache for each text
        cached_embeddings = []
        texts_to_encode = []
        text_indices = []
        for i, txt in enumerate(texts):
            cached_embedding = self.cache_manager.get_cached_embedding(txt, self.model_name)
            if cached_embedding is not None:
                all_embeddings[i] = cached_embedding
            else:
                texts_to_encode.append(txt)
                text_indices.append(i)
        # Batch encode uncached texts
        if texts_to_encode:
            try:
                import torch
                for batch_start in range(0, len(texts_to_encode), self.batch_size):
                    batch = texts_to_encode[batch_start:batch_start + self.batch_size]
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                    batch_embeddings = embeddings.cpu().numpy().tolist()
                    for txt, emb in zip(batch, batch_embeddings):
                        idx = text_indices[texts_to_encode.index(txt)]
                        all_embeddings[idx] = emb
                        self.cache_manager.cache_embedding(txt, emb, self.model_name)
            except Exception as e:
                logger.error(f"Error encoding texts with HuggingFace: {e}")
                raise
        return all_embeddings[0] if is_single else all_embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.config.hidden_size
    
    def get_model_name(self) -> str:
        """Get model name for caching"""
        return self.model_name

# Global embedding service instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get global embedding service instance"""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = create_embedding_service()
    
    return _embedding_service

def create_embedding_service(
    service_type: str = None,
    config: dict = None
) -> EmbeddingService:
    """
    Create embedding service instance
    
    Args:
        service_type: Type of embedding service
        config: Configuration for the service
        
    Returns:
        Embedding service instance
    """
    
    service_type = service_type or settings.EMBEDDING_SERVICE
    config = config or settings.get_embedding_config()
    
    logger.info(f"Creating embedding service: {service_type}")
    
    if service_type == "sentence_transformers":
        return SentenceTransformerEmbeddingService(
            model_name=config.get("model")
        )
    elif service_type == "openai":
        return OpenAIEmbeddingService(
            api_key=config.get("api_key"),
            model=config.get("model")
        )
    elif service_type == "huggingface":
        return HuggingFaceEmbeddingService(
            model_name=config.get("model")
        )
    else:
        raise ValueError(f"Unknown embedding service type: {service_type}")