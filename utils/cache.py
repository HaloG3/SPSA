import logging
import json
import hashlib
import time
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import threading

from config.settings import settings

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Production-ready cache manager with Redis support
    Handles caching for embeddings, LLM responses, and search results
    """
    
    def __init__(self, redis_url: str = None, default_ttl: int = None):
        """
        Initialize cache manager
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.default_ttl = default_ttl or settings.CACHE_TTL
        self.redis_client = None
        self.enabled = settings.CACHE_ENABLED
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        self.stats_lock = threading.Lock()
        
        if self.enabled:
            self._connect_redis()
        
        logger.info(f"Cache manager initialized (enabled: {self.enabled})")
    
    def _connect_redis(self):
        """Connect to Redis"""
        try:
            import redis
            
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
            
        except ImportError:
            logger.warning("Redis library not installed, caching disabled")
            self.enabled = False
            self.redis_client = None
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, caching disabled")
            self.enabled = False
            self.redis_client = None
    
    def set(self, key: str, value: Any, ttl: int = None, cache_type: str = "general") -> bool:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            cache_type: Type of cache for key prefixing
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            # Prepare cache key with prefix
            cache_key = f"{cache_type}:{key}"
            
            # Serialize value
            serialized_value = json.dumps(value, default=str)
            
            # Set TTL
            ttl = ttl or self.default_ttl
            
            # Store in Redis
            result = self.redis_client.setex(cache_key, ttl, serialized_value)
            
            if result:
                with self.stats_lock:
                    self.stats['sets'] += 1
                logger.debug(f"Cached: {cache_key} (TTL: {ttl}s)")
                return True
            else:
                logger.warning(f"Failed to cache: {cache_key}")
                return False
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
    
    def get(self, key: str, default: Any = None, cache_type: str = "general") -> Any:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            default: Default value if not found
            cache_type: Type of cache for key prefixing
            
        Returns:
            Cached value or default
        """
        
        if not self.enabled or not self.redis_client:
            return default
        
        try:
            # Prepare cache key with prefix
            cache_key = f"{cache_type}:{key}"
            
            # Get from Redis
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data is not None:
                # Deserialize value
                value = json.loads(cached_data)
                
                with self.stats_lock:
                    self.stats['hits'] += 1
                
                logger.debug(f"Cache hit: {cache_key}")
                return value
            
            else:
                with self.stats_lock:
                    self.stats['misses'] += 1
                
                logger.debug(f"Cache miss: {cache_key}")
                return default
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return default
    
    def delete(self, key: str, cache_type: str = "general") -> bool:
        """
        Delete a key from cache
        
        Args:
            key: Cache key
            cache_type: Type of cache for key prefixing
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = f"{cache_type}:{key}"
            result = self.redis_client.delete(cache_key)
            
            if result:
                with self.stats_lock:
                    self.stats['deletes'] += 1
                logger.debug(f"Cache delete: {cache_key}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
    
    def cache_embedding(self, text: str, embedding: list, model_name: str) -> bool:
        """
        Cache embedding for text
        
        Args:
            text: Input text
            embedding: Embedding vector
            model_name: Model name
            
        Returns:
            True if successful
        """
        
        cache_key = self._generate_embedding_key(text, model_name)
        return self.set(cache_key, embedding, ttl=settings.EMBEDDING_CACHE_TTL, cache_type="embedding")
    
    def get_cached_embedding(self, text: str, model_name: str) -> Optional[list]:
        """
        Get cached embedding for text
        
        Args:
            text: Input text
            model_name: Model name
            
        Returns:
            Cached embedding or None
        """
        
        cache_key = self._generate_embedding_key(text, model_name)
        return self.get(cache_key, cache_type="embedding")
    
    def cache_llm_response(
        self, 
        prompt: str, 
        response: str, 
        model_name: str,
        deal_id: str = None,
        analysis_type: str = None,
        include_rag_context: bool = None,
        cache_version: str = "v1"
    ) -> bool:
        """
        Cache LLM response with comprehensive context
        
        Args:
            prompt: LLM prompt
            response: LLM response
            model_name: Model name
            deal_id: Deal ID for context
            analysis_type: Type of analysis (sales/client)
            include_rag_context: Whether RAG context was included
            cache_version: Cache version for invalidation
            
        Returns:
            True if successful
        """
        
        cache_key = self._generate_llm_key(
            prompt=prompt,
            model_name=model_name,
            deal_id=deal_id,
            analysis_type=analysis_type,
            include_rag_context=include_rag_context,
            cache_version=cache_version
        )
        return self.set(cache_key, response, ttl=settings.LLM_CACHE_TTL, cache_type="llm")
    
    def get_cached_llm_response(
        self, 
        prompt: str, 
        model_name: str,
        deal_id: str = None,
        analysis_type: str = None,
        include_rag_context: bool = None,
        cache_version: str = "v1"
    ) -> Optional[str]:
        """
        Get cached LLM response with comprehensive context
        
        Args:
            prompt: LLM prompt
            model_name: Model name
            deal_id: Deal ID for context
            analysis_type: Type of analysis (sales/client)
            include_rag_context: Whether RAG context was included
            cache_version: Cache version for invalidation
            
        Returns:
            Cached response or None
        """
        
        cache_key = self._generate_llm_key(
            prompt=prompt,
            model_name=model_name,
            deal_id=deal_id,
            analysis_type=analysis_type,
            include_rag_context=include_rag_context,
            cache_version=cache_version
        )
        return self.get(cache_key, cache_type="llm")
    
    def _generate_embedding_key(self, text: str, model_name: str) -> str:
        """Generate cache key for embeddings"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{model_name}_{text_hash}"
    
    def _generate_llm_key(
        self,
        prompt: str,
        model_name: str,
        deal_id: str = None,
        analysis_type: str = None,
        include_rag_context: bool = None,
        cache_version: str = "v1"
    ) -> str:
        """
        Generate comprehensive cache key for LLM responses
        
        Args:
            prompt: LLM prompt
            model_name: Model name
            deal_id: Deal ID for context
            analysis_type: Type of analysis (sales/client)
            include_rag_context: Whether RAG context was included
            cache_version: Cache version for invalidation
            
        Returns:
            Unique cache key
        """
        
        # Create context dictionary with all relevant parameters
        context_dict = {
            'prompt_hash': hashlib.md5(prompt.encode('utf-8')).hexdigest(),
            'model_name': model_name,
            'deal_id': deal_id or 'unknown',
            'analysis_type': analysis_type or 'unknown',
            'include_rag_context': include_rag_context,
            'cache_version': cache_version
        }
        
        # Convert to sorted JSON string for consistent hashing
        context_json = json.dumps(context_dict, sort_keys=True, separators=(',', ':'))
        
        # Generate final hash
        cache_key = hashlib.sha256(context_json.encode('utf-8')).hexdigest()
        
        logger.debug(f"Generated cache key: {cache_key[:16]}... for deal {deal_id}, analysis {analysis_type}")
        
        return cache_key
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health check results
        """
        
        if not self.enabled:
            return {
                'status': 'disabled',
                'message': 'Caching is disabled'
            }
        
        if not self.redis_client:
            return {
                'status': 'unhealthy',
                'message': 'Redis client not available'
            }
        
        try:
            # Test Redis connection
            self.redis_client.ping()
            
            # Get Redis info
            info = self.redis_client.info()
            
            return {
                'status': 'healthy',
                'redis_version': info.get('redis_version'),
                'used_memory': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'uptime_seconds': info.get('uptime_in_seconds')
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics
        """
        
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        hit_rate = (stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'enabled': self.enabled,
            'hits': stats['hits'],
            'misses': stats['misses'],
            'sets': stats['sets'],
            'deletes': stats['deletes'],
            'errors': stats['errors'],
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    def clear_cache(self, cache_type: str = None) -> bool:
        """
        Clear cache by type or all
        
        Args:
            cache_type: Type of cache to clear, or None for all
            
        Returns:
            True if successful
        """
        
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            if cache_type:
                # Clear specific cache type
                pattern = f"{cache_type}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(f"Cleared {deleted} keys for cache type: {cache_type}")
                    return True
            else:
                # Clear all cache
                self.redis_client.flushdb()
                logger.info("Cleared all cache")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def create_cache_manager(redis_url: str = None, default_ttl: int = None) -> CacheManager:
    """Create cache manager instance"""
    return CacheManager(redis_url=redis_url, default_ttl=default_ttl)