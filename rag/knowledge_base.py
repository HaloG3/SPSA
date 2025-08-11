import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import hashlib

from core.embedding_service import EmbeddingService
from core.vector_store import VectorStore
from core.data_processor import DealDataProcessor
from models.schemas import DealPattern, ProcessedDealData
from config.settings import settings

logger = logging.getLogger(__name__)

class KnowledgeBaseBuilder:
    """
    Knowledge base builder for processing deals and building searchable index
    Focused on creating a comprehensive knowledge base for RAG retrieval
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        data_processor: DealDataProcessor = None
    ):
        """
        Initialize knowledge base builder
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector database for storing patterns
            data_processor: Data processor for deal activities
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.data_processor = data_processor or DealDataProcessor(embedding_service)
        
        # Configuration from settings
        self.batch_size = settings.KB_BATCH_SIZE
        self.enable_incremental = settings.KB_ENABLE_INCREMENTAL
        
        logger.info("Knowledge Base Builder initialized")
    
    def build_knowledge_base(
        self,
        data_file_path: str = None,
        rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Build knowledge base from deal data
        
        Args:
            data_file_path: Path to deal data file
            rebuild: Whether to rebuild existing knowledge base
            
        Returns:
            Build results and statistics
        """
        
        data_file_path = data_file_path or settings.DATA_PATH
        
        try:
            logger.info(f"Starting knowledge base build from {data_file_path}")
            
            # Load deal data
            deal_data = self._load_deal_data(data_file_path)
            
            if not deal_data:
                raise ValueError("No deal data found")
            
            # Check if rebuild is needed
            if not rebuild and not self._should_rebuild(deal_data):
                logger.info("Knowledge base is up to date")
                return {
                    'status': 'up_to_date',
                    'total_deals': len(deal_data),
                    'message': 'Knowledge base is already up to date'
                }
            
            # Clear existing data if rebuilding
            if rebuild:
                logger.info("Clearing existing knowledge base")
                self.vector_store.clear()
            
            # Process deals in batches
            build_stats = self._process_deals_in_batches(deal_data)
            
            # Create build summary
            build_summary = {
                'status': 'completed',
                'total_deals_processed': build_stats['total_processed'],
                'successful_embeddings': build_stats['successful'],
                'failed_embeddings': build_stats['failed'],
                'build_duration_seconds': build_stats['duration'],
                'knowledge_base_statistics': self.vector_store.get_stats(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Knowledge base build completed: {build_summary}")
            return build_summary
            
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _load_deal_data(self, data_file_path: str) -> List[Dict[str, Any]]:
        """Load deal data from file"""
        
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} deals from {data_file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading deal data: {e}")
            raise
    
    def _should_rebuild(self, deal_data: List[Dict[str, Any]]) -> bool:
        """Check if knowledge base should be rebuilt"""
        
        if not self.enable_incremental:
            return True
        
        try:
            # Check if vector store has data
            stats = self.vector_store.get_stats()
            stored_count = stats.get('total_vectors', 0)
            
            # If no data in vector store, rebuild
            if stored_count == 0:
                return True
            
            # Check data hash to detect changes
            current_hash = self._calculate_data_hash(deal_data)
            stored_hash = self._get_stored_data_hash()
            
            if current_hash != stored_hash:
                logger.info("Data has changed, rebuilding knowledge base")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking rebuild status: {e}")
            return True
    
    def _calculate_data_hash(self, deal_data: List[Dict[str, Any]]) -> str:
        """Calculate hash of deal data for change detection"""
        
        try:
            data_str = json.dumps(deal_data, sort_keys=True)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating data hash: {e}")
            return str(datetime.utcnow().timestamp())
    
    def _get_stored_data_hash(self) -> str:
        """Get stored data hash for comparison"""
        
        try:
            # Try to get hash from vector store metadata
            stats = self.vector_store.get_stats()
            return stats.get('data_hash', '')
        except Exception:
            return ''
    
    def _process_deals_in_batches(self, deal_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process deals in batches for efficiency"""
        
        start_time = datetime.utcnow()
        
        stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'duration': 0
        }
        
        # Process in batches
        for i in range(0, len(deal_data), self.batch_size):
            batch = deal_data[i:i + self.batch_size]
            
            logger.info(f"Processing batch {i//self.batch_size + 1} ({len(batch)} deals)")
            
            batch_results = self._process_deal_batch(batch)
            
            stats['total_processed'] += batch_results['processed']
            stats['successful'] += batch_results['successful']
            stats['failed'] += batch_results['failed']
        
        # Calculate duration
        end_time = datetime.utcnow()
        stats['duration'] = (end_time - start_time).total_seconds()
        
        return stats
    
    def _process_deal_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of deals with batch embedding generation"""
        batch_stats = {
            'processed': len(batch),
            'successful': 0,
            'failed': 0
        }
        deal_patterns = []
        processed_deals = []
        texts_to_embed = []
        deal_indices = []
        # First, process deals to extract combined_texts
        for idx, deal in enumerate(batch):
            try:
                processed_deal = self.data_processor.process_deal(deal)
                processed_deals.append(processed_deal)
                texts_to_embed.append(processed_deal.combined_text)
                deal_indices.append(idx)
            except Exception as e:
                logger.error(f"Error processing deal {deal.get('deal_id', 'unknown')}: {e}")
                batch_stats['failed'] += 1
                processed_deals.append(None)
        # Batch embed all combined_texts
        embeddings = []
        if texts_to_embed:
            try:
                embeddings = self.embedding_service.encode(texts_to_embed)
            except Exception as e:
                logger.error(f"Error batch embedding: {e}")
                # If batch fails, fallback to single embedding per text
                embeddings = []
                for text in texts_to_embed:
                    try:
                        emb = self.embedding_service.encode(text)
                        embeddings.append(emb)
                    except Exception as e2:
                        logger.error(f"Error embedding single text: {e2}")
                        embeddings.append(None)
        # Assign embeddings to processed_deals
        emb_idx = 0
        for idx, processed_deal in enumerate(processed_deals):
            if processed_deal is not None:
                embedding = embeddings[emb_idx] if emb_idx < len(embeddings) else None
                processed_deal.embedding = embedding
                try:
                    deal_pattern = self._create_deal_pattern(processed_deal)
                    deal_patterns.append(deal_pattern)
                    batch_stats['successful'] += 1
                except Exception as e:
                    logger.error(f"Error creating deal pattern: {e}")
                    batch_stats['failed'] += 1
                emb_idx += 1
        # Store batch in vector database
        if deal_patterns:
            try:
                self.vector_store.add_patterns(deal_patterns)
                logger.debug(f"Stored {len(deal_patterns)} deal patterns")
            except Exception as e:
                logger.error(f"Error storing deal patterns: {e}")
                batch_stats['failed'] += len(deal_patterns)
                batch_stats['successful'] -= len(deal_patterns)
        return batch_stats
    
    def _create_deal_pattern(self, processed_deal: ProcessedDealData) -> DealPattern:
        """Create deal pattern for vector storage"""
        
        return DealPattern(
            deal_id=processed_deal.deal_id,
            combined_text=processed_deal.combined_text,
            activities_count=len(processed_deal.processed_activities),
            activity_types=[activity.activity_type for activity in processed_deal.processed_activities],
            time_span_days=processed_deal.deal_metrics.time_span_days,
            metadata={
                'deal_amount': processed_deal.deal_characteristics.deal_amount,
                'deal_stage': processed_deal.deal_characteristics.deal_stage,
                'deal_type': processed_deal.deal_characteristics.deal_type,
                'deal_probability': processed_deal.deal_characteristics.deal_probability,
                'outcome': processed_deal.deal_characteristics.deal_outcome,
                'is_won': processed_deal.deal_characteristics.is_won,
                'is_lost': processed_deal.deal_characteristics.is_lost,
                'total_activities': processed_deal.deal_metrics.total_activities,
                'response_time_avg': processed_deal.deal_metrics.response_time_metrics.get('avg_response_time_hours', 0),
                'communication_gaps': processed_deal.deal_metrics.communication_gaps_count,
                'business_hours_ratio': processed_deal.deal_metrics.business_hours_ratio,
                'activity_frequency_trend': processed_deal.deal_metrics.activity_frequency_trend,
                'processing_timestamp': processed_deal.processing_timestamp.isoformat()
            },
            embedding=processed_deal.embedding
        )
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        
        try:
            vector_stats = self.vector_store.get_stats()
            
            return {
                'vector_store_stats': vector_stats,
                'configuration': {
                    'batch_size': self.batch_size,
                    'incremental_enabled': self.enable_incremental,
                    'embedding_service': settings.EMBEDDING_SERVICE,
                    'vector_db': settings.VECTOR_DB
                },
                'data_file': settings.DATA_PATH,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {'error': str(e)}
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear knowledge base"""
        
        try:
            logger.warning("Clearing knowledge base")
            self.vector_store.clear()
            
            return {
                'status': 'cleared',
                'message': 'Knowledge base cleared successfully',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def add_deal_to_knowledge_base(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add single deal to knowledge base"""
        
        try:
            # Process deal
            processed_deal = self.data_processor.process_deal(deal_data)
            
            # Create deal pattern
            deal_pattern = self._create_deal_pattern(processed_deal)
            
            # Add to vector store
            self.vector_store.add_patterns([deal_pattern])
            
            logger.info(f"Added deal {deal_data.get('deal_id', 'unknown')} to knowledge base")
            
            return {
                'status': 'added',
                'deal_id': processed_deal.deal_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error adding deal to knowledge base: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Factory function
def create_knowledge_base_builder(
    embedding_service: EmbeddingService = None,
    vector_store: VectorStore = None,
    data_processor: DealDataProcessor = None
) -> KnowledgeBaseBuilder:
    """Create knowledge base builder instance"""
    
    if not embedding_service:
        from core.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
    
    if not vector_store:
        from core.vector_store import get_vector_store
        vector_store = get_vector_store()
    
    if not data_processor:
        from core.data_processor import DealDataProcessor
        data_processor = DealDataProcessor(embedding_service)
    
    return KnowledgeBaseBuilder(
        embedding_service=embedding_service,
        vector_store=vector_store,
        data_processor=data_processor
    )