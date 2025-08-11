import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from llm.llm_clients import LLMClient
from rag.retriever import RAGRetriever
from core.data_processor import DealDataProcessor
from models.schemas import DealAnalysisResult
from config.settings import settings
from core.incremental_processor import IncrementalActivityProcessor
from utils.metrics import metrics_collector

logger = logging.getLogger(__name__)

class SalesSentimentAnalyzer:
    """
    Sales sentiment analyzer focused on RAG-based sentiment analysis
    Uses RAG retriever to get relevant examples for context
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        rag_retriever: RAGRetriever,
        data_processor: DealDataProcessor = None
    ):
        """
        Initialize sales sentiment analyzer
        
        Args:
            llm_client: LLM client for sentiment analysis
            rag_retriever: RAG retriever for historical context
            data_processor: Data processor for parsing deal activities
        """
        self.llm_client = llm_client
        self.rag_retriever = rag_retriever
        self.data_processor = data_processor
        self.incremental_processor = IncrementalActivityProcessor()
        
        logger.info("Sales Sentiment Analyzer initialized")
    
    def analyze_deal_sentiment(
        self,
        deal_data: Dict[str, Any],
        include_rag_context: bool = True
    ) -> Dict[str, Any]:
        """
        Perform sentiment analysis on a deal
        
        Args:
            deal_data: Deal data including activities and metadata
            include_rag_context: Whether to include RAG context
            
        Returns:
            Sentiment analysis result
        """
        
        op_name = "analyze_deal_sentiment"
        with metrics_collector.track_operation(op_name):
            try:
                deal_id = deal_data.get('deal_id', 'unknown')
                logger.info(f"Analyzing sales sentiment for deal {deal_id}")
                
                # Always extract RAG metadata directly from deal_data root level
                rag_metadata = {
                    'deal_amount': self._safe_float(deal_data.get('amount', 0)),
                    'deal_stage': deal_data.get('dealstage', 'unknown'),
                    'deal_type': deal_data.get('dealtype', 'unknown'),
                    'deal_probability': self._safe_float(deal_data.get('deal_stage_probability', 0)),
                    'outcome': self._determine_outcome(deal_data.get('dealstage', 'unknown')),
                    'total_activities': len(deal_data.get('activities', [])),
                    'time_span_days': 0,  # Will be calculated if needed
                    'communication_gaps_count': 0  # Will be calculated if needed
                }
                
                # Incremental activity processing
                activities = deal_data.get('activities', [])
                new_activities = self.incremental_processor.detect_new_or_updated_activities(deal_id, activities)
                previous_result = None
                cache_key = f"sales_sentiment_result:{deal_id}"
                # Try to get previous analysis result from cache
                from utils.cache import get_cache_manager
                cache = get_cache_manager()
                previous_result = cache.get(cache_key, default=None, cache_type="incremental")

                if not new_activities and previous_result:
                    logger.info(f"No new activities for deal {deal_id}, returning cached result.")
                    return previous_result

                # Process only new/updated activities
                if self.data_processor:
                    # Create a deal_data copy with only new activities for processing
                    new_deal_data = deal_data.copy()
                    new_deal_data['activities'] = new_activities
                    processed_deal = self.data_processor.process_deal(new_deal_data)
                    processed_activities = [
                        {
                            'activity_type': activity.activity_type,
                            'content': activity.content,
                            'timestamp': activity.timestamp.isoformat() if activity.timestamp else None,
                            'direction': activity.direction,
                            'metadata': activity.metadata
                        }
                        for activity in processed_deal.processed_activities
                    ]
                    # Update RAG metadata with processed metrics
                    rag_metadata.update({
                        'deal_amount': processed_deal.deal_characteristics.deal_amount,
                        'deal_stage': processed_deal.deal_characteristics.deal_stage,
                        'deal_type': processed_deal.deal_characteristics.deal_type,
                        'deal_probability': processed_deal.deal_characteristics.deal_probability,
                        'outcome': processed_deal.deal_characteristics.deal_outcome,
                        'total_activities': processed_deal.deal_metrics.total_activities,
                        'time_span_days': processed_deal.deal_metrics.time_span_days,
                        'communication_gaps_count': processed_deal.deal_metrics.communication_gaps_count
                    })
                    activities_text = processed_deal.combined_text
                else:
                    # Use raw data if no processor
                    processed_activities = [
                        {
                            'activity_type': activity.get('activity_type', 'unknown'),
                            'content': self._extract_raw_activity_content(activity),
                            'timestamp': activity.get('sent_at') or activity.get('createdate') or activity.get('meeting_start_time') or activity.get('lastmodifieddate'),
                            'direction': activity.get('direction') or activity.get('call_direction') or 'unknown',
                            'metadata': activity
                        }
                        for activity in new_activities
                    ]
                    activities_text = self._create_activities_text(new_activities)
                
                # Get RAG context if requested (specify sales analysis type)
                rag_context = ""
                if include_rag_context:
                    try:
                        rag_context = self.rag_retriever.retrieve_relevant_examples(
                            deal_id=deal_id,
                            activities=processed_activities,
                            metadata=rag_metadata,
                            analysis_type="sales"  # Specify sales analysis
                        )
                    except Exception as e:
                        logger.error(f"Error retrieving RAG context: {e}")
                        rag_context = "Error retrieving historical context."
                
                # Analyze sentiment using LLM
                sentiment_result = self.llm_client.analyze_sentiment(
                    deal_id=deal_id,
                    activities_text=activities_text,
                    rag_context=rag_context,
                    activity_frequency=len(processed_activities),
                    total_activities=len(processed_activities)
                )
                
                # Add analysis metadata
                sentiment_result.update({
                    'analysis_metadata': {
                        'deal_id': deal_id,
                        'analysis_timestamp': datetime.utcnow().isoformat(),
                        'llm_provider': self.llm_client.provider.get_provider_name(),
                        'included_rag_context': include_rag_context,
                        'total_activities_analyzed': len(processed_activities),
                        'rag_context_length': len(rag_context) if rag_context else 0,
                        'analysis_type': 'sales_sentiment',
                        'incremental': True,
                        'new_activities_count': len(new_activities)
                    }
                })
                
                logger.info(f"Sales sentiment analysis completed for deal {deal_id}")
                # Merge with previous result if exists
                merged_result = self.incremental_processor.merge_results(previous_result, sentiment_result)
                # Update state in cache
                self.incremental_processor.update_state_after_processing(deal_id, activities)
                cache.set(cache_key, merged_result, ttl=86400, cache_type="incremental")
                return merged_result
            except Exception as e:
                logger.error(f"Error analyzing sales sentiment for deal {deal_data.get('deal_id', 'unknown')}: {e}")
                metrics_collector.record_error(type(e).__name__)
                return {
                    'error': str(e),
                    'deal_id': deal_data.get('deal_id', 'unknown'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'analysis_type': 'sales_sentiment'
                }
    
    def _extract_raw_activity_content(self, activity: Dict[str, Any]) -> str:
        """Extract content from raw activity"""
        activity_type = activity.get('activity_type', 'unknown')
        content_parts = []
        
        if activity_type == 'email':
            subject = (activity.get('subject') or '').strip()
            body = (activity.get('body') or '').strip()
            
            if subject:
                content_parts.append(f"Subject: {subject}")
            if body:
                content_parts.append(f"Body: {body}")
        
        elif activity_type == 'call':
            title = (activity.get('call_title') or '').strip()
            body = (activity.get('call_body') or '').strip()
            
            if title:
                content_parts.append(f"Call: {title}")
            if body:
                content_parts.append(f"Notes: {body}")
        
        elif activity_type == 'meeting':
            title = (activity.get('meeting_title') or '').strip()
            notes = (activity.get('internal_meeting_notes') or '').strip()
            
            if title:
                content_parts.append(f"Meeting: {title}")
            if notes:
                content_parts.append(f"Notes: {notes}")
        
        elif activity_type == 'note':
            body = (activity.get('note_body') or '').strip()
            if body:
                content_parts.append(f"Note: {body}")
        
        elif activity_type == 'task':
            subject = (activity.get('task_subject') or '').strip()
            body = (activity.get('task_body') or '').strip()
            
            if subject:
                content_parts.append(f"Task: {subject}")
            if body:
                content_parts.append(f"Details: {body}")
        
        return '\n'.join(content_parts)
    
    def _create_activities_text(self, activities: List[Dict[str, Any]]) -> str:
        """Create combined activities text from raw activities"""
        
        text_parts = []
        
        for activity in activities:
            activity_type = activity.get('activity_type', 'unknown')
            
            # Extract content based on activity type
            if activity_type == 'email':
                subject = (activity.get('subject') or '').strip()
                body = (activity.get('body') or '').strip()
                if subject:
                    text_parts.append(f"[EMAIL] Subject: {subject}")
                if body:
                    text_parts.append(f"Body: {body}")
            
            elif activity_type == 'call':
                title = (activity.get('call_title') or '').strip()
                body = (activity.get('call_body') or '').strip()
                if title:
                    text_parts.append(f"[CALL] Call: {title}")
                if body:
                    text_parts.append(f"Notes: {body}")
            
            elif activity_type == 'meeting':
                title = (activity.get('meeting_title') or '').strip()
                notes = (activity.get('internal_meeting_notes') or '').strip()
                if title:
                    text_parts.append(f"[MEETING] Meeting: {title}")
                if notes:
                    text_parts.append(f"Notes: {notes}")
            
            elif activity_type == 'note':
                body = (activity.get('note_body') or '').strip()
                if body:
                    text_parts.append(f"[NOTE] Note: {body}")
            
            elif activity_type == 'task':
                subject = (activity.get('task_subject') or '').strip()
                body = (activity.get('task_body') or '').strip()
                if subject:
                    text_parts.append(f"[TASK] Task: {subject}")
                if body:
                    text_parts.append(f"Details: {body}")
        
        return "\n".join(text_parts)
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _determine_outcome(self, deal_stage: str) -> str:
        """Determine deal outcome from stage"""
        stage_lower = deal_stage.lower()
        
        if 'won' in stage_lower or 'closed won' in stage_lower:
            return 'won'
        elif 'lost' in stage_lower or 'closed lost' in stage_lower:
            return 'lost'
        elif 'closed' in stage_lower:
            return 'closed'
        else:
            return 'open'
    
    def analyze_batch_sentiment(
        self,
        deals_data: List[Dict[str, Any]],
        include_rag_context: bool = True
    ) -> Dict[str, Any]:
        """
        Perform batch sentiment analysis on multiple deals
        
        Args:
            deals_data: List of deal data
            include_rag_context: Whether to include RAG context
            
        Returns:
            Batch analysis results
        """
        
        op_name = "analyze_batch_sentiment"
        with metrics_collector.track_operation(op_name):
            start_time = datetime.utcnow()
            
            results = []
            successful_analyses = 0
            failed_analyses = 0
            
            logger.info(f"Starting batch sales sentiment analysis for {len(deals_data)} deals")
            
            for i, deal_data in enumerate(deals_data):
                try:
                    result = self.analyze_deal_sentiment(deal_data, include_rag_context)
                    
                    if 'error' not in result:
                        successful_analyses += 1
                    else:
                        failed_analyses += 1
                    
                    results.append(result)
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(deals_data)} deals")
                    
                except Exception as e:
                    logger.error(f"Error in batch analysis for deal {deal_data.get('deal_id', 'unknown')}: {e}")
                    failed_analyses += 1
                    results.append({
                        'error': str(e),
                        'deal_id': deal_data.get('deal_id', 'unknown'),
                        'timestamp': datetime.utcnow().isoformat(),
                        'analysis_type': 'sales_sentiment'
                    })
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            batch_summary = {
                'total_deals': len(deals_data),
                'successful_analyses': successful_analyses,
                'failed_analyses': failed_analyses,
                'results': results,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.utcnow().isoformat(),
                'analysis_type': 'sales_sentiment_batch'
            }
            
            logger.info(f"Batch sales sentiment analysis completed: {successful_analyses} successful, {failed_analyses} failed")
            return batch_summary
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        
        try:
            return {
                'llm_provider': self.llm_client.provider.get_provider_name(),
                'rag_retriever_stats': self.rag_retriever.get_retrieval_stats(),
                'configuration': {
                    'llm_max_tokens': settings.LLM_MAX_TOKENS,
                    'llm_temperature': settings.LLM_TEMPERATURE,
                    'rag_top_k': settings.RAG_TOP_K,
                    'rag_similarity_threshold': settings.RAG_SIMILARITY_THRESHOLD
                },
                'analysis_type': 'sales_sentiment',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting analyzer stats: {e}")
            return {'error': str(e)}

# Factory function
def create_sales_sentiment_analyzer(
    llm_provider: str = None,
    llm_config: Dict[str, Any] = None
) -> SalesSentimentAnalyzer:
    """Create sales sentiment analyzer instance"""
    
    # Get LLM provider and config from settings if not provided
    if not llm_provider:
        llm_provider = settings.LLM_PROVIDER
    
    if not llm_config:
        llm_config = settings.get_llm_config()
    
    # Create LLM client specifically for sales sentiment analysis
    from llm.llm_clients import create_llm_client
    llm_client = create_llm_client(
        provider_name=llm_provider,
        prompt_file_path="prompts/sales_sentiment_prompt.txt",
        analysis_type="sales",
        **llm_config
    )
    
    # Create RAG retriever with both context builders
    from rag.retriever import create_rag_retriever
    rag_retriever = create_rag_retriever()
    
    # Create data processor
    from core.embedding_service import get_embedding_service
    from core.data_processor import DealDataProcessor
    embedding_service = get_embedding_service()
    data_processor = DealDataProcessor(embedding_service)
    
    return SalesSentimentAnalyzer(
        llm_client=llm_client,
        rag_retriever=rag_retriever,
        data_processor=data_processor
    )