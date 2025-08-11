import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from llm.llm_clients import LLMClient, create_llm_client
from rag.retriever import RAGRetriever
from core.data_processor import DealDataProcessor
from config.settings import settings
from utils.security import sanitize_input
from core.incremental_processor import IncrementalActivityProcessor
from utils.metrics import metrics_collector

logger = logging.getLogger(__name__)

class ClientSentimentAnalyzer:
    """
    Client sentiment analyzer focused on analyzing client engagement and buying intent
    Uses RAG retriever to get relevant examples for context
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        rag_retriever: RAGRetriever,
        data_processor: DealDataProcessor = None
    ):
        """
        Initialize client sentiment analyzer
        
        Args:
            llm_client: LLM client for sentiment analysis (configured for client sentiment)
            rag_retriever: RAG retriever for historical context
            data_processor: Data processor for parsing deal activities
        """
        self.llm_client = llm_client
        self.rag_retriever = rag_retriever
        self.data_processor = data_processor
        self.incremental_processor = IncrementalActivityProcessor()
        
        logger.info("Client Sentiment Analyzer initialized")
    
    def analyze_client_sentiment(
        self,
        deal_data: Dict[str, Any],
        include_rag_context: bool = True
    ) -> Dict[str, Any]:
        op_name = "analyze_client_sentiment"
        with metrics_collector.track_operation(op_name):
            try:
                deal_id = deal_data.get('deal_id', 'unknown')
                logger.info(f"Analyzing client sentiment for deal {deal_id}")
                
                # Always extract RAG metadata directly from deal_data root level
                rag_metadata = {
                    'deal_amount': self._safe_float(deal_data.get('amount', 0)),
                    'deal_stage': deal_data.get('dealstage', 'unknown'),
                    'deal_type': deal_data.get('dealtype', 'unknown'),
                    'deal_probability': self._safe_float(deal_data.get('deal_stage_probability', 0)),
                    'outcome': self._determine_outcome(deal_data.get('dealstage', 'unknown')),
                    'total_activities': len(deal_data.get('activities', [])),
                    'time_span_days': 0,
                    'communication_gaps_count': 0
                }
                
                # Extract client-initiated activities
                raw_activities = deal_data.get('activities', [])
                client_activities = self._filter_client_activities(raw_activities)
                # Incremental activity processing
                new_activities = self.incremental_processor.detect_new_or_updated_activities(deal_id, client_activities)
                previous_result = None
                cache_key = f"client_sentiment_result:{deal_id}"
                from utils.cache import get_cache_manager
                cache = get_cache_manager()
                previous_result = cache.get(cache_key, default=None, cache_type="incremental")

                if not new_activities and previous_result:
                    logger.info(f"No new client activities for deal {deal_id}, returning cached result.")
                    return previous_result

                # Process only new/updated client activities
                if self.data_processor:
                    client_deal_data = deal_data.copy()
                    client_deal_data['activities'] = new_activities
                    processed_deal = self.data_processor.process_deal(client_deal_data)
                    activities = [
                        {
                            'activity_type': activity.activity_type,
                            'content': activity.content,
                            'timestamp': activity.timestamp.isoformat() if activity.timestamp else None,
                            'direction': activity.direction,
                            'metadata': activity.metadata
                        }
                        for activity in processed_deal.processed_activities
                    ]
                    rag_metadata.update({
                        'total_activities': len(client_activities),
                        'time_span_days': processed_deal.deal_metrics.time_span_days,
                        'communication_gaps_count': processed_deal.deal_metrics.communication_gaps_count
                    })
                    activities_text = processed_deal.combined_text
                else:
                    activities = [
                        {
                            'activity_type': activity.get('activity_type', 'unknown'),
                            'content': self._extract_raw_activity_content(activity),
                            'timestamp': activity.get('sent_at') or activity.get('createdate') or activity.get('meeting_start_time') or activity.get('lastmodifieddate'),
                            'direction': activity.get('direction') or activity.get('call_direction') or 'unknown',
                            'metadata': activity
                        }
                        for activity in new_activities
                    ]
                    activities_text = self._create_client_activities_text(new_activities)
                
                # Sanitize activities text before sending to LLM
                from utils.security import sanitize_input
                sanitized_activities_text = sanitize_input(activities_text, f"client_sentiment_activities_{deal_id}").cleaned_content
                
                # Get RAG context if requested (specify client analysis type)
                rag_context = ""
                if include_rag_context:
                    try:
                        all_activities = [
                            {
                                'activity_type': activity.get('activity_type', 'unknown'),
                                'content': self._extract_raw_activity_content(activity),
                                'timestamp': activity.get('sent_at') or activity.get('createdate'),
                                'direction': activity.get('direction') or 'unknown',
                                'metadata': activity
                            }
                            for activity in raw_activities
                        ]
                        rag_context = self.rag_retriever.retrieve_relevant_examples(
                            deal_id=deal_id,
                            activities=all_activities,
                            metadata=rag_metadata,
                            analysis_type="client"
                        )
                    except Exception as e:
                        logger.error(f"Error retrieving RAG context: {e}")
                        rag_context = "Error retrieving historical context."
                
                # Sanitize RAG context
                sanitized_rag_context = sanitize_input(rag_context, f"client_sentiment_rag_{deal_id}").cleaned_content
                
                # Analyze client sentiment using client-specific LLM
                sentiment_result = self.llm_client.analyze_sentiment(
                    deal_id=deal_id,
                    activities_text=sanitized_activities_text,
                    rag_context=sanitized_rag_context,
                    activity_frequency=len(new_activities),
                    total_activities=len(new_activities)
                )
                
                # Add client-specific analysis metadata
                sentiment_result.update({
                    'analysis_metadata': {
                        'deal_id': deal_id,
                        'analysis_timestamp': datetime.utcnow().isoformat(),
                        'llm_provider': self.llm_client.provider.get_provider_name(),
                        'included_rag_context': include_rag_context,
                        'total_client_activities_analyzed': len(new_activities),
                        'total_deal_activities': len(raw_activities),
                        'rag_context_length': len(rag_context) if rag_context else 0,
                        'analysis_type': 'client_sentiment',
                        'client_activity_filter_applied': True,
                        'incremental': True,
                        'new_activities_count': len(new_activities)
                    }
                })
                
                logger.info(f"Client sentiment analysis completed for deal {deal_id}")
                # Merge with previous result if exists
                merged_result = self.incremental_processor.merge_results(previous_result, sentiment_result)
                # Update state in cache
                self.incremental_processor.update_state_after_processing(deal_id, client_activities)
                cache.set(cache_key, merged_result, ttl=86400, cache_type="incremental")
                return merged_result
            except Exception as e:
                logger.error(f"Error analyzing client sentiment for deal {deal_data.get('deal_id', 'unknown')}: {e}")
                metrics_collector.record_error(type(e).__name__)
                return {
                    'error': str(e),
                    'deal_id': deal_data.get('deal_id', 'unknown'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'analysis_type': 'client_sentiment'
                }
    
    def _filter_client_activities(self, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter activities to include only client-initiated or client-focused activities
        
        Args:
            activities: All deal activities
            
        Returns:
            Filtered client activities
        """
        
        client_activities = []
        
        for activity in activities:
            activity_type = activity.get('activity_type', 'unknown')
            direction = activity.get('direction', '').lower()
            
            # Include client-initiated activities
            if direction == 'incoming':
                client_activities.append(activity)
            
            # Include meetings (client participation)
            elif activity_type == 'meeting':
                client_activities.append(activity)
            
            # Include calls that are incoming or client-initiated
            elif activity_type == 'call':
                call_direction = activity.get('call_direction', '').lower()
                if call_direction == 'inbound' or direction == 'incoming':
                    client_activities.append(activity)
        
        logger.info(f"Filtered {len(client_activities)} client activities from {len(activities)} total activities")
        return client_activities
    
    def _extract_raw_activity_content(self, activity: Dict[str, Any]) -> str:
        """Extract content from raw activity (same as sales analyzer)"""
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
    
    def _create_client_activities_text(self, activities: List[Dict[str, Any]]) -> str:
        """Create combined activities text from client activities"""
        
        text_parts = []
        
        for activity in activities:
            activity_type = activity.get('activity_type', 'unknown')
            
            if activity_type == 'email':
                subject = (activity.get('subject') or '').strip()
                body = (activity.get('body') or '').strip()
                if subject:
                    text_parts.append(f"[CLIENT EMAIL] Subject: {subject}")
                if body:
                    text_parts.append(f"Body: {body}")
            
            elif activity_type == 'call':
                title = (activity.get('call_title') or '').strip()
                body = (activity.get('call_body') or '').strip()
                if title:
                    text_parts.append(f"[CLIENT CALL] Call: {title}")
                if body:
                    text_parts.append(f"Notes: {body}")
            
            elif activity_type == 'meeting':
                title = (activity.get('meeting_title') or '').strip()
                notes = (activity.get('internal_meeting_notes') or '').strip()
                if title:
                    text_parts.append(f"[CLIENT MEETING] Meeting: {title}")
                if notes:
                    text_parts.append(f"Notes: {notes}")
        
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
    
    def analyze_batch_client_sentiment(
        self,
        deals_data: List[Dict[str, Any]],
        include_rag_context: bool = True
    ) -> Dict[str, Any]:
        op_name = "analyze_batch_client_sentiment"
        with metrics_collector.track_operation(op_name):
            start_time = datetime.utcnow()
            
            results = []
            successful_analyses = 0
            failed_analyses = 0
            
            logger.info(f"Starting batch client sentiment analysis for {len(deals_data)} deals")
            
            for i, deal_data in enumerate(deals_data):
                try:
                    result = self.analyze_client_sentiment(deal_data, include_rag_context)
                    
                    if 'error' not in result:
                        successful_analyses += 1
                    else:
                        failed_analyses += 1
                    
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(deals_data)} deals")
                    
                except Exception as e:
                    logger.error(f"Error in batch client analysis for deal {deal_data.get('deal_id', 'unknown')}: {e}")
                    failed_analyses += 1
                    results.append({
                        'error': str(e),
                        'deal_id': deal_data.get('deal_id', 'unknown'),
                        'timestamp': datetime.utcnow().isoformat(),
                        'analysis_type': 'client_sentiment'
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
                'analysis_type': 'client_sentiment_batch'
            }
            
            logger.info(f"Batch client sentiment analysis completed: {successful_analyses} successful, {failed_analyses} failed")
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
                'analysis_type': 'client_sentiment',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting client analyzer stats: {e}")
            return {'error': str(e)}

# Factory function
def create_client_sentiment_analyzer(
    llm_provider: str = None,
    llm_config: Dict[str, Any] = None
) -> ClientSentimentAnalyzer:
    """Create client sentiment analyzer instance"""
    
    if not llm_provider:
        llm_provider = settings.LLM_PROVIDER
    
    if not llm_config:
        llm_config = settings.get_llm_config()
    
    # Create LLM client specifically for client sentiment analysis
    llm_client = create_llm_client(
        provider_name=llm_provider, 
        prompt_file_path="prompts/client_sentiment_prompt.txt",
        analysis_type="client",
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
    
    return ClientSentimentAnalyzer(
        llm_client=llm_client,
        rag_retriever=rag_retriever,
        data_processor=data_processor
    )