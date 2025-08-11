import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json
from pathlib import Path

from models.schemas import ProcessedActivity, DealPattern, ProcessedDealData, DealMetrics, DealCharacteristics
from core.embedding_service import EmbeddingService
from config.settings import settings
from utils.security import SecurityMiddleware, get_default_sanitizer

logger = logging.getLogger(__name__)

class DealDataProcessor:
    """Simplified deal data processor focused on core functionality"""
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize processor with embedding service
        
        Args:
            embedding_service: Initialized embedding service instance
        """
        self.embedding_service = embedding_service
        self.security_middleware = SecurityMiddleware(get_default_sanitizer())
        logger.info("Deal Data Processor initialized with security middleware")
    
    def load_deal_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load deal data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} deals from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def process_deal(self, deal_data: Dict[str, Any]) -> ProcessedDealData:
        """
        Process a single deal into structured format
        
        Args:
            deal_data: Raw deal data
            
        Returns:
            Processed deal data
        """
        
        # Sanitize input data to prevent prompt injection attacks
        sanitized_deal_data = self.security_middleware.sanitize_deal_data(deal_data)
        
        # logger.info(sanitized_deal_data)

        deal_id = str(sanitized_deal_data.get('deal_id', 'unknown'))
        
        try:
            # Process activities
            activities = sanitized_deal_data.get('activities', [])
            processed_activities = [
                self._parse_activity(activity, deal_id) 
                for activity in activities
            ]
            
            # Filter out None activities
            processed_activities = [a for a in processed_activities if a is not None]
            
            # Calculate metrics
            deal_metrics = self._calculate_deal_metrics(processed_activities, sanitized_deal_data)
            
            # Extract characteristics
            deal_characteristics = self._extract_deal_characteristics(sanitized_deal_data, deal_metrics)
            
            # logger.info(f"Deal id: {deal_id} -- {deal_characteristics}")

            # Create combined text
            combined_text = self._create_combined_text(processed_activities)
            
            # Generate embedding
            embedding = None
            if combined_text.strip():
                try:
                    embedding = self.embedding_service.encode(combined_text)
                except Exception as e:
                    logger.error(f"Error generating embedding for deal {deal_id}: {e}")
            
            return ProcessedDealData(
                deal_id=deal_id,
                raw_deal_data=sanitized_deal_data,
                processed_activities=processed_activities,
                deal_metrics=deal_metrics,
                deal_characteristics=deal_characteristics,
                combined_text=combined_text,
                embedding=embedding,
                processing_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error processing deal {deal_id}: {e}")
            raise
    
    def _parse_activity(self, activity: Dict[str, Any], deal_id: str) -> Optional[ProcessedActivity]:
        """Parse a single activity into structured format"""
        
        try:
            activity_type = activity.get('activity_type', 'unknown')
            
            # Extract content based on activity type
            content = self._extract_activity_content(activity)
            
            if not content.strip():
                return None
            
            # Extract timestamp
            timestamp = self._extract_activity_timestamp(activity)
            
            # Extract direction
            direction = activity.get('direction') or activity.get('call_direction') or 'unknown'
            
            # Create metadata based on activity type
            metadata = {
                'original_activity_type': activity_type
            }
            
            # Add type-specific metadata based on your actual structure
            if activity_type == 'email':
                metadata.update({
                    'sent_at': activity.get('sent_at'),
                    'from': activity.get('from'),
                    'to': activity.get('to'),
                    'subject': activity.get('subject'),
                    'state': activity.get('state'),
                    'direction': activity.get('direction')
                })
            elif activity_type == 'call':
                metadata.update({
                    'id': activity.get('id'),
                    'createdate': activity.get('createdate'),
                    'call_title': activity.get('call_title'),
                    'call_direction': activity.get('call_direction'),
                    'call_duration': activity.get('call_duration'),
                    'call_status': activity.get('call_status')
                })
            elif activity_type == 'meeting':
                metadata.update({
                    'id': activity.get('id'),
                    'meeting_title': activity.get('meeting_title'),
                    'meeting_location': activity.get('meeting_location'),
                    'meeting_location_type': activity.get('meeting_location_type'),
                    'meeting_outcome': activity.get('meeting_outcome'),
                    'meeting_start_time': activity.get('meeting_start_time'),
                    'meeting_end_time': activity.get('meeting_end_time')
                })
            elif activity_type == 'note':
                metadata.update({
                    'id': activity.get('id'),
                    'createdate': activity.get('createdate'),
                    'lastmodifieddate': activity.get('lastmodifieddate')
                })
            elif activity_type == 'task':
                metadata.update({
                    'id': activity.get('id'),
                    'createdate': activity.get('createdate'),
                    'task_priority': activity.get('task_priority'),
                    'task_status': activity.get('task_status'),
                    'task_type': activity.get('task_type'),
                    'task_subject': activity.get('task_subject')
                })
            
            return ProcessedActivity(
                deal_id=deal_id,
                activity_type=activity_type,
                timestamp=timestamp,
                content=content,
                direction=direction,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Error parsing activity for deal {deal_id}: {e}")
            return None
    
    def _extract_activity_content(self, activity: Dict[str, Any]) -> str:
        """Extract content from activity based on type"""
        
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
        
        # Combine content parts
        combined_content = '\n'.join(content_parts)
        
        # Apply additional sanitization to the extracted content
        if combined_content.strip():
            sanitized_content = self.security_middleware.sanitize_activity_content(
                combined_content, 
                activity_type
            )
            return sanitized_content
        
        return combined_content
    
    def _extract_activity_timestamp(self, activity: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from activity"""
        
        activity_type = activity.get('activity_type', 'unknown')
        
        # Define timestamp fields for each activity type
        timestamp_fields = {
            'email': 'sent_at',
            'call': 'createdate',
            'meeting': 'meeting_start_time',
            'note': 'lastmodifieddate',
            'task': 'createdate'
        }
        
        timestamp_field = timestamp_fields.get(activity_type, 'createdate')
        timestamp_str = activity.get(timestamp_field)
        
        if not timestamp_str:
            return None
        
        try:
            # Handle timezone - if no timezone, assume UTC
            if timestamp_str.endswith('Z'):
                return datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
            elif '+' in timestamp_str or timestamp_str.count('-') > 2:
                return datetime.fromisoformat(timestamp_str)
            else:
                # No timezone info, assume UTC
                return datetime.fromisoformat(timestamp_str + '+00:00')
        except:
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return None
    
    def _calculate_deal_metrics(self, activities: List[ProcessedActivity], deal_data: Dict[str, Any]) -> DealMetrics:
        """Calculate comprehensive metrics for a deal"""
        
        deal_id = deal_data.get('deal_id', 'unknown')
        
        # Basic counts
        total_activities = len(activities)
        activity_type_counts = {}
        
        for activity in activities:
            activity_type = activity.activity_type
            activity_type_counts[activity_type] = activity_type_counts.get(activity_type, 0) + 1
        
        # Time span calculation
        timestamps = [a.timestamp for a in activities if a.timestamp]
        if timestamps:
            timestamps.sort()
            time_span = (timestamps[-1] - timestamps[0]).days
        else:
            time_span = 0
        
        # Calculate average time between activities
        avg_time_between_activities_hours = 0.0
        if len(timestamps) > 1:
            total_hours = sum(
                (timestamps[i] - timestamps[i-1]).total_seconds() / 3600 
                for i in range(1, len(timestamps))
            )
            avg_time_between_activities_hours = total_hours / (len(timestamps) - 1)
        
        # Calculate response time metrics
        response_time_metrics = self._calculate_response_times(activities)
        
        # Calculate communication gaps (periods > 7 days without activity)
        communication_gaps_count = 0
        for i in range(1, len(timestamps)):
            gap_days = (timestamps[i] - timestamps[i-1]).days
            if gap_days > 7:
                communication_gaps_count += 1
        
        # Calculate business hours ratio
        business_hours_ratio = self._calculate_business_hours_ratio(activities)
        
        # Calculate weekend activity ratio
        weekend_activity_ratio = self._calculate_weekend_ratio(activities)
        
        # Determine activity frequency trend
        activity_frequency_trend = self._calculate_frequency_trend(activities)
        
        # Calculate email ratio
        email_ratio = activity_type_counts.get('email', 0) / max(total_activities, 1)
        
        return DealMetrics(
            deal_id=deal_id,
            total_activities=total_activities,
            activity_type_counts=activity_type_counts,
            time_span_days=time_span,
            avg_time_between_activities_hours=avg_time_between_activities_hours,
            response_time_metrics=response_time_metrics,
            communication_gaps_count=communication_gaps_count,
            business_hours_ratio=business_hours_ratio,
            weekend_activity_ratio=weekend_activity_ratio,
            activity_frequency_trend=activity_frequency_trend,
            email_ratio=email_ratio
        )
    
    def _calculate_response_times(self, activities: List[ProcessedActivity]) -> Dict[str, float]:
        """Calculate response time metrics"""
        response_times = []
        
        # Sort activities by timestamp
        sorted_activities = sorted(
            [a for a in activities if a.timestamp], 
            key=lambda x: x.timestamp
        )
        
        for i in range(1, len(sorted_activities)):
            current = sorted_activities[i]
            previous = sorted_activities[i-1]
            
            # Calculate response time if direction changes (indicating a response)
            if (current.direction != previous.direction and 
                current.direction in ['incoming', 'outgoing']):
                response_time_hours = (current.timestamp - previous.timestamp).total_seconds() / 3600
                response_times.append(response_time_hours)
        
        if response_times:
            return {
                'avg_response_time_hours': sum(response_times) / len(response_times),
                'min_response_time_hours': min(response_times),
                'max_response_time_hours': max(response_times)
            }
        else:
            return {'avg_response_time_hours': 0, 'min_response_time_hours': 0, 'max_response_time_hours': 0}
    
    def _calculate_business_hours_ratio(self, activities: List[ProcessedActivity]) -> float:
        """Calculate ratio of activities during business hours (9 AM - 5 PM weekdays)"""
        business_hours_count = 0
        
        for activity in activities:
            if activity.timestamp:
                # Convert to local time (assuming UTC stored)
                hour = activity.timestamp.hour
                weekday = activity.timestamp.weekday()  # 0=Monday, 6=Sunday
                
                if weekday < 5 and 9 <= hour <= 17:  # Weekdays 9 AM - 5 PM
                    business_hours_count += 1
        
        return business_hours_count / max(len(activities), 1)
    
    def _calculate_weekend_ratio(self, activities: List[ProcessedActivity]) -> float:
        """Calculate ratio of activities during weekends"""
        weekend_count = 0
        
        for activity in activities:
            if activity.timestamp:
                weekday = activity.timestamp.weekday()  # 0=Monday, 6=Sunday
                if weekday >= 5:  # Saturday (5) or Sunday (6)
                    weekend_count += 1
        
        return weekend_count / max(len(activities), 1)
    
    def _calculate_frequency_trend(self, activities: List[ProcessedActivity]) -> str:
        """Calculate activity frequency trend"""
        timestamps = [a.timestamp for a in activities if a.timestamp]
        
        if len(timestamps) < 4:
            return "insufficient_data"
        
        timestamps.sort()
        
        # Split into first half and second half
        mid_point = len(timestamps) // 2
        first_half = timestamps[:mid_point]
        second_half = timestamps[mid_point:]
        
        # Calculate activity density (activities per day)
        first_half_days = (first_half[-1] - first_half[0]).days + 1
        second_half_days = (second_half[-1] - second_half[0]).days + 1
        
        first_half_density = len(first_half) / first_half_days
        second_half_density = len(second_half) / second_half_days
        
        # Determine trend
        if second_half_density > first_half_density * 1.2:
            return "accelerating"
        elif second_half_density < first_half_density * 0.8:
            return "declining"
        else:
            return "stable"
    
    def _extract_deal_characteristics(self, deal_data: Dict[str, Any], deal_metrics: DealMetrics) -> DealCharacteristics:
        """Extract deal characteristics from your deal structure"""
        
        # logger.info(f"Deal : {deal_data}")
        # Extract directly from deal_data (your actual structure)
        deal_amount = self._safe_float(deal_data.get('amount', 0))
        deal_stage = str(deal_data.get('dealstage', 'unknown'))
        deal_type = str(deal_data.get('dealtype', 'unknown'))
        deal_probability = self._safe_float(deal_data.get('deal_stage_probability', 0))
        
        # Determine outcome from deal stage
        deal_outcome = self._determine_outcome(deal_stage)
        
        # Extract dates
        create_date = deal_data.get('createdate', '')
        close_date = deal_data.get('closedate', '')
        
        # Calculate deal age
        deal_age_days = deal_metrics.time_span_days
        
        # Determine deal status
        is_closed = deal_outcome in ['won', 'lost']
        is_won = deal_outcome == 'won'
        is_lost = deal_outcome == 'lost'
        is_open = not is_closed
        is_new_business = deal_type == 'newbusiness'
        
        # Dynamic deal size categorization based on amount distribution
        deal_size_category = self._categorize_deal_size(deal_amount)
        
        # Categorize probability
        if deal_probability >= 0.8:
            probability_category = 'high'
        elif deal_probability >= 0.5:
            probability_category = 'medium'
        else:
            probability_category = 'low'
        
        return DealCharacteristics(
            deal_id=deal_metrics.deal_id,
            deal_amount=deal_amount,
            deal_stage=deal_stage,
            deal_type=deal_type,
            deal_probability=deal_probability,
            deal_outcome=deal_outcome,
            deal_size_category=deal_size_category,
            probability_category=probability_category,
            create_date=create_date,
            close_date=close_date,
            deal_age_days=deal_age_days,
            is_closed=is_closed,
            is_won=is_won,
            is_lost=is_lost,
            is_open=is_open,
            is_new_business=is_new_business
        )
    
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
    
    def _categorize_deal_size(self, amount: float) -> str:
        """Categorize deal size dynamically"""
        if amount <= 0:
            return 'unknown'
        elif amount < 10000:
            return 'small'
        elif amount < 50000:
            return 'medium'
        else:
            return 'large'
    
    def _create_combined_text(self, activities: List[ProcessedActivity]) -> str:
        """Create combined text from activities"""
        
        text_parts = []
        
        for activity in activities:
            if activity.content.strip():
                text_parts.append(f"[{activity.activity_type.upper()}] {activity.content}")
        
        return '\n'.join(text_parts)
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def process_batch(self, deals_data: List[Dict[str, Any]]) -> List[ProcessedDealData]:
        """Process multiple deals in batch"""
        
        processed_deals = []
        
        for deal_data in deals_data:
            try:
                processed_deal = self.process_deal(deal_data)
                processed_deals.append(processed_deal)
            except Exception as e:
                logger.error(f"Error processing deal {deal_data.get('deal_id', 'unknown')}: {e}")
                continue
        
        return processed_deals