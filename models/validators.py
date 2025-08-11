"""
Pydantic validators for LLM response validation
Ensures all LLM responses are properly structured and contain valid data
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum

logger = logging.getLogger(__name__)

class SentimentLevel(str, Enum):
    """Valid sentiment levels"""
    EXCEPTIONAL_POSITIVE = "exceptional_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    CRITICAL_NEGATIVE = "critical_negative"

class ResponsePattern(str, Enum):
    """Valid response patterns"""
    QUICK = "quick"
    NORMAL = "normal"
    DELAYED = "delayed"

class InitiativeLevel(str, Enum):
    """Valid initiative levels"""
    PROACTIVE = "proactive"
    RESPONSIVE = "responsive"
    PASSIVE = "passive"

class DecisionReadiness(str, Enum):
    """Valid decision readiness levels"""
    READY = "ready"
    EVALUATING = "evaluating"
    EARLY_STAGE = "early_stage"

class DecisionTimeline(str, Enum):
    """Valid decision timeline values"""
    IMMEDIATE = "immediate"
    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"
    UNCLEAR = "unclear"

class RiskLevel(str, Enum):
    """Valid risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ActivityBreakdownItem(BaseModel):
    """Validation model for activity breakdown items"""
    sentiment: str = Field(..., description="Sentiment for this activity type")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score between -1.0 and 1.0")
    key_indicators: List[str] = Field(default_factory=list, description="Key indicators for this activity type")
    count: int = Field(..., ge=0, description="Number of activities of this type")

    @validator('sentiment')
    def validate_sentiment(cls, v):
        """Validate sentiment values"""
        valid_sentiments = ['positive', 'negative', 'neutral', 'exceptional_positive', 'critical_negative']
        if v.lower() not in valid_sentiments:
            logger.warning(f"Invalid sentiment value: {v}, defaulting to neutral")
            return 'neutral'
        return v.lower()

    @validator('key_indicators')
    def validate_key_indicators(cls, v):
        """Ensure key indicators are strings and not empty"""
        if not isinstance(v, list):
            return []
        return [str(indicator).strip() for indicator in v if str(indicator).strip()]

class ClientEngagementIndicators(BaseModel):
    """Validation model for client engagement indicators"""
    response_pattern: ResponsePattern = Field(..., description="Client response pattern")
    initiative_level: InitiativeLevel = Field(..., description="Client initiative level")
    decision_readiness: DecisionReadiness = Field(..., description="Client decision readiness")

class DealMomentumIndicators(BaseModel):
    """Validation model for deal momentum indicators"""
    engagement_trend: str = Field(..., description="Engagement trend")
    decision_velocity: str = Field(..., description="Decision velocity")
    stakeholder_involvement: str = Field(..., description="Stakeholder involvement level")
    urgency_level: str = Field(..., description="Urgency level")

class BaseSentimentResponse(BaseModel):
    """Base validation model for all sentiment responses"""
    overall_sentiment: SentimentLevel = Field(..., description="Overall sentiment level")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score between -1.0 and 1.0")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    activity_breakdown: Dict[str, ActivityBreakdownItem] = Field(..., description="Breakdown by activity type")
    reasoning: str = Field(..., min_length=10, description="Detailed reasoning for the analysis")
    
    # Optional fields with defaults
    buying_signals: List[str] = Field(default_factory=list, description="Positive buying signals observed")
    concern_indicators: List[str] = Field(default_factory=list, description="Concerns or hesitation signals")
    engagement_opportunities: List[str] = Field(default_factory=list, description="Ways to increase engagement")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    analysis_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis metadata")

    @validator('reasoning')
    def validate_reasoning(cls, v):
        """Ensure reasoning is not empty and has minimum length"""
        if not v or len(v.strip()) < 10:
            logger.warning("Reasoning too short, providing default")
            return "Analysis based on available activity data."
        return v.strip()

    @validator('buying_signals', 'concern_indicators', 'engagement_opportunities', 'recommended_actions')
    def validate_list_fields(cls, v):
        """Ensure list fields contain valid strings"""
        if not isinstance(v, list):
            return []
        return [str(item).strip() for item in v if str(item).strip()]

    @root_validator
    def validate_activity_breakdown(cls, values):
        """Validate activity breakdown structure"""
        activity_breakdown = values.get('activity_breakdown', {})
        if not isinstance(activity_breakdown, dict):
            logger.warning("Invalid activity_breakdown format, using empty dict")
            values['activity_breakdown'] = {}
        return values

class ClientSentimentResponse(BaseSentimentResponse):
    """Validation model for client sentiment analysis responses"""
    client_engagement_indicators: ClientEngagementIndicators = Field(..., description="Client engagement indicators")
    decision_timeline: DecisionTimeline = Field(..., description="Expected decision timeline")
    client_risk_level: RiskLevel = Field(..., description="Client risk level")

    @validator('client_engagement_indicators', pre=True)
    def validate_engagement_indicators(cls, v):
        """Handle missing or malformed engagement indicators"""
        if not isinstance(v, dict):
            logger.warning("Invalid client_engagement_indicators, using defaults")
            return {
                'response_pattern': 'normal',
                'initiative_level': 'responsive',
                'decision_readiness': 'evaluating'
            }
        return v

class SalesSentimentResponse(BaseSentimentResponse):
    """Validation model for sales sentiment analysis responses"""
    deal_momentum_indicators: DealMomentumIndicators = Field(..., description="Deal momentum indicators")
    temporal_trend: str = Field(..., description="Temporal trend analysis")
    professional_gaps: List[str] = Field(default_factory=list, description="Professional gaps identified")
    excellence_indicators: List[str] = Field(default_factory=list, description="Excellence indicators")
    risk_indicators: List[str] = Field(default_factory=list, description="Risk indicators")
    opportunity_indicators: List[str] = Field(default_factory=list, description="Opportunity indicators")
    benchmark_comparison: str = Field(default="", description="Benchmark comparison")
    context_analysis_notes: List[str] = Field(default_factory=list, description="Context analysis notes")

    @validator('deal_momentum_indicators', pre=True)
    def validate_momentum_indicators(cls, v):
        """Handle missing or malformed momentum indicators"""
        if not isinstance(v, dict):
            logger.warning("Invalid deal_momentum_indicators, using defaults")
            return {
                'engagement_trend': 'stable',
                'decision_velocity': 'normal',
                'stakeholder_involvement': 'moderate',
                'urgency_level': 'low'
            }
        return v

    @validator('temporal_trend')
    def validate_temporal_trend(cls, v):
        """Validate temporal trend values"""
        valid_trends = ['improving', 'declining', 'stable', 'volatile', 'accelerating', 'slowing']
        if not v or v.lower() not in valid_trends:
            logger.warning(f"Invalid temporal_trend: {v}, defaulting to stable")
            return 'stable'
        return v.lower()

class ErrorResponse(BaseModel):
    """Validation model for error responses"""
    error: str = Field(..., description="Error message")
    deal_id: str = Field(..., description="Deal ID")
    timestamp: str = Field(..., description="Error timestamp")
    analysis_type: str = Field(..., description="Analysis type")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional error details")

class ResponseValidator:
    """Main validator class for LLM responses"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_client_response(self, response_data: Dict[str, Any], deal_id: str) -> ClientSentimentResponse:
        """
        Validate client sentiment analysis response
        
        Args:
            response_data: Raw response data from LLM
            deal_id: Deal ID for logging context
            
        Returns:
            Validated ClientSentimentResponse
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Pre-process the data to handle common issues
            processed_data = self._preprocess_response_data(response_data)
            
            # Validate with Pydantic model
            validated_response = ClientSentimentResponse(**processed_data)
            
            self.logger.info(f"Client response validation successful for deal {deal_id}")
            return validated_response
            
        except Exception as e:
            self.logger.error(f"Client response validation failed for deal {deal_id}: {e}")
            # Return a safe default response
            return self._create_default_client_response(deal_id, str(e))
    
    def validate_sales_response(self, response_data: Dict[str, Any], deal_id: str) -> SalesSentimentResponse:
        """
        Validate sales sentiment analysis response
        
        Args:
            response_data: Raw response data from LLM
            deal_id: Deal ID for logging context
            
        Returns:
            Validated SalesSentimentResponse
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Pre-process the data to handle common issues
            processed_data = self._preprocess_response_data(response_data)
            
            # Validate with Pydantic model
            validated_response = SalesSentimentResponse(**processed_data)
            
            self.logger.info(f"Sales response validation successful for deal {deal_id}")
            return validated_response
            
        except Exception as e:
            self.logger.error(f"Sales response validation failed for deal {deal_id}: {e}")
            # Return a safe default response
            return self._create_default_sales_response(deal_id, str(e))
    
    def _preprocess_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-process response data to handle common issues
        
        Args:
            data: Raw response data
            
        Returns:
            Processed data ready for validation
        """
        processed = data.copy()
        
        # Ensure required fields exist with defaults
        processed.setdefault('overall_sentiment', 'neutral')
        processed.setdefault('sentiment_score', 0.0)
        processed.setdefault('confidence', 0.5)
        processed.setdefault('activity_breakdown', {})
        processed.setdefault('reasoning', 'Analysis based on available data.')
        
        # Ensure list fields are lists
        for field in ['buying_signals', 'concern_indicators', 'engagement_opportunities', 'recommended_actions']:
            if field not in processed or not isinstance(processed[field], list):
                processed[field] = []
        
        # Ensure activity_breakdown is properly structured
        if 'activity_breakdown' in processed:
            processed['activity_breakdown'] = self._normalize_activity_breakdown(processed['activity_breakdown'])
        
        return processed
    
    def _normalize_activity_breakdown(self, breakdown: Any) -> Dict[str, ActivityBreakdownItem]:
        """
        Normalize activity breakdown data
        
        Args:
            breakdown: Raw activity breakdown data
            
        Returns:
            Normalized activity breakdown
        """
        if not isinstance(breakdown, dict):
            return {}
        
        normalized = {}
        for activity_type, data in breakdown.items():
            if isinstance(data, dict):
                try:
                    normalized[activity_type] = ActivityBreakdownItem(**data)
                except Exception as e:
                    self.logger.warning(f"Invalid activity breakdown for {activity_type}: {e}")
                    # Create default item
                    normalized[activity_type] = ActivityBreakdownItem(
                        sentiment='neutral',
                        sentiment_score=0.0,
                        key_indicators=[],
                        count=0
                    )
        
        return normalized
    
    def _create_default_client_response(self, deal_id: str, error_msg: str) -> ClientSentimentResponse:
        """Create a safe default client response"""
        return ClientSentimentResponse(
            overall_sentiment='neutral',
            sentiment_score=0.0,
            confidence=0.1,
            activity_breakdown={},
            reasoning=f"Analysis failed: {error_msg}",
            client_engagement_indicators=ClientEngagementIndicators(
                response_pattern='normal',
                initiative_level='responsive',
                decision_readiness='evaluating'
            ),
            decision_timeline='unclear',
            client_risk_level='medium',
            analysis_metadata={
                'validation_error': True,
                'error_message': error_msg,
                'deal_id': deal_id
            }
        )
    
    def _create_default_sales_response(self, deal_id: str, error_msg: str) -> SalesSentimentResponse:
        """Create a safe default sales response"""
        return SalesSentimentResponse(
            overall_sentiment='neutral',
            sentiment_score=0.0,
            confidence=0.1,
            activity_breakdown={},
            reasoning=f"Analysis failed: {error_msg}",
            deal_momentum_indicators=DealMomentumIndicators(
                engagement_trend='stable',
                decision_velocity='normal',
                stakeholder_involvement='moderate',
                urgency_level='low'
            ),
            temporal_trend='stable',
            analysis_metadata={
                'validation_error': True,
                'error_message': error_msg,
                'deal_id': deal_id
            }
        )
    
    def validate_error_response(self, response_data: Dict[str, Any]) -> ErrorResponse:
        """
        Validate error response
        
        Args:
            response_data: Raw error response data
            
        Returns:
            Validated ErrorResponse
        """
        try:
            return ErrorResponse(**response_data)
        except Exception as e:
            self.logger.error(f"Error response validation failed: {e}")
            return ErrorResponse(
                error="Validation failed",
                deal_id="unknown",
                timestamp=datetime.utcnow().isoformat(),
                analysis_type="unknown",
                details={'validation_error': str(e)}
            )

# Global validator instance
_response_validator = None

def get_response_validator() -> ResponseValidator:
    """Get the global response validator instance"""
    global _response_validator
    if _response_validator is None:
        _response_validator = ResponseValidator()
    return _response_validator

def validate_llm_response(response_data: Dict[str, Any], analysis_type: str, deal_id: str) -> Union[ClientSentimentResponse, SalesSentimentResponse]:
    """
    Convenience function to validate LLM responses
    
    Args:
        response_data: Raw response data from LLM
        analysis_type: Type of analysis ('client' or 'sales')
        deal_id: Deal ID for logging context
        
    Returns:
        Validated response object
    """
    validator = get_response_validator()
    
    if analysis_type.lower() == 'client':
        return validator.validate_client_response(response_data, deal_id)
    elif analysis_type.lower() == 'sales':
        return validator.validate_sales_response(response_data, deal_id)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
