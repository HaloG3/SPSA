import re
import logging
import html
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class SanitizationResult:
    """Result of input sanitization process"""
    cleaned_content: str
    is_safe: bool
    detected_threats: List[str]
    original_length: int
    cleaned_length: int
    sanitization_timestamp: datetime

class InputSanitizer:
    """
    Comprehensive input sanitizer to protect against prompt injection attacks
    Detects suspicious patterns, removes injection attempts, and escapes special characters
    """
    
    def __init__(self, max_content_length: int = 10000):
        """
        Initialize the input sanitizer
        
        Args:
            max_content_length: Maximum allowed content length in characters
        """
        self.max_content_length = max_content_length
        self._initialize_threat_patterns()
        logger.info("Input Sanitizer initialized with security patterns")
    
    def _initialize_threat_patterns(self):
        """Initialize patterns for detecting prompt injection attempts"""
        
        # Direct instruction injection patterns
        self.instruction_patterns = [
            r'(?i)ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|prompts?)',
            r'(?i)forget\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|prompts?)',
            r'(?i)disregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|prompts?)',
            r'(?i)stop\s+(?:following\s+)?(?:previous\s+)?(?:instructions?|prompts?)',
            r'(?i)new\s+(?:instructions?|prompts?|rules?)',
            r'(?i)override\s+(?:previous\s+)?(?:instructions?|prompts?)',
            r'(?i)replace\s+(?:all\s+)?(?:instructions?|prompts?)',
            r'(?i)system\s+(?:prompt|instruction)',
            r'(?i)role\s+(?:play|playing)',
            r'(?i)act\s+as\s+(?:if|though)',
            r'(?i)pretend\s+to\s+be',
            r'(?i)you\s+are\s+now',
            r'(?i)from\s+now\s+on',
            r'(?i)starting\s+now',
            r'(?i)begin\s+new\s+(?:conversation|session)',
        ]
        
        # Rating manipulation patterns
        self.rating_patterns = [
            r'(?i)rate\s+(?:everything\s+)?(?:as\s+)?(?:1\.0|5\.0|10\.0|positive|negative)',
            r'(?i)always\s+(?:rate|score|give)\s+(?:1\.0|5\.0|10\.0|positive|negative)',
            r'(?i)never\s+(?:rate|score|give)\s+(?:anything\s+)?(?:below|above)',
            r'(?i)give\s+(?:only\s+)?(?:1\.0|5\.0|10\.0|positive|negative)\s+(?:ratings?|scores?)',
            r'(?i)score\s+(?:everything\s+)?(?:as\s+)?(?:1\.0|5\.0|10\.0)',
        ]
        
        # Output format manipulation patterns
        self.output_patterns = [
            r'(?i)output\s+(?:only\s+)?(?:json|xml|html|text)',
            r'(?i)respond\s+(?:only\s+)?(?:with|in)\s+(?:json|xml|html|text)',
            r'(?i)format\s+(?:response\s+)?(?:as\s+)?(?:json|xml|html|text)',
            r'(?i)return\s+(?:only\s+)?(?:json|xml|html|text)',
        ]
        
        # System access patterns
        self.system_patterns = [
            r'(?i)access\s+(?:system|file|database)',
            r'(?i)read\s+(?:file|database|system)',
            r'(?i)execute\s+(?:command|code|script)',
            r'(?i)run\s+(?:command|code|script)',
            r'(?i)shell\s+(?:command|access)',
            r'(?i)terminal\s+(?:command|access)',
            r'(?i)sudo\s+(?:command|access)',
            r'(?i)admin\s+(?:access|privileges?)',
            r'(?i)root\s+(?:access|privileges?)',
        ]
        
        # Encoding and obfuscation patterns
        self.encoding_patterns = [
            r'(?i)base64\s+(?:encode|decode)',
            r'(?i)hex\s+(?:encode|decode)',
            r'(?i)url\s+(?:encode|decode)',
            r'(?i)rot13',
            r'(?i)caesar\s+cipher',
            r'(?i)substitution\s+cipher',
        ]
        
        # Compile all patterns for efficiency
        self.all_patterns = {
            'instruction_injection': [re.compile(p) for p in self.instruction_patterns],
            'rating_manipulation': [re.compile(p) for p in self.rating_patterns],
            'output_manipulation': [re.compile(p) for p in self.output_patterns],
            'system_access': [re.compile(p) for p in self.system_patterns],
            'encoding_obfuscation': [re.compile(p) for p in self.encoding_patterns],
        }
    
    def sanitize_input(self, content: str, context: str = "general") -> SanitizationResult:
        """
        Sanitize input content to prevent prompt injection attacks
        
        Args:
            content: Raw input content to sanitize
            context: Context for logging (e.g., "email", "meeting_notes", "task_description")
            
        Returns:
            SanitizationResult with cleaned content and threat detection info
        """
        if not content or not isinstance(content, str):
            return SanitizationResult(
                cleaned_content="",
                is_safe=True,
                detected_threats=[],
                original_length=0,
                cleaned_length=0,
                sanitization_timestamp=datetime.utcnow()
            )
        
        original_length = len(content)
        detected_threats = []
        cleaned_content = content
        
        # Step 1: Check content length
        if original_length > self.max_content_length:
            detected_threats.append(f"Content length ({original_length}) exceeds maximum ({self.max_content_length})")
            cleaned_content = cleaned_content[:self.max_content_length]
        
        # Step 2: Detect suspicious patterns
        threats = self._detect_threats(cleaned_content)
        detected_threats.extend(threats)
        
        # Step 3: Clean content based on detected threats
        if threats:
            cleaned_content = self._clean_content(cleaned_content, threats)
        
        # Step 4: Escape special characters
        cleaned_content = self._escape_special_characters(cleaned_content)
        
        # Step 5: Final length check
        cleaned_length = len(cleaned_content)
        if cleaned_length > self.max_content_length:
            cleaned_content = cleaned_content[:self.max_content_length]
            cleaned_length = self.max_content_length
        
        # Step 6: Log security events
        if detected_threats:
            self._log_security_event(context, detected_threats, original_length, cleaned_length)
        
        return SanitizationResult(
            cleaned_content=cleaned_content,
            is_safe=len(detected_threats) == 0,
            detected_threats=detected_threats,
            original_length=original_length,
            cleaned_length=cleaned_length,
            sanitization_timestamp=datetime.utcnow()
        )
    
    def _detect_threats(self, content: str) -> List[str]:
        """Detect potential threats in the content"""
        threats = []
        
        for threat_type, patterns in self.all_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    match = pattern.search(content)
                    threats.append(f"{threat_type}: '{match.group()}' at position {match.start()}")
        
        # Additional heuristic checks
        if self._check_suspicious_heuristics(content):
            threats.append("suspicious_heuristics: Content contains suspicious patterns")
        
        return threats
    
    def _check_suspicious_heuristics(self, content: str) -> bool:
        """Additional heuristic checks for suspicious content"""
        
        # Check for excessive repetition
        words = content.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # If any word appears more than 30% of the time
            max_freq = max(word_freq.values()) if word_freq else 0
            if max_freq > len(words) * 0.3:
                return True
        
        # Check for excessive punctuation
        if content.count('!') > len(content) * 0.1 or content.count('?') > len(content) * 0.1:
            return True
        
        # Check for excessive capitalization
        if sum(1 for c in content if c.isupper()) > len(content) * 0.5:
            return True
        
        return False
    
    def _clean_content(self, content: str, threats: List[str]) -> str:
        """Clean content based on detected threats"""
        cleaned = content
        
        # Remove or neutralize instruction injection attempts
        for threat in threats:
            if 'instruction_injection' in threat:
                # Remove the specific matched pattern
                for pattern in self.all_patterns['instruction_injection']:
                    cleaned = pattern.sub('[REDACTED]', cleaned)
            
            elif 'rating_manipulation' in threat:
                # Remove rating manipulation attempts
                for pattern in self.all_patterns['rating_manipulation']:
                    cleaned = pattern.sub('[REDACTED]', cleaned)
            
            elif 'output_manipulation' in threat:
                # Remove output format manipulation
                for pattern in self.all_patterns['output_manipulation']:
                    cleaned = pattern.sub('[REDACTED]', cleaned)
            
            elif 'system_access' in threat:
                # Remove system access attempts
                for pattern in self.all_patterns['system_access']:
                    cleaned = pattern.sub('[REDACTED]', cleaned)
        
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _escape_special_characters(self, content: str) -> str:
        """Escape special characters that could be used for injection"""
        
        # HTML escape to prevent HTML injection
        content = html.escape(content)
        
        # Escape common prompt injection characters
        content = content.replace('{', '\\{')
        content = content.replace('}', '\\}')
        content = content.replace('[', '\\[')
        content = content.replace(']', '\\]')
        
        # Escape backticks that could be used for code blocks
        content = content.replace('`', '\\`')
        
        return content
    
    def _log_security_event(self, context: str, threats: List[str], original_length: int, cleaned_length: int):
        """Log security events for monitoring"""
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'context': context,
            'threats': threats,
            'original_length': original_length,
            'cleaned_length': cleaned_length,
            'threat_count': len(threats)
        }
        
        logger.warning(f"Security threat detected: {json.dumps(event_data, indent=2)}")
    
    def validate_sanitization(self, original: str, sanitized: str) -> bool:
        """
        Validate that sanitization was effective
        
        Args:
            original: Original content
            sanitized: Sanitized content
            
        Returns:
            True if sanitization appears effective
        """
        # Check if any threats remain in sanitized content
        remaining_threats = self._detect_threats(sanitized)
        if remaining_threats:
            logger.error(f"Sanitization validation failed: {remaining_threats}")
            return False
        
        # Check if content was overly sanitized (too much removed)
        if len(sanitized) < len(original) * 0.5:  # More than 50% removed
            logger.warning("Sanitization may have been too aggressive")
            return False
        
        return True
    
    def get_sanitization_stats(self) -> Dict[str, Any]:
        """Get statistics about sanitization patterns"""
        return {
            'max_content_length': self.max_content_length,
            'pattern_counts': {
                threat_type: len(patterns) 
                for threat_type, patterns in self.all_patterns.items()
            },
            'total_patterns': sum(len(patterns) for patterns in self.all_patterns.values())
        }


class SecurityMiddleware:
    """
    Middleware for applying security sanitization to data processing pipeline
    """
    
    def __init__(self, sanitizer: InputSanitizer):
        """
        Initialize security middleware
        
        Args:
            sanitizer: Input sanitizer instance
        """
        self.sanitizer = sanitizer
        logger.info("Security Middleware initialized")
    
    def sanitize_deal_data(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize deal data before processing
        
        Args:
            deal_data: Raw deal data
            
        Returns:
            Sanitized deal data
        """
        sanitized_data = deal_data.copy()
        
        # Sanitize deal metadata
        if 'deal_id' in sanitized_data:
            result = self.sanitizer.sanitize_input(str(sanitized_data['deal_id']), "deal_id")
            sanitized_data['deal_id'] = result.cleaned_content
        
        # Sanitize activities
        if 'activities' in sanitized_data:
            sanitized_activities = []
            for i, activity in enumerate(sanitized_data['activities']):
                sanitized_activity = activity.copy()
                
                # Sanitize activity content
                if 'content' in activity:
                    result = self.sanitizer.sanitize_input(
                        str(activity['content']), 
                        f"activity_{i}_content"
                    )
                    sanitized_activity['content'] = result.cleaned_content
                
                # Sanitize activity subject
                if 'subject' in activity:
                    result = self.sanitizer.sanitize_input(
                        str(activity['subject']), 
                        f"activity_{i}_subject"
                    )
                    sanitized_activity['subject'] = result.cleaned_content
                
                sanitized_activities.append(sanitized_activity)
            
            sanitized_data['activities'] = sanitized_activities
        
        return sanitized_data
    
    def sanitize_activity_content(self, content: str, activity_type: str = "unknown") -> str:
        """
        Sanitize individual activity content
        
        Args:
            content: Raw activity content
            activity_type: Type of activity for logging context
            
        Returns:
            Sanitized content
        """
        result = self.sanitizer.sanitize_input(content, f"activity_{activity_type}")
        return result.cleaned_content


# Global sanitizer instance
_default_sanitizer = None

def get_default_sanitizer() -> InputSanitizer:
    """Get the default sanitizer instance"""
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = InputSanitizer()
    return _default_sanitizer

def sanitize_input(content: str, context: str = "general") -> SanitizationResult:
    """
    Convenience function to sanitize input using default sanitizer
    
    Args:
        content: Content to sanitize
        context: Context for logging
        
    Returns:
        SanitizationResult
    """
    sanitizer = get_default_sanitizer()
    return sanitizer.sanitize_input(content, context)
