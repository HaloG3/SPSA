import json
import logging
import os
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from datetime import datetime
import time
import re
from pathlib import Path
import openai
import anthropic
from groq import Groq

from config.settings import settings
from models.validators import validate_llm_response, get_response_validator, ClientSentimentResponse, SalesSentimentResponse
from utils.cache import get_cache_manager
from utils.token_manager import TokenManager
from utils.metrics import metrics_collector

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        if not openai:
            raise ImportError("openai library not installed. Run: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 3000) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            logger.info(f"Sending prompt to LLM - Length: {len(prompt)} characters")
            logger.info(f"Full prompt sent to LLM:\n{prompt}")
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"OpenAI ({self.model})"

class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        if not anthropic:
            raise ImportError("anthropic library not installed. Run: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Anthropic provider with model: {model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using Anthropic Claude"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            logger.info(f"Sending prompt to LLM - Length: {len(prompt)} characters")
            logger.info(f"Full prompt sent to LLM:\n{prompt}")
            return message.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"Anthropic ({self.model})"

class GroqProvider(LLMProvider):
    """Groq LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        if not Groq:
            raise ImportError("groq library not installed. Run: pip install groq")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Groq provider with model: {model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 3000) -> str:
        """Generate response using Groq"""
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            logger.info(f"Sending prompt to LLM - Length: {len(prompt)} characters")
            logger.info(f"Full prompt sent to LLM:\n{prompt}")
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"Groq ({self.model})"

class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM Provider using responses API"""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str = "2024-02-15-preview"):
        if not openai:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        if not api_key:
            raise ValueError("Azure OpenAI API key is required")
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        if not deployment_name:
            raise ValueError("Azure OpenAI deployment name is required")
        
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.endpoint = endpoint
        logger.info(f"Initialized Azure OpenAI provider with deployment: {deployment_name}")
    
    def _extract_content_safely(self, response_dict: dict) -> str:
        """Safely extract content from Azure OpenAI response with proper error handling"""
        try:
            # Log the response structure for debugging
            # logger.info(f"Response structure keys: {list(response_dict.keys())}")
            
            # Check if 'output' exists
            if 'output' not in response_dict:
                raise ValueError("Response missing 'output' field")
            
            output = response_dict['output']
            if not isinstance(output, list):
                raise ValueError(f"Expected 'output' to be a list, got {type(output)}")
            
            if len(output) == 0:
                raise ValueError("Response 'output' list is empty")
            
            # Find messages with type="message"
            messages = [item for item in output if isinstance(item, dict) and item.get('type') == "message"]
            
            if len(messages) == 0:
                # Fallback: try to find any item with 'content'
                messages = [item for item in output if isinstance(item, dict) and 'content' in item]
                if len(messages) == 0:
                    raise ValueError("No messages found with type='message' or 'content' field")
            
            # Get the first message
            message = messages[0]
            
            # Check if message has 'content'
            if 'content' not in message:
                raise ValueError("Message missing 'content' field")
            
            content = message['content']
            if not isinstance(content, list):
                raise ValueError(f"Expected 'content' to be a list, got {type(content)}")
            
            if len(content) == 0:
                raise ValueError("Message 'content' list is empty")
            
            # Get the first content item
            content_item = content[0]
            if not isinstance(content_item, dict):
                raise ValueError(f"Expected content item to be a dict, got {type(content_item)}")
            
            # Check if content item has 'text'
            if 'text' not in content_item:
                raise ValueError("Content item missing 'text' field")
            
            text = content_item['text']
            if not isinstance(text, str):
                raise ValueError(f"Expected 'text' to be a string, got {type(text)}")
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            logger.error(f"Response structure: {response_dict}")
            raise
    
    def _clean_json_response(self, raw_content: str) -> str:
        """
        Clean Azure OpenAI response to extract pure JSON.
        """
        import re
        import json
        
        if not raw_content or not raw_content.strip():
            raise ValueError("Empty or whitespace-only response")
        
        # Log the raw content for debugging
        # logger.info(f"Raw Azure response : {raw_content}\n{type(raw_content)}")
        
        # Step 1: Basic cleanup - remove leading/trailing whitespace
        content = raw_content.strip()
        
        # Step 2: If it's already valid JSON, return as-is
        try:
            json.loads(content)
            logger.debug("Content is already valid JSON")
            return content
        except json.JSONDecodeError:
            logger.debug("Content is not valid JSON, attempting to clean")
        
        # Step 3: Remove markdown code blocks if present
        markdown_pattern = r'```(?:json)?\s*(.*?)\s*```'
        markdown_match = re.search(markdown_pattern, content, re.DOTALL | re.IGNORECASE)
        if markdown_match:
            content = markdown_match.group(1).strip()
            logger.debug("Removed markdown code block")
            
            # Check if it's valid JSON after markdown removal
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                pass
        
        # Step 4: Look for JSON pattern in the text
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in json_matches:
            try:
                # Test if this match is valid JSON
                json.loads(match)
                logger.debug("Found valid JSON using regex pattern")
                return match
            except json.JSONDecodeError:
                continue
        
        # Step 5: Try to find JSON by looking for lines that start and end with braces
        lines = content.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            stripped_line = line.strip()
            if not in_json and stripped_line.startswith('{'):
                in_json = True
                json_lines = [line]
                brace_count = stripped_line.count('{') - stripped_line.count('}')
            elif in_json:
                json_lines.append(line)
                brace_count += stripped_line.count('{') - stripped_line.count('}')
                if brace_count == 0:
                    # Found complete JSON object
                    potential_json = '\n'.join(json_lines)
                    try:
                        json.loads(potential_json)
                        logger.debug("Found valid JSON using line-by-line parsing")
                        return potential_json
                    except json.JSONDecodeError:
                        pass
                    in_json = False
                    json_lines = []
                    brace_count = 0
        
        # Step 6: Last resort - try to extract any string that looks like JSON
        if '"overall_sentiment"' in content or '"sentiment_score"' in content:
            start_brace = content.find('{')
            end_brace = content.rfind('}')
            
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                potential_json = content[start_brace:end_brace + 1]
                try:
                    json.loads(potential_json)
                    logger.debug("Found valid JSON using brace extraction")
                    return potential_json
                except json.JSONDecodeError:
                    pass
        
        # If all cleaning attempts fail, log the content and raise an error
        logger.error(f"Failed to extract valid JSON from Azure response")
        logger.error(f"Full response content: {raw_content}")
        raise ValueError(f"Could not extract valid JSON from Azure OpenAI response. Content starts with: {raw_content[:100]}")
    
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        """Generate response using Azure OpenAI responses API"""
        try:
            # Add stronger JSON instruction to the prompt for Azure
            enhanced_prompt = f"""{prompt}

CRITICAL: Your response must be valid JSON only. Do not include any explanatory text, markdown formatting, or other content outside the JSON structure. Return only the JSON object."""
            
            response = self.client.responses.create(
                model=self.deployment_name,
                input=enhanced_prompt,
                max_output_tokens=max_tokens
            )
            
            # Convert response to dict for processing
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
            
            # Extract content safely
            raw_content = self._extract_content_safely(response_dict)
            
            # Clean and extract JSON
            cleaned_content = self._clean_json_response(raw_content)
            
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"Azure OpenAI ({self.deployment_name})"
    
class PromptManager:
    """Manages prompt templates for sentiment analysis"""
    
    def __init__(self, prompt_file_path: str = ""):
        self.prompt_file_path = prompt_file_path
        self.template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from file"""
        try:
            prompt_path = Path(self.prompt_file_path)
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found at {self.prompt_file_path}, using default")
                return self._get_default_prompt_template()
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            logger.info(f"Loaded prompt template from {self.prompt_file_path}")
            return template
            
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """Fallback default prompt template"""
        return """You are an expert sales psychology analyst. Analyze the salesperson sentiment from the following activities:

Deal ID: {deal_id}
Activities: {activities_text}
RAG Context: {rag_context}

Provide analysis in JSON format with sentiment score, reasoning, and recommendations."""
    
    def format_prompt(
        self,
        deal_id: str,
        activities_text: str,
        rag_context: str = "",
        activity_frequency: int = 0,
        total_activities: int = 0,
        **kwargs
    ) -> str:
        """Format the prompt template with provided data"""
        
        try:
            formatted_prompt = self.template.format(
                deal_id=deal_id,
                activities_text=activities_text,
                rag_context=rag_context,
                activity_frequency=activity_frequency,
                total_activities=total_activities,
                **kwargs
            )
            
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"Missing required prompt parameter: {e}")
            raise ValueError(f"Missing required prompt parameter: {e}")
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            raise

class LLMClient:
    """Multi-provider LLM client for sentiment analysis"""
    
    def __init__(
        self, 
        provider: LLMProvider,
        prompt_file_path: str = "",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        analysis_type: str = "sales"
    ):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider instance
            prompt_file_path: Path to prompt template file
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            analysis_type: Type of analysis (sales or client)
        """
        self.provider = provider
        self.prompt_manager = PromptManager(prompt_file_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.analysis_type = analysis_type
        self.cache_manager = get_cache_manager()
        self.token_manager = TokenManager(getattr(provider, 'model', getattr(provider, 'deployment_name', 'gpt-3.5-turbo')))
        logger.info(f"LLM Client initialized for {analysis_type} analysis")
    
    def analyze_sentiment(
        self,
        deal_id: str,
        activities_text: str,
        rag_context: str = "",
        activity_frequency: int = 0,
        total_activities: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        op_name = f"analyze_sentiment_{self.analysis_type}"
        token_count = 0
        cache_result = None
        with metrics_collector.track_operation(op_name) as _:
            try:
                # Format prompt
                prompt = self.prompt_manager.format_prompt(
                    deal_id=deal_id,
                    activities_text=activities_text,
                    rag_context=rag_context,
                    activity_frequency=activity_frequency,
                    total_activities=total_activities,
                    **kwargs
                )
                # Token management: validate and truncate if needed
                max_output_tokens = kwargs.get('max_tokens', 2000)
                if not self.token_manager.validate_context_window(prompt, max_output_tokens):
                    logger.warning(f"Prompt too long for model context window, truncating for deal {deal_id}")
                    # Truncate activities_text and reformat prompt
                    allowed_tokens = self.token_manager.get_token_limit() - max_output_tokens
                    truncated_activities = self.token_manager.truncate_text(activities_text, allowed_tokens, keep_head=128, keep_tail=128)
                    prompt = self.prompt_manager.format_prompt(
                        deal_id=deal_id,
                        activities_text=truncated_activities,
                        rag_context=rag_context,
                        activity_frequency=activity_frequency,
                        total_activities=total_activities,
                        **kwargs
                    )
                # Check cache first
                include_rag_context = bool(rag_context.strip())
                cached_response = self.cache_manager.get_cached_llm_response(
                    prompt=prompt,
                    model_name=self.provider.get_provider_name(),
                    deal_id=deal_id,
                    analysis_type=self.analysis_type,
                    include_rag_context=include_rag_context,
                    cache_version="v1"
                )
                
                if cached_response:
                    logger.info(f"Cache hit for deal {deal_id} ({self.analysis_type} analysis)")
                    cache_result = "hit"
                    # Parse and validate cached response
                    result = self._parse_and_validate_response(cached_response, deal_id)
                    result['cache_metadata'] = {
                        'cached': True,
                        'cache_timestamp': datetime.utcnow().isoformat(),
                        'deal_id': deal_id,
                        'analysis_type': self.analysis_type
                    }
                    result['token_metrics'] = self.token_manager.get_usage_metrics()
                    token_count = result['token_metrics'].get('input_tokens', 0) + result['token_metrics'].get('output_tokens', 0)
                    metrics_collector.record_token_usage(op_name, token_count)
                    metrics_collector.record_cache(cache_result)
                    return result
                
                # Generate new response
                logger.info(f"Cache miss for deal {deal_id} ({self.analysis_type} analysis), generating new response")
                cache_result = "miss"
                response_text, input_tokens, output_tokens = self._generate_with_retries_token(prompt, max_output_tokens)
                # Track token usage
                token_count = input_tokens + output_tokens
                metrics_collector.record_token_usage(op_name, token_count)
                # Cache the response
                cache_success = self.cache_manager.cache_llm_response(
                    prompt=prompt,
                    response=response_text,
                    model_name=self.provider.get_provider_name(),
                    deal_id=deal_id,
                    analysis_type=self.analysis_type,
                    include_rag_context=include_rag_context,
                    cache_version="v1"
                )
                
                if cache_success:
                    logger.info(f"Cached response for deal {deal_id} ({self.analysis_type} analysis)")
                else:
                    logger.warning(f"Failed to cache response for deal {deal_id} ({self.analysis_type} analysis)")
                metrics_collector.record_cache(cache_result)
                
                # Parse and validate response
                result = self._parse_and_validate_response(response_text, deal_id)
                result['cache_metadata'] = {
                    'cached': False,
                    'cache_timestamp': datetime.utcnow().isoformat(),
                    'deal_id': deal_id,
                    'analysis_type': self.analysis_type
                }
                result['token_metrics'] = self.token_manager.get_usage_metrics()
                return result
                
            except Exception as e:
                logger.error(f"Error in sentiment analysis for deal {deal_id}: {e}")
                metrics_collector.record_error(type(e).__name__)
                raise
    
    def _generate_with_retries_token(self, prompt: str, max_output_tokens: int) -> (str, int, int):
        """Generate response with retry logic and token counting"""
        
        for attempt in range(self.max_retries):
            try:
                input_tokens = self.token_manager.count_tokens(prompt)
                response = self.provider.generate_response(prompt, max_tokens=max_output_tokens)
                output_tokens = self.token_manager.count_tokens(response)
                
                if not response or not response.strip():
                    raise ValueError("Empty response from LLM")
                
                return response, input_tokens, output_tokens
                
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise
                
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        raise RuntimeError("Failed to generate response after all retries")
    
    def _parse_and_validate_response(self, response_text: str, deal_id: str) -> Dict[str, Any]:
        """Parse and validate the LLM response using Pydantic validation"""
        
        try:
            # Clean response text (remove markdown formatting if present)
            cleaned_response = self._clean_response_text(response_text)
            
            # Parse JSON
            result = json.loads(cleaned_response)
            
            # Use Pydantic validation system
            validated_response = validate_llm_response(result, self.analysis_type, deal_id)
            
            # Convert validated response back to dict for compatibility
            response_dict = validated_response.dict()
            
            # Add validation metadata
            response_dict['validation_metadata'] = {
                'validated': True,
                'validation_timestamp': datetime.utcnow().isoformat(),
                'analysis_type': self.analysis_type,
                'deal_id': deal_id
            }
            
            logger.info(f"Response validation successful for deal {deal_id} using {self.analysis_type} analysis")
            return response_dict
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for deal {deal_id}: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Return a safe default response
            validator = get_response_validator()
            if self.analysis_type == "client":
                default_response = validator._create_default_client_response(deal_id, f"Invalid JSON: {e}")
            else:
                default_response = validator._create_default_sales_response(deal_id, f"Invalid JSON: {e}")
            
            return default_response.dict()
        
        except Exception as e:
            logger.error(f"Response validation failed for deal {deal_id}: {e}")
            
            # Return a safe default response
            validator = get_response_validator()
            if self.analysis_type == "client":
                default_response = validator._create_default_client_response(deal_id, str(e))
            else:
                default_response = validator._create_default_sales_response(deal_id, str(e))
            
            return default_response.dict()
    
    def _clean_response_text(self, response_text: str) -> str:
        """Clean response text to extract JSON"""
        
        # Remove markdown code blocks
        if response_text.startswith('```'):
            response_text = re.sub(r'^```(?:json)?\s*\n', '', response_text)
            response_text = re.sub(r'\n```\s*$', '', response_text)
        
        # Extract JSON if wrapped in other text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        return response_text.strip()

def create_llm_client(
    provider_name: str, 
    prompt_file_path: str = "",
    analysis_type: str = "sales",
    **provider_kwargs
) -> LLMClient:
    """
    Factory function to create LLM client with specified provider
    
    Args:
        provider_name: Name of provider ('openai', 'anthropic', 'groq', 'azure')
        prompt_file_path: Path to prompt template file
        analysis_type: Type of analysis (sales or client)
        **provider_kwargs: Provider-specific configuration
        
    Returns:
        Configured LLM client
    """
    
    provider_name = provider_name.lower()
    
    if provider_name == 'openai':
        provider = OpenAIProvider(**provider_kwargs)
    elif provider_name == 'anthropic':
        provider = AnthropicProvider(**provider_kwargs)
    elif provider_name == 'groq':
        provider = GroqProvider(**provider_kwargs)
    elif provider_name == 'azure':
        provider = AzureOpenAIProvider(**provider_kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return LLMClient(
        provider=provider,
        prompt_file_path=prompt_file_path,
        analysis_type=analysis_type
    )

# Priority 2.3 is also finished