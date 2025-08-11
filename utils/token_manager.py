import tiktoken
import re
from typing import List, Dict, Optional

MODEL_TOKEN_LIMITS = {
    # OpenAI
    'gpt-4-turbo-preview': 128000,
    'gpt-4-1106-preview': 128000,
    'gpt-4': 8192,
    'gpt-3.5-turbo': 4096,
    'text-davinci-003': 4097,
    # Anthropic
    'claude-3-opus-20240229': 200000,
    'claude-3-sonnet-20240229': 200000,
    'claude-3-haiku-20240307': 200000,
    'claude-v2': 100000,
    # Groq (Llama3, Mixtral, etc.)
    'llama-3-70b-8192': 8192,
    'llama-3-8b-8192': 8192,
    'mixtral-8x7b-32768': 32768,
    # Add more as needed
}

MODEL_COSTS = {
    # Example costs per 1K tokens (USD)
    'gpt-4-turbo-preview': {'input': 0.01, 'output': 0.03},
    'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002},
    'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
    # ...
}

def get_token_limit(model: str) -> int:
    return MODEL_TOKEN_LIMITS.get(model, 4096)

def get_model_cost(model: str) -> Dict[str, float]:
    return MODEL_COSTS.get(model, {'input': 0.0, 'output': 0.0})

class TokenManager:
    def __init__(self, model: str):
        self.model = model
        self.token_limit = get_token_limit(model)
        self.costs = get_model_cost(model)
        self.usage_metrics = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost_usd': 0.0,
            'calls': 0
        }
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoding = tiktoken.get_encoding('cl100k_base')

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return max(1, len(text) // 4)

    def validate_context_window(self, prompt: str, max_output_tokens: int) -> bool:
        input_tokens = self.count_tokens(prompt)
        return (input_tokens + max_output_tokens) <= self.token_limit

    def truncate_text(self, text: str, max_tokens: int, keep_head: int = 0, keep_tail: int = 0) -> str:
        """
        Truncate text to fit max_tokens. Optionally keep head/tail tokens for context.
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        if keep_head + keep_tail >= max_tokens:
            keep_head = max_tokens // 2
            keep_tail = max_tokens - keep_head
        head_tokens = tokens[:keep_head] if keep_head else []
        tail_tokens = tokens[-keep_tail:] if keep_tail else []
        main_tokens = tokens[keep_head: max_tokens - keep_tail] if keep_head < max_tokens - keep_tail else []
        new_tokens = head_tokens + main_tokens + tail_tokens
        return self.encoding.decode(new_tokens)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1000) * self.costs['input']
        output_cost = (output_tokens / 1000) * self.costs['output']
        return input_cost + output_cost

    def track_usage(self, input_tokens: int, output_tokens: int):
        self.usage_metrics['total_input_tokens'] += input_tokens
        self.usage_metrics['total_output_tokens'] += output_tokens
        self.usage_metrics['total_cost_usd'] += self.estimate_cost(input_tokens, output_tokens)
        self.usage_metrics['calls'] += 1

    def get_usage_metrics(self) -> Dict[str, float]:
        return self.usage_metrics.copy()

    def get_token_limit(self) -> int:
        return self.token_limit

    def get_model(self) -> str:
        return self.model
