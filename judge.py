import os
import re
from typing import Dict, List, Optional
import math
from functools import lru_cache
from pathlib import Path
import yaml
import numpy as np

# Try to import OpenAI (optional)
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

# Try to import Anthropic (optional)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Set up credentials if available
openai = None
if OPENAI_AVAILABLE:
    try:
        from config import setup_credentials
        config = setup_credentials()
        openai = AsyncOpenAI()
    except Exception:
        pass  # OpenAI not configured



class OpenAiJudge:
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly 
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model = model
        assert eval_type in ["0_100", "0_10", "binary", "binary_text"], "eval_type must be either 0_100 or binary"
        self.eval_type = eval_type

        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score
        elif self.eval_type == "0_10":
            self.aggregate_score = self._aggregate_0_10_score
        elif self.eval_type == "binary":
            self.aggregate_score = self._aggregate_binary_score
        elif self.eval_type == "binary_text":
            self.aggregate_score = self._aggregate_binary_text_score
        else:
            raise ValueError(f"Invalid eval_type: {self.eval_type}")

        self.prompt_template = prompt_template
        
    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        if self.eval_type == "binary_text":
            response_text = await self.query_full_text(messages)
            score = self.aggregate_score(response_text) # aggregate_score is _aggregate_binary_text_score
        else:
            logprobs = await self.logprob_probs(messages)
            score = self.aggregate_score(logprobs) # aggregate_score is one of the other three
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = await openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))
        
        return result
    
    async def query_full_text(self, messages) -> str:
        """Requests a full text completion. Used for binary_text eval_type."""
        completion = await openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            seed=0
        )
        try:
            return completion.choices[0].message.content
        except (IndexError, AttributeError):
            return ""

    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total
    
    def _aggregate_0_10_score(self, score: dict) -> float:
        #   NOTE: we check for refusals explcitly.
        #   If "REFUSAL" has the highest probability, we return None.
        #   Otherwise, we aggregate scores from 0-9.
        if "REFUSAL" in score and score["REFUSAL"] > max(score.get(str(i), 0) for i in range(10)):
            return None

        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 9:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25: # Similar threshold as 0_100
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    def _aggregate_binary_score(self, score: dict) -> float:
        #   NOTE: we check for refusals explicitly.
        #   If "REFUSAL" has a higher probability than both "YES" and "NO", we return None.
        #   Otherwise, calculates a score P(YES) / (P(YES) + P(NO)).
        #   Returns None if the combined probability of YES and NO is below a threshold (0.25).
        
        yes_prob = score.get("YES", 0.0)
        no_prob = score.get("NO", 0.0)
        refusal_prob = score.get("REFUSAL", 0.0)

        # If REFUSAL has a higher probability than both YES and NO, consider it a refusal.
        if refusal_prob > yes_prob and refusal_prob > no_prob:
            return None
        
        denominator = yes_prob + no_prob

        # If the combined probability of YES and NO is too low (e.g., model outputted something else,
        # or was not confident in YES/NO), return None.
        if denominator < 0.25:  # Using 0.25 to be consistent with other aggregation methods
            return None
            
        return yes_prob / denominator

    def _aggregate_binary_text_score(self, response_text: str) -> bool:
        if "<answer>REFUSAL</answer>" in response_text:
            return None
        elif "<answer>NO</answer>" in response_text:
            return 0
        elif "<answer>YES</answer>" in response_text:
            return 1
        return None # Invalid response

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)


class ClaudeJudge:
    """Claude-based judge for scoring persona responses.

    Uses Anthropic API to evaluate responses. Since Claude doesn't provide
    logprobs, we use direct text output and parse the score.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        prompt_template: str = "",
        eval_type: str = "0_100",
        api_key: Optional[str] = None,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package required for ClaudeJudge. "
                "Install with: pip install anthropic"
            )

        self.model = model
        self.eval_type = eval_type
        self.prompt_template = prompt_template
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in environment or pass api_key parameter."
            )

        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _parse_score(self, response_text: str) -> Optional[float]:
        """Parse a numeric score from Claude's response."""
        # Clean the response
        text = response_text.strip()

        # Check for refusal
        if "REFUSAL" in text.upper():
            return None

        # Try to extract a number
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            score = float(numbers[0])
            if self.eval_type == "0_100":
                return max(0, min(100, score))
            elif self.eval_type == "0_10":
                return max(0, min(10, score))

        return None

    async def judge(self, **kwargs) -> Optional[float]:
        """Score a response using Claude."""
        prompt = self.prompt_template.format(**kwargs)

        # Use sync client in async context (anthropic doesn't have native async)
        import asyncio
        loop = asyncio.get_event_loop()

        def _call_api():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
                system="You are a scoring assistant. Respond with only a number between 0 and 100. Nothing else.",
            )
            return response.content[0].text.strip()

        response_text = await loop.run_in_executor(None, _call_api)
        return self._parse_score(response_text)

    async def __call__(self, **kwargs) -> Optional[float]:
        return await self.judge(**kwargs)
