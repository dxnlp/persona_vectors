"""
Judge implementations for scoring student answers.
Includes local LLM judge with steering and Claude API judge.

Uses score-based approach with logprobs for consistent, reliable scoring.
"""

import torch
import torch.nn.functional as F
import os
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from activation_steer import ActivationSteerer
from .config import SteeringConfig, ESSAY_SET_INFO
from .student import GeneratedAnswer
from .data_loader import Essay


@dataclass
class ScoringResult:
    """Result of scoring a student answer."""
    essay_id: int
    essay_set: int
    student_config: str  # Student steering configuration ("original" for original essays)
    judge_config: str  # Judge configuration (steering or "api")
    predicted_score: int
    ground_truth_score: float
    reasoning: Optional[str] = None  # Score probabilities or API response
    score_probs: Optional[Dict[int, float]] = None  # Probability distribution over scores
    quality_feedback: Optional[str] = None  # Text-based quality judgment
    answer_text: Optional[str] = None  # The answer that was scored


class BaseJudge(ABC):
    """Abstract base class for judges."""

    # Score-based prompt - ends with "Score:" to get logprobs for next token
    SCORING_PROMPT_TEMPLATE = """You are an expert essay grader. Score the following student answer.

Essay Prompt: {essay_prompt}

Rubric:
- Score range: {min_score} to {max_score}
- {min_score} = Poor/No understanding
- {max_score} = Excellent/Complete understanding
- Evaluate: content accuracy, clarity, relevance to prompt

Student Answer:
{student_answer}

Based on the rubric, this answer deserves a score of (choose one number from {min_score} to {max_score}):"""

    @abstractmethod
    def score(self, answer: GeneratedAnswer) -> ScoringResult:
        """Score a single student answer."""
        pass

    @abstractmethod
    def score_batch(self, answers: List[GeneratedAnswer]) -> List[ScoringResult]:
        """Score multiple student answers."""
        pass

    def _get_score_range(self, essay_set: int) -> Tuple[int, int]:
        """Get the score range for an essay set."""
        info = ESSAY_SET_INFO.get(essay_set, {})
        return info.get("score_range", (0, 4))

    def _build_scoring_prompt(self, answer: GeneratedAnswer) -> str:
        """Build the scoring prompt for an answer."""
        min_score, max_score = self._get_score_range(answer.essay_set)
        return self.SCORING_PROMPT_TEMPLATE.format(
            essay_prompt=answer.prompt,
            min_score=min_score,
            max_score=max_score,
            student_answer=answer.generated_answer,
        )

    def _build_scoring_prompt_for_essay(self, essay: Essay) -> str:
        """Build the scoring prompt for an original essay."""
        min_score, max_score = self._get_score_range(essay.essay_set)
        return self.SCORING_PROMPT_TEMPLATE.format(
            essay_prompt=essay.prompt,
            min_score=min_score,
            max_score=max_score,
            student_answer=essay.essay_text,
        )

    def score_essay(self, essay: Essay) -> ScoringResult:
        """Score an original essay from the dataset.

        This is used for calculating QWK against ground truth.
        """
        raise NotImplementedError("Subclasses must implement score_essay")

    def score_essays(self, essays: List[Essay]) -> List[ScoringResult]:
        """Score multiple original essays."""
        raise NotImplementedError("Subclasses must implement score_essays")


class LocalJudge(BaseJudge):
    """Local LLM judge with optional persona steering.

    Uses logprobs-based scoring for consistent, reliable results,
    plus generates text-based quality feedback.
    """

    SYSTEM_PROMPT = """You are an expert essay grader. Score student answers objectively."""

    # Prompt for generating quality feedback after scoring
    FEEDBACK_PROMPT_TEMPLATE = """You are an expert essay grader. Provide a brief quality assessment.

Essay Prompt: {essay_prompt}

Student Answer:
{student_answer}

You gave this answer a score of {score} out of {max_score}.

In 1-2 sentences, explain the key strengths and weaknesses of this answer. Be specific and constructive.

Feedback:"""

    def __init__(
        self,
        model,
        tokenizer,
        steering_config: Optional[SteeringConfig] = None,
        generate_feedback: bool = True,
        max_feedback_tokens: int = 100,
    ):
        """Initialize the local judge.

        Args:
            model: The HuggingFace model
            tokenizer: The tokenizer
            steering_config: Optional steering configuration
            generate_feedback: Whether to generate text quality feedback
            max_feedback_tokens: Maximum tokens for feedback generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.steering_config = steering_config or SteeringConfig.unsteered()
        self.generate_feedback = generate_feedback
        self.max_feedback_tokens = max_feedback_tokens

        # Load steering vector if needed
        self.steering_vector = None
        if self.steering_config.coef != 0 and self.steering_config.vector_path:
            vectors = torch.load(self.steering_config.vector_path, weights_only=False)
            self.steering_vector = vectors[self.steering_config.layer]

        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # Cache score token IDs for efficiency
        self._score_token_cache: Dict[int, List[int]] = {}

    def _get_score_token_ids(self, min_score: int, max_score: int) -> Dict[int, List[int]]:
        """Get token IDs for each possible score.

        Returns dict mapping score -> list of token IDs that represent that score.
        Handles cases where "3" might tokenize differently than " 3".
        """
        cache_key = (min_score, max_score)
        if cache_key in self._score_token_cache:
            return self._score_token_cache[cache_key]

        score_to_tokens = {}
        for score in range(min_score, max_score + 1):
            token_ids = set()
            # Try different representations
            for text in [str(score), f" {score}", f"{score} ", f" {score} "]:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if tokens:
                    token_ids.add(tokens[0])  # First token is the score
            score_to_tokens[score] = list(token_ids)

        self._score_token_cache[cache_key] = score_to_tokens
        return score_to_tokens

    def _build_messages(self, answer: GeneratedAnswer) -> List[Dict[str, str]]:
        """Build chat messages for scoring."""
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_scoring_prompt(answer)},
        ]

    def _get_score_from_logits(
        self,
        logits: torch.Tensor,
        min_score: int,
        max_score: int,
    ) -> Tuple[int, Dict[int, float]]:
        """Extract score from model logits using token probabilities.

        Args:
            logits: Logits for the next token [vocab_size]
            min_score: Minimum possible score
            max_score: Maximum possible score

        Returns:
            Tuple of (predicted_score, probability_dict)
        """
        score_token_ids = self._get_score_token_ids(min_score, max_score)

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Calculate probability for each score (sum over all token variants)
        score_probs = {}
        for score, token_ids in score_token_ids.items():
            if token_ids:
                score_probs[score] = sum(probs[tid].item() for tid in token_ids)
            else:
                score_probs[score] = 0.0

        # Normalize probabilities
        total = sum(score_probs.values())
        if total > 0:
            score_probs = {k: v / total for k, v in score_probs.items()}

        # Get the most likely score
        predicted_score = max(score_probs, key=score_probs.get)

        return predicted_score, score_probs

    def _generate_feedback(
        self,
        answer: GeneratedAnswer,
        predicted_score: int,
    ) -> str:
        """Generate text-based quality feedback for the answer.

        Args:
            answer: The student's answer
            predicted_score: The score already assigned

        Returns:
            Text feedback about answer quality
        """
        min_score, max_score = self._get_score_range(answer.essay_set)

        # Add instruction to disable thinking mode for Qwen3 models
        feedback_prompt = self.FEEDBACK_PROMPT_TEMPLATE.format(
            essay_prompt=answer.prompt,
            student_answer=answer.generated_answer[:500],  # Truncate for efficiency
            score=predicted_score,
            max_score=max_score,
        ) + "\n\nRespond directly without any thinking process. Be concise."

        messages = [
            {"role": "system", "content": "You are an expert essay grader. Provide direct, concise feedback without any preamble or thinking."},
            {"role": "user", "content": feedback_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.steering_vector is not None and self.steering_config.coef != 0:
                with ActivationSteerer(
                    self.model,
                    self.steering_vector,
                    coeff=self.steering_config.coef,
                    layer_idx=self.steering_config.layer - 1,
                    positions="response",
                ):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_feedback_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_feedback_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        prompt_len = inputs["input_ids"].shape[1]
        feedback = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        # Clean up feedback - remove thinking tags if present (for Qwen3 models)
        if "<think>" in feedback:
            # Extract content after </think> if present
            if "</think>" in feedback:
                parts = feedback.split("</think>")
                feedback = parts[-1].strip()
                # If still empty, try to extract useful content from thinking
                if not feedback and len(parts) > 1:
                    # Get the thinking content as fallback
                    thinking = parts[0].replace("<think>", "").strip()
                    # Extract last sentence from thinking as summary
                    sentences = [s.strip() for s in thinking.split('.') if s.strip()]
                    if sentences:
                        feedback = sentences[-1] + "."
            else:
                feedback = feedback.split("<think>")[0].strip()

        # If still empty, provide a default based on score
        if not feedback:
            if predicted_score >= max_score * 0.8:
                feedback = "Strong response that addresses the prompt effectively."
            elif predicted_score >= max_score * 0.5:
                feedback = "Adequate response with room for improvement in depth or clarity."
            else:
                feedback = "Response needs improvement in addressing the prompt requirements."

        return feedback.strip()

    def score(self, answer: GeneratedAnswer) -> ScoringResult:
        """Score a single student answer using logprobs + generate feedback."""
        min_score, max_score = self._get_score_range(answer.essay_set)

        messages = self._build_messages(answer)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.steering_vector is not None and self.steering_config.coef != 0:
                with ActivationSteerer(
                    self.model,
                    self.steering_vector,
                    coeff=self.steering_config.coef,
                    layer_idx=self.steering_config.layer - 1,
                    positions="all",  # Apply to all positions for scoring
                ):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        # Get logits for the last token (next token prediction)
        last_logits = outputs.logits[0, -1, :]

        # Get score from logits
        predicted_score, score_probs = self._get_score_from_logits(
            last_logits, min_score, max_score
        )

        # Format probabilities as readable string
        prob_str = ", ".join([f"{s}:{p:.2%}" for s, p in sorted(score_probs.items())])

        # Generate text feedback if enabled
        quality_feedback = None
        if self.generate_feedback:
            quality_feedback = self._generate_feedback(answer, predicted_score)

        return ScoringResult(
            essay_id=answer.essay_id,
            essay_set=answer.essay_set,
            student_config=answer.steering_config,
            judge_config=self.steering_config.name,
            predicted_score=predicted_score,
            ground_truth_score=answer.ground_truth_score,
            reasoning=f"Score probabilities: {prob_str}",
            score_probs=score_probs,
            quality_feedback=quality_feedback,
        )

    def score_batch(self, answers: List[GeneratedAnswer]) -> List[ScoringResult]:
        """Score multiple student answers."""
        from tqdm import tqdm

        results = []
        for answer in tqdm(answers, desc=f"Scoring ({self.steering_config.name})"):
            result = self.score(answer)
            results.append(result)
        return results

    def score_essay(self, essay: Essay, generate_feedback: bool = False) -> ScoringResult:
        """Score an original essay from the dataset for QWK calculation.

        Args:
            essay: Original essay from ASAP-SAS dataset
            generate_feedback: Whether to generate quality feedback

        Returns:
            ScoringResult with predicted score vs ground truth
        """
        min_score, max_score = self._get_score_range(essay.essay_set)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_scoring_prompt_for_essay(essay)},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.steering_vector is not None and self.steering_config.coef != 0:
                with ActivationSteerer(
                    self.model,
                    self.steering_vector,
                    coeff=self.steering_config.coef,
                    layer_idx=self.steering_config.layer - 1,
                    positions="all",
                ):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        # Get logits for the last token
        last_logits = outputs.logits[0, -1, :]

        # Get score from logits
        predicted_score, score_probs = self._get_score_from_logits(
            last_logits, min_score, max_score
        )

        # Format probabilities
        prob_str = ", ".join([f"{s}:{p:.2%}" for s, p in sorted(score_probs.items())])

        # Generate feedback if requested (slower)
        quality_feedback = None
        if generate_feedback and self.generate_feedback:
            # Create a temporary GeneratedAnswer for feedback generation
            temp_answer = GeneratedAnswer(
                essay_id=essay.essay_id,
                essay_set=essay.essay_set,
                prompt=essay.prompt,
                generated_answer=essay.essay_text,
                steering_config="original",
                ground_truth_score=essay.avg_score,
            )
            quality_feedback = self._generate_feedback(temp_answer, predicted_score)

        return ScoringResult(
            essay_id=essay.essay_id,
            essay_set=essay.essay_set,
            student_config="original",  # Mark as original essay, not generated
            judge_config=self.steering_config.name,
            predicted_score=predicted_score,
            ground_truth_score=essay.avg_score,
            reasoning=f"Score probabilities: {prob_str}",
            score_probs=score_probs,
            quality_feedback=quality_feedback,
            answer_text=essay.essay_text[:500] + "..." if len(essay.essay_text) > 500 else essay.essay_text,
        )

    def score_essays(self, essays: List[Essay], generate_feedback: bool = False) -> List[ScoringResult]:
        """Score multiple original essays for QWK calculation.

        Args:
            essays: List of original essays from ASAP-SAS dataset
            generate_feedback: Whether to generate quality feedback

        Returns:
            List of ScoringResults
        """
        from tqdm import tqdm

        results = []
        for essay in tqdm(essays, desc=f"Scoring essays ({self.steering_config.name})"):
            result = self.score_essay(essay, generate_feedback=generate_feedback)
            results.append(result)
        return results


class ClaudeAPIJudge(BaseJudge):
    """Claude API judge for unbiased baseline scoring.

    Uses constrained output to get consistent score-based responses,
    plus generates text-based quality feedback.
    """

    # More constrained prompt for API - asks for just the number
    SCORING_PROMPT_TEMPLATE = """You are an expert essay grader. Score the following student answer.

Essay Prompt: {essay_prompt}

Rubric:
- Score range: {min_score} to {max_score}
- {min_score} = Poor/No understanding
- {max_score} = Excellent/Complete understanding

Student Answer:
{student_answer}

Respond with ONLY a single integer from {min_score} to {max_score}. No explanation, just the number."""

    # Prompt for generating quality feedback
    FEEDBACK_PROMPT_TEMPLATE = """You are an expert essay grader providing constructive feedback.

Essay Prompt: {essay_prompt}

Student Answer:
{student_answer}

You gave this answer a score of {score} out of {max_score}.

In 1-2 sentences, explain the key strengths and weaknesses of this answer. Be specific and constructive."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        generate_feedback: bool = True,
    ):
        """Initialize the Claude API judge.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Claude model to use
            generate_feedback: Whether to generate text quality feedback
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.generate_feedback = generate_feedback
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required for Claude API judge. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def _build_scoring_prompt(self, answer: GeneratedAnswer) -> str:
        """Build the scoring prompt for Claude API (uses constrained format)."""
        min_score, max_score = self._get_score_range(answer.essay_set)
        return self.SCORING_PROMPT_TEMPLATE.format(
            essay_prompt=answer.prompt,
            min_score=min_score,
            max_score=max_score,
            student_answer=answer.generated_answer,
        )

    def _parse_score(self, response: str, min_score: int, max_score: int) -> int:
        """Parse the score from Claude's response."""
        # Extract first number from response
        numbers = re.findall(r'\d+', response.strip())
        if numbers:
            score = int(numbers[0])
            return max(min_score, min(max_score, score))
        # Default to middle of range
        return (min_score + max_score) // 2

    def _generate_feedback(self, answer: GeneratedAnswer, predicted_score: int) -> str:
        """Generate text-based quality feedback using Claude API."""
        min_score, max_score = self._get_score_range(answer.essay_set)

        feedback_prompt = self.FEEDBACK_PROMPT_TEMPLATE.format(
            essay_prompt=answer.prompt,
            student_answer=answer.generated_answer[:1000],  # Truncate for efficiency
            score=predicted_score,
            max_score=max_score,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[
                {"role": "user", "content": feedback_prompt}
            ],
            system="You are an expert essay grader providing brief, constructive feedback.",
        )

        return response.content[0].text.strip()

    def score(self, answer: GeneratedAnswer) -> ScoringResult:
        """Score a single student answer using Claude API."""
        min_score, max_score = self._get_score_range(answer.essay_set)
        prompt = self._build_scoring_prompt(answer)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=5,  # Only need a single number
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are an essay grader. Respond with only a single integer score.",
        )

        response_text = response.content[0].text.strip()
        predicted_score = self._parse_score(response_text, min_score, max_score)

        # Generate text feedback if enabled
        quality_feedback = None
        if self.generate_feedback:
            try:
                quality_feedback = self._generate_feedback(answer, predicted_score)
            except Exception as e:
                quality_feedback = f"Feedback generation failed: {e}"

        return ScoringResult(
            essay_id=answer.essay_id,
            essay_set=answer.essay_set,
            student_config=answer.steering_config,
            judge_config=f"api_{self.model}",
            predicted_score=predicted_score,
            ground_truth_score=answer.ground_truth_score,
            reasoning=f"API response: {response_text}",
            quality_feedback=quality_feedback,
        )

    def score_batch(self, answers: List[GeneratedAnswer]) -> List[ScoringResult]:
        """Score multiple student answers using Claude API."""
        from tqdm import tqdm
        import time

        results = []
        for answer in tqdm(answers, desc="Scoring (Claude API)"):
            try:
                result = self.score(answer)
                results.append(result)
            except Exception as e:
                print(f"Error scoring essay {answer.essay_id}: {e}")
                # Add a placeholder result with mid-range score
                min_score, max_score = self._get_score_range(answer.essay_set)
                results.append(ScoringResult(
                    essay_id=answer.essay_id,
                    essay_set=answer.essay_set,
                    student_config=answer.steering_config,
                    judge_config=f"api_{self.model}",
                    predicted_score=(min_score + max_score) // 2,
                    ground_truth_score=answer.ground_truth_score,
                    reasoning=f"Error: {e}",
                ))
            # Small delay to avoid rate limiting
            time.sleep(0.1)

        return results

    def score_essay(self, essay: Essay, generate_feedback: bool = False) -> ScoringResult:
        """Score an original essay from the dataset for QWK calculation."""
        min_score, max_score = self._get_score_range(essay.essay_set)

        prompt = self.SCORING_PROMPT_TEMPLATE.format(
            essay_prompt=essay.prompt,
            min_score=min_score,
            max_score=max_score,
            student_answer=essay.essay_text,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
            system="You are an essay grader. Respond with only a single integer score.",
        )

        response_text = response.content[0].text.strip()
        predicted_score = self._parse_score(response_text, min_score, max_score)

        # Generate feedback if requested
        quality_feedback = None
        if generate_feedback and self.generate_feedback:
            feedback_prompt = self.FEEDBACK_PROMPT_TEMPLATE.format(
                essay_prompt=essay.prompt,
                student_answer=essay.essay_text[:1000],
                score=predicted_score,
                max_score=max_score,
            )
            try:
                feedback_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=150,
                    messages=[{"role": "user", "content": feedback_prompt}],
                    system="You are an expert essay grader providing brief, constructive feedback.",
                )
                quality_feedback = feedback_response.content[0].text.strip()
            except Exception as e:
                quality_feedback = f"Feedback generation failed: {e}"

        return ScoringResult(
            essay_id=essay.essay_id,
            essay_set=essay.essay_set,
            student_config="original",
            judge_config=f"api_{self.model}",
            predicted_score=predicted_score,
            ground_truth_score=essay.avg_score,
            reasoning=f"API response: {response_text}",
            quality_feedback=quality_feedback,
            answer_text=essay.essay_text[:500] + "..." if len(essay.essay_text) > 500 else essay.essay_text,
        )

    def score_essays(self, essays: List[Essay], generate_feedback: bool = False) -> List[ScoringResult]:
        """Score multiple original essays for QWK calculation."""
        from tqdm import tqdm
        import time

        results = []
        for essay in tqdm(essays, desc="Scoring essays (Claude API)"):
            try:
                result = self.score_essay(essay, generate_feedback=generate_feedback)
                results.append(result)
            except Exception as e:
                print(f"Error scoring essay {essay.essay_id}: {e}")
                min_score, max_score = self._get_score_range(essay.essay_set)
                results.append(ScoringResult(
                    essay_id=essay.essay_id,
                    essay_set=essay.essay_set,
                    student_config="original",
                    judge_config=f"api_{self.model}",
                    predicted_score=(min_score + max_score) // 2,
                    ground_truth_score=essay.avg_score,
                    reasoning=f"Error: {e}",
                ))
            time.sleep(0.1)

        return results
