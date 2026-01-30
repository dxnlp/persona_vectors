"""
Student answer generator with optional persona steering.
"""

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from activation_steer import ActivationSteerer
from .data_loader import Essay
from .config import SteeringConfig


@dataclass
class GeneratedAnswer:
    """Represents a generated student answer."""
    essay_id: int
    essay_set: int
    prompt: str
    generated_answer: str
    steering_config: str  # "good", "evil", or "unsteered"
    ground_truth_score: float


class StudentGenerator:
    """Generates student answers with optional persona steering."""

    SYSTEM_PROMPT = """You are a student taking an exam. Answer the question based on the provided context and prompt.
Your answer should be a short, focused response that directly addresses the question.
Write naturally as a student would, with appropriate depth for the question."""

    ANSWER_PROMPT_TEMPLATE = """Based on the following essay prompt, write a short answer essay.

Essay Prompt: {essay_prompt}

Context/Source Text (if applicable):
{essay_text}

Write your answer below. Be concise and directly address the prompt.

Answer:"""

    def __init__(
        self,
        model,
        tokenizer,
        steering_config: Optional[SteeringConfig] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ):
        """Initialize the student generator.

        Args:
            model: The HuggingFace model
            tokenizer: The tokenizer
            steering_config: Optional steering configuration
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model = model
        self.tokenizer = tokenizer
        self.steering_config = steering_config or SteeringConfig.unsteered()
        self.max_tokens = max_tokens
        self.temperature = temperature

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

    def _build_prompt(self, essay: Essay) -> str:
        """Build the prompt for answer generation.

        Args:
            essay: The essay object

        Returns:
            The formatted prompt string
        """
        return self.ANSWER_PROMPT_TEMPLATE.format(
            essay_prompt=essay.prompt,
            essay_text=essay.essay_text,
        )

    def _build_messages(self, essay: Essay) -> List[Dict[str, str]]:
        """Build chat messages for the model.

        Args:
            essay: The essay object

        Returns:
            List of message dictionaries
        """
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(essay)},
        ]

    def generate_answer(self, essay: Essay) -> GeneratedAnswer:
        """Generate a single answer for an essay.

        Args:
            essay: The essay to answer

        Returns:
            Generated answer with metadata
        """
        messages = self._build_messages(essay)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_kwargs = {
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "max_new_tokens": self.max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }

        with torch.no_grad():
            if self.steering_vector is not None and self.steering_config.coef != 0:
                with ActivationSteerer(
                    self.model,
                    self.steering_vector,
                    coeff=self.steering_config.coef,
                    layer_idx=self.steering_config.layer - 1,  # 0-indexed
                    positions="response",
                ):
                    outputs = self.model.generate(**inputs, **generate_kwargs)
            else:
                outputs = self.model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        )

        return GeneratedAnswer(
            essay_id=essay.essay_id,
            essay_set=essay.essay_set,
            prompt=essay.prompt,
            generated_answer=generated_text.strip(),
            steering_config=self.steering_config.name,
            ground_truth_score=essay.avg_score,
        )

    def generate_answers(
        self,
        essays: List[Essay],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[GeneratedAnswer]:
        """Generate answers for multiple essays.

        Args:
            essays: List of essays to answer
            batch_size: Batch size for generation
            show_progress: Whether to show progress bar

        Returns:
            List of generated answers
        """
        from tqdm import tqdm

        answers = []
        iterator = tqdm(essays, desc=f"Generating ({self.steering_config.name})") if show_progress else essays

        for essay in iterator:
            answer = self.generate_answer(essay)
            answers.append(answer)

        return answers

    def generate_batch(
        self,
        essays: List[Essay],
        batch_size: int = 4,
    ) -> List[GeneratedAnswer]:
        """Generate answers for a batch of essays.

        More efficient than generate_answers for larger batches.

        Args:
            essays: List of essays to answer
            batch_size: Batch size for generation

        Returns:
            List of generated answers
        """
        from tqdm import trange

        answers = []

        for i in trange(0, len(essays), batch_size, desc=f"Batches ({self.steering_config.name})"):
            batch_essays = essays[i:i + batch_size]

            # Build prompts for all essays in batch
            messages_list = [self._build_messages(e) for e in batch_essays]
            prompts = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_list
            ]

            # Tokenize batch
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            generate_kwargs = {
                "do_sample": self.temperature > 0,
                "temperature": self.temperature if self.temperature > 0 else None,
                "max_new_tokens": self.max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "use_cache": True,
            }

            with torch.no_grad():
                if self.steering_vector is not None and self.steering_config.coef != 0:
                    with ActivationSteerer(
                        self.model,
                        self.steering_vector,
                        coeff=self.steering_config.coef,
                        layer_idx=self.steering_config.layer - 1,
                        positions="response",
                    ):
                        outputs = self.model.generate(**inputs, **generate_kwargs)
                else:
                    outputs = self.model.generate(**inputs, **generate_kwargs)

            prompt_len = inputs["input_ids"].shape[1]

            for j, essay in enumerate(batch_essays):
                generated_text = self.tokenizer.decode(
                    outputs[j][prompt_len:], skip_special_tokens=True
                )
                answers.append(GeneratedAnswer(
                    essay_id=essay.essay_id,
                    essay_set=essay.essay_set,
                    prompt=essay.prompt,
                    generated_answer=generated_text.strip(),
                    steering_config=self.steering_config.name,
                    ground_truth_score=essay.avg_score,
                ))

        return answers
