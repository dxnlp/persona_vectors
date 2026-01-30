"""
Experiment configuration for education scoring research.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for education scoring experiments."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B"
    vector_path: Optional[str] = None  # Path to persona vectors directory
    trait: str = "evil"  # Trait to use for steering (evil vector, negate for good)

    # Steering configuration
    steering_coef: float = 2.0  # Coefficient for steering vector
    layer: int = 15  # Layer to apply steering (Qwen3-4B has 40 layers)
    steering_type: str = "response"  # "response", "prompt", or "all"

    # Data configuration
    data_path: str = "asap-sas/train.tsv"
    sample_size: int = 20  # Number of essays to sample per set
    essay_sets: List[int] = field(default_factory=lambda: [1, 2])  # Which essay sets to use

    # API configuration
    use_api_judge: bool = True
    api_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: Optional[str] = None

    # Generation configuration
    max_tokens: int = 500
    temperature: float = 0.7

    # Output configuration
    output_dir: str = "experiments/education/results"
    experiment_name: str = "persona_scoring"

    # Test mode
    test_mode: bool = False  # If True, use 1 essay only for quick validation

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.test_mode:
            self.sample_size = 1
            self.essay_sets = [1]

        if self.vector_path is None:
            # Default vector path based on model name
            model_short = self.model_name.split("/")[-1]
            self.vector_path = f"persona_vectors/{model_short}"

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_vector_path(self, vector_type: str = "response_avg") -> str:
        """Get the full path to a specific vector file.

        Args:
            vector_type: Type of vector ("response_avg", "prompt_avg", or "prompt_last")

        Returns:
            Full path to the vector file
        """
        return f"{self.vector_path}/{self.trait}_{vector_type}_diff.pt"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        import yaml
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


@dataclass
class SteeringConfig:
    """Configuration for a specific steering setup."""
    name: str  # "good", "evil", or "unsteered"
    coef: float  # Steering coefficient (0 for unsteered, positive for evil, negative for good)
    layer: int
    vector_path: Optional[str] = None

    @classmethod
    def good(cls, layer: int, coef: float, vector_path: str) -> "SteeringConfig":
        """Create a good-steered configuration (negative coefficient)."""
        return cls(name="good", coef=-abs(coef), layer=layer, vector_path=vector_path)

    @classmethod
    def evil(cls, layer: int, coef: float, vector_path: str) -> "SteeringConfig":
        """Create an evil-steered configuration (positive coefficient)."""
        return cls(name="evil", coef=abs(coef), layer=layer, vector_path=vector_path)

    @classmethod
    def unsteered(cls) -> "SteeringConfig":
        """Create an unsteered configuration."""
        return cls(name="unsteered", coef=0.0, layer=0, vector_path=None)


# ASAP-SAS Essay Set Information
ESSAY_SET_INFO = {
    1: {
        "topic": "Effects of computers on society",
        "score_range": (0, 3),
        "prompt": "Describe the effects computers have had on the way people communicate.",
    },
    2: {
        "topic": "Censorship in libraries",
        "score_range": (0, 4),
        "prompt": "Should libraries censor content? Discuss the pros and cons.",
    },
    3: {
        "topic": "Source-based writing",
        "score_range": (0, 3),
        "prompt": "Based on the article, describe the process and explain the significance.",
    },
    4: {
        "topic": "Source-based writing",
        "score_range": (0, 3),
        "prompt": "Based on the article, explain the key findings.",
    },
    5: {
        "topic": "Narrative writing",
        "score_range": (0, 4),
        "prompt": "Write a narrative about a significant experience.",
    },
    6: {
        "topic": "Narrative writing",
        "score_range": (0, 4),
        "prompt": "Write a narrative about a memorable event.",
    },
    7: {
        "topic": "Persuasive writing",
        "score_range": (0, 30),
        "prompt": "Write a persuasive essay on a given topic.",
    },
    8: {
        "topic": "Persuasive writing",
        "score_range": (0, 60),
        "prompt": "Write a persuasive essay on a given topic.",
    },
}
