"""
ASAP-SAS Essay Set Configurations loaded from extracted JSON.

The configurations are extracted from DOCX files using extract_essay_configs.py.
Each set contains:
- prompt: The question/task for the student
- context: Reading passage or experiment description (if applicable)
- rubric: Scoring criteria for the judge
- score_range: (min, max) scores
- topic: Brief description

Usage:
    from experiments.education.essay_sets import ESSAY_SETS, normalize_score, get_all_set_ids

    essay_set = ESSAY_SETS[1]
    print(essay_set["prompt"])
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_essay_configs() -> Dict[int, dict]:
    """Load essay configurations from JSON file."""
    config_path = Path(__file__).parent / "essay_configs.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Essay configs not found at {config_path}. "
            "Run 'python -m experiments.education.extract_essay_configs' first."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    # Convert string keys to int and tuple score_range
    result = {}
    for key, value in configs.items():
        set_id = int(key)
        value["score_range"] = tuple(value["score_range"])
        result[set_id] = value

    return result


# Load configurations at module import time
ESSAY_SETS = _load_essay_configs()


def get_essay_set(set_id: int) -> Optional[dict]:
    """Get essay set configuration by ID."""
    return ESSAY_SETS.get(set_id, None)


def get_all_set_ids() -> List[int]:
    """Get all available essay set IDs."""
    return sorted(ESSAY_SETS.keys())


def normalize_score(score: int, set_id: int) -> float:
    """Normalize a score to 0-1 range based on the set's score range."""
    min_score, max_score = ESSAY_SETS[set_id]["score_range"]
    if max_score == min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)


def denormalize_score(normalized: float, set_id: int) -> int:
    """Convert normalized score back to original range."""
    min_score, max_score = ESSAY_SETS[set_id]["score_range"]
    return round(normalized * (max_score - min_score) + min_score)


def reload_configs() -> None:
    """Reload configurations from JSON file (useful after re-extraction)."""
    global ESSAY_SETS
    ESSAY_SETS = _load_essay_configs()


if __name__ == "__main__":
    # Print summary when run directly
    print("ASAP-SAS Essay Set Configurations")
    print("=" * 60)

    for set_id in get_all_set_ids():
        config = ESSAY_SETS[set_id]
        print(f"\nSet {set_id}: {config['topic']}")
        print(f"  Score range: {config['score_range']}")
        print(f"  Context: {len(config['context'])} chars")
        print(f"  Prompt: {config['prompt'][:60]}...")
        print(f"  Rubric: {len(config['rubric'])} chars")
