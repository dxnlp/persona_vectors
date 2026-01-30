# Education experiment package for persona vectors research
# Studying how evil/good persona steering affects automated essay scoring

from .config import ExperimentConfig
from .data_loader import ASAPDataLoader
from .student import StudentGenerator
from .judge import LocalJudge, ClaudeAPIJudge
from .metrics import calculate_qwk, calculate_agreement_stats
from .run_experiment import run_experiment

__all__ = [
    "ExperimentConfig",
    "ASAPDataLoader",
    "StudentGenerator",
    "LocalJudge",
    "ClaudeAPIJudge",
    "calculate_qwk",
    "calculate_agreement_stats",
    "run_experiment",
]
