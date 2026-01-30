"""
Data loader for ASAP-SAS (Automated Student Assessment Prize - Short Answer Scoring) dataset.

Dataset: https://www.kaggle.com/c/asap-sas/data
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import ESSAY_SET_INFO


@dataclass
class Essay:
    """Represents a single essay from the ASAP-SAS dataset."""
    essay_id: int
    essay_set: int
    essay_text: str
    score1: int
    score2: Optional[int] = None

    @property
    def avg_score(self) -> float:
        """Average of the two scores (or just score1 if score2 is missing)."""
        if self.score2 is not None:
            return (self.score1 + self.score2) / 2
        return float(self.score1)

    @property
    def score_range(self) -> Tuple[int, int]:
        """Get the valid score range for this essay set."""
        return ESSAY_SET_INFO.get(self.essay_set, {}).get("score_range", (0, 4))

    @property
    def prompt(self) -> str:
        """Get the essay prompt for this essay set."""
        return ESSAY_SET_INFO.get(self.essay_set, {}).get("prompt", "Write an essay on the given topic.")


class ASAPDataLoader:
    """Loader for the ASAP-SAS dataset."""

    # Expected column names (may vary by dataset version)
    COLUMN_MAPPINGS = {
        "essay_id": ["essay_id", "EssayId", "Id"],
        "essay_set": ["essay_set", "EssaySet", "prompt"],
        "essay_text": ["essay", "EssayText", "essay_text", "text"],
        "score1": ["score1", "Score1", "domain1_score", "rater1_domain1"],
        "score2": ["score2", "Score2", "domain2_score", "rater2_domain1"],
    }

    def __init__(self, data_path: str):
        """Initialize the data loader.

        Args:
            data_path: Path to the ASAP-SAS CSV file
        """
        self.data_path = Path(data_path)
        self._df: Optional[pd.DataFrame] = None

    def _find_column(self, df: pd.DataFrame, target: str) -> Optional[str]:
        """Find the actual column name in the dataframe.

        Args:
            df: The dataframe to search
            target: The target column type (e.g., "essay_id")

        Returns:
            The actual column name, or None if not found
        """
        possible_names = self.COLUMN_MAPPINGS.get(target, [target])
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def load(self) -> pd.DataFrame:
        """Load the dataset from disk.

        Returns:
            The loaded dataframe with standardized column names
        """
        if self._df is not None:
            return self._df

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download the ASAP-SAS dataset from "
                "https://www.kaggle.com/c/asap-sas/data and place it in the data directory."
            )

        # Load the CSV
        df = pd.read_csv(self.data_path)

        # Standardize column names
        column_map = {}
        for target in ["essay_id", "essay_set", "essay_text", "score1", "score2"]:
            actual = self._find_column(df, target)
            if actual and actual != target:
                column_map[actual] = target

        if column_map:
            df = df.rename(columns=column_map)

        # Validate required columns
        required = ["essay_id", "essay_set", "essay_text", "score1"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        self._df = df
        return df

    def get_essays(
        self,
        essay_sets: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> List[Essay]:
        """Get essays from the dataset.

        Args:
            essay_sets: List of essay set IDs to include (None = all)
            sample_size: Number of essays to sample per set (None = all)
            random_state: Random seed for reproducibility

        Returns:
            List of Essay objects
        """
        df = self.load()

        # Filter by essay set
        if essay_sets is not None:
            df = df[df["essay_set"].isin(essay_sets)]

        # Sample if requested
        if sample_size is not None:
            sampled_dfs = []
            for essay_set in df["essay_set"].unique():
                set_df = df[df["essay_set"] == essay_set]
                n_samples = min(sample_size, len(set_df))
                sampled_dfs.append(set_df.sample(n=n_samples, random_state=random_state))
            df = pd.concat(sampled_dfs, ignore_index=True)

        # Convert to Essay objects
        essays = []
        for _, row in df.iterrows():
            score2 = row.get("score2")
            if pd.isna(score2):
                score2 = None
            else:
                score2 = int(score2)

            essays.append(Essay(
                essay_id=int(row["essay_id"]),
                essay_set=int(row["essay_set"]),
                essay_text=str(row["essay_text"]),
                score1=int(row["score1"]),
                score2=score2,
            ))

        return essays

    def get_essay_set_info(self, essay_set: int) -> Dict:
        """Get information about a specific essay set.

        Args:
            essay_set: The essay set ID

        Returns:
            Dictionary with topic, score_range, and prompt
        """
        return ESSAY_SET_INFO.get(essay_set, {
            "topic": f"Essay Set {essay_set}",
            "score_range": (0, 4),
            "prompt": "Write an essay on the given topic.",
        })

    def get_available_sets(self) -> List[int]:
        """Get list of available essay set IDs in the dataset."""
        df = self.load()
        return sorted(df["essay_set"].unique().tolist())

    def get_stats(self) -> Dict:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        df = self.load()
        stats = {
            "total_essays": len(df),
            "essay_sets": {},
        }

        for essay_set in df["essay_set"].unique():
            set_df = df[df["essay_set"] == essay_set]
            stats["essay_sets"][int(essay_set)] = {
                "count": len(set_df),
                "score1_mean": set_df["score1"].mean(),
                "score1_std": set_df["score1"].std(),
                "score_range": ESSAY_SET_INFO.get(essay_set, {}).get("score_range", (0, 4)),
            }

        return stats


def create_sample_dataset(output_path: str, n_samples: int = 20):
    """Create a sample dataset for testing when full dataset is not available.

    This creates a minimal CSV with synthetic essays for testing the pipeline.

    Args:
        output_path: Path to save the sample CSV
        n_samples: Number of samples to create per essay set
    """
    import random

    data = []
    essay_id = 1

    sample_essays = [
        "Computers have changed communication by making it faster and easier. We can now send emails and messages instantly. However, some people argue that this has made communication less personal.",
        "Libraries should not censor books because everyone has the right to access information. Censorship limits freedom of thought and expression. Students need access to diverse viewpoints.",
        "The experiment showed that temperature affects the rate of reaction. When we increased the temperature, the reaction happened faster. This supports our hypothesis.",
        "My most memorable experience was climbing Mount Washington. The trail was difficult but the view from the top was amazing. I learned that persistence pays off.",
    ]

    for essay_set in [1, 2, 3, 4]:
        score_range = ESSAY_SET_INFO.get(essay_set, {}).get("score_range", (0, 4))
        for i in range(n_samples):
            base_essay = sample_essays[(essay_set - 1) % len(sample_essays)]
            # Add some variation
            essay_text = base_essay + f" (Sample {i+1} for set {essay_set})"

            score1 = random.randint(score_range[0], score_range[1])
            score2 = random.randint(score_range[0], score_range[1])

            data.append({
                "essay_id": essay_id,
                "essay_set": essay_set,
                "essay_text": essay_text,
                "score1": score1,
                "score2": score2,
            })
            essay_id += 1

    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset with {len(df)} essays at {output_path}")
