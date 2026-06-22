"""
src/data/normalizer.py

Per-country z-score normalization.
Critical: normalize per country, not globally.
India's case counts and Germany's are incomparable raw.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class CountryStats:
    """Stores normalization parameters per country for inverse transform."""
    iso_code: str
    mean: float
    std:  float


@dataclass 
class Normalizer:
    stats: dict = field(default_factory=dict)   # iso_code → CountryStats

    def fit_transform(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fit per-country stats and normalize in one pass.
        Stores stats for inverse transform at inference time.
        """
        df = df.copy()
        for iso, group in df.groupby("iso_code"):
            vals  = group[col].dropna()
            mean  = vals.mean()
            std   = vals.std()
            std   = std if std > 1e-8 else 1.0   # avoid division by zero

            self.stats[iso] = CountryStats(iso, mean, std)
            mask = df["iso_code"] == iso
            df.loc[mask, col] = (df.loc[mask, col] - mean) / std

        return df

    def inverse_transform(self, iso: str, values: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale."""
        s = self.stats[iso]
        return values * s.std + s.mean