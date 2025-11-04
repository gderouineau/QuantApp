# strategies/base.py

from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class BaseStrategy(ABC):
    """
    Contrat minimal :
        evaluate(asset_data: pd.DataFrame, indicators: pd.DataFrame) -> dict
    Retour attendu :
        {
            'signal': bool,
            'score': float (0..100),
            'strength': 'WEAK' | 'MEDIUM' | 'STRONG',
            'details': dict
        }
    """

    def __init__(self, parameters: Dict | None = None):
        self.parameters = parameters or {}

    @abstractmethod
    def evaluate(self, asset_data: pd.DataFrame, indicators: pd.DataFrame) -> Dict:
        raise NotImplementedError

    # Helpers communs (facultatif mais pratique)
    @staticmethod
    def strength_from_score(score: float, medium: float = 50.0, strong: float = 70.0) -> str:
        if score >= strong:
            return "STRONG"
        if score >= medium:
            return "MEDIUM"
        return "WEAK"

    def get_latest_score(self, asset_data: pd.DataFrame, indicators: pd.DataFrame) -> float:
        result = self.evaluate(asset_data, indicators)
        try:
            return float(result.get('score', 0.0))
        except Exception:
            return 0.0
