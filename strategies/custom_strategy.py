# strategies/custom_strategy.py

import pandas as pd
from typing import Dict
from .base import BaseStrategy


class CustomStrategy(BaseStrategy):
    """
    Exécute du code Python stocké en base (ADMIN ONLY).
    Sandbox à faire plus tard.
    """

    def __init__(self, code: str, parameters: Dict | None = None):
        super().__init__(parameters)
        self.code = code or ""

    def evaluate(self, asset_data: pd.DataFrame, indicators: pd.DataFrame) -> Dict:
        # Contexte d'exécution : volontairement simple pour tests
        context: Dict = {
            'asset_data': asset_data,
            'indicators': indicators,
            'parameters': self.parameters,
            'pd': pd,
            # Valeurs par défaut (l’admin les écrase dans son code)
            'signal': False,
            'score': 0.0,
            'strength': 'WEAK',
            'details': {},
        }

        try:
            # ⚠️ Non-sécurisé : réservé à l'admin. Sandbox viendra ensuite.
            exec(self.code, context)

            return {
                'signal': bool(context.get('signal', False)),
                'score': float(context.get('score', 0.0)),
                'strength': str(context.get('strength', 'WEAK')),
                'details': context.get('details', {}) or {},
            }

        except Exception as e:
            return {
                'signal': False,
                'score': 0.0,
                'strength': 'WEAK',
                'details': {'error': str(e)},
            }
