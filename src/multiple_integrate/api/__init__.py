"""Curated public API for MultipleIntegrate."""

from multiple_integrate.api.integration import integrate_result
from multiple_integrate.core import clear_cache, integrate
from multiple_integrate.results import (
    IntegralCase,
    IntegrationResult,
    IntegrationStatus,
    StrategyAttempt,
    UnevaluatedRegionIntegral,
)

__all__ = [
    "integrate",
    "integrate_result",
    "clear_cache",
    "IntegralCase",
    "IntegrationResult",
    "IntegrationStatus",
    "UnevaluatedRegionIntegral",
    "StrategyAttempt",
]
