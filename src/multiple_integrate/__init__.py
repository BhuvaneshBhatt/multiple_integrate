"""multiple_integrate
====================

Symbolic definite integration for Python, with an emphasis on structured multiple/multivariate integrals.

The main public entry point is ``integrate``.
"""

from multiple_integrate.api.integration import (
    integrate_result,
    regularized_result,
)
from multiple_integrate.core import clear_cache, integrate
from multiple_integrate.numerical_verification import (
    NumericalVerification,
    verify_finite_result_numerically,
)
from multiple_integrate.parameter_cells import ParameterCell, integrate_cells
from multiple_integrate.policy import IntegrationPolicy, StrategyBudget, StrategyBudgetExceeded
from multiple_integrate.principal_value import (
    CauchyPrincipalValue,
    GeometricPrincipalValue,
    HypersurfacePrincipalValue,
    IteratedPrincipalValue,
)
from multiple_integrate.regions import Boole, region_from_ranges
from multiple_integrate.regularization import (
    HadamardFinitePart,
    RegularizationResult,
    RegularizationVerification,
    verify_regularization_result,
)
from multiple_integrate.results import (
    ConvergenceEvidence,
    CoverageEvidence,
    IntegrationOutcome,
    IntegrationResult,
    IntegrationStatus,
    UnevaluatedRegionIntegral,
)

__all__ = [
    "integrate",
    "NumericalVerification",
    "verify_finite_result_numerically",
    "integrate_cells",
    "ParameterCell",
    "integrate_result",
    "regularized_result",
    "IntegrationResult",
    "IntegrationOutcome",
    "CoverageEvidence",
    "ConvergenceEvidence",
    "IntegrationPolicy",
    "StrategyBudget",
    "StrategyBudgetExceeded",
    "IntegrationStatus",
    "UnevaluatedRegionIntegral",
    "clear_cache",
    "CauchyPrincipalValue",
    "IteratedPrincipalValue",
    "GeometricPrincipalValue",
    "HypersurfacePrincipalValue",
    "HadamardFinitePart",
    "RegularizationResult",
    "RegularizationVerification",
    "verify_regularization_result",
    "Boole",
    "region_from_ranges",
]
