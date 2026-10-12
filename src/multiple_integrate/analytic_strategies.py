"""Private import facade for analytic integration strategies."""

from multiple_integrate.layer_cake_strategies import (  # noqa: F401
    _try_general_layercake,
    _try_poly_layercake,
    _try_superellipse,
)
from multiple_integrate.pushforward_strategies import (  # noqa: F401
    _try_linear_pushforward,
    _try_monotone_change,
    _try_piecewise_change,
)
from multiple_integrate.quadratic_gaussian import (  # noqa: F401
    _try_fullspace_quadratic,
    _try_gauss_linear_map,
    _try_halfspace_quadratic,
)
from multiple_integrate.region_strategies import (  # noqa: F401
    _region_shortcut,
    _try_graph_reversal,
)
from multiple_integrate.separable_strategies import (  # noqa: F401
    _try_additive_separable,
    _try_product_separable,
)
from multiple_integrate.strategy_common import (  # noqa: F401
    _const_result,
    _iterated_integrate,
)
