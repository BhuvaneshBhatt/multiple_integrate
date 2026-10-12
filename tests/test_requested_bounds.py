"""Coordinate restrictions must precede native geometry/cache dispatch."""

import pytest
import sympy as sp

from multiple_integrate import (
    GeometricPrincipalValue,
    HadamardFinitePart,
    HypersurfacePrincipalValue,
    clear_cache,
    integrate,
    regularized_result,
)
from multiple_integrate.region_types import BoxRegion

x, y = sp.symbols("x y", real=True)
box = BoxRegion(((x, -1, 1), (y, -1, 1)))


@pytest.mark.parametrize(
    "bounds",
    [
        {x: (-sp.Rational(1, 2), sp.Rational(1, 2))},
        {"x": (-sp.Rational(1, 2), sp.Rational(1, 2))},
        [(x, -sp.Rational(1, 2), sp.Rational(1, 2))],
    ],
)
@pytest.mark.parametrize(
    "prescription", [None, HypersurfacePrincipalValue(x * y), GeometricPrincipalValue(point=(0, 0))]
)
def test_native_paths_and_cached_values_use_clipped_domain(bounds, prescription):
    clear_cache()
    assert integrate(1, region=box) == 4
    assert integrate(1, region=box, bounds=bounds, principal_value=prescription) == 2


def test_product_finite_part_scales_with_clipped_width():
    result = regularized_result(
        1 / (x * y) ** 2,
        region=box,
        bounds={x: (-sp.Rational(1, 2), sp.Rational(1, 2))},
        regularization=HadamardFinitePart(hypersurface=x * y),
    )
    assert result.value == 8
    assert result.provenance.details["widths"] == (sp.Rational(1, 2), 1)


def test_bound_on_unknown_coordinate_is_validation_error():
    with pytest.raises(ValueError, match="coordinate"):
        integrate(1, region=box, bounds={sp.Symbol("other"): (0, 1)})


def test_flat_regularization_does_not_retain_provenance():
    from multiple_integrate.regularization import _regularization_trace

    before = _regularization_trace()
    for _ in range(12):
        assert integrate(x**-2, (x, -1, 1), regularization=HadamardFinitePart()) == -2
    assert _regularization_trace() == before == ()


def test_structured_provenance_remains_scoped_and_replayable():
    from multiple_integrate import verify_regularization_result
    from multiple_integrate.regularization import _regularization_trace

    result = regularized_result(x**-2, (x, -1, 1), regularization=HadamardFinitePart())
    assert result.provenance.cutoff_integral is not None
    assert verify_regularization_result(result).valid
    assert _regularization_trace() == ()
