# Geometry guide

See the [method reference guide](method-references.md) for the identities, theorem sources,
implementation map and attribution limits behind these methods.

The detailed guides and executable examples are collected in the [geometry casebook](geometry-casebook.md).

- [Parameter geometry over complete cells](geometry/parameter-geometry.md)
- [Cusp contact and vanishing proofs](geometry/cusp-proofs.md)
- [Exact half-integer radial series](geometry/radial-series.md)
- [Clipped polygon and graph cutoffs](geometry/clipped-cutoffs.md)

Regression references: `tests/test_geometry_extensions.py`, `tests/test_geometry_casebook_properties.py`, `tests/test_serious_boundary_regressions.py`. Bernstein certificates and the contact threshold `s<1` are sufficient proofs for the supported families. An exact `Sum` and a proved cutoff window have different semantics.
