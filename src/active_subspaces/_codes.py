"""Static registry of error and warning codes (spec 06 §4.3).

``CODES`` maps every code to ``(class name, one-line meaning)``. It is the
complete v1.0 table: nothing registers codes at run time, so the table does
not depend on which modules have been imported. Codes are permanent; a new
code is added here (and to spec 06) through the consistency pass.

Layer 0: this module imports nothing from the package.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = ()

CODES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        # InputError
        "input.dtype": (
            "InputError",
            "unsupported array dtype (bool, complex, object, string, datetime)",
        ),
        "input.shape": ("InputError", "wrong number of dimensions or wrong shape"),
        "input.vector_qoi": ("InputError", "f has more than one column"),
        "input.size_mismatch": ("InputError", "arrays disagree on M"),
        "input.nonfinite": ("InputError", "NaN or +-inf in an input"),
        "input.out_of_space": ("InputError", "point outside the support of the input space"),
        "input.out_of_domain": (
            "InputError",
            "an active-variable point y lies outside the active domain",
        ),
        "input.weights": (
            "InputError",
            "quadrature weights negative or not summing to 1, or sample weights negative or all "
            "zero",
        ),
        "input.parameter": (
            "InputError",
            (
                "invalid scalar, choice or constructor argument; details['reason'] distinguishes "
                "space_mismatch, subspace_mismatch and other spec-defined cases"
            ),
        ),
        "input.too_few_samples": ("InputError", "fewer samples than the method needs"),
        "input.no_bootstrap": (
            "InputError",
            "the operation needs bootstrap replicates and the result has none",
        ),
        "input.unknown_model": ("InputError", "no model of that name is in the registry"),
        "input.duplicate_model": (
            "InputError",
            "a registry would contain two models with the same name",
        ),
        # SchemaError
        "schema.unknown_type": ("SchemaError", "unknown result type tag"),
        "schema.too_new": ("SchemaError", "schema version newer than this library"),
        "schema.invalid": (
            "SchemaError",
            "malformed dict or file, unknown or missing field, or broken invariants",
        ),
        # UnsupportedError
        "unsupported.space": (
            "UnsupportedError",
            "operation not defined for this kind of input space",
        ),
        "unsupported.model_gradient": (
            "UnsupportedError",
            "gradients requested from a model that has neither grad nor value_and_grad",
        ),
        # NumericalError
        "numerical.nonfinite": (
            "NumericalError",
            "a computation produced NaN or +-inf from finite inputs",
        ),
        "numerical.singular": ("NumericalError", "a required factorization or solve failed"),
        "numerical.infeasible_draw": (
            "NumericalError",
            "a sampled point of the inverse map violates its constraints beyond tolerance",
        ),
        # SolverError
        "solver.failed": (
            "SolverError",
            "backend reported failure or an iteration limit (status and message in details)",
        ),
        "solver.infeasible": ("SolverError", "constraints infeasible"),
        "solver.unbounded": ("SolverError", "the problem is unbounded"),
        # ModelError
        "model.raised": (
            "ModelError",
            "the user's model raised; the original exception is __cause__",
        ),
        "model.nonfinite": ("ModelError", "the user's model returned NaN or +-inf"),
        "model.shape": ("ModelError", "the user's model returned the wrong shape"),
        "model.dtype": (
            "ModelError",
            "the user's model returned a dtype that is not integer or floating",
        ),
        "model.executor": (
            "ModelError",
            "the executor failed to run a batch (e.g. a worker process died)",
        ),
        # ValidityWarning
        "validity.assumption": (
            "ValidityWarning",
            "estimator, input space and dimension rule outside their assumptions",
        ),
        "validity.bootstrap_weights": (
            "ValidityWarning",
            "bootstrap requested with non-uniform quadrature weights; i.i.d. resampling does not "
            "hold",
        ),
        # AccuracyWarning
        "accuracy.curvature_clipped": (
            "AccuracyWarning",
            "lift curvature projected onto the PSD cone",
        ),
        "accuracy.correlated_draws": (
            "AccuracyWarning",
            "hit-and-run draws are correlated beyond what the error report accounts for: drawn "
            "without thinning "
            "(thin == 1), or chains too short for tau-hat (2 <= N_MC < 20)",
        ),
        "accuracy.degenerate_fiber": (
            "AccuracyWarning",
            "the inverse map returned a fixed point for a fiber with empty interior",
        ),
        "accuracy.degenerate_matrix": (
            "AccuracyWarning",
            "the estimator's matrix is exactly zero, so the eigenvectors are arbitrary (W = I "
            "returned)",
        ),
        "accuracy.rank_deficient": (
            "AccuracyWarning",
            "a least-squares fit is rank-deficient; the minimum-norm solution is used",
        ),
        "accuracy.zero_dimension": (
            "AccuracyWarning",
            "the ladle chose k = 0; n is set to 1",
        ),
        "accuracy.loo_undefined": (
            "AccuracyWarning",
            "leave-one-out residuals are undefined because a point fully determines its own fit",
        ),
        "accuracy.empty_bounds": (
            "AccuracyWarning",
            "the published bounds on the RBF hyperparameter are empty; the upper bound is used",
        ),
        "accuracy.flat_objective": (
            "AccuracyWarning",
            "the RBF likelihood is flat over at least four decades of the hyperparameter; the "
            "selected value is arbitrary (pass f_variance or bounds)",
        ),
        "accuracy.extrapolation": (
            "AccuracyWarning",
            "a Gaussian-case minimizer lies beyond the 99.9 % ball of the reference density",
        ),
        "accuracy.weight_deficit": (
            "AccuracyWarning",
            "quadrature weights renormalized after Monte Carlo samples fell outside the hull",
        ),
        "accuracy.large_dimension": (
            "AccuracyWarning",
            "the automatically chosen dimension exceeds the documented envelope",
        ),
        # ConvergenceWarning
        "convergence.max_iter": (
            "ConvergenceWarning",
            "an iteration limit was reached; best iterate returned",
        ),
        "convergence.failed_starts": (
            "ConvergenceWarning",
            "some multistart starts failed; the rest were used",
        ),
        # ExperimentalWarning
        "experimental.module": (
            "ExperimentalWarning",
            "active_subspaces.experimental was imported; issued once",
        ),
        # ActiveSubspacesDeprecationWarning: the deprecated.* family has no codes in v1.0 [D-152]
    }
)
