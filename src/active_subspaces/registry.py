"""An immutable table of named models.

The built-in test problems are entries of a registry built by
``active_subspaces.test_functions.builtin_models()``; an application builds
its own with ``builtin_models().with_models(...)``. There is no module-level
mutable registry, and nothing here imports or executes code named by a
string or received as data.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import MappingProxyType
from typing import Any, cast

from active_subspaces.errors import InputError
from active_subspaces.models import Model

__all__ = ("ModelRegistry",)


def _check_model(model: object) -> Model:
    if not isinstance(model, Model):
        raise InputError(
            f"models: expected Model instances, got {type(model).__name__}",
            code="input.parameter",
            details={"param": "models", "got": type(model).__name__},
        )
    if model.name is None:
        raise InputError(
            "models: a registered model needs a name",
            code="input.parameter",
            details={"param": "models", "reason": "unnamed"},
        )
    return model


def _duplicate(name: str) -> InputError:
    return InputError(
        f"a registry cannot hold two models named {name!r}",
        code="input.duplicate_model",
        details={"name": name},
    )


class ModelRegistry:
    """An immutable table of named models.

    Parameters
    ----------
    models : iterable of Model, default ()
        Models with unique, non-``None`` names.

    Raises
    ------
    InputError
        ``input.parameter`` if ``models`` is not an iterable of ``Model`` or a
        model has no name; ``input.duplicate_model`` if two models share a
        name.

    See Also
    --------
    active_subspaces.Model : The registered objects.
    active_subspaces.test_functions.builtin_models : The registry of built-in test problems.

    Notes
    -----
    ``registry[name]``, ``name in registry``, ``len(registry)`` and iteration
    over the names (in sorted order) work as for a mapping. An unknown name
    raises ``InputError`` (``input.unknown_model``), not ``KeyError``; the
    class is deliberately not a ``collections.abc.Mapping``. A registry
    pickles by its models.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(2)
    >>> first = ac.Model(space=space, value=lambda X: X[:, 0], name="first")
    >>> registry = ac.ModelRegistry([first])
    >>> "first" in registry, len(registry), list(registry)
    (True, 1, ['first'])
    >>> registry["first"] is first
    True
    """

    __slots__ = ("_models",)
    _models: MappingProxyType[str, Model]

    def __init__(self, models: Iterable[Model] = ()) -> None:
        if isinstance(models, Model) or not hasattr(models, "__iter__"):
            raise InputError(
                "models: expected an iterable of Model",
                code="input.parameter",
                details={"param": "models"},
            )
        table: dict[str, Model] = {}
        for model in models:
            _check_model(model)
            name = cast(str, model.name)  # _check_model rejects None
            if name in table:
                raise _duplicate(name)
            table[name] = model
        object.__setattr__(self, "_models", MappingProxyType(dict(sorted(table.items()))))

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse to set attributes (immutable)."""
        raise AttributeError("ModelRegistry is immutable; use with_models to build a new one")

    def __delattr__(self, name: str) -> None:
        """Refuse to delete attributes (immutable)."""
        raise AttributeError("ModelRegistry is immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by the registered models."""
        return (ModelRegistry, (tuple(self._models.values()),))

    def get(self, name: str) -> Model:
        """Return the model called ``name``.

        Parameters
        ----------
        name : str
            A registered name.

        Returns
        -------
        Model
            The registered model.

        Raises
        ------
        InputError
            ``input.unknown_model`` for an unknown name; ``details["known"]``
            lists the registered names.

        See Also
        --------
        names : The registered names.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> registry = ac.test_functions.builtin_models()
        >>> registry.get("borehole").dim
        8
        """
        try:
            return self._models[name]
        except (KeyError, TypeError):
            raise InputError(
                f"no model named {name!r} in the registry; known names: {list(self._models)}",
                code="input.unknown_model",
                details={
                    "name": repr(name) if not isinstance(name, str) else name,
                    "known": list(self._models),
                },
            ) from None

    def __getitem__(self, name: str) -> Model:
        """Return the model registered as ``name``."""
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is registered."""
        return isinstance(name, str) and name in self._models

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered names."""
        return iter(tuple(self._models))

    def __len__(self) -> int:
        """Return the number of registered models."""
        return len(self._models)

    def names(self) -> tuple[str, ...]:
        """Return the registered names, sorted.

        Returns
        -------
        tuple of str
            The names.

        See Also
        --------
        get : Look up a model by name.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.test_functions.builtin_models().names()[:3]
        ('borehole', 'otl_circuit', 'piston')
        """
        return tuple(self._models)

    def with_models(self, *models: Model, replace: bool = False) -> ModelRegistry:
        """Return a new registry with ``models`` added.

        Parameters
        ----------
        *models : Model
            Named models to add.
        replace : bool, default False, keyword-only
            Let the new models replace registered models of the same name.

        Returns
        -------
        ModelRegistry
            A new registry; this one is unchanged.

        Raises
        ------
        InputError
            ``input.duplicate_model`` if a name is already registered (and
            ``replace`` is false) or repeats among ``models``;
            ``input.parameter`` for an unnamed model or a non-bool ``replace``.

        See Also
        --------
        ModelRegistry : The constructor.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> base = ac.test_functions.builtin_models()
        >>> mine = ac.Model(space=ac.Uniform.reference(2), value=lambda X: X[:, 0], name="mine")
        >>> extended = base.with_models(mine)
        >>> len(extended) - len(base), "mine" in base
        (1, False)
        """
        if not isinstance(replace, bool):
            raise InputError(
                "replace: expected a bool", code="input.parameter", details={"param": "replace"}
            )
        table = dict(self._models)
        seen: set[str] = set()
        for model in models:
            _check_model(model)
            name = model.name
            assert name is not None
            if name in seen or (name in table and not replace):
                raise _duplicate(name)
            seen.add(name)
            table[name] = model
        return ModelRegistry(table.values())

    def describe(self) -> list[dict[str, Any]]:
        """Return JSON-compatible metadata of every model.

        Returns
        -------
        list of dict
            :meth:`Model.describe` of each model, in sorted name order.

        See Also
        --------
        active_subspaces.Model.describe : The metadata of one model.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> [d["name"] for d in ac.test_functions.builtin_models().describe()][:2]
        ['borehole', 'otl_circuit']
        """
        return [m.describe() for m in self._models.values()]

    def __repr__(self) -> str:
        """Return a summary listing the registered names."""
        return f"ModelRegistry({list(self._models)})"
