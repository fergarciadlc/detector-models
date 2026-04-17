# src/evaluation/evaluators/factory.py
"""Evaluator factory — resolves model output type to the right evaluator."""

from __future__ import annotations

from src.evaluation.evaluators._base import BaseEvaluator
from src.evaluation.evaluators.binary import BinaryEvaluator
from src.evaluation.evaluators.multitask import MultitaskEvaluator

_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "binary": BinaryEvaluator,
    "multitask": MultitaskEvaluator,
}


def create_evaluator(output_type: str) -> BaseEvaluator:
    """Return the evaluator for the given model output type.

    Args:
        output_type: Value of ``cfg.model.output.type``
            (e.g. ``"binary"``, ``"multitask"``).

    Raises:
        ValueError: If ``output_type`` is not registered.
    """
    cls = _REGISTRY.get(output_type)
    if cls is None:
        supported = list(_REGISTRY)
        raise ValueError(
            f"No evaluator registered for output type '{output_type}'. "
            f"Supported: {supported}"
        )
    return cls()
