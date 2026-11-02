"""Shared model/runtime contracts for AIv2 mother and aiv2_node."""

from aiv2_model_contract.introduction import (
    INTRODUCTION_SCHEMA_VERSION,
    build_introduction,
    effective_serve,
    normalize_introduction,
    project_introduction_for_catalog,
)
from aiv2_model_contract.reasoning_contract import (
    apply_reasoning_to_metadata,
    normalize_reasoning,
    reasoning_catalog_aliases,
)

__all__ = [
    "INTRODUCTION_SCHEMA_VERSION",
    "apply_reasoning_to_metadata",
    "build_introduction",
    "effective_serve",
    "normalize_introduction",
    "normalize_reasoning",
    "project_introduction_for_catalog",
    "reasoning_catalog_aliases",
]

__version__ = "0.1.0"
