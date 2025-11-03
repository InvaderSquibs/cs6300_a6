"""
Edge and routing functions for the LangGraph workflow.

This package contains routing functions that determine workflow control flow
based on state conditions. Routing functions are used with conditional edges
to create dynamic workflows.

All routers follow the pattern:
    (state: GraphState) -> Literal[str, ...]

They return string literals that map to node names in the workflow.
"""
from .routers import (
    route_after_context_check,
    route_after_relevance_check
)

__all__ = [
    "route_after_context_check",
    "route_after_relevance_check",
]

