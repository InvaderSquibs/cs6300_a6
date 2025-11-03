"""
Workflow builder with automatic dependency injection.

This module provides the WorkflowBuilder class which automatically injects
dependencies into node functions based on their function signatures and type
hints. This enables a clean, declarative workflow construction while
keeping node dependencies explicit and self-documenting.

The builder uses Python's inspect module to analyze function signatures and
type hints, then matches them against a dependency registry to automatically
create node wrappers with dependencies injected.
"""
from inspect import signature
from typing import get_type_hints, Callable, Dict, Any, Optional
from functools import wraps


class WorkflowBuilder:
    """
    Builder for LangGraph workflows with automatic dependency injection.
    
    This class analyzes node function signatures and automatically injects
    dependencies based on type hints. This enables:
    - Explicit dependencies in node function signatures (self-documenting)
    - Automatic wiring without manual partial application
    - Type-safe dependency resolution
    - Clear error messages when dependencies are missing
    
    The builder expects node functions to have the pattern:
        def node_func(state: GraphState, dependency1: Type1, ...) -> GraphState
    
    Dependencies are identified by their type hints and matched against
    the dependency registry provided to __init__.
    
    Example:
        .. code-block:: python
        
            # Define dependencies
            dependencies = {
                "BaseChatModel": llm_instance,
                "VectorDBManager": vector_db_instance,
            }
            
            # Create builder
            builder = WorkflowBuilder(dependencies)
            
            # Create node with auto-injected dependencies
            wrapped_node = builder.create_node(some_node_function)
            # wrapped_node now has llm and vector_db automatically injected
            
            # Use in workflow
            workflow.add_node("my_node", wrapped_node)
    
    Attributes:
        dependencies: Dictionary mapping type names to dependency instances.
            Keys should match the class names used in node function type hints.
    """
    
    def __init__(self, dependencies: Dict[str, Any]):
        """
        Initialize the workflow builder with dependency registry.
        
        Args:
            dependencies: Dictionary mapping type/class names to dependency
                instances. Keys should be the exact class names as they appear
                in node function type hints (e.g., "BaseChatModel", "VectorDBManager").
                
                Example:
                    {
                        "BaseChatModel": ChatOpenAI(...),
                        "VectorDBManager": VectorDBManager(...),
                        "ArxivSearcher": ArxivSearcher(...),
                    }
        
        Raises:
            ValueError: If dependencies dictionary is empty or invalid.
        """
        if not dependencies:
            raise ValueError("Dependencies dictionary cannot be empty")
        
        self.dependencies = dependencies
    
    def create_node(self, node_func: Callable) -> Callable:
        """
        Create a node wrapper with dependencies automatically injected.
        
        This method analyzes the node function's signature and type hints,
        matches them against the dependency registry, and returns a wrapper
        function that automatically injects dependencies when called by LangGraph.
        
        The wrapper function has signature: (state: GraphState) -> GraphState,
        which matches LangGraph's expected node signature.
        
        Args:
            node_func: The node function to wrap. Must have:
                - First parameter named 'state' of type GraphState
                - Additional parameters with type hints matching dependency keys
                - Return type GraphState
        
        Returns:
            Wrapped function with signature (state: GraphState) -> GraphState.
            When called, it automatically injects dependencies and calls the
            original node_func.
        
        Raises:
            ValueError: If required dependencies are not found in registry.
            TypeError: If node_func signature is invalid or missing type hints.
        
        Example:
            .. code-block:: python
            
                def my_node(state: GraphState, llm: BaseChatModel) -> GraphState:
                    # Use llm here
                    return state
                
                builder = WorkflowBuilder({"BaseChatModel": my_llm})
                wrapped = builder.create_node(my_node)
                
                # wrapped(state) automatically passes my_llm as llm parameter
        """
        # Get function signature and type hints
        sig = signature(node_func)
        hints = get_type_hints(node_func)
        
        # Extract parameter names (excluding 'state')
        param_names = [name for name in sig.parameters.keys() if name != 'state']
        
        # Build dependency dictionary for this node
        node_deps = {}
        missing_deps = []
        
        for param_name in param_names:
            param_type = hints.get(param_name)
            if param_type is None:
                raise TypeError(
                    f"Node function {node_func.__name__} parameter '{param_name}' "
                    f"must have type hint"
                )
            
            # Get type name (handle both classes and strings)
            if hasattr(param_type, '__name__'):
                type_name = param_type.__name__
            elif isinstance(param_type, str):
                type_name = param_type
            else:
                # Handle TYPE_CHECKING string annotations
                type_name = str(param_type).replace("'", "")
                # Try to extract class name from string like "BaseChatModel"
                if '.' in type_name:
                    type_name = type_name.split('.')[-1]
            
            # Find matching dependency
            if type_name in self.dependencies:
                node_deps[param_name] = self.dependencies[type_name]
            else:
                missing_deps.append(f"{param_name}: {type_name}")
        
        if missing_deps:
            available = ", ".join(self.dependencies.keys())
            raise ValueError(
                f"Node {node_func.__name__} requires dependencies not in registry:\n"
                f"  Missing: {', '.join(missing_deps)}\n"
                f"  Available: {available}"
            )
        
        # Create wrapper function
        @wraps(node_func)
        def wrapper(state):
            """
            Wrapper that injects dependencies and calls the original node function.
            
            This wrapper is called by LangGraph with just the state, and it
            automatically passes the appropriate dependencies to the node function.
            """
            return node_func(state, **node_deps)
        
        return wrapper
    
    def get_dependencies(self) -> Dict[str, Any]:
        """
        Get the current dependency registry.
        
        Returns:
            Copy of the dependencies dictionary.
        """
        return self.dependencies.copy()
    
    def add_dependency(self, type_name: str, instance: Any) -> None:
        """
        Add or update a dependency in the registry.
        
        Args:
            type_name: Type name (class name) to register
            instance: Dependency instance to store
        """
        self.dependencies[type_name] = instance

