"""Registrable mixin class."""

from __future__ import annotations
from typing import TypeVar, Generic, cast
import inspect

T = TypeVar('T', covariant=True)

class Registrable(Generic[T]):
    """A mixin class that provides a class with a registry for its subclasses."""

    _registry: dict[str, type[T]] = {}
    """Dictionary mapping identifiers to their corresponding subclass type."""

    identifier: str
    """The string identifier of the subclass."""

    aliases: list[str] = []
    """Optional list of additional string identifiers of the subclass."""

    def __init_subclass__(cls) -> None:
        """Initialisees a separate registry for each base class family"""
        super().__init_subclass__()

        # Direct children of Registrable get their own registry
        if Registrable in cls.__bases__:
            cls._registry = {}

        # Skips registration for abstract classes
        if inspect.isabstract(cls):
            return
        
        # Ensures the class has an identifier
        if not hasattr(cls, 'identifier') or not isinstance(cls.identifier, str):
            raise TypeError(f"Class '{cls.__name__}' must define an `identifier` string as a class variable.")
                
        for identifier in [cls.identifier] + cls.aliases:
            # Ensures the identifier is unique
            if identifier in cls._registry:
                existing = cls._registry[identifier].__name__
                raise TypeError(
                    f"Cannot register class '{cls.__name__}': its identifier "
                    f"'{identifier}' is already used by class '{existing}'."
                )
            cls._registry[identifier] = cast(type[T], cls)

    @classmethod
    def get_registry(cls) -> dict[str, type[T]]:
        """Returns the registry specific to this class family."""
        return cls._registry
    
    @classmethod
    def get(cls, identifier: str | T) -> T:
        """Retrieves an instance from a string identifier or returns the given instance."""        
        if not isinstance(identifier, str):
            return identifier
        
        key = identifier.lower()
        registry = cls.get_registry()

        if key in registry:
            return registry[key]()
        
        raise ValueError(
            f"Unknown {cls.__name__.lower()}: '{identifier}'.\n\t"
            f"Available options are: {', '.join(list(registry.keys()))}"
        )