"""Representable mixin class."""

class Representable:
    """
    Mixin class that provides a default __repr__ implementation for its subclasses.
    
    The representation will automatically find and include all public attributes
    (those not starting with '_').
    """

    def __repr__(self) -> str:
        # Finds attributes to be included
        if hasattr(self, '__dict__'):
            attrs = [key for key in self.__dict__ if not key.startswith('_')]
        elif hasattr(self, '__slots__'):
            slots: list[str] = getattr(self, '__slots__', [])
            attrs = [key for key in slots if not key.startswith('_')]
        else:
            return super().__repr__()

        attrs_str = ', '.join([f'{key}={getattr(self, key)!r}' for key in attrs])
        return f'{self.__class__.__name__}({attrs_str})'
    