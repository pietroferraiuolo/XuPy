from typing import Any, Optional, Union, Protocol, runtime_checkable
from numpy.typing import NDArray, ArrayLike, DTypeLike
from numpy.ma import masked_array

# Type aliases for better readability
Array = NDArray[Any]
Scalar = Union[int, float, complex]

@runtime_checkable
class XupyMaskedArrayProtocol(Protocol):
    """Protocol defining the interface for XuPy masked arrays."""
    data: Array
    mask: Array
    
    def __init__(self, data: ArrayLike, mask: Optional[ArrayLike] = None, dtype: Optional[DTypeLike] = None) -> None: ...
    
    # Core properties
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
    @property
    def size(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    
    # Conversion methods
    def asmarray(self, **kwargs: Any) -> masked_array: ...
    
    # String representation
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# Main type for XuPy masked arrays
XupyMaskedArray = XupyMaskedArrayProtocol