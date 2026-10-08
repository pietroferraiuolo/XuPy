from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)
from numpy.typing import NDArray, ArrayLike, DTypeLike
from numpy.ma import masked_array

if TYPE_CHECKING:  # cupy is never imported at runtime here
    import cupy  # type: ignore

    Array = Union[NDArray[Any], "cupy.ndarray"]
else:
    Array = Union[NDArray[Any], Any]

# Type aliases for better readability
Scalar = Union[int, float, complex]


@runtime_checkable
class XupyMaskedArrayProtocol(Protocol):
    """Protocol defining the interface for XuPy masked arrays."""

    data: Array
    _mask: Array

    def __init__(
        self,
        data: ArrayLike,
        mask: Optional[ArrayLike] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> None: ...

    # Core properties
    @property
    def mask(self) -> Array: ...
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
MaskedArray = XupyMaskedArrayProtocol  # alias for annotations
