from typing import Any, Optional, Union, TypeVar, Protocol, runtime_checkable
from numpy.typing import NDArray
from numpy.ma import masked_array

@runtime_checkable
class XupyMaskedArray_P(Protocol):
    def __init__(self, data: Union[NDArray[Any], Any], mask: Optional[NDArray[Any]] = None, **kwargs: Any) -> None: ...
    @property
    def _mask(self) -> NDArray[Any]: ...
    @_mask.setter
    def _mask(self, value: NDArray[Any]) -> None: ...

XupyMaskedArray = TypeVar("XupyMaskedArray", bound=XupyMaskedArray_P)