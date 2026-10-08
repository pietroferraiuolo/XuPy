"""
Domain classes and ufunc tables for masked array operations.

This module defines the domain checks used by masked ufuncs to identify
invalid inputs, and maintains lookup tables keyed by ufunc names for
querying domains and fill values. Domain-aware code (e.g., in _ops.py)
uses these tables to propagate masks correctly and independently of the
array backend (numpy or cupy).
"""
from __future__ import annotations

import numpy as _np

from ._backend import get_xp


# ============================================================================
# Domain classes
# ============================================================================

class _DomainCheckInterval:
    """
    Define a valid interval, so that ``domain(x)`` is True where
    ``x < a`` or ``x > b``.
    """

    def __init__(self, a, b):
        if a > b:
            (a, b) = (b, a)
        self.a = a
        self.b = b

    def __call__(self, x):
        """Return True where x is outside [a, b]."""
        xp = get_xp(x)
        with _np.errstate(invalid='ignore'):
            return xp.logical_or(xp.greater(x, self.b),
                                 xp.less(x, self.a))


class _DomainTan:
    """
    Define a valid interval for the `tan` function, so that
    ``domain(x)`` is True where ``abs(cos(x)) < eps``.
    """

    def __init__(self, eps):
        self.eps = eps

    def __call__(self, x):
        """Return True where abs(cos(x)) < eps."""
        xp = get_xp(x)
        with _np.errstate(invalid='ignore'):
            return xp.less(xp.absolute(xp.cos(x)), self.eps)


class _DomainSafeDivide:
    """
    Define a domain for safe division: return True where the divisor
    would cause overflow or division by zero.
    """

    def __init__(self, tolerance=None):
        self.tolerance = tolerance

    def __call__(self, a, b):
        """Return True where abs(a) * tolerance >= abs(b)."""
        xp = get_xp(a, b)
        # Delay tolerance selection to reduce overhead.
        if self.tolerance is None:
            # Use np.finfo even for integer dtypes (numpy.ma does this).
            self.tolerance = _np.finfo(float).tiny
        # Convert to arrays on the target backend.
        a = xp.asarray(a)
        b = xp.asarray(b)
        with _np.errstate(all='ignore'):
            return xp.absolute(a) * self.tolerance >= xp.absolute(b)


class _DomainGreater:
    """
    Domain check: ``domain(x)`` is True where ``x <= critical_value``.
    """

    def __init__(self, critical_value):
        self.critical_value = critical_value

    def __call__(self, x):
        """Return True where x <= critical_value."""
        xp = get_xp(x)
        with _np.errstate(invalid='ignore'):
            return xp.less_equal(x, self.critical_value)


class _DomainGreaterEqual:
    """
    Domain check: ``domain(x)`` is True where ``x < critical_value``.
    """

    def __init__(self, critical_value):
        self.critical_value = critical_value

    def __call__(self, x):
        """Return True where x < critical_value."""
        xp = get_xp(x)
        with _np.errstate(invalid='ignore'):
            return xp.less(x, self.critical_value)


# ============================================================================
# Ufunc tables: domain and fill value lookup by ufunc
# ============================================================================

# These dicts are populated at import time by _ops.py from the UNARY_OPS,
# BINARY_OPS and DOMAINED_BINARY_OPS tables below. They are keyed by the ufunc
# NAME (a string), since numpy and cupy ufuncs with the same name are
# different objects; ``ufunc.__name__`` is the lookup key.

ufunc_domain = {}  # ufunc name -> domain instance or None
ufunc_fills = {}   # ufunc name -> fill value or (fill_a, fill_b) tuple


# ============================================================================
# Public tables: keyed by ufunc name (string)
# ============================================================================

# Domain instances (shared across backends)
_domain_safe_divide = _DomainSafeDivide()
_domain_greater_zero = _DomainGreater(0.0)
_domain_greater_equal_zero = _DomainGreaterEqual(0.0)
_domain_greater_equal_one = _DomainGreaterEqual(1.0)
_domain_check_interval_neg1_1 = _DomainCheckInterval(-1.0, 1.0)
_domain_check_interval_arctanh = _DomainCheckInterval(-1.0 + 1e-15, 1.0 - 1e-15)
_domain_tan = _DomainTan(1e-35)


# Unary operations: (ufunc_name, fill_value, domain)
UNARY_OPS = {
    'sqrt': ('sqrt', 0.0, _domain_greater_equal_zero),
    'log': ('log', 1.0, _domain_greater_zero),
    'log2': ('log2', 1.0, _domain_greater_zero),
    'log10': ('log10', 1.0, _domain_greater_zero),
    'exp': ('exp', 0, None),
    'conjugate': ('conjugate', 0, None),
    'sin': ('sin', 0, None),
    'cos': ('cos', 0, None),
    'tan': ('tan', 0.0, _domain_tan),
    'arctan': ('arctan', 0, None),
    'arcsin': ('arcsin', 0.0, _domain_check_interval_neg1_1),
    'arccos': ('arccos', 0.0, _domain_check_interval_neg1_1),
    'arcsinh': ('arcsinh', 0, None),
    'arccosh': ('arccosh', 1.0, _domain_greater_equal_one),
    'arctanh': ('arctanh', 0.0, _domain_check_interval_arctanh),
    'sinh': ('sinh', 0, None),
    'cosh': ('cosh', 0, None),
    'tanh': ('tanh', 0, None),
    'absolute': ('absolute', 0, None),
    'abs': ('absolute', 0, None),  # alias
    'fabs': ('fabs', 0, None),
    'negative': ('negative', 0, None),
    'floor': ('floor', 0, None),
    'ceil': ('ceil', 0, None),
    'around': ('around', 0, None),
    'logical_not': ('logical_not', 0, None),
}

# Binary operations (no domain): (ufunc_name, fill_a, fill_b)
BINARY_OPS = {
    'add': ('add', 0, 0),
    'subtract': ('subtract', 0, 0),
    'multiply': ('multiply', 1, 1),
    'arctan2': ('arctan2', 0.0, 1.0),
    'equal': ('equal', 0, 0),
    'not_equal': ('not_equal', 0, 0),
    'less_equal': ('less_equal', 0, 0),
    'greater_equal': ('greater_equal', 0, 0),
    'less': ('less', 0, 0),
    'greater': ('greater', 0, 0),
    'logical_and': ('logical_and', 0, 0),
    'logical_or': ('logical_or', 0, 0),
    'logical_xor': ('logical_xor', 0, 0),
    'bitwise_and': ('bitwise_and', 0, 0),
    'bitwise_or': ('bitwise_or', 0, 0),
    'bitwise_xor': ('bitwise_xor', 0, 0),
    'hypot': ('hypot', 0, 0),
}

# Binary operations with domain (division-like): (ufunc_name, domain, fill_a, fill_b)
DOMAINED_BINARY_OPS = {
    'divide': ('divide', _domain_safe_divide, 0, 1),
    'true_divide': ('divide', _domain_safe_divide, 0, 1),  # same ufunc as divide
    'floor_divide': ('floor_divide', _domain_safe_divide, 0, 1),
    'remainder': ('remainder', _domain_safe_divide, 0, 1),
    'mod': ('remainder', _domain_safe_divide, 0, 1),  # alias
    'fmod': ('fmod', _domain_safe_divide, 0, 1),
}
