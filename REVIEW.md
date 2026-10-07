# XuPy code review: v1.7.3

**Scope:** `xupy/_core.py`, `xupy/ma/` (core and extras), `xupy/_cupy_install/`, `typings.py`, packaging, CI and tests.

**Method:** the source was read in full. Claims were then checked by running code against `numpy`/`numpy.ma`.

**Test environments:**

| env | numpy | cupy | GPU |
|---|---|---|---|
| `specula` | 2.5.3 | 14.2.0 | RTX 5080, CUDA 13.2 |
| `opticalib` | 2.5.1 | 14.1.1 | RTX 5080, CUDA 13.2 |
| `aosim` | 2.5.3 | none | none |

No repository files were changed during the review.

**Tags:** **[V]** = verified by running code. **[R]** = inferred from reading the source.

---

## 0. Executive summary

XuPy already *runs* on numpy 2.5 + cupy 14. With `CUDA_PATH` set, the whole suite passes on GPU. What it exposes, though, is not numpy 2.x:

- **The top-level namespace is numpy 1.x in GPU mode.** It is built from `dir(cupy)`, which still carries the numpy‑1 aliases (`NaN`, `float_`, `in1d`, `trapz`, `row_stack`, …). It also lacks several numpy‑2 names (`vecdot`, `unstack`, `errstate`, `strings`, …). Code written on the GPU can break on the CPU, and the reverse.
- **`xupy.ma` covers only part of `numpy.ma`.** It has about 60 % of the API surface. The `nomask` handling, the `__getattr__` fallback and incomplete NumPy protocols (`__array_ufunc__`, `__array_function__`, `__array__(copy=)`) cause silently wrong results.
- **`import xupy` has side effects.** It can block on `input()`, `pip install` into the wrong environment, exit the interpreter, and rewrite its own source file.

The fix for the first and third points is architectural but small: a resolver-based namespace with no import-time install. The second is a refactor of `ma/core.py` around one binary-op helper and one reduction helper. That removes about 1,200 lines and fixes most of the bugs at once.

---

## 1. Strengths

### Backend layer
- **Clear layering for switching.** Backend names are applied first and mode-specific overrides after (`_core.py:1022-1163`, ordering at `1136-1140`). `use_gpu()` fails with a clear message when CuPy is absent (`_core.py:1204-1207`).
- **Working CPU fallback.** `_CPUMemoryContext` (`_core.py:174-300`) mirrors the GPU memory API, treats `psutil` as optional, and has GPU-free tests.
- **Builtins are protected.** `_b = builtins` (`_core.py:11`) avoids `min`/`max`/`sum` being shadowed by the star import.
- **No removed numpy-1 aliases in XuPy's own core code.** `_core.py`, `typings.py` and the tests use none [V, grep].
- **Multi-GPU awareness** is built in from the start (device detection at `_core.py:31-47`).
- **Clean packaging metadata.** PEP 621 metadata, `py.typed`, and the wheel correctly excludes `test/` [V].

### Masked arrays (`xupy.ma`)
- **The right building blocks exist:**
  - `_combine_masks` (`ma/core.py:711-734`) handles `nomask` properly.
  - Axis `mean`/`sum` (`ma/core.py:826-850`, `913-917`) use `where` + reduce, which stays fully on the device with no sync. This is the pattern everything else should follow.
- **GPU-aware `repr`.** For large 1-D and 2-D arrays it copies only the edges to the host (`ma/core.py:449-615`).
- **The constructor respects the input dtype and broadcasts masks**, with clear errors (`ma/core.py:273-309`).
- **`asmarray()`** (`ma/core.py:2863-2931`) is a clean bridge to `numpy.ma` that keeps `fill_value` and `hard_mask`.
- **`extras` mirrors numpy's own `_fromnxfunction` design** (`extras.py:781-807`):
  - `stack`/`vstack`/`hstack`/`atleast_*` are correct.
  - `prod` fills masked entries with 1 via `where`.
  - `average` validates weight shapes the way numpy does.
- **The repr helpers already use the numpy-2 `np._core` layout** (`ma/core.py:383,436`).
- **The tests compare against numpy.** `test_arithmetic_compatibility.py` checks results against `numpy.ma` instead of hard-coded values.

---

## 2. Must fix: correctness and safety

### 2.1 Import-time CuPy installer (Critical)
**Where:** `_core.py:21`, `_cupy_install/__check_availability__.py:35-86`, `_cupy_install/__install_cupy__.py`

- **Non-interactive import crashes [V].** `input()` (`__check_availability__.py:57`) raises `EOFError` under CI, cron, daemons, or any process without a tty.
- **An empty answer means "yes" [V].** It runs `pip install cupy-cudaXXx` via `shell=True`, using whatever `pip` is first on `PATH`, so it can target the wrong environment.
- **A failed install exits the interpreter.** On failure the installer calls `sys.exit(1)`, and `SystemExit` is not caught by `except Exception`, so **`import xupy` exits Python** [V].
- **It rewrites its own installed `.py`** (`:58-61`) through a hard-coded `code[4]` line index.
  - On a read-only install this raises an uncaught `PermissionError`.
  - It invalidates the wheel's RECORD hashes.
  - The `.gitignore` entry does nothing because the file is tracked.
  - The flag has been committed as `True` before.

**Fix:**
- Remove `xupy_init()` from the import path.
- Ship extras: `pip install xupy[cuda12]` / `xupy[cuda13]`.
- Optionally add an explicit CLI, `python -m xupy.install_cupy`, that runs `[sys.executable, "-m", "pip", ...]` without a shell.
- At import, emit at most one `warnings.warn` or `logging.info`.

### 2.2 `nomask` crashes most of `xupy.ma` (Critical) [V]
A `masked_array(data)` created without a mask stores the custom `nomask` singleton. Methods then call array methods on it:

- **`AttributeError`:** `reshape`, `flatten`, `squeeze`, `transpose`, `swapaxes`, `repeat`, `tile`, `copy`, `astype` (`ma/core.py:656-708, 1568, 1601`).
- **`TypeError`:**
  - `std`, `var`, `any`, `all`, `count`, because `~nomask` is invalid.
  - All unary ufuncs, including `np.sqrt(a)`, because of `_xp.where(..., nomask)` at `:1171`.
  - The reflected and in-place `//`, `%`, `**` and `@` operators, because of raw `| nomask`.
- **`.T` silently drops the mask.** The `AttributeError` raised inside the property is caught by `__getattr__`, which returns the raw `data.T`.

The test suite works around this ("Add a mask to avoid nomask issues", `test_ma_core.py:307-358`).

**Fix:** use `numpy.ma.nomask` (which is `np.False_`) as the sentinel, so `~`, `|` and broadcasting just work. Route every mask operation through `_combine_masks` or a `getmaskarray`-style helper.

Related problems:
- `keep_mask=False` with no mask leaves `_mask` unset (`:280-293`).
- A mask-retrieval failure is `print()`ed and ignored (`:290`).

### 2.3 `__getattr__` forwarding silently ignores the mask (Critical) [V]
`ma/core.py:2699-2705` forwards any unknown attribute to the raw cupy/numpy array, bypassing the mask:

| call | xupy | numpy.ma |
|---|---|---|
| `argmax()` | `1` | `2` |
| `ptp()` | `8` | `2` |
| `sort()` | data reordered, mask not → `['--', 2, 3]` (corrupted) | `[1, 2, --]` |

- `cumsum`, `prod`, `argsort`, `nonzero`, `clip`, `real`, `imag`, `view` and `flat` also ignore the mask.
- `__cuda_array_interface__` is forwarded, so `cupy.asarray(xma)` drops the mask silently.
- `copy.copy` and `pickle` hit `RecursionError`; `copy.deepcopy` returns a bare cupy array.
- The API changes with the backend. `ndarray.ptp` was removed in numpy 2, so `.ptp()` works on GPU and raises on CPU.

**Fix:**
- Delete `__getattr__`, or restrict it to a whitelist of read-only metadata (`nbytes`, `itemsize`, `strides`, `flags`).
- Implement the masked methods explicitly.
- Add `__reduce__`, `__copy__` and `__deepcopy__`.

### 2.4 NumPy interoperability protocols (Critical) [V]
**`__array_ufunc__`** (`ma/core.py:2688-2697`) handles only unary `__call__` and drops `out=`, `where=` and `dtype=`.
- `np.ones(4) + a`, `np.add(nd, a)`, `np.maximum(a, a)` and `np.add.reduce(a)` all raise `TypeError`.

**`__array_function__` does not exist**, so numpy functions fall back to `__array__`, which returns the data *without the mask*:
- `np.ma.median(a)` returns **2.5 instead of 3.0**.
- `np.concatenate` and `np.where` lose the mask and silently copy to the host.

**Fix:**
- Implement a full `__array_ufunc__`: all methods, masks combined per ufunc domain, and `NotImplemented` for anything unsupported.
- Implement `__array_function__` with a dispatch table to `xupy.ma` functions.

### 2.5 Scalars and Python protocols (Critical) [V]
- **Indexing returns a 0-d masked array.** `a[i]` on an unmasked element returns a 0-d `_XupyMaskedArray`; numpy returns a numpy scalar.
- **Number protocols are missing.** There is no `__bool__`, `__float__`, `__int__`, `__index__` or `__complex__`.
  - `if a[0] > 5:` raises `TypeError: len() of unsized object`.
  - `float(a[0])` raises.
- **Bitwise and some binary operators are missing.** There is no `__invert__`, `__and__`, `__or__`, `__xor__`, shift operators or `__divmod__`.
  - `~(a > 2)` and `(a > 1) & (a < 4)` both raise. These are core idioms for building masks.
- **Reductions return 0-d cupy arrays** (`mean`, `sum`, `count`, `std`, …) where numpy returns numpy scalars.

### 2.6 Arithmetic and mask-propagation semantics (High) [V]
**In-place operators:**
- `c += d` overwrites data under masked positions: `[11, 12, 13]` vs numpy `[1, 2, 13]`.
- When `self` has `nomask`, `self._mask = other._mask` *aliases* the other array's mask, so later masking `c` also masks `d`.

**NaN/Inf auto-masking** (`_detect_nan_inf`, `:736-771`):
- `+ - * ** // %` mask NaN/Inf *results*; numpy.ma does not.
- It is also asymmetric: `a + 1` masks, `1 + a` doesn't.

**Domain handling:**
- Integer `//` and `%` by 0 are not masked (numpy masks them).
- `log(0)` is not masked (numpy masks the domain).

**Division by a cupy array always fails.** `a / cupy_array` raises because it falls into the scalar branch at `if other == 0:` (`:2501`).

**`matmul`:**
- It combines masks element-wise (`:2160`), which fails for any non-square case.
- It casts `other` to `self.dtype`, so `int32 @ 0.5` gives zeros.

**Fix:** collapse the roughly 1,200 lines of copy-pasted operator code (`:1655-2673`) into one `_binary_op(ufunc, other, domain=None, reflected=False, inplace=False)`. Port numpy.ma's domain classes (`_DomainSafeDivide`, `_DomainGreater`, `_DomainCheckInterval`) and use `ma.dot` semantics for `@`.

### 2.7 Reductions (High) [V]
- **Integer arrays fail along an axis.** `min`/`max`/`std`/`var` raise `cannot convert float NaN to integer`, because NaN is written into masked slots (`:963, 1019, 1078, 1131`).
- **Real NaNs are swallowed.** Using `nan*` functions silently ignores genuine unmasked NaNs.
- **`axis=None` ignores every keyword.**
  - `std(ddof=1)` gives 1.247 vs 1.528.
  - `sum(dtype='f4')` gives float64.
  - `out=` is ignored.
- **Fully-masked slices are handled wrongly:**
  - Fully-masked `std`/`var` return `nan` instead of `masked`.
  - The fully-masked shortcut does `result_shape.pop(axis)` (`:808, 887`), so tuple axes raise and `keepdims` is ignored.
  - Fully-masked integer `mean(axis=0)` returns int64.
- **`any`/`all` along an axis** return plain arrays, not masked arrays.

**Fix:** use one generic reduction helper:
1. `filled = where(mask, identity or min/max fill value, data)`
2. reduce `filled`
3. `result_mask = all(mask, axis, keepdims)`
4. for std/var: `sum((x - mean)² · valid) / (count - ddof)`.

### 2.8 `__setitem__`, metadata and copies (High) [V]
**`__setitem__`:**
- With a hard mask it raises `ValueError`; numpy silently leaves masked slots unchanged.
- With a soft mask, assigning an unmasked value never unmasks (`mask[key] |= value.mask`, `:2800`).

**`dtype`, `astype`, `copy()`:**
- `dtype` returns whatever the user passed (`float`, `'f4'`) instead of an `np.dtype` (`:272, 340`).
- `astype` to the same dtype aliases the data (`:1600`).
- `copy()` and every op result drop `fill_value` and `hard_mask`.

**`fill_value` and the `mask` setter:**
- `fill_value` is never cast or validated: an int array keeps `fill_value=1.5`.
- The `mask` setter (`:327-330`) stores raw input, so `a.mask = True` or `a.mask = [T, F, F]` breaks later ops.

**Other functions:**
- `getmaskarray` builds the mask with the data dtype (`:3020`).
- `tolist()` ignores the mask; numpy gives `None` for masked entries.
- `fill(v)` only fills masked slots; numpy's `fill` sets all elements.
- `extras.mr_` uses `vstack`, so 1-D inputs become 2-D.
- `masked_all` defaults to float32; numpy uses float64.
- `average` forces float32 weight sums, which loses precision (`extras.py:626-658`).

### 2.9 Backend switching (High) [V]
**Namespace corruption.** `use_cpu()`/`use_gpu()` turn `xupy.__version__` into a *module*:
- `_core.py:1154-1157` copies every `xupy.*` entry from `sys.modules` into the package.
- Separately, `numpy.__all__` contains `__version__`.

**Not thread-safe.**
- Names are popped from the module and then re-added (`_core.py:1109-1149`). In between, they are missing.
- A worker thread calling `xp.zeros` during 200 toggles got about 4.9M `AttributeError`s.

**Stale references:**
- `from xupy._core import on_gpu` keeps `True` after `use_cpu()`.
- Arrays created before a switch can't mix with new ones.
- `xp.ma` is `numpy.ma` or `xupy.ma` depending on import order and on which backend was active at import.

**Device helpers:**
- On single-GPU machines (the common case), `with xp.on_device(0):` raises `RuntimeError: Only one GPU available`.
- In CPU mode `on_device` is a one-argument lambda, so `with xp.on_device(0):` fails.
- In CPU-only installs, `array_size` raises `NameError`, because `_B2mb_` is defined inside the GPU `try` block.

**`float`/`cfloat` overrides** (`_core.py:1041-1044` vs `1074-1077`):
- `xp.float` is float32 on GPU but float64 on CPU.
- `xp.cfloat` is complex64 on GPU, which was already wrong for numpy 1.26 (complex128).
- `_core` has no `__all__`, so `from xupy import *` rebinds the builtin `float`.

### 2.10 GPU probe is too shallow (High) [V]
The availability probe only allocates memory (`_core.py:56-58`); it never compiles a kernel. On a machine where cupy cannot find the CUDA headers, XuPy chooses the GPU anyway, and every elementwise op then fails (52 + 109 test failures without `CUDA_PATH=/usr`).

**Fix:** probe with `(_xp.arange(4) + 1).sum().item()` inside the `try`. On failure, fall back to CPU with a warning.

---

## 3. Should improve: quality, performance, maintainability

### Performance on GPU [V]
Benchmark: 20M-element float32.

| operation | raw cupy | xupy.ma |
|---|---|---|
| `a + b` | 0.45 ms | 1.07 ms |
| `a.sum()` | 0.44 ms | 1.08 ms |
| `a.std()` | — | 13.1 ms |
| `item(0)` | — | 7.4 ms (copies the whole array to host) |

Causes:
- **Host syncs** from data-dependent `if _xp.any(...)` branches (`ma/core.py:761, 801, 880, 1854, 1876, 2456, 2478, 2755, 2769`). Compute masks unconditionally instead.
- **Boolean compaction** (`data[~mask]`, which syncs and copies) in every `axis=None` reduction. Use `where` + reduce.
- **`item()`** goes through `asmarray()`, which copies the whole array; `__iter__` syncs once per element.
- **In extras:** `int(valid.sum())`, `bool(mask_result)`, and `nonzero` + `unique` in `mask_rowcols` (`m |= m.any(axis=1, keepdims=True)` needs no sync).

### `MemoryContext` [V unless marked]
- **MB is converted twice.** `get_memory_info` already returns MB, but `__exit__`, `__repr__` and `monitor_memory` divide by 1,024,000 again, so the output reads "Memory delta: 0.00 MB" after allocating 1 GiB.
- **The unit constant is wrong.** `_B2mb_ = 1024*1000` is neither MB nor MiB.
- **The device is not restored when `auto_cleanup=False`** [R].
- **`force_memory_deallocation` never triggers.** Its threshold is 100 GiB, while the comment says 100 MB.
- **`_cleanup_gpu_objects` mutates tracked user objects** (`obj.data = None`).
- **Each default `__exit__` costs about 55 ms** (three sleeps plus syncs) and prints to stdout.

### Repr and str [V]
- **Masked elements print quoted:** `[1.0 '--' 3.0]`. Use `np.ma.masked_print_option` instead of the string `"--"`.
- **The `fill_value=` line is missing.**
- **`nomask` is displayed as `mask=nomask`**; numpy shows `mask=False`.
- **Arrays with 3 or more dimensions copy everything to the host** (`:617-653`, which also has dead variables).

### Singletons [V]
- **`masked` is a plain object.**
  - `masked + 1` raises.
  - `b[1] is np.ma.masked` is False.
  - `b == masked` raises.
- **`np.ma.isMaskedArray(x)` is False** for XuPy arrays.

**Fix:** reuse `np.ma.masked` / `np.ma.nomask`, or make `masked` a 0-d XuPy masked array that supports arithmetic.

### Hygiene
- **Stdout banners** print at import and on every switch. Use `logging` or `warnings`.
- **`except Exception` is too broad** (`_core.py:63`). Any bug in the GPU init branch silently becomes CPU mode.
- **`nvcc --version` / `nvidia-smi` run on every import.** Use `cupy.cuda.runtime.runtimeGetVersion()` lazily instead.
- **`_core` has no `__all__`**, so `gc`, `gpu`, `gpu_name`, `line1`, `n_gpus` and `typings` leak into `xupy`.
- **Outdated docstrings:** the class docstring (`ma/core.py:84-154`) describes a float32 default, reductions via `asmarray`, and a `fill_value(value)` method, none of which exist anymore.
- **`_array_size`** truncates per dimension, treats any unit other than `'MB'` as GB, and fails on `shape=()` and on numpy ints.
- **Typing:**
  - `typings.Array = NDArray` is wrong for cupy.
  - `tuple[int]` should be `tuple[int, ...]`.
  - There is no `__init__.pyi`, so IDEs and type checkers can't see `xp.zeros` and friends.

### Packaging and CI
- **`setup.py`:**
  - `import tomllib` requires Python 3.11, but `requires-python >= 3.10`.
  - It duplicates the pyproject metadata, and its `CustomInstall` never runs under pip/PEP 517.
  - **Delete `setup.py` and `custom_install.py`.**
- **`setuptools>=69` is too low** for `license = "MIT"`; PEP 639 license strings need ≥ 77.
- **`numpy` is unpinned**, but `ma/core.py:383,436` needs numpy ≥ 2.0 (`np._core`).
- **The version is duplicated** in `pyproject.toml` and `__version__.py`. Use `dynamic = ["version"]`.
- **The CI workflow (`publish.yml`) runs no tests** and uses `setup-python@v4` with a twine token. Add a test workflow and switch to trusted publishing.

### Tests
- **The CPU path of `xupy.ma` is never tested.** Both ma test files skip entirely without cupy (`test_ma_core.py:28`, `test_ma_extras.py:64`).
- **Skip conditions check the wrong thing.** They test whether cupy is importable, not `xp.on_gpu`, which causes 6 failures in CPU mode with cupy installed.
- **`test_core.py:149`** expects float32 on CPU.
- **Assertions are too loose:**
  - They accept 0-d arrays as scalars.
  - `repr` is only checked for substrings.
  - Fixtures add masks to avoid the `nomask` bugs.
- **Missing: differential tests against `numpy.ma`.** Run the same inputs through both and compare data, mask, type and dtype. They should cover:
  - `nomask`, hard and soft masks
  - in-place ops on masked slots
  - integer dtypes
  - `ddof`, `keepdims`, tuple axes
  - fully-masked reductions
  - `ndarray op xma` and `np.func(xma)`
  - `copy`/`pickle`, `bool()`/`float()`
  - backend round trips, non-interactive import, single-GPU `on_device`, thread safety
- **`test_large_array_printing.py`** allocates several 800 MB arrays. Mark it `slow`.

---

## 4. Moving the frontend from numpy 1.26 to numpy 2.x

### 4.1 Root cause: the GPU namespace is `dir(cupy)` [V]
In GPU mode the public namespace is everything public in `dir(cupy)` (`_core.py:61, 78-81, 1117-1120`), and CuPy 14.2 has no `__all__`.

**Names present on GPU but absent from numpy 2.x:**
- `Inf`, `Infinity`, `NaN`, `NAN`, `PINF`, `NINF`, `PZERO`, `NZERO`, `infty`
- `float_`, `complex_`, `cfloat`, `singlecomplex`
- `alltrue`, `sometrue`, `product`, `cumproduct`, `round_`
- `in1d`, `trapz`, `row_stack`, `msort`, `asfarray`
- `find_common_type`, `issubclass_`, `issctype`, `obj2sctype`, `sctype2char`
- `set_string_function`, `byte_bounds`, `disp`, `who`, `safe_eval`
- `ComplexWarning`, `RankWarning`, `VisibleDeprecationWarning`, `AxisError`, `TooHardError`
- even `annotations` (the `__future__` feature)

**Names in numpy 2.5 but missing on GPU:**
- `errstate`, `seterr`, `geterr`
- `vecdot`, `vecmat`, `matvec`, `unstack`
- `strings`, `char`, `dtypes`, `rec`, `emath`
- `object_`, `str_`, `bytes_`, `datetime64`, `timedelta64`, `longdouble`
- `geomspace`, `insert`, `block`, `nanquantile`, `nanpercentile`
- `__array_namespace_info__`, `__array_api_version__`
- in `linalg`: `matrix_norm`, `vector_norm`, `vecdot`, `diagonal`, `trace`, `svdvals`

**Consequence:** code written and tested on the GPU can raise `AttributeError` on the CPU, and the reverse. This is the main reason XuPy "feels like 1.26".

**Recommended architecture.** This one change also fixes 2.9 (namespace corruption, thread safety, stale references).
1. **Canonical name set:**
   ```
   _PUBLIC = set(numpy.__all__) - _EXCLUDE | _XUPY_EXTRAS
   ```
   - `_EXCLUDE` = dunders, `core`, `f2py`, `test`, `testing`, `typing`, `show_config`, `show_runtime`. This also stops the star import from eagerly loading `numpy.f2py` and clobbering `__version__`.
   - `_XUPY_EXTRAS` = `asnumpy`, `asmarray`, `MemoryContext`, `use_cpu`, `use_gpu`, `on_device`, … plus an allowlist of CuPy-only tools (`get_array_module`, `cuda`, `fuse`, `ElementwiseKernel`, `RawKernel`, `RawModule`, `ReductionKernel`, `get_default_memory_pool`, …).
2. **Per-backend resolver table,** built once. Each name resolves to the first that applies:
   1. the native name, unless it is in a `_NUMPY2_REMOVED` denylist;
   2. a numpy‑2 shim (see 4.2);
   3. a stub that raises `NotImplementedError("numpy.X has no CuPy equivalent")`.
3. **PEP 562 module `__getattr__` / `__dir__`** that read `_state.table`. `use_cpu()`/`use_gpu()` then swap one reference under a `threading.Lock`. Optionally, a `contextvars.ContextVar` enables `with xp.backend("cpu"):`.
4. **`on_gpu` becomes dynamic**: resolve it via `__getattr__`, or make it a function.

### 4.2 Shims needed for GPU mode
| numpy 2.x | GPU-mode action |
|---|---|
| `errstate`, `seterr`, `geterr` | map to `cupyx.errstate` / `cupyx.seterr` / `cupyx.geterr` |
| dtype and scalar types (`object_`, `str_`, `datetime64`, `typecodes`, …), `dtypes`, `strings`, `char`, `rec` | forward numpy's objects (cupy uses numpy dtypes) |
| `vecdot`, `vecmat`, `matvec`, `unstack`, `linalg.matrix_norm` / `vector_norm` / `vecdot` / `svdvals` / … | take from `array_api_compat.cupy` (optional dependency), or write thin wrappers |
| `sort(stable=, descending=)`, `argsort(stable=)` | wrapper: `stable=True` → `kind="stable"`; `descending` → flip (cupy 14 rejects these keywords [V]) |
| `unique(sorted=, equal_nan=)` | wrapper (cupy rejects `sorted=` [V]) |
| `__array_api_version__`, `__array_namespace_info__` | expose per backend |

### 4.3 XuPy's own non-numpy names
Delete the `float`, `cfloat`, `double` and `cdouble` overrides (`_core.py:1041-1044, 1074-1077`):
- `np.float` was removed in 1.24 and `np.cfloat` in 2.0.
- `double` and `cdouble` already come from the backend.

Update `test_core.py:144-157` to match.

### 4.4 `copy=` semantics (NumPy 2 copy keyword)
- **`ma/core.py:2684` `__array__(self, dtype=None)`.** Numpy 2 emits a DeprecationWarning for this signature [V].
  - Change it to `__array__(self, dtype=None, copy=None)`.
  - Raise `ValueError` for `copy=False` on GPU, since the device-to-host copy is unavoidable.
  - Apply `dtype`.
  - Decide whether it returns `filled()` data or raises.
- **The masked-array constructor** lacks `copy`, `subok`, `ndmin` and `shrink`. In numpy 2, `copy=False` means "never copy" and `copy=None` means "copy if needed".
- **`astype`** needs `copy=`, and must copy by default even for the same dtype.
- **GPU `asarray(np_arr, copy=False)`** silently copies under cupy, while numpy 2 raises [V]. Document this, or wrap `asarray` to raise.

### 4.5 NEP 50 type promotion
- **CuPy 14 already follows NEP 50** [V]:
  - `int8 + 300` raises `OverflowError`.
  - `float32 * 1.5` stays float32.
  - `float32 + np.float64(1)` gives float64.
- **CuPy 13.x does not fully follow it.** This could not be tested here; it is the reviewer's understanding. Pin `cupy>=14` in the GPU extras.
- **`xupy.ma` differs from `np.ma` here [V].** `np.ma` turns Python scalars into 0-d float64 arrays, so float32 MA × `1e300` promotes to float64; xupy keeps float32. Pick a rule and document it. Mirror `np.ma` if parity is the goal.
- **Add CI cases** that check promotion on both backends.

### 4.6 Scalars and repr
- **numpy 2 reprs scalars as `np.float64(2.5)`.** XuPy (via cupy) returns 0-d arrays that print `array(2.5)` [V].
  - In `xupy.ma`, return numpy scalars from reductions and scalar indexing: `_np.dtype(r.dtype).type(r.get())`.
  - Optionally keep a lazy device-scalar mode for performance.
- **The masked-array repr should match numpy 2's:** add `fill_value=…`, print `--` unquoted, and show `mask=False` for `nomask`.
- **The helpers at `ma/core.py:383,436`** use the private `np._core.arrayprint` API. That is fine once numpy ≥ 2 is pinned, but formatting through a small host-side `np.ma` array would be more robust.

### 4.7 `numpy.ma` 2.x API changes
| item | needed change |
|---|---|
| `np.AxisError` removed (now `np.exceptions.AxisError`) | `extras.py:608` raises `AttributeError` today [V]; change it |
| `ndarray.ptp` removed, but `MaskedArray.ptp` kept | implement a mask-aware `MaskedArray.ptp` / `ma.ptp` (`max - min`) instead of forwarding |
| `newbyteorder`, `itemset`, `tostring` removed | remove `__getattr__` forwarding (2.3) so these stay absent on both backends |
| `std`/`var(mean=…)`, `correction=` | add both; `mean=` currently goes to `nanvar` and raises `TypeError` [V]; apply to `extras.std` / `extras.var` too |
| `sort(axis, kind, order, endwith, fill_value, *, stable, descending)`, `argsort(stable=)` | implement mask-aware: fill with min/max fill value according to `endwith`, argsort, reorder data and mask |
| `__array_wrap__(obj, context=None, return_scalar=False)` | needed only if a fallback wrap path remains; prefer a full `__array_ufunc__` |
| `ma.round_` deprecated | don't add it; add `ma.round` / `ma.around` |
| `ma.isin` primary, `ma.in1d` legacy | implement `isin`; make `in1d` a thin wrapper; never call `_xp.in1d` / `_xp.row_stack` (absent on the numpy-2 CPU backend) |
| `fill_value` validated and cast against the dtype (NEP-50 overflow errors) | port `_check_fill_value` |
| Fully-masked reductions return `np.ma.masked`; others return numpy scalars | align every reduction |
| Array-API attributes `mT`, `device`, `to_device`, `__array_namespace__`, `__dlpack__` | add explicitly (`mT` must transpose the mask too) |

### 4.8 Missing `numpy.ma` surface
**About 183 public functions are missing [V]. Highest value first:**
1. **Creation:** `array`, `asarray`, `asanyarray`, `zeros`, `ones`, `empty`, `arange`
2. **Masking helpers:**
   - `masked_where`, `masked_equal`, `masked_greater`, `masked_less`, `masked_inside`, `masked_outside`, `masked_invalid`, `masked_values`
   - `getdata`, `filled`, `is_masked`, `isMaskedArray`, `make_mask`, `mask_or`
3. **Joining and selection:** `concatenate`, `append`, `where`, `choose`, `take`, `put`, `compressed`
4. **Sorting and scans:** `sort`, `argsort`, `argmax`, `argmin`, `cumsum`, `cumprod`, `clip`, `round`
5. **Masked ufunc wrappers:** `sqrt`, `log`, `exp`, `add`, `maximum`, …
6. **Linear algebra and signal:** `dot`, `outer`, `inner`, `diff`, `median`, `cov`, `corrcoef`, `correlate`, `convolve`, `polyfit`
7. **Set operations:** `unique`, `isin`, `intersect1d`, `union1d`, `setdiff1d`
8. **Axis and run utilities:** `apply_along_axis`, `clump_masked`, `clump_unmasked`, `notmasked_edges`, `notmasked_contiguous`, `ndenumerate`
9. **Fill values and the rest:**
   - `default_fill_value`, `set_fill_value`, `common_fill_value`
   - `allclose`, `allequal`, `ndim`, `shape`, `size`, `MAError`, `MaskError`

**About 60 `MaskedArray` methods are missing,** or reachable only through the unsafe forwarding:
- `argmax`, `argmin`, `argsort`, `sort`, `cumsum`, `cumprod`, `prod`, `ptp`, `clip`
- `compress`, `take`, `put`, `nonzero`, `dot`, `trace`, `diagonal`, `resize`, `view`, `anom`
- `real`, `imag`, `get_fill_value`, `set_fill_value`
- `hardmask`, `sharedmask`, `unshare_mask`, `shrink_mask`, `searchsorted`, `partition`
- `mT`, `device`, `to_device`

### 4.9 Dependencies
```toml
[project]
dependencies = ["numpy>=2.0"]

[project.optional-dependencies]
cuda12 = ["cupy-cuda12x>=14"]
cuda13 = ["cupy-cuda13x>=14"]
array-api = ["array-api-compat>=1.11"]
test = ["pytest", "psutil"]
```
Also add CI for:
- **CPU:** numpy 2.0, latest numpy 2.x, Python 3.10–3.13.
- **Windows:** numpy 2's default int is int64 there, so check that cupy's default int matches.

---

## 5. Suggested roadmap

| phase | work | fixes |
|---|---|---|
| **1. Safety (patch release)** | Remove the import-time installer; deeper GPU probe; delete the `float`/`cfloat` overrides; `np.exceptions.AxisError`; `__array__(copy=)`; pin `numpy>=2`; drop `setup.py`; add a CPU test job in CI | 2.1, 2.10, 4.3, 4.4 (part), packaging |
| **2. Namespace** | Resolver table + PEP 562 `__getattr__`; `_NUMPY2_REMOVED` denylist; numpy-2 shims for cupy; lock or ContextVar switching; `__all__` | 2.9, 4.1, 4.2 |
| **3. `ma` core refactor** | `nomask = np.ma.nomask`; drop `__getattr__`; `_binary_op` + domains; generic `_reduce`; scalar returns; number and bitwise dunders; full `__array_ufunc__` + `__array_function__`; pickle/copy | 2.2–2.8, 4.6, part of 4.7 |
| **4. Parity** | Missing `np.ma` functions and methods (priority order in 4.8); `sort`/`std`/`var` numpy-2 signatures; repr parity; differential test suite against `numpy.ma` on both backends | 4.7, 4.8, tests |
| **5. Performance** | Remove data-dependent host syncs and boolean compaction; benchmark suite | 3 (performance) |

Phase 3 is the largest piece of work. Collapsing the operator and reduction code into two helpers should shrink `ma/core.py` considerably. Most of the semantic divergences from `numpy.ma` then become one-line fixes in those helpers, rather than about 30 copies of each.
