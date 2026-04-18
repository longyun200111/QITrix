# QITrix API Reference

`QITrix` provides a small set of finite-dimensional quantum information
primitives built on top of NumPy, SciPy sparse arrays, and selected CVXPY
expressions.

The public API is exported from `qitrix.__init__`:

```python
from qitrix import (
    __version__,
    Operator,
    Space,
    SpaceList,
    extend,
    fock,
    fock_dm,
    operator,
    permute,
    ptrace,
    ptrans,
    tensor,
)
```

## Installation

For local development:

```bash
uv sync
```

To run the test suite:

```bash
uv run --group dev python -m pytest
```

## Backend Support

Most linear-algebra functions in `QITrix` accept one or more of the following
container types:

- `numpy.ndarray`
- `scipy.sparse` matrices or sparse arrays
- `cvxpy.Expression` for functions that represent linear matrix maps

When multiple backends are possible, `QITrix` keeps the most expressive backend:

- If any operand is a `cvxpy.Expression`, the result is built with CVXPY.
- Otherwise, sparse inputs are kept sparse with SciPy.
- Otherwise, dense NumPy arrays are returned.

Internally, backend-dependent operations are routed through a registry of
adapters. The built-in adapters cover NumPy, SciPy sparse, and CVXPY. This
keeps the public function API unchanged while making future backend extensions
explicit.

## `Space`

```python
Space(label, dim)
```

Represent a single tensor factor.

### Parameters

- `label`: subsystem name
- `dim`: subsystem dimension

### Properties

- `label`
- `dim`

### Example

```python
from qitrix import Space

a = Space("A", 2)
```

## `SpaceList`

```python
SpaceList(spaces)
```

Represent an ordered tensor-product space built from `Space` factors.

### Parameters

- `spaces`: iterable of `Space` objects

### Properties

- `labels`: ordered subsystem names
- `dims`: ordered subsystem dimensions
- `dim`: total Hilbert-space dimension

### Methods

- `index(label)`: position of one subsystem
- `indices(labels)`: positions of several subsystems
- `reordered(labels)`: new `SpaceList` with permuted subsystem order
- `drop(labels)`: new `SpaceList` with selected subsystems removed
- `select(labels)`: new `SpaceList` keeping only selected subsystems

### Example

```python
from qitrix import Space, SpaceList

space = SpaceList([Space("A", 2), Space("B", 3), Space("C", 2)])
ac = space.select(["A", "C"])
```

## `Operator`

```python
Operator(data, space=None, *, input_space=None, output_space=None)
operator(data, space=None, *, input_space=None, output_space=None)
```

Attach input/output `Space` metadata to a matrix-like object.

### Parameters

- `data`: dense, sparse, or CVXPY matrix-like object
- `space`: shorthand for square operators with identical input/output spaces
- `input_space`, `output_space`: spaces for rectangular maps

### Methods

- `as_raw()`: return the underlying matrix object
- `tensor(other)`: tensor product with another `Operator`
- `ptrace(labels)`: partial trace by subsystem labels
- `ptrans(labels)`: partial transpose by subsystem labels
- `permute(labels, direction=...)`: subsystem permutation by labels
- `extend(...)`: embed the operator into larger named spaces

### Example

```python
import numpy as np
from qitrix import Operator, Space, SpaceList

rho = Operator(
    np.eye(6, dtype=complex) / 6,
    space=SpaceList([Space("A", 2), Space("B", 3)]),
)

rho_a = rho.ptrace("B")
rho_ba = rho.permute(["B", "A"])
```

## Conventions and Validation

- `dims` is the ordered list of subsystem dimensions.
- `axes` identifies subsystem indices in the same order as `dims`.
- Negative subsystem indices are allowed in `ptrace`, `ptrans`, and `extend`
  helper arguments, and are normalized in the Python style.
- Repeated subsystem indices are rejected.
- `dims` must contain only positive integers.
- For bipartite or multipartite operators, the total Hilbert-space dimension is
  `prod(dims)`.
- `permute(..., perm=...)` requires `perm` to be a permutation of
  `range(len(dims))`.
- `ptrace` and `ptrans` require a square input operator of shape
  `(prod(dims), prod(dims))`.
- `fock(dim, n)` and `fock_dm(dim, n)` require `dim >= 1` and `0 <= n < dim`.

## `tensor`

```python
tensor(*factors)
```

Compute the tensor (Kronecker) product of multiple factors from left to right.

### Parameters

- `factors`: sequence of matrix-like operands.

### Returns

- `numpy.ndarray`, sparse SciPy object, or `cvxpy.Expression`

### Raises

- `ValueError` if no factor is provided

### Example

```python
import numpy as np
from qitrix import tensor

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

op = tensor(sigma_x, sigma_z)
```

## `permute`

```python
permute(rho, dims, perm, direction="both")
permute(op, labels, direction="both")
```

Permute subsystem order in an operator or rectangular matrix.

### Parameters

- `rho`: input matrix-like object
- `dims`: subsystem dimensions before permutation
- `perm`: target subsystem order
- `direction`: one of `"both"`, `"left"`, or `"right"`

### Semantics

- `direction="both"` applies the same subsystem permutation to row and column
  tensor factors
- `direction="left"` permutes only row-space tensor factors
- `direction="right"` permutes only column-space tensor factors

### Returns

- object in the same backend family as `rho`

### Validation

- `dims` must contain only positive integers
- `perm` must be a permutation of `range(len(dims))`
- `direction="both"` requires `rho.shape == (prod(dims), prod(dims))`
- `direction="left"` requires the row dimension to equal `prod(dims)`
- `direction="right"` requires the column dimension to equal `prod(dims)`

### Example

```python
import numpy as np
from qitrix import permute

rho = np.arange(16).reshape(4, 4)
rho_swapped = permute(rho, dims=[2, 2], perm=[1, 0])
```

For `Operator` inputs, use subsystem labels instead of integer permutations:

```python
from qitrix import Operator, Space, SpaceList, permute

op = Operator(np.eye(6), space=SpaceList([Space("A", 2), Space("B", 3)]))
swapped = permute(op, ["B", "A"])
```

## `extend`

```python
extend(
    op,
    dims=None,
    axes=None,
    input_dims=None,
    output_dims=None,
    input_axes=None,
    output_axes=None,
)
```

Embed an operator acting on selected subsystems into a larger multipartite
space.

### Parameters

- `op`: operator to be embedded
- `dims`: shorthand for setting both `input_dims` and `output_dims`
- `axes`: shorthand for setting both `input_axes` and `output_axes`
- `input_dims`, `output_dims`: full subsystem dimensions of the input and
  output spaces
- `input_axes`, `output_axes`: subsystem positions on which `op` acts

### Returns

- extended operator on the full multipartite space

### Requirements

- You must provide either `dims` or both `input_dims` and `output_dims`.
- You must provide either `axes` or both `input_axes` and `output_axes`.
- The shape of `op` must equal
  `(prod(output_dims[output_axes]), prod(input_dims[input_axes]))`.
- Untouched input and output subsystem dimensions must match, because the
  extension is implemented by tensoring identities on those subsystems.

### Example

```python
import numpy as np
from qitrix import extend

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

# Extend X acting on qubit 1 to a 3-qubit Hilbert space.
full_op = extend(sigma_x, dims=[2, 2, 2], axes=[1])
```

Rectangular operators are also supported:

```python
import numpy as np
from qitrix import extend

isometry = np.array([[1.0, 0.0]], dtype=complex)
full_isometry = extend(
    isometry,
    input_dims=[2, 2],
    output_dims=[1, 2],
    input_axes=[0],
    output_axes=[0],
)
```

## `ptrace`

```python
ptrace(rho, dims, axes)
ptrace(op, labels)
```

Compute the partial trace over selected subsystems.

### Parameters

- `rho`: operator on the full tensor-product Hilbert space
- `dims`: subsystem dimensions satisfying
  `prod(dims) == rho.shape[0] == rho.shape[1]`
- `axes`: subsystem indices to trace out

### Returns

- reduced operator after tracing out the selected subsystems

### Backend-specific implementation

- `numpy.ndarray`: tensor reshape plus `numpy.trace`
- `cvxpy.Expression`: sparse superoperator acting on `vec(rho)`
- other supported matrix-like inputs: summation formula with sparse matrix
  multiplications

### Example

```python
import numpy as np
from qitrix import ptrace, tensor

rho_a = np.array([[1, 0], [0, 0]], dtype=complex)
rho_b = np.eye(3, dtype=complex) / 3
rho_ab = tensor(rho_a, rho_b)

reduced_a = ptrace(rho_ab, dims=[2, 3], axes=[1])
```

With named spaces:

```python
from qitrix import Operator, Space, SpaceList, ptrace

rho_ab = Operator(rho_ab, space=SpaceList([Space("A", 2), Space("B", 3)]))
reduced_a = ptrace(rho_ab, "B")
```

### Structural property

`ptrace` preserves the full trace:

```python
np.trace(ptrace(rho, dims, axes)) == np.trace(rho)
```

up to floating-point roundoff.

## `ptrans`

```python
ptrans(rho, dims, axes)
ptrans(op, labels)
```

Compute the partial transpose over selected subsystems.

### Parameters

- `rho`: operator on the full tensor-product Hilbert space
- `dims`: subsystem dimensions satisfying
  `prod(dims) == rho.shape[0] == rho.shape[1]`
- `axes`: subsystem indices to transpose

### Returns

- partially transposed operator

### Backend-specific implementation

- `numpy.ndarray`: tensor reshape plus bra/ket index swaps
- `cvxpy.Expression`: sparse linear map acting on `vec(rho)`
- other supported matrix-like inputs: summation formula with sparse matrix
  multiplications

### Example

```python
import numpy as np
from qitrix import ptrans

rho = np.eye(6, dtype=complex) / 6
rho_pt = ptrans(rho, dims=[2, 3], axes=[1])
```

### Structural property

Applying the same partial transpose twice returns the original operator:

```python
ptrans(ptrans(rho, dims, axes), dims, axes)
```

up to floating-point roundoff.

## `fock`

```python
fock(dim, n, format="dense")
```

Return the Fock basis ket `|n>` in a finite-dimensional Hilbert space.

### Parameters

- `dim`: Hilbert-space dimension
- `n`: occupation-number index
- `format`: `"dense"` for a NumPy array, or a SciPy sparse format such as
  `"csr"` or `"coo"`

### Returns

- dense or sparse column vector of shape `(dim, 1)`

### Raises

- `ValueError` if `dim < 1` or `n` is not in `0, ..., dim - 1`

### Example

```python
from qitrix import fock

ket_1_dense = fock(4, 1)
ket_1 = fock(4, 1, format="csr")
```

## `fock_dm`

```python
fock_dm(dim, n, format="dense")
```

Return the rank-one projector `|n><n|`.

### Parameters

- `dim`: Hilbert-space dimension
- `n`: occupation-number index
- `format`: `"dense"` for a NumPy array, or a SciPy sparse format such as
  `"csr"` or `"coo"`

### Returns

- dense or sparse matrix of shape `(dim, dim)`

### Raises

- `ValueError` if `dim < 1` or `n` is not in `0, ..., dim - 1`

### Example

```python
from qitrix import fock_dm

rho_1_dense = fock_dm(4, 1)
rho_1 = fock_dm(4, 1, format="csr")
```

## Version

```python
from qitrix import __version__
```

Package version string.
