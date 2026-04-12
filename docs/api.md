# qupy API Reference

`qupy` provides finite-dimensional quantum information primitives built on top
of NumPy, SciPy sparse arrays, and selected CVXPY expressions.

The public API is exported from `qupy.__init__`:

```python
from qupy import (
    __version__,
    extend,
    fock,
    fock_dm,
    permute,
    ptrace,
    ptrans,
    tensor,
)
```

## Installation

```bash
pip install .
```

## Backend Support

Most linear-algebra functions in `qupy` accept one or more of the following
container types:

- `numpy.ndarray`
- `scipy.sparse` matrices or sparse arrays
- `cvxpy.Expression` for functions that represent linear matrix maps

When multiple backends are possible, `qupy` keeps the most expressive backend:

- If any operand is a `cvxpy.Expression`, the result is built with CVXPY.
- Otherwise, sparse inputs are kept sparse with SciPy.
- Otherwise, dense NumPy arrays are returned.

## Conventions

- `dims` is the ordered list of subsystem dimensions.
- `axes` identifies subsystem indices in the same order as `dims`.
- Negative subsystem indices are allowed in `ptrace`, `ptrans`, and `extend`
  helper arguments, and are normalized in the Python style.
- Repeated subsystem indices are rejected.
- For bipartite or multipartite operators, the total Hilbert-space dimension is
  `prod(dims)`.

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
from qupy import tensor

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

op = tensor(sigma_x, sigma_z)
```

## `permute`

```python
permute(rho, dims, perm, direction="both")
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

### Example

```python
import numpy as np
from qupy import permute

rho = np.arange(16).reshape(4, 4)
rho_swapped = permute(rho, dims=[2, 2], perm=[1, 0])
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
from qupy import extend

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

# Extend X acting on qubit 1 to a 3-qubit Hilbert space.
full_op = extend(sigma_x, dims=[2, 2, 2], axes=[1])
```

## `ptrace`

```python
ptrace(rho, dims, axes)
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
from qupy import ptrace

rho = np.eye(6, dtype=complex) / 6
rho_a = ptrace(rho, dims=[2, 3], axes=[1])
```

## `ptrans`

```python
ptrans(rho, dims, axes)
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
- `cvxpy.Expression`: sparse superoperator acting on `vec(rho)`
- other supported matrix-like inputs: summation formula with sparse matrix
  multiplications

### Example

```python
import numpy as np
from qupy import ptrans

rho = np.eye(6, dtype=complex) / 6
rho_pt = ptrans(rho, dims=[2, 3], axes=[1])
```

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

- `ValueError` if `n >= dim`

### Example

```python
from qupy import fock

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

- `ValueError` if `n >= dim`

### Example

```python
from qupy import fock_dm

rho_1_dense = fock_dm(4, 1)
rho_1 = fock_dm(4, 1, format="csr")
```

## Version

```python
from qupy import __version__
```

Package version string.

## Quick Example

```python
import numpy as np
from qupy import ptrace, ptrans, tensor

zero = np.array([[1.0], [0.0]], dtype=complex)
one = np.array([[0.0], [1.0]], dtype=complex)
psi = (tensor(zero, zero) + tensor(one, one)) / np.sqrt(2.0)
rho = psi @ psi.conj().T

rho_a = ptrace(rho, dims=[2, 2], axes=[1])
rho_pt = ptrans(rho, dims=[2, 2], axes=[1])
```
