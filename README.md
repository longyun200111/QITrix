# QITrix

`QITrix` is a small Python library for finite-dimensional quantum information
calculations.

It focuses on a few linear maps that appear constantly in finite-dimensional
quantum information:

- tensor products across dense, sparse, and CVXPY backends
- subsystem permutations and operator extension
- partial trace and partial transpose
- finite-dimensional Fock basis vectors and projectors
- optional `Space`/`Operator` wrappers for subsystem bookkeeping

## Included functionality

- `QITrix.tensor`: tensor/Kronecker products across NumPy, SciPy sparse, and CVXPY
- `QITrix.permute`: subsystem permutation for operators and rectangular matrices
- `QITrix.extend`: extension of subsystem operators to larger multipartite spaces
- `QITrix.ptrace`: partial trace
- `QITrix.ptrans`: partial transpose
- `QITrix.fock`, `QITrix.fock_dm`: sparse Fock basis states and projectors
- `QITrix.Space`: named tensor-factor metadata
- `QITrix.Operator`: matrix wrapper carrying input/output spaces

## Backend behavior

Most functions accept one or more of:

- `numpy.ndarray`
- `scipy.sparse` matrices or sparse arrays
- `cvxpy.Expression` where the operation is linear in the input

When several backends are possible, `QITrix` keeps the most expressive one:

- if any argument is a `cvxpy.Expression`, the result is a CVXPY expression
- otherwise, sparse inputs stay sparse
- otherwise, the result is a dense NumPy array

Internally this dispatch now goes through a backend registry. The built-in
adapters cover NumPy, SciPy sparse, and CVXPY. Additional matrix libraries can
be integrated by registering a new backend adapter.

## Conventions

- `dims` lists subsystem dimensions in tensor-factor order
- `axes` refers to subsystem positions in that same order
- `perm` must be a permutation of `range(len(dims))`
- `ptrace` and `ptrans` require a square operator of shape
  `(prod(dims), prod(dims))`
- `fock(dim, n)` and `fock_dm(dim, n)` require `dim >= 1` and `0 <= n < dim`

## Examples

Partial trace and partial transpose:

```python
import numpy as np
from qitrix import ptrace, ptrans

rho = np.eye(6, dtype=complex) / 6

rho_a = ptrace(rho, [2, 3], [1])
rho_pt = ptrans(rho, [2, 3], [1])
```

Operator extension:

```python
import numpy as np
from qitrix import extend

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

# Extend X on subsystem 1 of a 3-qubit space.
full_x = extend(sigma_x, dims=[2, 2, 2], axes=[1])
```

Sparse Fock states:

```python
from qitrix import fock, fock_dm

ket_2 = fock(5, 2, format="csr")
proj_2 = fock_dm(5, 2, format="csr")
```

Named subsystem spaces:

```python
import numpy as np
from qitrix import Operator, Space, SpaceList, ptrace

space = SpaceList([Space("A", 2), Space("B", 3)])
rho = Operator(np.eye(6, dtype=complex) / 6, space=space)

rho_a = ptrace(rho, "B")
rho_swapped = rho.permute(["B", "A"])
```

`Space` only manages tensor-factor labels and dimensions. Matrix arithmetic
still happens in the underlying backend object stored in `Operator.data`.

## Install

From GitHub with `pip`:

```bash
python -m pip install "git+https://github.com/longyun200111/QITrix.git"
```

From GitHub with `uv`:

```bash
uv add git+https://github.com/longyun200111/QITrix.git
```

To install into the current `uv` environment without adding a project dependency:

```bash
uv pip install git+https://github.com/longyun200111/QITrix.git
```

To pin a branch or tag:

```bash
python -m pip install "git+https://github.com/longyun200111/QITrix.git@main"
uv add git+https://github.com/longyun200111/QITrix.git --tag v0.2.0
```

For local development:

```bash
git clone https://github.com/longyun200111/QITrix.git
cd QITrix
uv sync
```

To run the test suite:

```bash
uv run --group dev python -m pytest
```

The project name is `QITrix`. The Python import name is `qitrix`.

## Documentation

- [API reference](docs/api.md)
- [Test workflow](.github/workflows/tests.yml)
