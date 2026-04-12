# qupy

`qupy` is a small Python library for finite-dimensional quantum information
calculations.

## Included functionality

- `qupy.tensor`: tensor/Kronecker products across NumPy, SciPy sparse, and CVXPY
- `qupy.permute`: subsystem permutation for operators and rectangular matrices
- `qupy.extend`: extension of subsystem operators to larger multipartite spaces
- `qupy.ptrace`: partial trace
- `qupy.ptrans`: partial transpose
- `qupy.fock`, `qupy.fock_dm`: sparse Fock basis states and projectors

## Example

```python
import numpy as np
from qupy import ptrace, ptrans

rho = np.eye(6, dtype=complex) / 6

rho_a = ptrace(rho, [2, 3], [1])
rho_pt = ptrans(rho, [2, 3], [1])
```

## Install

From the repository root:

```bash
pip install .
```
