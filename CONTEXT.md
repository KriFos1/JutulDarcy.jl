# JutulDarcy Matérn SPDE

This context describes the Matérn SPDE precision-matrix builder used for finite-volume reservoir grids.

## Language

**Precision Matrix**:
The sparse matrix `Q` whose inverse is the Matérn prior covariance on active reservoir cells.
_Avoid_: Regularization matrix, covariance inverse

**SPDE Operator**:
The finite-volume matrix `K = L_H + diag(V*kappa^2)` that represents the square-root precision before mass weighting.
_Avoid_: Filter matrix, A matrix

**Mass Matrix**:
The finite-volume volume matrix `C = diag(V)`.
_Avoid_: FEM mass matrix when discussing JutulDarcy finite-volume assembly

**Diffusion Tensor**:
The symmetric positive-definite tensor field `Theta` that controls anisotropy in the Matérn operator.
_Avoid_: Permeability, except when describing reuse of reservoir transmissibility assembly

**Practical Range**:
The distance parameter `rho` in the convention `kappa = sqrt(8nu)/rho`.
_Avoid_: Correlation length when the convention is ambiguous

**avgMPFA**:
Jutul's linear averaged multi-point flux approximation used for full rotated 3D diffusion tensors.
_Avoid_: fvm9, 27-point stencil

**TPFA**:
The two-point flux approximation used when the diffusion tensor is axis-aligned with the grid geometry.
_Avoid_: MPFA

**Robin Boundary Correction**:
A boundary-face diagonal contribution used to reduce boundary variance artifacts without adding synthetic halo cells.
_Avoid_: Halo padding

**Halo Padding**:
Synthetic boundary cells added outside the model domain for TPFA-only Matérn diagnostics and sampling.
_Avoid_: Robin correction

**Nonstationary Field**:
A Matérn parameter field that varies by active cell, such as variance, principal range, or orientation.
_Avoid_: Heterogeneous field
