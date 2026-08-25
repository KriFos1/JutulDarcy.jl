# avgMPFA For Rotated 3D Matérn

Rotated full-tensor 3D Matérn precision assembly uses Jutul's existing NFVM `:avgmpfa` linear discretization instead of a hand-rolled structured 27-point stencil. This reuses the reservoir-grid full-tensor machinery on corner-point and unstructured grids, but means `K` is generally non-symmetric, so the precision is assembled as `Q = Dtau * K' * C^-1 * K * Dtau` and factored directly; TPFA halo padding is not supported for the avgMPFA path.
