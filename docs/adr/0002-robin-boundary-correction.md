# Robin Boundary Correction

The exposed 3D Matérn precision API defaults to a hippylib-style Robin boundary correction rather than halo padding. Robin terms compose with both TPFA and avgMPFA full-tensor discretizations, while halo padding remains available only for the direct TPFA operator path where synthetic exterior cells are well-defined.
