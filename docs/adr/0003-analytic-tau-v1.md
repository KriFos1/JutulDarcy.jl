# Analytic Tau In 3D v1

The first exposed 3D precision builder uses the analytic Matérn `tau` normalization from the Whittle SPDE formula and does not run a 3D Gauss-Newton range or variance calibration. This keeps the `.DATA` to CSC builder deterministic and scalable while leaving empirical calibration as a later, separate design decision.
