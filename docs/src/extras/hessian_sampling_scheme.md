# Low-Rank Hessian Estimation via L-BFGS Recycling and Complement Probing

## Problem Statement

After solving the inverse problem via L-BFGS optimization, we want a low-rank approximation of the Hessian at the optimum $\theta^*$:

$$H(\theta^*) \approx \hat{U}_r \Lambda_r \hat{U}_r^T$$

This enables posterior sampling via:

$$\theta_{\text{sample}} \sim \mathcal{N}\!\left(\theta^*,\; \left(\Gamma_{\text{prior}}^{-1} + H_{\text{misfit}}\right)^{-1}\right)$$

## Available Information After L-BFGS

During optimization, L-BFGS accumulates secant pairs:

$$s_k = \theta_{k+1} - \theta_k, \quad y_k = \nabla G(\theta_{k+1}) - \nabla G(\theta_k)$$

Near convergence, these approximate Hessian actions:

$$y_k \approx H(\theta^*)\, s_k + \mathcal{O}(\|s_k\|^2)$$

We retain the last $K_0$ pairs where $\|\theta_k - \theta^*\| < \delta$ (pairs collected near the optimum).

## Algorithm: Complement Probing with L-BFGS Warm-Start

### Phase 1: Extract L-BFGS Subspace

From the retained pairs, form the initial sketch:

$$\Omega_1 = [s_{K-K_0+1},\; \ldots,\; s_K] \in \mathbb{R}^{n \times K_0}$$
$$Y_1 = [y_{K-K_0+1},\; \ldots,\; y_K] \in \mathbb{R}^{n \times K_0}$$

Orthogonalize the range:

$$Q_{\text{LBFGS}} = \text{orth}(Y_1) \in \mathbb{R}^{n \times K_0}$$

### Phase 2: Probe the Complement

Generate $r$ probing directions in the **complement** of the L-BFGS subspace:

$$\tilde{v}_i \sim \mathcal{N}(0, \Gamma_{\text{prior}}), \quad v_i = (I - Q_{\text{LBFGS}}\, Q_{\text{LBFGS}}^T)\, \tilde{v}_i$$

For each $v_i$, compute a finite-difference Hessian action:

$$H v_i \approx \frac{\nabla G(\theta^* + \epsilon\, v_i) - \nabla G(\theta^*)}{\epsilon}$$

Each evaluation costs one forward simulation + one adjoint solve.

### Phase 3: Combine and Decompose

Assemble the full sketch:

$$\Omega = [\Omega_1 \;|\; v_1 \;\ldots\; v_r] \in \mathbb{R}^{n \times (K_0 + r)}$$
$$Y = [Y_1 \;|\; Hv_1 \;\ldots\; Hv_r] \in \mathbb{R}^{n \times (K_0 + r)}$$

Compute the low-rank approximation via randomized SVD:

1. Orthogonalize: $Q, R = \text{qr}(Y)$
2. Form the small projected matrix: $B = Q^T Y (\Omega^T Q)^{-1} Q^T Y^T \in \mathbb{R}^{(K_0+r) \times (K_0+r)}$
   - or equivalently, solve $B = (Q^T \Omega)^{-1} (Q^T Y)$
3. Symmetrize: $B \leftarrow (B + B^T)/2$
4. Eigendecompose: $B = V \Lambda V^T$
5. Recover full eigenvectors: $\hat{U}_r = Q V$

**Result:**

$$H(\theta^*) \approx \hat{U}_r\, \Lambda_r\, \hat{U}_r^T$$

## Option: Single-Pass Estimator (Zero Additional Cost)

If no additional gradient evaluations are affordable, use the Tropp et al. (2017) single-pass estimator from the L-BFGS pairs alone:

$$H \approx Y_1\, (\Omega_1^T Y_1)^{-1}\, Y_1^T$$

This is a rank-$K_0$ approximation that requires **no extra PDE solves**. Its quality depends on how well $\{s_k\}$ span the dominant eigenspace of $H(\theta^*)$.

**Procedure:**

1. Form $M = \Omega_1^T Y_1 \in \mathbb{R}^{K_0 \times K_0}$
2. Symmetrize: $M \leftarrow (M + M^T)/2$
3. Cholesky or eigen-decompose $M$ for stability
4. Approximate: $H \approx Y_1 M^{-1} Y_1^T$
5. Eigendecompose: $Q_Y \Sigma V^T = \text{svd}(Y_1)$, then combine with $M^{-1}$

**Limitations:**
- Rank limited to $K_0$ (typically 10–20 usable pairs)
- Eigenvalues systematically underestimated due to trajectory contamination
- Eigenvectors biased toward gradient/descent directions

Use this as a **quick diagnostic** or when computational budget is exhausted.

## Cost Summary

| Variant | Additional PDE solves | Achievable rank | Eigenvalue accuracy |
|---------|----------------------|-----------------|---------------------|
| Single-pass (L-BFGS only) | 0 | $K_0 \approx 10$–$20$ | Approximate |
| Complement probing ($r$ extra) | $2r$ | $K_0 + r$ | High for dominant modes |
| Full randomized SVD (no reuse) | $2(r + p)$ | $r$ | High |

Typical recommendation: $K_0 \approx 10$–$15$ reusable pairs + $r = 30$–$50$ complement probes for a rank-50 approximation, costing 60–100 additional PDE solves.

## Perturbation Size Selection

The finite-difference step $\epsilon$ should satisfy:

$$\epsilon \approx \sqrt{\eta_{\text{mach}}} \cdot \frac{\|\theta^*\|}{\|v_i\|}$$

where $\eta_{\text{mach}} \approx 10^{-16}$ for Float64. In practice, $\epsilon \in [10^{-5}, 10^{-7}]$ relative to parameter magnitude works well. Verify via a Taylor test:

$$\left\| \nabla G(\theta^* + \epsilon v) - \nabla G(\theta^*) - \epsilon\, H v \right\| = \mathcal{O}(\epsilon^2)$$

## References

- Halko, Martinsson, Tropp (2011). "Finding structure with randomness." SIAM Review.
- Tropp, Yurtsever, Udell, Cevher (2017). "Practical sketching algorithms for low-rank matrix approximation." SIAM J. Matrix Anal. Appl.
- Bui-Thanh, Ghattas, Martin, Stadler (2013). "A computational framework for infinite-dimensional Bayesian inverse problems." J. Comp. Phys.
- Nocedal, Wright (2006). "Numerical Optimization." Springer. Chapter 7 (L-BFGS).
