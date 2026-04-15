#!/usr/bin/env julia
# Diagnostic script: verify K-based Q⁻¹ diagonal matches sampling variance

using Jutul, JutulDarcy
using Jutul: si_unit
using SparseArrays, LinearAlgebra, Statistics, Random
using GeoEnergyIO

function load_norne_domain()
    norne_dir = GeoEnergyIO.test_input_file_path("NORNE_NOHYST")
    data = parse_data_file(joinpath(norne_dir, "NORNE_NOHYST.DATA"))
    return reservoir_domain(data)
end

function build_prior_aniso(domain; a = 0.5, b = 0.0)
    nc = number_of_cells(domain)
    ρ0 = 1500.0 * si_unit(:meter)
    Ba = ones(Float64, nc, 1)
    Bb = ones(Float64, nc, 1)
    prior = MaternSPDE3Prior(nc; ρ0 = ρ0, σ0 = 1.0,
                             anisotropy_a_basis = Ba, anisotropy_b_basis = Bb)
    θ = (anisotropy_a = [Float64(a)], anisotropy_b = [Float64(b)])
    return prior, θ
end

println("=== Verifying K-based Q⁻¹ diagonal ===\n")
domain = load_norne_domain()
prior, θ = build_prior_aniso(domain; a = 0.5, b = 0.0)
halo_spec = MaternHaloSpec(layers = 3, growth = 1.5)

# Build operator WITHOUT compensation
op = matern_spde_operator(domain, prior, θ; compensated = false, halo = halo_spec)
nc = prior.nc
n  = size(op.K, 1)
println("System: nc=$nc, n=$n")

test_cells = [1, nc÷4, nc÷2, 3*nc÷4, nc]

# K-based diagonal (new method)
println("\n--- K-based Q⁻¹ diagonal ---")
kdiag = JutulDarcy._qinv_diagonal_via_K(op.K, op.C, op.fields.τ, test_cells)
for (i, c) in enumerate(test_cells)
    println("  cell $c: $(kdiag[i])")
end

# Empirical sampling
println("\n--- Empirical variance from 200 samples ---")
Nsamp = 200
rng = Random.MersenneTwister(42)
samp = zeros(nc, Nsamp)
for s in 1:Nsamp
    ξ = randn(rng, n)
    sqrtC_ξ = sqrt.(op.C.diag) .* ξ
    y = op.K \ sqrtC_ξ
    samp[:, s] = y[1:nc] ./ op.fields.τ[1:nc]
end
emp_var = vec(var(samp; dims = 2))
println("  Overall: mean_var=$(mean(emp_var))")

println("\n--- Comparison (K-based vs empirical) ---")
for (i, c) in enumerate(test_cells)
    r = emp_var[c] / kdiag[i]
    println("  cell $c: K-based=$(kdiag[i])  empirical=$(emp_var[c])  ratio=$r")
end

# Now test compensation
println("\n--- Testing compensation with K-based method ---")
prior2, θ2 = build_prior_aniso(domain; a = 0.5, b = 0.0)
matern_sd_compensation!(prior2, domain, θ2; mode = :mean, anchors = :all, halo = halo_spec)
offset = prior2.sd_compensation.logtau_offset
println("  Compensation offset: $offset  (τ multiplier: $(exp(offset)))")

op2 = matern_spde_operator(domain, prior2, θ2; compensated = true, halo = halo_spec)
Nsamp2 = 200
rng2 = Random.MersenneTwister(42)
samp2 = zeros(nc, Nsamp2)
for s in 1:Nsamp2
    ξ = randn(rng2, n)
    sqrtC_ξ = sqrt.(op2.C.diag) .* ξ
    y = op2.K \ sqrtC_ξ
    samp2[:, s] = y[1:nc] ./ op2.fields.τ[1:nc]
end
emp_var2 = vec(var(samp2; dims = 2))
println("  After compensation: mean_var=$(mean(emp_var2))  mean_std=$(mean(sqrt.(emp_var2)))")
println("  Target: σ²=1.0, σ=1.0")

println("\nDone.")
