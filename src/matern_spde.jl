mutable struct MaternSPDE2Prior{T<:AbstractFloat, M<:AbstractMatrix{T}}
    nc::Int
    ν::T
    range_basis::M
    sd_basis::M
    anisotropy_u_basis::M
    anisotropy_v_basis::M
    ρ0::T
    σ0::T
    κ_min::T
    laplace_from::Symbol
    version::Symbol
    sd_compensation::Any
    range_compensation::Any
end

struct MaternHaloSpec{T<:AbstractFloat}
    layers::Int
    growth::T
    total_padding::Any
    extension::Symbol
    include_corners::Bool
end

struct MaternSDCompensation{T<:AbstractFloat, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    mode::Symbol
    anchors::Vector{Int}
    logtau_offset::V
    range_sensitivity::M
    anisotropy_u_sensitivity::M
    anisotropy_v_sensitivity::M
    fd_delta::T
end

"""
    MaternHaloSpec(; layers = 3, growth = 1.5, total_padding = :boundary_range, extension = :nearest, include_corners = true)

Specification for synthetic halo padding used to move the artificial Matérn
boundary away from the reservoir grid.
"""
function MaternHaloSpec(
        ;
        layers::Integer = 3,
        growth::Real = 1.5,
        total_padding = :boundary_range,
        extension::Symbol = :nearest,
        include_corners::Bool = true
    )
    layers > 0 || throw(ArgumentError("layers must be positive, got $layers."))
    growth > 0 || throw(ArgumentError("growth must be positive, got $growth."))
    extension == :nearest || throw(ArgumentError("Only extension = :nearest is supported in this first implementation, got $extension."))
    if !(total_padding isa Symbol || total_padding isa Real)
        throw(ArgumentError("total_padding must be a Symbol or Real, got $(typeof(total_padding))."))
    end
    return MaternHaloSpec(Int(layers), Float64(growth), total_padding, extension, include_corners)
end

struct MaternRangeCompensation{T<:AbstractFloat, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    mode::Symbol
    anchors::Vector{Int}
    logkappa_offset::V
    anisotropy_amplitude_offset::V
    range_sensitivity::M
    anisotropy_u_sensitivity::M
    anisotropy_v_sensitivity::M
    amplitude_range_sensitivity::M
    amplitude_u_sensitivity::M
    amplitude_v_sensitivity::M
    corr_level::T
    fd_delta::T
end

"""
    MaternSPDE2Prior(nc; ν = 1.0, range_basis = nothing, sd_basis = nothing, anisotropy_u_basis = nothing, anisotropy_v_basis = nothing, ρ0 = 1.0, σ0 = 1.0, κ_min = 1e-8, laplace_from = :geometry, version = :xyz)

Specification for a 2D Matérn SPDE prior with `α = 2`, low-rank basis fields,
and optional cached variance/range compensation.

The current implementation targets 2D fields. Single-layer 3D grids are
supported by embedding the 2D anisotropy tensor in the horizontal block of a
3D permeability tensor.
"""
function MaternSPDE2Prior(
        nc::Integer;
        ν::Real = 1.0,
        range_basis = nothing,
        sd_basis = nothing,
        anisotropy_u_basis = nothing,
        anisotropy_v_basis = nothing,
        ρ0::Real = 1.0,
        σ0::Real = 1.0,
        κ_min::Real = 1e-8,
        laplace_from::Symbol = :geometry,
        version::Symbol = :xyz
    )
    T = Float64
    νf = T(ν)
    isapprox(νf, 1.0; atol = 1e-12) || throw(ArgumentError("This implementation only supports α = 2 in 2D, which requires ν = 1.0 (got $ν)."))
    ρ0f = T(ρ0)
    σ0f = T(σ0)
    κminf = T(κ_min)
    ρ0f > 0 || throw(ArgumentError("ρ0 must be positive, got $ρ0."))
    σ0f > 0 || throw(ArgumentError("σ0 must be positive, got $σ0."))
    κminf > 0 || throw(ArgumentError("κ_min must be positive, got $κ_min."))
    laplace_from in (:geometry, :flow, :unit) || throw(ArgumentError("laplace_from must be one of :geometry, :flow, :unit (got $laplace_from)."))
    version in (:xyz, :ijk) || throw(ArgumentError("version must be :xyz or :ijk (got $version)."))

    Bρ = _matern_basis_matrix(range_basis, nc, T)
    Bσ = _matern_basis_matrix(sd_basis, nc, T)
    Bu = _matern_basis_matrix(anisotropy_u_basis, nc, T)
    Bv = _matern_basis_matrix(anisotropy_v_basis, nc, T)

    return MaternSPDE2Prior(
        Int(nc),
        νf,
        Bρ,
        Bσ,
        Bu,
        Bv,
        ρ0f,
        σ0f,
        κminf,
        laplace_from,
        version,
        nothing,
        nothing
    )
end

"""
    matern_parameter_fields(prior, θ = NamedTuple())

Construct target Matérn parameter fields from a prior specification and basis
coefficients `θ`.

The coefficients are supplied as a named tuple with optional fields:
`range`, `sd`, `anisotropy_u`, and `anisotropy_v`.
"""
function matern_parameter_fields(prior::MaternSPDE2Prior, θ = NamedTuple())
    θn = _normalize_matern_theta(prior, θ)

    logρ = log(prior.ρ0) .+ prior.range_basis*θn.range
    logσ = log(prior.σ0) .+ prior.sd_basis*θn.sd
    ρ = exp.(logρ)
    σ = exp.(logσ)
    u = prior.anisotropy_u_basis*θn.anisotropy_u
    v = prior.anisotropy_v_basis*θn.anisotropy_v

    κ = max.(sqrt(8*prior.ν) ./ ρ, prior.κ_min)
    τ_nominal = _matern_tau_nominal(prior.ν, κ, σ)
    H = _matern_anisotropy_tensor(u, v)
    r = sqrt.(u.^2 .+ v.^2)
    orientation = 0.5 .* atan.(v, u)
    ρ_major = ρ .* exp.(0.5 .* r)
    ρ_minor = ρ .* exp.(-0.5 .* r)

    return (
        θ = θn,
        ρ = ρ,
        σ = σ,
        u = u,
        v = v,
        r = r,
        orientation = orientation,
        H = H,
        κ = κ,
        τ_nominal = τ_nominal,
        ρ_major = ρ_major,
        ρ_minor = ρ_minor
    )
end

"""
    matern_spde_operator(domain, prior, θ = NamedTuple(); compensated = true, halo = nothing)

Assemble the finite-volume Matérn SPDE operator for `α = 2`:

`Q = Dτ * K * C^-1 * K * Dτ`, with `K = L_H + Diagonal(volumes .* κ.^2)`.
"""
function matern_spde_operator(domain::DataDomain, prior::MaternSPDE2Prior, θ = NamedTuple(); compensated::Bool = true, halo = nothing)
    _validate_matern_domain(domain, prior)
    if !isnothing(halo)
        halo isa MaternHaloSpec || throw(ArgumentError("halo must be nothing or a MaternHaloSpec, got $(typeof(halo))."))
        _validate_matern_halo(domain, prior, halo, compensated)
    end
    fields = matern_parameter_fields(prior, θ)
    κ_eff, u_eff, v_eff, H_eff, logkappa_offset, anisotropy_amplitude_offset = _apply_range_compensation(prior, fields; compensated = compensated)

    T = _matern_diffusion_transmissibility(domain, prior, fields, H_eff)
    N = domain[:neighbors]
    nc = prior.nc
    volumes = domain[:volumes]

    logτ = log.(fields.τ_nominal)
    if compensated && !isnothing(prior.sd_compensation)
        sdcomp = prior.sd_compensation::MaternSDCompensation
        θn = fields.θ
        logτ .+= sdcomp.logtau_offset
        logτ .+= sdcomp.range_sensitivity*θn.range
        logτ .+= sdcomp.anisotropy_u_sensitivity*θn.anisotropy_u
        logτ .+= sdcomp.anisotropy_v_sensitivity*θn.anisotropy_v
    end
    τ_eff = exp.(logτ)
    interior_idx = collect(1:nc)

    if isnothing(halo)
        L_H = tpfa_laplacian(N, T, nc)
        volumes_eff = Vector{Float64}(volumes)
        κ_all = κ_eff
        τ_all = τ_eff
        halo_idx = Int[]
        halo_meta = nothing
        transmissibilities_ext = T
        fields_out = _matern_extend_fields(fields, Int[], κ_eff, τ_eff, u_eff, v_eff, H_eff, logkappa_offset, anisotropy_amplitude_offset)
    else
        halo_assembly = _build_matern_halo(domain, halo, fields, κ_eff, τ_eff, u_eff, v_eff, H_eff, logkappa_offset, anisotropy_amplitude_offset, T)
        L_H = _edge_laplacian(
            vcat(halo_assembly.interior_left, halo_assembly.extra_left),
            vcat(halo_assembly.interior_right, halo_assembly.extra_right),
            vcat(halo_assembly.interior_weight, halo_assembly.extra_weight),
            halo_assembly.total_cells
        )
        volumes_eff = halo_assembly.volumes
        κ_all = halo_assembly.κ_eff
        τ_all = halo_assembly.τ
        halo_idx = halo_assembly.halo_idx
        halo_meta = halo_assembly.meta
        transmissibilities_ext = vcat(halo_assembly.interior_weight, halo_assembly.extra_weight)
        fields_out = halo_assembly.fields
    end

    C = Diagonal(volumes_eff)
    Kmat = sparse(L_H + Diagonal(volumes_eff .* κ_all.^2))
    K = Symmetric(Kmat)
    Dτ = Diagonal(τ_all)
    Cinv = Diagonal(1.0 ./ volumes_eff)
    Q = Symmetric(sparse(Dτ*Kmat*Cinv*Kmat*Dτ))

    return (
        L_H = L_H,
        C = C,
        K = K,
        Q = Q,
        transmissibilities = T,
        transmissibilities_ext = transmissibilities_ext,
        interior_idx = interior_idx,
        halo_idx = halo_idx,
        halo_meta = halo_meta,
        fields = fields_out
    )
end

"""
    matern_realized_variance(domain, prior, θ = NamedTuple(); method = :diag_qinv, anchors = :all, compensated = true)

Compute realized marginal variances for the selected anchor cells from the
precision matrix.
"""
function matern_realized_variance(
        domain::DataDomain,
        prior::MaternSPDE2Prior,
        θ = NamedTuple();
        method::Symbol = :diag_qinv,
        anchors = :all,
        compensated::Bool = true,
        halo = nothing
    )
    method == :diag_qinv || throw(ArgumentError("Only method = :diag_qinv is supported, got $method."))
    op = matern_spde_operator(domain, prior, θ; compensated = compensated, halo = halo)
    idx = _resolve_matern_anchors(prior.nc, anchors)
    diagvals = _selected_qinv_diagonal(op.Q, idx)
    return (anchors = idx, realized = diagvals, target = op.fields.σ[idx].^2)
end

"""
    matern_realized_ranges(domain, prior, θ = NamedTuple(); anchors, directions = :local_eigen, corr_level = 0.1, compensated = true)

Estimate realized directional practical ranges at the selected anchor cells.
"""
function matern_realized_ranges(
        domain::DataDomain,
        prior::MaternSPDE2Prior,
        θ = NamedTuple();
        anchors = :all,
        directions::Symbol = :local_eigen,
        corr_level::Real = 0.1,
        compensated::Bool = true,
        halo = nothing
    )
    directions == :local_eigen || throw(ArgumentError("Only directions = :local_eigen is supported, got $directions."))
    0 < corr_level < 1 || throw(ArgumentError("corr_level must lie in (0, 1), got $corr_level."))

    op = matern_spde_operator(domain, prior, θ; compensated = compensated, halo = halo)
    idx = _resolve_matern_anchors(prior.nc, anchors)
    if prior.nc <= 256
        vars = _selected_qinv_diagonal(op.Q, collect(1:prior.nc))
    else
        vars = copy(op.fields.σ[1:prior.nc].^2)
        vars[idx] .= _selected_qinv_diagonal(op.Q, idx)
    end
    covcols = _selected_qinv_columns(op.Q, idx)[1:prior.nc, :]
    pts = _matern_points(domain)

    major = Vector{Float64}(undef, length(idx))
    minor = Vector{Float64}(undef, length(idx))
    dir_major = Matrix{Float64}(undef, 2, length(idx))
    dir_minor = Matrix{Float64}(undef, 2, length(idx))

    for (k, cell) in enumerate(idx)
        Hk = _compact_tensor_to_matrix(op.fields.H_eff[:, cell])
        e_major, e_minor = _principal_directions(Hk)
        dir_major[:, k] .= e_major
        dir_minor[:, k] .= e_minor
        covcol = covcols[:, k]
        corr = covcol ./ sqrt.(max.(vars .* vars[cell], eps(Float64)))
        major[k] = _average_directional_range(pts, cell, corr, e_major, corr_level)
        minor[k] = _average_directional_range(pts, cell, corr, e_minor, corr_level)
    end

    return (
        anchors = idx,
        realized_major = major,
        realized_minor = minor,
        target_major = op.fields.ρ_major[idx],
        target_minor = op.fields.ρ_minor[idx],
        direction_major = dir_major,
        direction_minor = dir_minor
    )
end

"""
    matern_sd_compensation!(prior, domain; mode = :local, diag_method = :chol, anchors = :all, fd_delta = 1e-3, halo = nothing)

Calibrate a cached additive correction to `log(τ)` from mesh-based realized
variance, with optional first-order sensitivity terms for range and anisotropy
coefficients.
"""
function matern_sd_compensation!(
        prior::MaternSPDE2Prior,
        domain::DataDomain;
        mode::Symbol = :local,
        diag_method::Symbol = :chol,
        anchors = :all,
        fd_delta::Real = 1e-3,
        halo = nothing
    )
    mode in (:local, :mean) || throw(ArgumentError("mode must be :local or :mean (got $mode)."))
    diag_method == :chol || throw(ArgumentError("Only diag_method = :chol is supported, got $diag_method."))
    fd = Float64(fd_delta)
    fd > 0 || throw(ArgumentError("fd_delta must be positive, got $fd_delta."))
    idx = _resolve_matern_anchors(prior.nc, anchors)
    if !isnothing(halo)
        halo isa MaternHaloSpec || throw(ArgumentError("halo must be nothing or a MaternHaloSpec, got $(typeof(halo))."))
        mode == :mean || throw(ArgumentError("Halo-compatible SD compensation only supports mode = :mean."))
        isnothing(prior.range_compensation) || throw(ArgumentError("Halo-compatible SD compensation cannot be combined with cached range compensation."))
    end

    sd_saved = prior.sd_compensation
    prior.sd_compensation = nothing
    θ0 = _zero_matern_theta(prior)
    base = matern_realized_variance(domain, prior, θ0; anchors = idx, compensated = true, halo = halo)
    target = matern_parameter_fields(prior, θ0).σ[idx].^2
    base_offset_anchor = 0.5 .* (log.(base.realized) .- log.(target))

    offset = _anchor_field_to_cells(domain, idx, base_offset_anchor; mode = mode)
    Sρ = _matern_variance_sensitivity(domain, prior, idx, :range, fd, base.realized; mode = mode, halo = halo)
    Su = _matern_variance_sensitivity(domain, prior, idx, :anisotropy_u, fd, base.realized; mode = mode, halo = halo)
    Sv = _matern_variance_sensitivity(domain, prior, idx, :anisotropy_v, fd, base.realized; mode = mode, halo = halo)

    prior.sd_compensation = MaternSDCompensation(mode, idx, offset, 0.5 .* Sρ, 0.5 .* Su, 0.5 .* Sv, fd)
    return prior
end

"""
    matern_range_compensation!(prior, domain; mode = :directional_local, anchors = :all, corr_level = 0.1, fd_delta = 1e-3, diag_method = :chol)

Calibrate cached corrections for the realized geometric-mean range and
anisotropy amplitude from local directional range diagnostics.
"""
function matern_range_compensation!(
        prior::MaternSPDE2Prior,
        domain::DataDomain;
        mode::Symbol = :directional_local,
        anchors = :all,
        corr_level::Real = 0.1,
        fd_delta::Real = 1e-3,
        diag_method::Symbol = :chol
    )
    mode == :directional_local || throw(ArgumentError("Only mode = :directional_local is supported, got $mode."))
    diag_method == :chol || throw(ArgumentError("Only diag_method = :chol is supported, got $diag_method."))
    fd = Float64(fd_delta)
    fd > 0 || throw(ArgumentError("fd_delta must be positive, got $fd_delta."))
    0 < corr_level < 1 || throw(ArgumentError("corr_level must lie in (0, 1), got $corr_level."))

    idx = _resolve_matern_anchors(prior.nc, anchors)
    range_saved = prior.range_compensation
    prior.range_compensation = nothing

    θ0 = _zero_matern_theta(prior)
    base = matern_realized_ranges(domain, prior, θ0; anchors = idx, corr_level = corr_level, compensated = true)
    fields0 = matern_parameter_fields(prior, θ0)
    logk_anchor0, amp_anchor0 = _range_correction_from_ranges(
        base.realized_major,
        base.realized_minor,
        fields0.ρ_major[idx],
        fields0.ρ_minor[idx]
    )

    logkappa_offset = _anchor_field_to_cells(domain, idx, logk_anchor0)
    amp_offset = _anchor_field_to_cells(domain, idx, amp_anchor0)

    Sρ_logk, Sρ_amp = _matern_range_sensitivity(domain, prior, idx, :range, fd, corr_level, logk_anchor0, amp_anchor0)
    Su_logk, Su_amp = _matern_range_sensitivity(domain, prior, idx, :anisotropy_u, fd, corr_level, logk_anchor0, amp_anchor0)
    Sv_logk, Sv_amp = _matern_range_sensitivity(domain, prior, idx, :anisotropy_v, fd, corr_level, logk_anchor0, amp_anchor0)

    damping = 0.5
    logkappa_offset .*= damping
    amp_offset .*= damping
    Sρ_logk .*= damping
    Su_logk .*= damping
    Sv_logk .*= damping
    Sρ_amp .*= damping
    Su_amp .*= damping
    Sv_amp .*= damping

    prior.range_compensation = MaternRangeCompensation(
        mode,
        idx,
        logkappa_offset,
        amp_offset,
        Sρ_logk,
        Su_logk,
        Sv_logk,
        Sρ_amp,
        Su_amp,
        Sv_amp,
        Float64(corr_level),
        fd
    )
    return prior
end

function _matern_basis_matrix(basis, nc::Integer, ::Type{T}) where T
    if isnothing(basis)
        return zeros(T, nc, 0)
    end
    M = Matrix{T}(basis)
    size(M, 1) == nc || throw(ArgumentError("Basis has $(size(M, 1)) rows, expected $nc."))
    return M
end

function _theta_block(theta::NamedTuple, name::Symbol, n::Integer)
    if haskey(theta, name)
        x = collect(theta[name])
        length(x) == n || throw(ArgumentError("θ.$name has length $(length(x)), expected $n."))
        return Float64.(x)
    else
        return zeros(Float64, n)
    end
end

function _normalize_matern_theta(prior::MaternSPDE2Prior, θ)
    θ isa NamedTuple || throw(ArgumentError("θ must be a named tuple with optional fields :range, :sd, :anisotropy_u, :anisotropy_v."))
    return (
        range = _theta_block(θ, :range, size(prior.range_basis, 2)),
        sd = _theta_block(θ, :sd, size(prior.sd_basis, 2)),
        anisotropy_u = _theta_block(θ, :anisotropy_u, size(prior.anisotropy_u_basis, 2)),
        anisotropy_v = _theta_block(θ, :anisotropy_v, size(prior.anisotropy_v_basis, 2))
    )
end

function _zero_matern_theta(prior::MaternSPDE2Prior)
    return (
        range = zeros(Float64, size(prior.range_basis, 2)),
        sd = zeros(Float64, size(prior.sd_basis, 2)),
        anisotropy_u = zeros(Float64, size(prior.anisotropy_u_basis, 2)),
        anisotropy_v = zeros(Float64, size(prior.anisotropy_v_basis, 2))
    )
end

function _matern_tau_nominal(ν::Real, κ::AbstractVector, σ::AbstractVector; d::Real = 2.0)
    α = ν + d/2
    g1 = _matern_gamma_halfint_or_int(Float64(ν))
    g2 = _matern_gamma_halfint_or_int(Float64(α))
    return sqrt.(g1 ./ (g2 .* (4π)^(d/2) .* κ.^(2ν) .* σ.^2))
end

function _matern_gamma_halfint_or_int(x::Real)
    x > 0 || throw(ArgumentError("gamma(x) requires x > 0, got $x."))
    two_x = 2x
    isapprox(two_x, round(two_x); atol = 1e-12) || throw(ArgumentError("gamma is only implemented for integer or half-integer inputs, got $x."))
    n2 = Int(round(two_x))
    if iseven(n2)
        n = n2 ÷ 2
        return float(factorial(n - 1))
    end
    m = (n2 - 1) ÷ 2
    g = sqrt(pi)
    for k in 1:m
        g *= (k - 0.5)
    end
    return g
end

function _matern_anisotropy_tensor(u::AbstractVector, v::AbstractVector)
    n = length(u)
    H = Matrix{Float64}(undef, 3, n)
    @inbounds for i in 1:n
        ui = u[i]
        vi = v[i]
        ri = hypot(ui, vi)
        if ri < 1e-14
            c = 1.0
            s = 1.0
        else
            c = cosh(ri)
            s = sinh(ri)/ri
        end
        H[1, i] = c + s*ui
        H[2, i] = s*vi
        H[3, i] = c - s*ui
    end
    return H
end

function _apply_range_compensation(prior::MaternSPDE2Prior, fields; compensated::Bool)
    κ_eff = copy(fields.κ)
    u_eff = copy(fields.u)
    v_eff = copy(fields.v)
    logkappa_offset = zeros(Float64, prior.nc)
    amplitude_offset = zeros(Float64, prior.nc)

    if compensated && !isnothing(prior.range_compensation)
        rc = prior.range_compensation::MaternRangeCompensation
        θn = fields.θ
        logkappa_offset .+= rc.logkappa_offset
        logkappa_offset .+= rc.range_sensitivity*θn.range
        logkappa_offset .+= rc.anisotropy_u_sensitivity*θn.anisotropy_u
        logkappa_offset .+= rc.anisotropy_v_sensitivity*θn.anisotropy_v

        amplitude_offset .+= rc.anisotropy_amplitude_offset
        amplitude_offset .+= rc.amplitude_range_sensitivity*θn.range
        amplitude_offset .+= rc.amplitude_u_sensitivity*θn.anisotropy_u
        amplitude_offset .+= rc.amplitude_v_sensitivity*θn.anisotropy_v

        logkappa_offset .= clamp.(logkappa_offset, -2.0, 2.0)
        amplitude_offset .= clamp.(amplitude_offset, -1.0, 1.0)
        κ_eff .= max.(κ_eff .* exp.(logkappa_offset), prior.κ_min)
        r = sqrt.(u_eff.^2 .+ v_eff.^2)
        mask = r .> 1e-12
        r_eff = clamp.(r .+ amplitude_offset, 0.0, 2.0)
        scale = ones(Float64, prior.nc)
        @inbounds for i in eachindex(scale)
            if mask[i]
                scale[i] = r_eff[i]/r[i]
            end
        end
        u_eff .*= scale
        v_eff .*= scale
    end

    H_eff = _matern_anisotropy_tensor(u_eff, v_eff)
    return κ_eff, u_eff, v_eff, H_eff, logkappa_offset, amplitude_offset
end

function _matern_diffusion_transmissibility(domain::DataDomain, prior::MaternSPDE2Prior, fields, H_eff::AbstractMatrix)
    has_anisotropy = maximum(abs.(H_eff[2, :])) > 1e-12 || maximum(abs.(H_eff[1, :] .- 1.0)) > 1e-12 || maximum(abs.(H_eff[3, :] .- 1.0)) > 1e-12
    if prior.laplace_from == :geometry
        prior.version == :xyz || throw(ArgumentError("Anisotropic Matérn SPDE assembly requires version = :xyz."))
        d2 = deepcopy(domain)
        d2[:permeability] = _embed_horizontal_tensor(domain, H_eff)
        # Geometry mode should represent a stationary grid-based operator, not the
        # reservoir flow model. Neutralize flow-specific transmissibility edits
        # such as NTG and fault multipliers here; use `laplace_from = :flow` to
        # keep those effects in the prior.
        if haskey(d2, :net_to_gross)
            d2[:net_to_gross] = ones(Float64, number_of_cells(d2))
        end
        if haskey(d2, :transmissibility_multiplier, Faces())
            d2[:transmissibility_multiplier, Faces()] = ones(Float64, number_of_faces(d2))
        end
        if haskey(d2, :transmissibility_override, Faces())
            d2[:transmissibility_override, Faces()] = fill(NaN, number_of_faces(d2))
        end
        return reservoir_transmissibility(d2; version = prior.version)
    elseif prior.laplace_from == :unit
        if has_anisotropy
            throw(ArgumentError("laplace_from = :unit does not support anisotropy. Use :geometry instead."))
        end
        return ones(Float64, size(domain[:neighbors], 2))
    elseif prior.laplace_from == :flow
        if has_anisotropy
            throw(ArgumentError("laplace_from = :flow with anisotropy is not supported in this first implementation. Use :geometry."))
        end
        return reservoir_transmissibility(domain; version = prior.version)
    else
        error("Unsupported laplace_from = $(prior.laplace_from)")
    end
end

function _embed_horizontal_tensor(domain::DataDomain, H::AbstractMatrix)
    D = size(domain[:cell_centroids], 1)
    nc = size(H, 2)
    if D == 2
        return H
    elseif D == 3
        perm = zeros(Float64, 6, nc)
        perm[1, :] .= H[1, :]
        perm[2, :] .= H[2, :]
        perm[3, :] .= 0.0
        perm[4, :] .= H[3, :]
        perm[5, :] .= 0.0
        perm[6, :] .= 1.0
        return perm
    else
        throw(ArgumentError("Only 2D or embedded single-layer 3D domains are supported, got centroid dimension $D."))
    end
end

function _validate_matern_domain(domain::DataDomain, prior::MaternSPDE2Prior)
    number_of_cells(domain) == prior.nc || throw(ArgumentError("Prior has nc = $(prior.nc), but domain has $(number_of_cells(domain)) cells."))
    D = size(domain[:cell_centroids], 1)
    D in (2, 3) || throw(ArgumentError("Only 2D or embedded single-layer 3D domains are supported, got dimension $D."))
    return nothing
end

function _validate_matern_halo(domain::DataDomain, prior::MaternSPDE2Prior, halo::MaternHaloSpec, compensated::Bool)
    prior.laplace_from == :geometry || throw(ArgumentError("Halo padding currently requires laplace_from = :geometry, got $(prior.laplace_from)."))
    prior.version == :xyz || throw(ArgumentError("Halo padding currently requires version = :xyz, got $(prior.version)."))
    if compensated
        if !isnothing(prior.range_compensation)
            throw(ArgumentError("Halo padding is not compatible with cached range compensation in this first implementation."))
        end
        if !isnothing(prior.sd_compensation)
            sdcomp = prior.sd_compensation::MaternSDCompensation
            sdcomp.mode == :mean || throw(ArgumentError("Halo padding only supports mean SD compensation in this first implementation."))
        end
    end
    haskey(domain, :boundary_centroids) || throw(ArgumentError("Domain is missing :boundary_centroids required for halo padding."))
    haskey(domain, :boundary_areas) || throw(ArgumentError("Domain is missing :boundary_areas required for halo padding."))
    haskey(domain, :boundary_neighbors) || throw(ArgumentError("Domain is missing :boundary_neighbors required for halo padding."))
    haskey(domain, :boundary_normals) || throw(ArgumentError("Domain is missing :boundary_normals required for halo padding."))
    return nothing
end

function _matern_extend_fields(fields, source::AbstractVector{<:Integer}, κ_eff, τ_eff, u_eff, v_eff, H_eff, logkappa_offset, amplitude_offset)
    return (
        θ = fields.θ,
        ρ = vcat(fields.ρ, fields.ρ[source]),
        σ = vcat(fields.σ, fields.σ[source]),
        u = vcat(fields.u, fields.u[source]),
        v = vcat(fields.v, fields.v[source]),
        r = vcat(fields.r, fields.r[source]),
        orientation = vcat(fields.orientation, fields.orientation[source]),
        H = hcat(fields.H, fields.H[:, source]),
        κ = vcat(fields.κ, fields.κ[source]),
        τ_nominal = vcat(fields.τ_nominal, fields.τ_nominal[source]),
        ρ_major = vcat(fields.ρ_major, fields.ρ_major[source]),
        ρ_minor = vcat(fields.ρ_minor, fields.ρ_minor[source]),
        κ_eff = vcat(κ_eff, κ_eff[source]),
        τ = vcat(τ_eff, τ_eff[source]),
        u_eff = vcat(u_eff, u_eff[source]),
        v_eff = vcat(v_eff, v_eff[source]),
        H_eff = hcat(H_eff, H_eff[:, source]),
        logkappa_offset = vcat(logkappa_offset, logkappa_offset[source]),
        anisotropy_amplitude_offset = vcat(amplitude_offset, amplitude_offset[source])
    )
end

function _edge_laplacian(left::AbstractVector{<:Integer}, right::AbstractVector{<:Integer}, weight::AbstractVector, nc::Integer)
    (length(left) == length(right) && length(right) == length(weight)) || throw(ArgumentError("Edge lists must have equal length."))
    diag = zeros(Float64, nc)
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, 3*length(weight) + nc)
    sizehint!(J, 3*length(weight) + nc)
    sizehint!(V, 3*length(weight) + nc)

    @inbounds for e in eachindex(weight)
        i = left[e]
        j = right[e]
        w = Float64(weight[e])
        1 <= i <= nc || throw(ArgumentError("Edge left index $i is outside 1:$nc."))
        1 <= j <= nc || throw(ArgumentError("Edge right index $j is outside 1:$nc."))
        i == j && continue
        push!(I, i); push!(J, j); push!(V, -w)
        push!(I, j); push!(J, i); push!(V, -w)
        diag[i] += w
        diag[j] += w
    end

    for i in 1:nc
        push!(I, i); push!(J, i); push!(V, diag[i])
    end
    return sparse(I, J, V, nc, nc)
end

function _halo_side(v::NTuple{2, Float64})
    if abs(v[1]) >= abs(v[2])
        return v[1] >= 0 ? :east : :west
    else
        return v[2] >= 0 ? :north : :south
    end
end

function _horizontal_boundary_faces(domain::DataDomain)
    cc = Matrix{Float64}(domain[:cell_centroids])
    bc = Matrix{Float64}(domain[:boundary_centroids])
    bn = Int.(vec(domain[:boundary_neighbors]))
    normals = Matrix{Float64}(domain[:boundary_normals])
    faces = Int[]
    face_side = Dict{Int, Symbol}()
    outward = Dict{Int, NTuple{2, Float64}}()
    tangential_coord = Dict{Int, Float64}()

    for f in eachindex(bn)
        cell = bn[f]
        dv = bc[1:2, f] .- cc[1:2, cell]
        nxy = normals[1:2, f]
        if norm(dv) <= 1e-10
            dv = nxy
        end
        normxy = norm(dv)
        normxy > 1e-10 || continue
        dir = (Float64(dv[1]/normxy), Float64(dv[2]/normxy))
        side = _halo_side(dir)
        push!(faces, f)
        face_side[f] = side
        outward[f] = dir
        tangential_coord[f] = side in (:east, :west) ? Float64(bc[2, f]) : Float64(bc[1, f])
    end

    side_faces = Dict(side => Int[] for side in (:west, :east, :south, :north))
    for f in faces
        push!(side_faces[face_side[f]], f)
    end
    for side in keys(side_faces)
        sort!(side_faces[side], by = f -> tangential_coord[f])
    end
    return (faces = faces, face_side = face_side, outward = outward, side_faces = side_faces)
end

function _halo_total_padding(spec, ρ_boundary::AbstractVector)
    if spec === :boundary_range
        val = mean(ρ_boundary)
    else
        val = Float64(spec)
    end
    isfinite(val) && val > 0 || throw(ArgumentError("Halo padding distance must be positive and finite, got $val."))
    return val
end

function _halo_layer_widths(total_padding::Real, layers::Integer, growth::Real)
    total_padding > 0 || throw(ArgumentError("total_padding must be positive, got $total_padding."))
    layers > 0 || throw(ArgumentError("layers must be positive, got $layers."))
    growth > 0 || throw(ArgumentError("growth must be positive, got $growth."))
    if abs(growth - 1.0) < 1e-12
        return fill(Float64(total_padding)/layers, layers)
    end
    base = Float64(total_padding)*(growth - 1.0)/(growth^layers - 1.0)
    return [base*growth^(k - 1) for k in 1:layers]
end

function _project_compact_tensor(h::AbstractVector, dir::NTuple{2, Float64})
    H = _compact_tensor_to_matrix(h)
    v = @SVector [dir[1], dir[2]]
    return max(dot(v, H*v), 1e-12)
end

function _interior_edge_transmissibility_map(neighbors::AbstractMatrix{<:Integer}, transmissibilities::AbstractVector)
    edge_map = Dict{Tuple{Int, Int}, Float64}()
    for f in axes(neighbors, 2)
        i = neighbors[1, f]
        j = neighbors[2, f]
        if i > 0 && j > 0
            key = i < j ? (i, j) : (j, i)
            edge_map[key] = get(edge_map, key, 0.0) + Float64(transmissibilities[f])
        end
    end
    return edge_map
end

function _add_halo_corner_connections!(left, right, weight, side_faces, face_to_chain, chain_normal_weights)
    seen = Set{Tuple{Int, Int}}()
    corner_pairs = Tuple{Int, Int}[]

    !isempty(side_faces[:south]) && !isempty(side_faces[:west]) && push!(corner_pairs, (first(side_faces[:south]), first(side_faces[:west])))
    !isempty(side_faces[:south]) && !isempty(side_faces[:east]) && push!(corner_pairs, (last(side_faces[:south]), first(side_faces[:east])))
    !isempty(side_faces[:north]) && !isempty(side_faces[:west]) && push!(corner_pairs, (first(side_faces[:north]), last(side_faces[:west])))
    !isempty(side_faces[:north]) && !isempty(side_faces[:east]) && push!(corner_pairs, (last(side_faces[:north]), last(side_faces[:east])))

    for (f1, f2) in corner_pairs
        chain1 = face_to_chain[f1]
        chain2 = face_to_chain[f2]
        for l in 1:min(length(chain1), length(chain2))
            i = chain1[l]
            j = chain2[l]
            i == j && continue
            key = i < j ? (i, j) : (j, i)
            key in seen && continue
            push!(seen, key)
            w = min(chain_normal_weights[f1][l], chain_normal_weights[f2][l])
            push!(left, i)
            push!(right, j)
            push!(weight, w)
        end
    end
    return nothing
end

function _build_matern_halo(domain::DataDomain, halo::MaternHaloSpec, fields, κ_eff, τ_eff, u_eff, v_eff, H_eff, logkappa_offset, amplitude_offset, transmissibilities)
    nc = number_of_cells(domain)
    cc = Matrix{Float64}(domain[:cell_centroids])
    bc = Matrix{Float64}(domain[:boundary_centroids])
    bn = Int.(vec(domain[:boundary_neighbors]))
    ba = Float64.(vec(domain[:boundary_areas]))
    D = size(cc, 1)

    binfo = _horizontal_boundary_faces(domain)
    faces = binfo.faces
    isempty(faces) && throw(ArgumentError("Could not identify any horizontal boundary faces for halo padding."))

    boundary_cells = unique(bn[faces])
    total_padding = _halo_total_padding(halo.total_padding, fields.ρ[boundary_cells])
    widths = _halo_layer_widths(total_padding, halo.layers, halo.growth)

    nh = length(faces)*halo.layers
    halo_idx = collect((nc + 1):(nc + nh))
    halo_centroids = zeros(Float64, D, nh)
    halo_volumes = zeros(Float64, nh)
    halo_source = zeros(Int, nh)
    face_to_chain = Dict{Int, Vector{Int}}()
    cell_side_to_chain = Dict{Tuple{Int, Symbol}, Vector{Int}}()
    chain_normal_weights = Dict{Int, Vector{Float64}}()

    extra_left = Int[]
    extra_right = Int[]
    extra_weight = Float64[]
    interior_left = Int[]
    interior_right = Int[]
    interior_weight = Float64[]
    for f in axes(domain[:neighbors], 2)
        i = domain[:neighbors][1, f]
        j = domain[:neighbors][2, f]
        if i > 0 && j > 0
            push!(interior_left, i)
            push!(interior_right, j)
            push!(interior_weight, Float64(transmissibilities[f]))
        end
    end

    cursor = 0
    for f in faces
        cell = bn[f]
        side = binfo.face_side[f]
        out = binfo.outward[f]
        area = ba[f]
        kproj = _project_compact_tensor(H_eff[:, cell], out)
        chain = Int[]
        normal_weights = Float64[]
        cumulative = 0.0
        prev_cell = cell
        interior_dist = max(norm(bc[1:2, f] .- cc[1:2, cell]), 1e-8)
        left_half = area*kproj/interior_dist

        for layer in 1:halo.layers
            cursor += 1
            gid = nc + cursor
            push!(chain, gid)
            halo_source[cursor] = cell
            halo_volumes[cursor] = area*widths[layer]
            halo_centroids[1, cursor] = bc[1, f] + out[1]*(cumulative + widths[layer]/2)
            halo_centroids[2, cursor] = bc[2, f] + out[2]*(cumulative + widths[layer]/2)
            if D == 3
                halo_centroids[3, cursor] = bc[3, f]
            end

            right_half = area*kproj/max(widths[layer]/2, 1e-8)
            w = 1.0/(1.0/max(left_half, 1e-12) + 1.0/max(right_half, 1e-12))
            push!(extra_left, prev_cell)
            push!(extra_right, gid)
            push!(extra_weight, w)
            push!(normal_weights, w)

            prev_cell = gid
            left_half = right_half
            cumulative += widths[layer]
        end

        face_to_chain[f] = chain
        cell_side_to_chain[(cell, side)] = chain
        chain_normal_weights[f] = normal_weights
    end

    edge_map = _interior_edge_transmissibility_map(domain[:neighbors], transmissibilities)
    for side in (:west, :east, :south, :north)
        side_list = binfo.side_faces[side]
        for k in 1:(length(side_list) - 1)
            f1 = side_list[k]
            f2 = side_list[k + 1]
            c1 = bn[f1]
            c2 = bn[f2]
            key = c1 < c2 ? (c1, c2) : (c2, c1)
            w = get(edge_map, key, NaN)
            if isfinite(w) && w > 0
                chain1 = face_to_chain[f1]
                chain2 = face_to_chain[f2]
                for layer in 1:halo.layers
                    push!(extra_left, chain1[layer])
                    push!(extra_right, chain2[layer])
                    push!(extra_weight, w)
                end
            end
        end
    end

    halo.include_corners && _add_halo_corner_connections!(extra_left, extra_right, extra_weight, binfo.side_faces, face_to_chain, chain_normal_weights)

    volumes_ext = vcat(Float64.(domain[:volumes]), halo_volumes)
    fields_ext = _matern_extend_fields(fields, halo_source, κ_eff, τ_eff, u_eff, v_eff, H_eff, logkappa_offset, amplitude_offset)

    return (
        total_cells = nc + nh,
        halo_idx = halo_idx,
        volumes = volumes_ext,
        κ_eff = fields_ext.κ_eff,
        τ = fields_ext.τ,
        fields = fields_ext,
        interior_left = interior_left,
        interior_right = interior_right,
        interior_weight = interior_weight,
        extra_left = extra_left,
        extra_right = extra_right,
        extra_weight = extra_weight,
        meta = (
            face_to_chain = face_to_chain,
            cell_side_to_chain = cell_side_to_chain,
            side_faces = binfo.side_faces,
            face_side = binfo.face_side,
            halo_centroids = halo_centroids,
            ext_centroids = hcat(cc, halo_centroids),
            halo_source = halo_source
        )
    )
end

function _resolve_matern_anchors(nc::Integer, anchors)
    if anchors === :all
        return collect(1:nc)
    elseif anchors isa Integer
        1 <= anchors <= nc || throw(ArgumentError("Anchor $anchors is outside 1:$nc."))
        return [Int(anchors)]
    elseif anchors isa AbstractVector{<:Integer}
        idx = unique!(sort!(Int.(collect(anchors))))
        all(1 .<= idx .<= nc) || throw(ArgumentError("All anchors must lie in 1:$nc."))
        return idx
    else
        throw(ArgumentError("anchors must be :all, an integer, or a vector of integers (got $(typeof(anchors)))."))
    end
end

_matern_points(domain::DataDomain) = Matrix{Float64}(domain[:cell_centroids][1:2, :])

function _anchor_field_to_cells(domain::DataDomain, anchors::AbstractVector{<:Integer}, values::AbstractVector; mode::Symbol = :local)
    nc = number_of_cells(domain)
    if mode == :mean
        return fill(mean(values), nc)
    elseif length(anchors) == nc
        return Float64.(values)
    end

    pts = _matern_points(domain)
    ap = pts[:, anchors]
    out = zeros(Float64, nc)
    anchor_lookup = Dict(a => k for (k, a) in enumerate(anchors))

    for c in 1:nc
        if haskey(anchor_lookup, c)
            out[c] = values[anchor_lookup[c]]
            continue
        end
        wsum = 0.0
        vsum = 0.0
        x = pts[:, c]
        for (k, a) in enumerate(anchors)
            d2 = sum(abs2, ap[:, k] .- x)
            w = inv(max(d2, 1e-12))
            wsum += w
            vsum += w*values[k]
        end
        out[c] = vsum/wsum
    end
    return out
end

function _anchor_sensitivity_to_cells(domain::DataDomain, anchors::AbstractVector{<:Integer}, S_anchor::AbstractMatrix; mode::Symbol = :local)
    nc = number_of_cells(domain)
    p = size(S_anchor, 2)
    if mode == :mean
        return repeat(reshape(vec(mean(S_anchor, dims = 1)), 1, p), nc, 1)
    elseif length(anchors) == nc
        return Matrix{Float64}(S_anchor)
    end
    S = Matrix{Float64}(undef, nc, p)
    for j in 1:p
        S[:, j] .= _anchor_field_to_cells(domain, anchors, view(S_anchor, :, j))
    end
    return S
end

function _factor_precision_matrix(Q::Symmetric; shift_rtol::Real = 1e-12, max_tries::Integer = 10)
    try
        return cholesky(Q)
    catch err
        err isa PosDefException || rethrow()
    end

    diag_scale = max(maximum(abs, diag(parent(Q))), 1.0)
    for k in 0:(max_tries - 1)
        shift = Float64(shift_rtol) * diag_scale * 10.0^k
        try
            return cholesky(Q; shift = shift, check = true)
        catch err
            err isa PosDefException || rethrow()
        end
    end
    throw(PosDefException(size(Q, 1)))
end

function _selected_qinv_diagonal(Q::Symmetric, idx::AbstractVector{<:Integer}; chunk_size::Int = 32)
    F = _factor_precision_matrix(Q)
    n = size(Q, 1)
    vals = zeros(Float64, length(idx))
    e = zeros(Float64, n, 0)
    for start in 1:chunk_size:length(idx)
        stop = min(start + chunk_size - 1, length(idx))
        block = idx[start:stop]
        E = zeros(Float64, n, length(block))
        for (k, j) in enumerate(block)
            E[j, k] = 1.0
        end
        X = F \ E
        for (k, j) in enumerate(block)
            vals[start + k - 1] = X[j, k]
        end
    end
    return vals
end

function _selected_qinv_columns(Q::Symmetric, idx::AbstractVector{<:Integer}; chunk_size::Int = 16)
    F = _factor_precision_matrix(Q)
    n = size(Q, 1)
    cols = Matrix{Float64}(undef, n, length(idx))
    for start in 1:chunk_size:length(idx)
        stop = min(start + chunk_size - 1, length(idx))
        block = idx[start:stop]
        E = zeros(Float64, n, length(block))
        for (k, j) in enumerate(block)
            E[j, k] = 1.0
        end
        cols[:, start:stop] .= F \ E
    end
    return cols
end

function _compact_tensor_to_matrix(h::AbstractVector)
    return @SMatrix [h[1] h[2]; h[2] h[3]]
end

function _principal_directions(H::AbstractMatrix)
    F = eigen(Symmetric(Matrix(H)))
    if F.values[2] >= F.values[1]
        e_major = F.vectors[:, 2]
        e_minor = F.vectors[:, 1]
    else
        e_major = F.vectors[:, 1]
        e_minor = F.vectors[:, 2]
    end
    return e_major ./ norm(e_major), e_minor ./ norm(e_minor)
end

function _local_spacing(pts::AbstractMatrix, anchor::Integer)
    x = pts[:, anchor]
    dmin = Inf
    for j in axes(pts, 2)
        j == anchor && continue
        dmin = min(dmin, norm(pts[:, j] - x))
    end
    return isfinite(dmin) ? dmin : 1.0
end

function _range_along_ray(pts::AbstractMatrix, anchor::Integer, corr::AbstractVector, dir::AbstractVector, corr_level::Real)
    x0 = pts[:, anchor]
    n = @SVector [-dir[2], dir[1]]
    spacing = _local_spacing(pts, anchor)
    tol = 0.55*spacing + 1e-12

    pairs = Tuple{Float64, Float64}[]
    for j in axes(pts, 2)
        dx = pts[:, j] - x0
        proj = dot(dx, dir)
        proj <= 0 && continue
        perp = abs(dot(dx, n))
        perp <= tol || continue
        push!(pairs, (proj, corr[j]))
    end
    isempty(pairs) && return NaN
    sort!(pairs, by = first)

    prev_d = 0.0
    prev_c = 1.0
    for (d, c) in pairs
        if c <= corr_level
            if abs(c - prev_c) < 1e-12
                return d
            end
            t = (corr_level - prev_c)/(c - prev_c)
            return prev_d + t*(d - prev_d)
        end
        prev_d = d
        prev_c = c
    end
    return last(pairs)[1]
end

function _average_directional_range(pts::AbstractMatrix, anchor::Integer, corr::AbstractVector, dir::AbstractVector, corr_level::Real)
    rp = _range_along_ray(pts, anchor, corr, dir, corr_level)
    rm = _range_along_ray(pts, anchor, corr, -dir, corr_level)
    vals = filter(isfinite, [rp, rm])
    isempty(vals) && return NaN
    return mean(vals)
end

function _range_correction_from_ranges(realized_major, realized_minor, target_major, target_minor)
    n = length(realized_major)
    logk = zeros(Float64, n)
    amp = zeros(Float64, n)
    for i in 1:n
        rg = sqrt(realized_major[i]*realized_minor[i])
        tg = sqrt(target_major[i]*target_minor[i])
        rr = realized_major[i]/realized_minor[i]
        tr = target_major[i]/target_minor[i]
        if all(isfinite, (rg, tg, rr, tr)) && rg > 0 && tg > 0 && rr > 0 && tr > 0
            logk[i] = log(rg/tg)
            amp[i] = log(tr/rr)
        end
    end
    return logk, amp
end

function _matern_variance_sensitivity(
        domain::DataDomain,
        prior::MaternSPDE2Prior,
        anchors::Vector{Int},
        block::Symbol,
        fd::Float64,
        base_realized::AbstractVector;
        mode::Symbol,
        halo = nothing
    )
    p = _matern_block_size(prior, block)
    S_anchor = zeros(Float64, length(anchors), p)
    for j in 1:p
        θ = _zero_matern_theta(prior)
        _set_theta_entry!(θ, block, j, fd)
        pert = matern_realized_variance(domain, prior, θ; anchors = anchors, compensated = true, halo = halo)
        S_anchor[:, j] .= (log.(pert.realized) .- log.(base_realized)) ./ fd
    end
    return _anchor_sensitivity_to_cells(domain, anchors, S_anchor; mode = mode)
end

function _matern_range_sensitivity(domain::DataDomain, prior::MaternSPDE2Prior, anchors::Vector{Int}, block::Symbol, fd::Float64, corr_level::Float64, base_logk::AbstractVector, base_amp::AbstractVector)
    p = _matern_block_size(prior, block)
    Slogk_anchor = zeros(Float64, length(anchors), p)
    Samp_anchor = zeros(Float64, length(anchors), p)
    for j in 1:p
        θ = _zero_matern_theta(prior)
        _set_theta_entry!(θ, block, j, fd)
        ranges = matern_realized_ranges(domain, prior, θ; anchors = anchors, corr_level = corr_level, compensated = true)
        fields = matern_parameter_fields(prior, θ)
        logk, amp = _range_correction_from_ranges(
            ranges.realized_major,
            ranges.realized_minor,
            fields.ρ_major[anchors],
            fields.ρ_minor[anchors]
        )
        Slogk_anchor[:, j] .= (logk .- base_logk) ./ fd
        Samp_anchor[:, j] .= (amp .- base_amp) ./ fd
    end
    return _anchor_sensitivity_to_cells(domain, anchors, Slogk_anchor), _anchor_sensitivity_to_cells(domain, anchors, Samp_anchor)
end

function _matern_block_size(prior::MaternSPDE2Prior, block::Symbol)
    if block == :range
        return size(prior.range_basis, 2)
    elseif block == :anisotropy_u
        return size(prior.anisotropy_u_basis, 2)
    elseif block == :anisotropy_v
        return size(prior.anisotropy_v_basis, 2)
    else
        error("Unsupported θ block $block")
    end
end

function _set_theta_entry!(θ::NamedTuple, block::Symbol, j::Integer, value::Real)
    θ[block][j] = value
    return θ
end
