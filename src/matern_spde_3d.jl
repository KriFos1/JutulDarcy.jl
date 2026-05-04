mutable struct MaternSPDE3Prior{T<:AbstractFloat, M<:AbstractMatrix{T}}
    nc::Int
    ν::T
    ρ0::T
    σ0::T
    κ_min::T
    laplace_from::Symbol
    version::Symbol
    anisotropy_a_basis::M
    anisotropy_b_basis::M
    sd_compensation::Any
end

struct MaternMeanSDCompensation{T<:AbstractFloat}
    mode::Symbol
    anchors::Vector{Int}
    logtau_offset::T
end

"""
    MaternSPDE3Prior(nc; ν = 0.5, ρ0 = 1.0, σ0 = 1.0, κ_min = 1e-8, laplace_from = :geometry, version = :xyz, anisotropy_a_basis = nothing, anisotropy_b_basis = nothing)

Specification for a 3D Matérn SPDE prior with `α = 2` and optional
axis-aligned anisotropy.

The anisotropy is parameterised by two scalars `a`, `b` that define a
diagonal diffusion tensor `H = diag(exp(a), exp(b), exp(-(a+b)))` with
`det(H) = 1`.  When `a = b = 0` the model is isotropic.  The effective
correlation ranges become `ρ_I = ρ₀ exp(a/2)`, `ρ_J = ρ₀ exp(b/2)`,
`ρ_K = ρ₀ exp(-(a+b)/2)`.  Spatially varying anisotropy is supported
through `anisotropy_a_basis` and `anisotropy_b_basis` matrices.
"""
function MaternSPDE3Prior(
        nc::Integer;
        ν::Real = 0.5,
        ρ0::Real = 1.0,
        σ0::Real = 1.0,
        κ_min::Real = 1e-8,
        laplace_from::Symbol = :geometry,
        version::Symbol = :xyz,
        anisotropy_a_basis = nothing,
        anisotropy_b_basis = nothing
    )
    T = Float64
    νf = T(ν)
    isapprox(νf, 0.5; atol = 1e-12) || throw(ArgumentError("This implementation only supports α = 2 in 3D, which requires ν = 0.5 (got $ν)."))
    ρ0f = T(ρ0)
    σ0f = T(σ0)
    κminf = T(κ_min)
    ρ0f > 0 || throw(ArgumentError("ρ0 must be positive, got $ρ0."))
    σ0f > 0 || throw(ArgumentError("σ0 must be positive, got $σ0."))
    κminf > 0 || throw(ArgumentError("κ_min must be positive, got $κ_min."))
    laplace_from in (:geometry, :flow, :unit) || throw(ArgumentError("laplace_from must be one of :geometry, :flow, :unit (got $laplace_from)."))
    version == :xyz || throw(ArgumentError("The 3D Matérn SPDE implementation currently only supports version = :xyz (got $version)."))
    Ba = _matern_basis_matrix(anisotropy_a_basis, nc, T)
    Bb = _matern_basis_matrix(anisotropy_b_basis, nc, T)
    return MaternSPDE3Prior(Int(nc), νf, ρ0f, σ0f, κminf, laplace_from, version, Ba, Bb, nothing)
end

function _normalize_matern_theta_3d(prior::MaternSPDE3Prior, θ)
    θ isa NamedTuple || throw(ArgumentError("θ must be a named tuple with optional fields :anisotropy_a, :anisotropy_b."))
    return (
        anisotropy_a = _theta_block(θ, :anisotropy_a, size(prior.anisotropy_a_basis, 2)),
        anisotropy_b = _theta_block(θ, :anisotropy_b, size(prior.anisotropy_b_basis, 2))
    )
end

function matern_parameter_fields(prior::MaternSPDE3Prior, θ = NamedTuple())
    θn = _normalize_matern_theta_3d(prior, θ)
    ρ = fill(prior.ρ0, prior.nc)
    σ = fill(prior.σ0, prior.nc)
    κ = fill(max(sqrt(8*prior.ν)/prior.ρ0, prior.κ_min), prior.nc)
    τ_nominal = _matern_tau_nominal(prior.ν, κ, σ; d = 3.0)

    a = prior.anisotropy_a_basis * θn.anisotropy_a
    b = prior.anisotropy_b_basis * θn.anisotropy_b
    H = _matern_diagonal_anisotropy_3d(a, b)
    return (
        θ = θn,
        ρ = ρ,
        σ = σ,
        κ = κ,
        τ_nominal = τ_nominal,
        a = a,
        b = b,
        H = H
    )
end

"""
    _matern_diagonal_anisotropy_3d(a, b)

Build diagonal diffusion tensor `H = diag(exp(a), exp(b), exp(-(a+b)))`
with `det(H) = 1`, stored as a `3 × n` matrix.
"""
function _matern_diagonal_anisotropy_3d(a::AbstractVector, b::AbstractVector)
    n = length(a)
    length(b) == n || throw(ArgumentError("a and b must have equal length."))
    H = Matrix{Float64}(undef, 3, n)
    @inbounds for i in 1:n
        H[1, i] = exp(a[i])
        H[2, i] = exp(b[i])
        H[3, i] = exp(-(a[i] + b[i]))
    end
    return H
end

function matern_realized_ranges(domain::DataDomain, prior::MaternSPDE3Prior, θ = NamedTuple(); kwargs...)
    throw(ArgumentError("Directional major/minor range diagnostics are only implemented for MaternSPDE2Prior. Use matern_realized_axis_ranges for MaternSPDE3Prior."))
end

function matern_range_compensation!(prior::MaternSPDE3Prior, domain::DataDomain; kwargs...)
    throw(ArgumentError("Range compensation is not implemented for MaternSPDE3Prior in this first version."))
end

function _validate_matern_domain(domain::DataDomain, prior::MaternSPDE3Prior)
    number_of_cells(domain) == prior.nc || throw(ArgumentError("Prior has nc = $(prior.nc), but domain has $(number_of_cells(domain)) cells."))
    D = size(domain[:cell_centroids], 1)
    D == 3 || throw(ArgumentError("The 3D Matérn SPDE implementation requires a 3D domain, got centroid dimension $D."))
    return nothing
end

function _validate_matern_halo(domain::DataDomain, prior::MaternSPDE3Prior, halo::MaternHaloSpec, compensated::Bool)
    prior.laplace_from == :geometry || throw(ArgumentError("3D halo padding currently requires laplace_from = :geometry, got $(prior.laplace_from)."))
    prior.version == :xyz || throw(ArgumentError("3D halo padding currently requires version = :xyz, got $(prior.version)."))
    if compensated && !isnothing(prior.sd_compensation)
        sdcomp = prior.sd_compensation::MaternMeanSDCompensation
        sdcomp.mode == :mean || throw(ArgumentError("3D halo padding only supports mean SD compensation in this first implementation."))
    end
    haskey(domain, :boundary_centroids) || throw(ArgumentError("Domain is missing :boundary_centroids required for 3D halo padding."))
    haskey(domain, :boundary_areas) || throw(ArgumentError("Domain is missing :boundary_areas required for 3D halo padding."))
    haskey(domain, :boundary_neighbors) || throw(ArgumentError("Domain is missing :boundary_neighbors required for 3D halo padding."))
    haskey(domain, :boundary_normals) || throw(ArgumentError("Domain is missing :boundary_normals required for 3D halo padding."))
    return nothing
end

function _neutralize_matern_flow_controls!(domain::DataDomain)
    if haskey(domain, :net_to_gross)
        domain[:net_to_gross] = ones(Float64, number_of_cells(domain))
    end
    if haskey(domain, :transmissibility_multiplier, Faces())
        domain[:transmissibility_multiplier, Faces()] = ones(Float64, number_of_faces(domain))
    end
    if haskey(domain, :transmissibility_override, Faces())
        domain[:transmissibility_override, Faces()] = fill(NaN, number_of_faces(domain))
    end
    return domain
end

function _matern_diffusion_transmissibility(domain::DataDomain, prior::MaternSPDE3Prior, fields)
    if prior.laplace_from == :geometry
        d2 = deepcopy(domain)
        d2[:permeability] = _embed_diagonal_tensor_3d(fields.H)
        _neutralize_matern_flow_controls!(d2)
        T = reservoir_transmissibility(d2; version = prior.version)
    elseif prior.laplace_from == :flow
        T = reservoir_transmissibility(domain; version = prior.version)
    elseif prior.laplace_from == :unit
        T = ones(Float64, size(domain[:neighbors], 2))
    else
        error("Unsupported laplace_from = $(prior.laplace_from)")
    end

    bad = 0
    for i in eachindex(T)
        if !isfinite(T[i])
            T[i] = 0.0
            bad += 1
        end
    end
    if bad > 0
        jutul_message("Matérn SPDE", "Replaced $bad non-finite face transmissibilities with zero during 3D operator assembly.")
    end
    return T
end

"""
    _embed_diagonal_tensor_3d(H::AbstractMatrix)

Convert a `3 × nc` diagonal tensor to the 3-row permeability format expected
by `reservoir_transmissibility`.
"""
function _embed_diagonal_tensor_3d(H::AbstractMatrix)
    size(H, 1) == 3 || throw(ArgumentError("H must have 3 rows, got $(size(H, 1))."))
    return copy(H)
end

function _matern_extend_fields_3d(fields, source::AbstractVector{<:Integer}, κ_eff::AbstractVector, τ_eff::AbstractVector, logtau_offset::Real)
    H_ext = isempty(source) ? fields.H : hcat(fields.H, fields.H[:, source])
    return (
        θ = fields.θ,
        ρ = vcat(fields.ρ, fields.ρ[source]),
        σ = vcat(fields.σ, fields.σ[source]),
        κ = vcat(fields.κ, fields.κ[source]),
        τ_nominal = vcat(fields.τ_nominal, fields.τ_nominal[source]),
        κ_eff = vcat(κ_eff, κ_eff[source]),
        τ = vcat(τ_eff, τ_eff[source]),
        logtau_offset = fill(Float64(logtau_offset), length(κ_eff) + length(source)),
        a = vcat(fields.a, isempty(source) ? Float64[] : fields.a[source]),
        b = vcat(fields.b, isempty(source) ? Float64[] : fields.b[source]),
        H = H_ext
    )
end

"""
    matern_spde_operator(domain, prior::MaternSPDE3Prior, θ = NamedTuple(); compensated = true, halo = nothing)

Assemble the 3D finite-volume Matérn SPDE operator for `α = 2`:

`Q = Dτ * K * C^-1 * K * Dτ`, with `K = L + Diagonal(volumes .* κ.^2)`.
"""
function matern_spde_operator(domain::DataDomain, prior::MaternSPDE3Prior, θ = NamedTuple(); compensated::Bool = true, halo = nothing)
    _validate_matern_domain(domain, prior)
    if !isnothing(halo)
        halo isa MaternHaloSpec || throw(ArgumentError("halo must be nothing or a MaternHaloSpec, got $(typeof(halo))."))
        _validate_matern_halo(domain, prior, halo, compensated)
    end

    fields = matern_parameter_fields(prior, θ)
    κ_eff = copy(fields.κ)
    T = _matern_diffusion_transmissibility(domain, prior, fields)
    N = domain[:neighbors]
    nc = prior.nc
    volumes = Float64.(domain[:volumes])

    logτ = log.(fields.τ_nominal)
    scalar_offset = 0.0
    if compensated && !isnothing(prior.sd_compensation)
        sdcomp = prior.sd_compensation::MaternMeanSDCompensation
        scalar_offset = Float64(sdcomp.logtau_offset)
        logτ .+= scalar_offset
    end
    τ_eff = exp.(logτ)
    interior_idx = collect(1:nc)

    if isnothing(halo)
        L_H = tpfa_laplacian(N, T, nc)
        volumes_eff = volumes
        κ_all = κ_eff
        τ_all = τ_eff
        halo_idx = Int[]
        halo_meta = nothing
        transmissibilities_ext = T
        fields_out = _matern_extend_fields_3d(fields, Int[], κ_eff, τ_eff, scalar_offset)
    else
        halo_assembly = _build_matern_halo_3d(domain, halo, fields, κ_eff, τ_eff, T, scalar_offset)
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
    _qinv_diagonal_via_K(K, C, τ, idx; chunk_size = 32)

Compute `diag(Q⁻¹)` at `idx` via the K factorization instead of factorizing Q
directly.  Since `Q = Dτ K C⁻¹ K Dτ`, we have
`Q⁻¹[j,j] = ‖√C · K⁻¹(eⱼ/τⱼ)‖²`, requiring only one K solve per cell.

This avoids forming Q explicitly, which can lose positive-definiteness when the
diagonal spans many orders of magnitude.
"""
function _qinv_diagonal_via_K(K::Symmetric, C::Diagonal, τ::AbstractVector, idx::AbstractVector{<:Integer}; chunk_size::Int = 32)
    FK = cholesky(K)
    n = size(K, 1)
    Vvec = C.diag
    vals = zeros(Float64, length(idx))
    for start in 1:chunk_size:length(idx)
        stop = min(start + chunk_size - 1, length(idx))
        block = idx[start:stop]
        E = zeros(Float64, n, length(block))
        for (k, j) in enumerate(block)
            E[j, k] = 1.0 / τ[j]
        end
        U = FK \ E   # U[:,k] = K⁻¹ (e_j / τ_j)
        for (k, _) in enumerate(block)
            s = 0.0
            @inbounds for i in 1:n
                s += Vvec[i] * U[i, k]^2
            end
            vals[start + k - 1] = s
        end
    end
    return vals
end

"""
    _qinv_columns_via_K(K, C, τ, idx; chunk_size = 16)

Compute full columns `Q⁻¹[:, j]` for each `j` in `idx` through K.
Requires two K solves per column: `Q⁻¹ eⱼ = Dτ⁻¹ K⁻¹ C K⁻¹ (eⱼ/τⱼ)`.
"""
function _qinv_columns_via_K(K::Symmetric, C::Diagonal, τ::AbstractVector, idx::AbstractVector{<:Integer}; chunk_size::Int = 16)
    FK = cholesky(K)
    n = size(K, 1)
    Vvec = C.diag
    cols = Matrix{Float64}(undef, n, length(idx))
    for start in 1:chunk_size:length(idx)
        stop = min(start + chunk_size - 1, length(idx))
        block = idx[start:stop]
        # First solve: u = K⁻¹ (e_j / τ_j)
        E = zeros(Float64, n, length(block))
        for (k, j) in enumerate(block)
            E[j, k] = 1.0 / τ[j]
        end
        U = FK \ E
        # Multiply by C: h = V .* u
        for (k, _) in enumerate(block)
            @inbounds for i in 1:n
                U[i, k] *= Vvec[i]
            end
        end
        # Second solve: p = K⁻¹ h, then divide by τ
        P = FK \ U
        for (k, _) in enumerate(block)
            @inbounds for i in 1:n
                P[i, k] /= τ[i]
            end
        end
        cols[:, start:stop] .= P
    end
    return cols
end

function matern_realized_variance(
        domain::DataDomain,
        prior::MaternSPDE3Prior,
        θ = NamedTuple();
        method::Symbol = :diag_qinv,
        anchors = :all,
        compensated::Bool = true,
        halo = nothing
    )
    method == :diag_qinv || throw(ArgumentError("Only method = :diag_qinv is supported, got $method."))
    op = matern_spde_operator(domain, prior, θ; compensated = compensated, halo = halo)
    idx = _resolve_matern_anchors(prior.nc, anchors)
    diagvals = _qinv_diagonal_via_K(op.K, op.C, op.fields.τ, idx)
    return (anchors = idx, realized = diagvals, target = op.fields.σ[idx].^2)
end

"""
    matern_sd_compensation!(prior::MaternSPDE3Prior, domain, θ = NamedTuple(); mode = :mean, diag_method = :chol, anchors = :all, fd_delta = 1e-3, halo = nothing)

Calibrate a cached scalar additive correction to `log(τ)` from mesh-based
realized variance. Only mean compensation is supported in 3D v1.

When using anisotropy, pass the same `θ` that will be used to build the
operator so that compensation is calibrated for the correct Laplacian.
"""
function matern_sd_compensation!(
        prior::MaternSPDE3Prior,
        domain::DataDomain,
        θ = NamedTuple();
        mode::Symbol = :mean,
        diag_method::Symbol = :chol,
        anchors = :all,
        fd_delta::Real = 1e-3,
        halo = nothing
    )
    mode == :mean || throw(ArgumentError("The 3D implementation only supports mode = :mean, got $mode."))
    diag_method == :chol || throw(ArgumentError("Only diag_method = :chol is supported, got $diag_method."))
    fd_delta > 0 || throw(ArgumentError("fd_delta must be positive, got $fd_delta."))
    idx = _resolve_matern_anchors(prior.nc, anchors)
    if !isnothing(halo)
        halo isa MaternHaloSpec || throw(ArgumentError("halo must be nothing or a MaternHaloSpec, got $(typeof(halo))."))
    end

    saved = prior.sd_compensation
    prior.sd_compensation = nothing
    base = matern_realized_variance(domain, prior, θ; anchors = idx, compensated = true, halo = halo)
    target = matern_parameter_fields(prior, θ).σ[idx].^2
    offset = mean(0.5 .* (log.(base.realized) .- log.(target)))
    prior.sd_compensation = MaternMeanSDCompensation(:mean, idx, Float64(offset))
    return prior
end

"""
    matern_realized_axis_ranges(domain, prior::MaternSPDE3Prior, θ = NamedTuple(); anchors = :all, corr_level = practical_range, compensated = true, halo = nothing)

Estimate realized practical ranges along the global x-, y-, and z-axes.
By default, `corr_level` is the correlation implied by the `sqrt(8ν)/ρ`
practical range convention.
"""
function matern_realized_axis_ranges(
        domain::DataDomain,
        prior::MaternSPDE3Prior,
        θ = NamedTuple();
        anchors = :all,
        corr_level::Real = _matern_practical_corr_level(prior.ν),
        compensated::Bool = true,
        halo = nothing
    )
    0 < corr_level < 1 || throw(ArgumentError("corr_level must lie in (0, 1), got $corr_level."))
    op = matern_spde_operator(domain, prior, θ; compensated = compensated, halo = halo)
    idx = _resolve_matern_anchors(prior.nc, anchors)
    if prior.nc <= 256
        vars = _qinv_diagonal_via_K(op.K, op.C, op.fields.τ, collect(1:prior.nc))
    else
        vars = copy(op.fields.σ[1:prior.nc].^2)
        vars[idx] .= _qinv_diagonal_via_K(op.K, op.C, op.fields.τ, idx)
    end
    covcols = _qinv_columns_via_K(op.K, op.C, op.fields.τ, idx)[1:prior.nc, :]
    pts = Matrix{Float64}(domain[:cell_centroids])

    rx = Vector{Float64}(undef, length(idx))
    ry = Vector{Float64}(undef, length(idx))
    rz = Vector{Float64}(undef, length(idx))
    ex = [1.0, 0.0, 0.0]
    ey = [0.0, 1.0, 0.0]
    ez = [0.0, 0.0, 1.0]

    for (k, cell) in enumerate(idx)
        covcol = covcols[:, k]
        corr = covcol ./ sqrt.(max.(vars .* vars[cell], eps(Float64)))
        rx[k] = _average_axis_range_3d(pts, cell, corr, ex, corr_level)
        ry[k] = _average_axis_range_3d(pts, cell, corr, ey, corr_level)
        rz[k] = _average_axis_range_3d(pts, cell, corr, ez, corr_level)
    end

    return (
        anchors = idx,
        realized_x = rx,
        realized_y = ry,
        realized_z = rz,
        target_x = op.fields.ρ[idx] .* exp.(0.5 .* op.fields.a[idx]),
        target_y = op.fields.ρ[idx] .* exp.(0.5 .* op.fields.b[idx]),
        target_z = op.fields.ρ[idx] .* exp.(-0.5 .* (op.fields.a[idx] .+ op.fields.b[idx]))
    )
end

function _boundary_face_nodes_3d(domain::DataDomain)
    g = physical_representation(domain)
    gu = g isa UnstructuredMesh ? g : UnstructuredMesh(g)
    faces_to_nodes = gu.boundary_faces.faces_to_nodes
    pos = faces_to_nodes.pos
    vals = faces_to_nodes.vals
    nf = length(gu.boundary_faces.neighbors)
    out = Vector{Vector{Int}}(undef, nf)
    for f in 1:nf
        out[f] = Int.(vals[pos[f]:(pos[f + 1] - 1)])
    end
    return out
end

function _boundary_surface_adjacency_3d(domain::DataDomain; include_corners::Bool = true)
    g = physical_representation(domain)
    nodes = _boundary_face_nodes_3d(domain)
    gu = g isa UnstructuredMesh ? g : UnstructuredMesh(g)
    pts = gu.node_points
    normals = Matrix{Float64}(domain[:boundary_normals])
    edge_to_faces = Dict{Tuple{Int, Int}, Vector{Int}}()
    edge_length = Dict{Tuple{Int, Int}, Float64}()

    for (f, f_nodes) in enumerate(nodes)
        nfn = length(f_nodes)
        nfn >= 2 || continue
        for k in 1:nfn
            a = f_nodes[k]
            b = f_nodes[k == nfn ? 1 : k + 1]
            a == b && continue
            key = a < b ? (a, b) : (b, a)
            push!(get!(edge_to_faces, key, Int[]), f)
            if !haskey(edge_length, key)
                pa = pts[a]
                pb = pts[b]
                edge_length[key] = norm(pb - pa)
            end
        end
    end

    pair_length = Dict{Tuple{Int, Int}, Float64}()
    for (edge, faces) in edge_to_faces
        length(faces) >= 2 || continue
        for i in 1:(length(faces) - 1), j in (i + 1):length(faces)
            f1 = faces[i]
            f2 = faces[j]
            if !include_corners
                n1 = normals[:, f1] / max(norm(normals[:, f1]), 1e-12)
                n2 = normals[:, f2] / max(norm(normals[:, f2]), 1e-12)
                abs(dot(n1, n2)) > 0.8 || continue
            end
            key = f1 < f2 ? (f1, f2) : (f2, f1)
            pair_length[key] = max(get(pair_length, key, 0.0), edge_length[edge])
        end
    end

    return [(first(pair), last(pair), shared_length) for (pair, shared_length) in pair_length]
end

function _build_matern_halo_3d(domain::DataDomain, halo::MaternHaloSpec, fields, κ_eff, τ_eff, transmissibilities, logtau_offset::Real)
    nc = number_of_cells(domain)
    cc = Matrix{Float64}(domain[:cell_centroids])
    bc = Matrix{Float64}(domain[:boundary_centroids])
    bn = Int.(vec(domain[:boundary_neighbors]))
    ba = Float64.(vec(domain[:boundary_areas]))
    normals = Matrix{Float64}(domain[:boundary_normals])
    nb = length(bn)

    # Skip boundary faces with non-finite centroids or normals (degenerate geometry)
    valid_bnd = Int[]
    for f in 1:nb
        if all(isfinite, bc[:, f]) && all(isfinite, normals[:, f])
            push!(valid_bnd, f)
        end
    end
    nb_valid = length(valid_bnd)
    if nb_valid < nb
        jutul_message("Matérn SPDE", "Skipped $(nb - nb_valid) boundary faces with non-finite centroids/normals during 3D halo construction.")
    end

    boundary_cells = unique(bn[valid_bnd])
    total_padding = _halo_total_padding(halo.total_padding, fields.ρ[boundary_cells])
    widths = _halo_layer_widths(total_padding, halo.layers, halo.growth)

    nh = nb_valid * halo.layers
    halo_centroids = zeros(Float64, 3, nh)
    halo_volumes = zeros(Float64, nh)
    halo_source = zeros(Int, nh)
    face_to_chain = Dict{Int, Vector{Int}}()
    face_normal_weights = Dict{Int, Vector{Float64}}()

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
    for f in valid_bnd
        cell = bn[f]
        out = bc[:, f] .- cc[:, cell]
        if norm(out) <= 1e-10
            out = normals[:, f]
        end
        out_norm = max(norm(out), 1e-12)
        out ./= out_norm
        area = max(ba[f], 1e-12)
        chain = Int[]
        normal_weights = Float64[]
        cumulative = 0.0
        prev_cell = cell
        interior_dist = max(norm(bc[:, f] .- cc[:, cell]), 1e-8)
        left_half = area / interior_dist

        for layer in 1:halo.layers
            cursor += 1
            gid = nc + cursor
            push!(chain, gid)
            halo_source[cursor] = cell
            halo_volumes[cursor] = area * widths[layer]
            halo_centroids[:, cursor] .= bc[:, f] .+ out .* (cumulative + widths[layer]/2)

            right_half = area / max(widths[layer]/2, 1e-8)
            w = 1.0 / (1.0 / max(left_half, 1e-12) + 1.0 / max(right_half, 1e-12))
            push!(extra_left, prev_cell)
            push!(extra_right, gid)
            push!(extra_weight, w)
            push!(normal_weights, w)

            prev_cell = gid
            left_half = right_half
            cumulative += widths[layer]
        end

        face_to_chain[f] = chain
        face_normal_weights[f] = normal_weights
    end

    halo_idx = collect((nc + 1):(nc + nh))
    adjacency = _boundary_surface_adjacency_3d(domain; include_corners = halo.include_corners)
    for (f1, f2, shared_len) in adjacency
        shared_len > 0 || continue
        haskey(face_to_chain, f1) && haskey(face_to_chain, f2) || continue
        chain1 = face_to_chain[f1]
        chain2 = face_to_chain[f2]
        base_dist = max(norm(bc[:, f1] .- bc[:, f2]), 1e-8)
        for layer in 1:halo.layers
            interface_measure = shared_len * widths[layer]
            w = interface_measure / base_dist
            push!(extra_left, chain1[layer])
            push!(extra_right, chain2[layer])
            push!(extra_weight, w)
        end
    end

    volumes_ext = vcat(Float64.(domain[:volumes]), halo_volumes)
    fields_ext = _matern_extend_fields_3d(fields, halo_source, κ_eff, τ_eff, logtau_offset)

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
            boundary_adjacency = adjacency,
            halo_centroids = halo_centroids,
            ext_centroids = hcat(cc, halo_centroids),
            halo_source = halo_source,
            face_normal_weights = face_normal_weights
        )
    )
end

function _range_along_ray_3d(pts::AbstractMatrix, anchor::Integer, corr::AbstractVector, dir::AbstractVector, corr_level::Real)
    x0 = pts[:, anchor]
    dirn = Float64.(dir) ./ max(norm(dir), 1e-12)
    spacing = _local_spacing(pts, anchor)
    tol = 0.55 * spacing + 1e-12

    pairs = Tuple{Float64, Float64}[]
    for j in axes(pts, 2)
        dx = pts[:, j] - x0
        proj = dot(dx, dirn)
        proj <= 0 && continue
        perp = norm(dx .- proj .* dirn)
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
            t = (corr_level - prev_c) / (c - prev_c)
            return prev_d + t * (d - prev_d)
        end
        prev_d = d
        prev_c = c
    end
    return last(pairs)[1]
end

function _average_axis_range_3d(pts::AbstractMatrix, anchor::Integer, corr::AbstractVector, dir::AbstractVector, corr_level::Real)
    rp = _range_along_ray_3d(pts, anchor, corr, dir, corr_level)
    rm = _range_along_ray_3d(pts, anchor, corr, -Float64.(dir), corr_level)
    vals = filter(isfinite, [rp, rm])
    isempty(vals) && return NaN
    return mean(vals)
end
