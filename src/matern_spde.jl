struct MaternHaloSpec{T<:AbstractFloat}
    layers::Int
    growth::T
    total_padding::Any
    extension::Symbol
    include_corners::Bool
end

Base.:(==)(a::MaternHaloSpec, b::MaternHaloSpec) = a.layers == b.layers &&
    a.growth == b.growth &&
    isequal(a.total_padding, b.total_padding) &&
    a.extension == b.extension &&
    a.include_corners == b.include_corners

struct MaternCalibration{T<:AbstractFloat, V<:AbstractVector{T}}
    signature::NamedTuple
    halo::Any
    anchors::Vector{Int}
    anchor_resolution::NTuple{2, Int}
    corr_level::T
    tolerances::NamedTuple
    logtau_correction::V
    logkappa_correction::V
    angle_correction::V
    logratio_correction::V
    diagnostics::NamedTuple
end

struct MaternSPDE2Prior{T<:AbstractFloat, M<:AbstractMatrix{T}}
    nc::Int
    ν::T
    basis::Any
    basis_matrix::M
    range0::T
    sd0::T
    angle0::T
    logratio0::T
    κ_min::T
    laplace_from::Symbol
    version::Symbol
    diffusion_scheme::Symbol
end

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
    extension == :nearest || throw(ArgumentError("Only extension = :nearest is supported, got $extension."))
    if !(total_padding isa Symbol || total_padding isa Real)
        throw(ArgumentError("total_padding must be a Symbol or Real, got $(typeof(total_padding))."))
    end
    return MaternHaloSpec(Int(layers), Float64(growth), total_padding, extension, include_corners)
end

function MaternSPDE2Prior(
        nc::Integer;
        ν::Real = 1.0,
        basis = nothing,
        range0::Real,
        sd0::Real,
        angle0::Real = 0.0,
        logratio0::Real = 0.0,
        κ_min::Real = 1e-8,
        laplace_from::Symbol = :geometry,
        version::Symbol = :xyz,
        diffusion_scheme::Symbol = :tpfa
    )
    T = Float64
    νf = T(ν)
    isapprox(νf, 1.0; atol = 1e-12) || throw(ArgumentError("This implementation only supports α = 2 in 2D, which requires ν = 1.0 (got $ν)."))
    range0f = T(range0)
    sd0f = T(sd0)
    κminf = T(κ_min)
    range0f > 0 || throw(ArgumentError("range0 must be positive, got $range0."))
    sd0f > 0 || throw(ArgumentError("sd0 must be positive, got $sd0."))
    κminf > 0 || throw(ArgumentError("κ_min must be positive, got $κ_min."))
    laplace_from in (:geometry, :flow, :unit) || throw(ArgumentError("laplace_from must be one of :geometry, :flow, :unit (got $laplace_from)."))
    version in (:xyz, :ijk) || throw(ArgumentError("version must be :xyz or :ijk (got $version)."))
    diffusion_scheme in (:tpfa, :fvm9) || throw(ArgumentError("diffusion_scheme must be :tpfa or :fvm9 (got $diffusion_scheme)."))
    B = _matern_basis_matrix(basis, nc, T)
    basis_out = isnothing(basis) ? nothing : (basis isa MaternBasis ? basis : MaternBasis(basis))
    return MaternSPDE2Prior(
        Int(nc),
        νf,
        basis_out,
        B,
        range0f,
        sd0f,
        T(angle0),
        T(logratio0),
        κminf,
        laplace_from,
        version,
        diffusion_scheme
    )
end

function _matern_practical_corr_level(nu::Real)
    nuf = Float64(nu)
    # Correlation at h = sqrt(8nu)/kappa for the supported alpha = 2 cases.
    if isapprox(nuf, 0.5; atol = 1e-12)
        return exp(-2.0)
    elseif isapprox(nuf, 1.0; atol = 1e-12)
        return 0.1396674740152931
    else
        throw(ArgumentError("No practical range correlation level is defined for nu = $nu."))
    end
end

function matern_parameter_fields(prior::MaternSPDE2Prior, θ = NamedTuple())
    θn = _normalize_matern_theta(prior, θ)
    B = prior.basis_matrix

    logρ = log(prior.range0) .+ B*θn.range
    logσ = log(prior.sd0) .+ B*θn.sd
    ρ = exp.(logρ)
    σ = exp.(logσ)

    raw_angle = fill(prior.angle0, prior.nc) .+ B*θn.angle
    raw_logratio = fill(prior.logratio0, prior.nc) .+ B*θn.logratio
    angle, logratio = _canonicalize_orientation_logratio(raw_angle, raw_logratio)
    ratio = exp.(logratio)
    u = logratio .* cos.(2 .* angle)
    v = logratio .* sin.(2 .* angle)

    κ = max.(sqrt(8*prior.ν) ./ ρ, prior.κ_min)
    τ_nominal = _matern_tau_nominal(prior.ν, κ, σ)
    H = _matern_anisotropy_tensor(u, v)
    ρ_major = ρ .* exp.(0.5 .* logratio)
    ρ_minor = ρ .* exp.(-0.5 .* logratio)

    return (
        θ = θn,
        ρ = ρ,
        σ = σ,
        angle = raw_angle,
        orientation = angle,
        raw_logratio = raw_logratio,
        logratio = logratio,
        ratio = ratio,
        u = u,
        v = v,
        r = logratio,
        H = H,
        κ = κ,
        τ_nominal = τ_nominal,
        ρ_major = ρ_major,
        ρ_minor = ρ_minor
    )
end

function matern_spde_operator(layer_or_domain, prior::MaternSPDE2Prior, θ = NamedTuple(); calibration = nothing, strict::Bool = false, halo = nothing)
    domain, layer = _coerce_matern_domain(layer_or_domain)
    _validate_matern_domain(domain, prior, layer)

    if isnothing(calibration)
        strict && throw(ArgumentError("No MaternCalibration was provided. Run `matern_calibrate` first or call with `strict = false` to accept an uncalibrated fallback."))
        @warn "No MaternCalibration provided; building an uncalibrated Matérn operator."
        corrections = _zero_matern_corrections(prior.nc)
        return _matern_operator_internal(domain, layer, prior, θ, corrections; halo = halo, calibration_status = :missing_fallback)
    else
        _validate_matern_calibration(calibration, layer_or_domain, prior, θ; halo = halo)
        corrections = (
            logtau = calibration.logtau_correction,
            logkappa = calibration.logkappa_correction,
            angle = calibration.angle_correction,
            logratio = calibration.logratio_correction
        )
        return _matern_operator_internal(domain, layer, prior, θ, corrections; halo = calibration.halo, calibration_status = :applied)
    end
end

function matern_realized_variance(
        layer_or_domain,
        prior::MaternSPDE2Prior,
        θ = NamedTuple();
        method::Symbol = :diag_qinv,
        anchors = :all,
        calibration = nothing,
        strict::Bool = false,
        halo = nothing
    )
    method == :diag_qinv || throw(ArgumentError("Only method = :diag_qinv is supported, got $method."))
    op = matern_spde_operator(layer_or_domain, prior, θ; calibration = calibration, strict = strict, halo = halo)
    idx = _resolve_matern_anchors(prior.nc, anchors)
    diagvals = _selected_qinv_diagonal(op.Q, idx)
    return (
        anchors = idx,
        realized = diagvals,
        target = op.fields.σ[idx].^2,
        calibration_status = op.calibration_status
    )
end

function matern_realized_ranges(
        layer_or_domain,
        prior::MaternSPDE2Prior,
        θ = NamedTuple();
        anchors = :all,
        directions::Symbol = :local_eigen,
        corr_level::Real = _matern_practical_corr_level(prior.ν),
        calibration = nothing,
        strict::Bool = false,
        halo = nothing
    )
    directions == :local_eigen || throw(ArgumentError("Only directions = :local_eigen is supported, got $directions."))
    0 < corr_level < 1 || throw(ArgumentError("corr_level must lie in (0, 1), got $corr_level."))
    domain, _ = _coerce_matern_domain(layer_or_domain)
    op = matern_spde_operator(layer_or_domain, prior, θ; calibration = calibration, strict = strict, halo = halo)
    idx = _resolve_matern_anchors(prior.nc, anchors)
    ranges = _matern_realized_ranges_from_operator(domain, op, idx; corr_level = corr_level)
    return merge(
        ranges,
        (
            anchors = idx,
            target_major = op.fields.ρ_major[idx],
            target_minor = op.fields.ρ_minor[idx],
            target_angle = op.fields.orientation[idx],
            target_ratio = op.fields.ratio[idx],
            calibration_status = op.calibration_status
        )
    )
end

function matern_precision_from_data_file(
        data_path::AbstractString;
        layer::Integer = 1,
        target_variance,
        target_range,
        target_rotation = nothing,
        target_anisotropy = nothing,
        angle_unit::Symbol = :degree,
        diffusion_scheme::Symbol = :auto,
        calibrate::Bool = true,
        anchor_resolution::Tuple{<:Integer, <:Integer} = (2, 2),
        tolerances = _default_matern_tolerances(),
        maxiter::Integer = 10
    )
    op = _matern_precision_operator_from_data_file(
        data_path;
        layer = layer,
        target_variance = target_variance,
        target_range = target_range,
        target_rotation = target_rotation,
        target_anisotropy = target_anisotropy,
        angle_unit = angle_unit,
        diffusion_scheme = diffusion_scheme,
        calibrate = calibrate,
        anchor_resolution = anchor_resolution,
        tolerances = tolerances,
        maxiter = maxiter
    )
    return sparse(op.Q)
end

function matern_precision_csc_from_data_file(data_path::AbstractString; nugget::Union{Nothing, Real} = nothing, kwargs...)
    op, layer_domain = _matern_precision_operator_from_data_file(data_path; kwargs...)
    Q = sparse(op.Q)
    if !isnothing(nugget)
        nug = Float64(nugget)
    else
        nug = 1e-8 * mean(diag(Q))
    end
    if nug > 0
        Q = Q + nug * sparse(I, size(Q)...)
    end
    domain_mask = falses(size(Q, 1))
    domain_mask[op.interior_idx] .= true
    ni, nj = layer_domain.parent_shape
    grid_mask = zeros(Int8, ni * nj)
    grid_mask[layer_domain.active_parent_linear] .= Int8(1)
    nhalo = length(op.halo_idx)
    if nhalo > 0
        grid_mask_halo_count = nhalo
    else
        grid_mask_halo_count = 0
    end
    return (
        shape = size(Q),
        colptr = copy(Q.colptr),
        rowval = copy(Q.rowval),
        nzval = copy(Q.nzval),
        domain_mask = domain_mask,
        interior_idx = copy(op.interior_idx),
        halo_idx = copy(op.halo_idx),
        grid_mask = grid_mask,
        grid_ni = ni,
        grid_nj = nj,
        parent_centroids = copy(layer_domain.parent_centroids),
        n_halo = grid_mask_halo_count
    )
end

function matern_calibrate(
        layer_or_domain,
        prior::MaternSPDE2Prior,
        θ = NamedTuple();
        halo = :auto,
        anchors = :auto,
        anchor_resolution::Tuple{<:Integer, <:Integer} = (2, 2),
        corr_level::Real = _matern_practical_corr_level(prior.ν),
        tolerances = _default_matern_tolerances(),
        maxiter::Integer = 100
    )
    domain, layer = _coerce_matern_domain(layer_or_domain)
    _validate_matern_domain(domain, prior, layer)
    halo_eff = _resolve_matern_calibration_halo(prior, halo)
    !isnothing(halo_eff) && _validate_matern_halo(domain, prior, halo_eff)
    fields = matern_parameter_fields(prior, θ)
    stationary = _matern_fields_stationary(fields)
    actual_anchor_resolution = anchors === :auto ? (Int(anchor_resolution[1]), Int(anchor_resolution[2])) : (Int(anchor_resolution[1]), Int(anchor_resolution[2]))
    idx = if anchors === :auto
        stationary ? _matern_central_anchor(layer_or_domain, fields.ρ_major) : _matern_auto_anchors(layer_or_domain, fields.ρ_major; resolution = actual_anchor_resolution)
    else
        _resolve_matern_anchors(prior.nc, anchors)
    end
    isempty(idx) && throw(ArgumentError("Calibration anchor set is empty."))

    controls = zeros(Float64, 4, length(idx))
    last_diag = nothing
    for iter in 1:maxiter
        cellcorr = _matern_anchor_controls_to_cells(domain, idx, controls)
        diag = _matern_anchor_diagnostics(domain, layer, prior, θ, idx, cellcorr; halo = halo_eff, corr_level = Float64(corr_level))
        last_diag = diag
        if _matern_calibration_converged(diag, tolerances)
            sig = _matern_calibration_signature(layer_or_domain, prior, θ, halo_eff)
            return MaternCalibration(
                sig,
                halo_eff,
                idx,
                actual_anchor_resolution,
                Float64(corr_level),
                tolerances,
                cellcorr.logtau,
                cellcorr.logkappa,
                cellcorr.angle,
                cellcorr.logratio,
                merge(diag.summary, (iterations = iter,))
            )
        end

        active_controls = _matern_active_calibration_controls(prior, diag, tolerances)
        J = _matern_fd_jacobian(domain, layer, prior, θ, idx, controls, diag.residual, active_controls; halo = halo_eff, corr_level = Float64(corr_level))
        normal = J' * J + 1e-6I
        rhs = J' * diag.residual
        step = -(normal \ rhs)
        controls = _matern_line_search_controls(domain, layer, prior, θ, idx, controls, step, diag; halo = halo_eff, corr_level = Float64(corr_level))
    end

    throw(ArgumentError("matern_calibrate failed to converge within $maxiter iterations. Last diagnostics = $(last_diag === nothing ? :none : last_diag.summary)."))
end

function matern_sd_compensation!(prior::MaternSPDE2Prior, layer_or_domain, θ = NamedTuple(); kwargs...)
    throw(ArgumentError("`matern_sd_compensation!` is not supported for `MaternSPDE2Prior`. Use `matern_calibrate(layer_or_domain, prior, θ; ...)` and pass the returned calibration via `calibration = cal`."))
end

function matern_range_compensation!(prior::MaternSPDE2Prior, layer_or_domain, θ = NamedTuple(); kwargs...)
    throw(ArgumentError("`matern_range_compensation!` is not supported for `MaternSPDE2Prior`. Use `matern_calibrate(layer_or_domain, prior, θ; ...)` and pass the returned calibration via `calibration = cal`."))
end

function _matern_operator_internal(domain::DataDomain, layer, prior::MaternSPDE2Prior, θ, corrections; halo = nothing, calibration_status::Symbol = :applied)
    fields = matern_parameter_fields(prior, θ)
    angle_eff, logratio_eff = _canonicalize_orientation_logratio(fields.orientation .+ corrections.angle, fields.logratio .+ corrections.logratio)
    ratio_eff = exp.(logratio_eff)
    u_eff = logratio_eff .* cos.(2 .* angle_eff)
    v_eff = logratio_eff .* sin.(2 .* angle_eff)
    H_eff = _matern_anisotropy_tensor(u_eff, v_eff)
    κ_eff = max.(fields.κ .* exp.(corrections.logkappa), prior.κ_min)
    τ_eff = exp.(log.(fields.τ_nominal) .+ corrections.logtau)

    _validate_matern_diffusion_scheme(prior, layer, H_eff, halo)
    N = domain[:neighbors]
    nc = prior.nc
    volumes = Float64.(domain[:volumes])
    interior_idx = collect(1:nc)

    if prior.diffusion_scheme == :fvm9
        fvm = _matern_filter_matrix_fvm9(layer, κ_eff, H_eff)
        L_H = fvm.L_H
        volumes_eff = volumes
        κ_all = κ_eff
        τ_all = τ_eff
        halo_idx = Int[]
        halo_meta = fvm.meta
        transmissibilities_ext = Float64[]
        fields_out = _matern_extend_fields(
            fields,
            Int[];
            κ_eff = κ_eff,
            τ_eff = τ_eff,
            angle_eff = angle_eff,
            logratio_eff = logratio_eff,
            ratio_eff = ratio_eff,
            u_eff = u_eff,
            v_eff = v_eff,
            H_eff = H_eff,
            logtau_correction = corrections.logtau,
            logkappa_correction = corrections.logkappa,
            angle_correction = corrections.angle,
            logratio_correction = corrections.logratio
        )
        Kmat = fvm.A
        K = Kmat
        C = Diagonal(volumes_eff)
        Dτ = Diagonal(τ_all)
        Cinv = Diagonal(1.0 ./ volumes_eff)
        Q = Symmetric(sparse(Dτ*(Kmat'*Cinv*Kmat)*Dτ))
        return (
            L_H = L_H,
            C = C,
            K = K,
            Q = Q,
            transmissibilities = Float64[],
            transmissibilities_ext = transmissibilities_ext,
            interior_idx = interior_idx,
            halo_idx = halo_idx,
            halo_meta = halo_meta,
            fields = fields_out,
            calibration_status = calibration_status
        )
    end

    T = _matern_diffusion_transmissibility(domain, prior, H_eff)
    if isnothing(halo)
        L_H = tpfa_laplacian(N, T, nc)
        volumes_eff = volumes
        κ_all = κ_eff
        τ_all = τ_eff
        halo_idx = Int[]
        halo_meta = nothing
        transmissibilities_ext = T
        fields_out = _matern_extend_fields(
            fields,
            Int[];
            κ_eff = κ_eff,
            τ_eff = τ_eff,
            angle_eff = angle_eff,
            logratio_eff = logratio_eff,
            ratio_eff = ratio_eff,
            u_eff = u_eff,
            v_eff = v_eff,
            H_eff = H_eff,
            logtau_correction = corrections.logtau,
            logkappa_correction = corrections.logkappa,
            angle_correction = corrections.angle,
            logratio_correction = corrections.logratio
        )
    else
        halo_assembly = _build_matern_halo(domain, halo, fields, κ_eff, τ_eff, angle_eff, logratio_eff, ratio_eff, u_eff, v_eff, H_eff, corrections, T)
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
        fields = fields_out,
        calibration_status = calibration_status
    )
end

function _matern_realized_ranges_from_operator(domain::DataDomain, op, idx::AbstractVector{<:Integer}; corr_level::Real, factor = nothing, anchor_vars = nothing)
    nc = op.interior_idx[end]
    F = isnothing(factor) ? _factor_precision_matrix(op.Q) : factor
    n_total = size(op.Q, 1)
    if nc <= 256
        vars = _selected_qinv_diagonal_from_factor(F, n_total, collect(1:nc))
    else
        vars = copy(op.fields.σ[1:nc].^2)
        vars[idx] .= isnothing(anchor_vars) ? _selected_qinv_diagonal_from_factor(F, n_total, idx) : Float64.(anchor_vars)
    end
    covcols = _selected_qinv_columns_from_factor(F, n_total, idx)[1:nc, :]
    pts = _matern_points(domain)

    major = Vector{Float64}(undef, length(idx))
    minor = Vector{Float64}(undef, length(idx))
    dir_major = Matrix{Float64}(undef, 2, length(idx))
    dir_minor = Matrix{Float64}(undef, 2, length(idx))
    realized_angle = Vector{Float64}(undef, length(idx))

    for (k, cell) in enumerate(idx)
        Hsource = hasproperty(op.fields, :H_eff) ? op.fields.H_eff : op.fields.H
        Hk = _compact_tensor_to_matrix(Hsource[:, cell])
        e_major, e_minor = _principal_directions(Hk)
        dir_major[:, k] .= e_major
        dir_minor[:, k] .= e_minor
        covcol = covcols[:, k]
        corr = covcol ./ sqrt.(max.(vars .* vars[cell], eps(Float64)))
        major[k] = _average_directional_range(pts, cell, corr, e_major, corr_level)
        minor[k] = _average_directional_range(pts, cell, corr, e_minor, corr_level)
        realized_angle[k] = _realized_covariance_orientation(pts, cell, corr)
    end

    return (
        realized_major = major,
        realized_minor = minor,
        direction_major = dir_major,
        direction_minor = dir_minor,
        realized_angle = realized_angle
    )
end

function _matern_anchor_diagnostics(domain::DataDomain, layer, prior::MaternSPDE2Prior, θ, anchors::Vector{Int}, corrections; halo, corr_level::Float64)
    op = _matern_operator_internal(domain, layer, prior, θ, corrections; halo = halo, calibration_status = :applied)
    F = _factor_precision_matrix(op.Q)
    vars = _selected_qinv_diagonal_from_factor(F, size(op.Q, 1), anchors)
    target_var = op.fields.σ[anchors].^2
    ranges = _matern_realized_ranges_from_operator(domain, op, anchors; corr_level = corr_level, factor = F, anchor_vars = vars)
    target_geom = sqrt.(op.fields.ρ_major[anchors] .* op.fields.ρ_minor[anchors])
    realized_geom = sqrt.(ranges.realized_major .* ranges.realized_minor)
    target_ratio = op.fields.ratio[anchors]
    realized_ratio = ranges.realized_major ./ ranges.realized_minor
    target_angle = op.fields.orientation[anchors]
    angle_active = (target_ratio .>= 1.05) .& (prior.diffusion_scheme == :fvm9)
    angle_residual = zeros(Float64, length(anchors))
    angle_error = zeros(Float64, length(anchors))
    for i in eachindex(anchors)
        if angle_active[i]
            δ = _orientation_difference(ranges.realized_angle[i], target_angle[i])
            angle_residual[i] = δ
            angle_error[i] = abs(δ)
        end
    end

    residual = vcat(
        log.(vars ./ target_var),
        log.(realized_geom ./ target_geom),
        log.(realized_ratio ./ target_ratio),
        angle_residual
    )
    all(isfinite, residual) || throw(ArgumentError("Non-finite calibration residual encountered while calibrating Matérn prior."))

    summary = (
        variance_error = abs.(vars ./ target_var .- 1.0),
        geometric_range_error = abs.(realized_geom ./ target_geom .- 1.0),
        axis_ratio_error = abs.(realized_ratio ./ target_ratio .- 1.0),
        angle_error = angle_error,
        angle_active = angle_active
    )
    return (
        op = op,
        residual = residual,
        summary = merge(summary, (
            max_variance_error = maximum(summary.variance_error),
            max_geometric_range_error = maximum(summary.geometric_range_error),
            max_axis_ratio_error = maximum(summary.axis_ratio_error),
            max_angle_error = any(angle_active) ? maximum(angle_error[angle_active]) : 0.0
        ))
    )
end

function _matern_fd_jacobian(domain::DataDomain, layer, prior::MaternSPDE2Prior, θ, anchors::Vector{Int}, controls::AbstractMatrix, residual0::AbstractVector, active_controls::AbstractMatrix{Bool}; halo, corr_level::Float64)
    n = length(anchors)
    J = zeros(Float64, length(residual0), 4*n)

    # logtau only rescales Q as Dτ*A*Dτ, so correlations/ranges are invariant
    # and log variance residuals have the exact derivative -2 at anchor cells.
    for j in 1:n
        active_controls[1, j] && (J[j, j] = -2.0)
    end

    deltas = (0.05, 0.05, π/180, 0.05)
    tasks = Tuple{Int, Int}[]
    for block in 2:4
        for j in 1:n
            active_controls[block, j] && push!(tasks, (block, j))
        end
    end

    function fill_fd_column!(block, j)
        δ = deltas[block]
        pert = copy(controls)
        pert[block, j] += δ
        cellcorr = _matern_anchor_controls_to_cells(domain, anchors, pert)
        residual = _matern_anchor_diagnostics(domain, layer, prior, θ, anchors, cellcorr; halo = halo, corr_level = corr_level).residual
        J[:, (block - 1)*n + j] .= (residual .- residual0) ./ δ
    end

    if Threads.nthreads() > 1 && length(tasks) > 1
        Threads.@threads for k in eachindex(tasks)
            block, j = tasks[k]
            fill_fd_column!(block, j)
        end
    else
        for (block, j) in tasks
            fill_fd_column!(block, j)
        end
    end
    return J
end

function _apply_matern_control_step!(controls::AbstractMatrix, step::AbstractVector; damping::Real = 0.75)
    n = size(controls, 2)
    damp = Float64(damping)
    for j in 1:n
        controls[1, j] += clamp(damp * step[j], -0.75, 0.75)
        controls[2, j] += clamp(damp * step[n + j], -0.75, 0.75)
        controls[3, j] += clamp(damp * step[2*n + j], -10π/180, 10π/180)
        controls[4, j] += clamp(damp * step[3*n + j], -0.75, 0.75)
    end
    return controls
end

function _matern_active_calibration_controls(prior::MaternSPDE2Prior, diag, tolerances)
    n = length(diag.summary.variance_error)
    active = falses(4, n)
    active[1, :] .= diag.summary.variance_error .> tolerances.variance
    active[2, :] .= diag.summary.geometric_range_error .> tolerances.geometric_range
    active[4, :] .= diag.summary.axis_ratio_error .> tolerances.axis_ratio
    if prior.diffusion_scheme == :fvm9 && any(diag.summary.angle_active)
        active[3, :] .= diag.summary.angle_active .& (diag.summary.angle_error .> tolerances.angle)
    end
    return active
end

function _matern_line_search_controls(domain::DataDomain, layer, prior::MaternSPDE2Prior, θ, anchors::Vector{Int}, controls::AbstractMatrix, step::AbstractVector, diag0; halo, corr_level::Float64)
    base = norm(diag0.residual)
    best_controls = copy(controls)
    best_norm = base
    for scale in (0.75, 0.375, 0.1875, 0.09375)
        candidate = copy(controls)
        _apply_matern_control_step!(candidate, step; damping = scale)
        cellcorr = _matern_anchor_controls_to_cells(domain, anchors, candidate)
        diag = _matern_anchor_diagnostics(domain, layer, prior, θ, anchors, cellcorr; halo = halo, corr_level = corr_level)
        nrm = norm(diag.residual)
        if isfinite(nrm) && nrm < best_norm
            best_norm = nrm
            best_controls = candidate
        end
    end
    if best_norm < base
        return best_controls
    else
        candidate = copy(controls)
        _apply_matern_control_step!(candidate, step; damping = 0.046875)
        return candidate
    end
end

function _matern_calibration_converged(diag, tolerances)
    angle_ok = true
    if any(diag.summary.angle_active)
        angle_ok = all(diag.summary.angle_error[diag.summary.angle_active] .<= tolerances.angle)
    end
    return all(diag.summary.variance_error .<= tolerances.variance) &&
        all(diag.summary.geometric_range_error .<= tolerances.geometric_range) &&
        all(diag.summary.axis_ratio_error .<= tolerances.axis_ratio) &&
        angle_ok
end

_default_matern_tolerances() = (variance = 0.05, geometric_range = 0.05, axis_ratio = 0.05, angle = 2π/180)

function _resolve_matern_calibration_halo(prior::MaternSPDE2Prior, halo)
    if halo === :auto
        return prior.diffusion_scheme == :fvm9 ? nothing : MaternHaloSpec(layers = 3, growth = 1.5, total_padding = :boundary_range)
    elseif isnothing(halo) || halo isa MaternHaloSpec
        return halo
    else
        throw(ArgumentError("halo must be :auto, nothing, or a MaternHaloSpec (got $(typeof(halo)))."))
    end
end

_matern_halo_signature(::Nothing) = nothing
_matern_halo_signature(halo::MaternHaloSpec) = (
    layers = halo.layers,
    growth = halo.growth,
    total_padding = halo.total_padding,
    extension = halo.extension,
    include_corners = halo.include_corners
)

function _matern_calibration_signature(layer_or_domain, prior::MaternSPDE2Prior, θ, halo)
    domain, layer = _coerce_matern_domain(layer_or_domain)
    active = if isnothing(layer)
        collect(1:number_of_cells(domain))
    else
        layer.active_parent_linear
    end
    return (
        nc = prior.nc,
        layer = isnothing(layer) ? 1 : layer.layer,
        parent_shape = isnothing(layer) ? (number_of_cells(domain), 1) : layer.parent_shape,
        diffusion_scheme = prior.diffusion_scheme,
        active_hash = _content_hash(active),
        centroid_hash = _content_hash(domain[:cell_centroids]),
        volume_hash = _content_hash(domain[:volumes]),
        halo = _matern_halo_signature(halo),
        theta_hash = _content_hash(_flatten_theta(_normalize_matern_theta(prior, θ)))
    )
end

function _validate_matern_calibration(cal::MaternCalibration, layer_or_domain, prior::MaternSPDE2Prior, θ; halo = nothing)
    sig = _matern_calibration_signature(layer_or_domain, prior, θ, cal.halo)
    sig == cal.signature || throw(ArgumentError("The supplied MaternCalibration does not match this domain/layer/halo/θ snapshot."))
    if !isnothing(halo)
        halo == cal.halo || throw(ArgumentError("The supplied halo keyword does not match the halo stored in the MaternCalibration."))
    end
    return nothing
end

function _matern_constant_field(x; rtol::Real = 1e-10, atol::Real = 1e-12)
    vals = Float64.(x)
    isempty(vals) && return true
    ref = first(vals)
    scale = max(maximum(abs, vals), abs(ref), 1.0)
    return maximum(abs.(vals .- ref)) <= Float64(atol) + Float64(rtol)*scale
end

function _matern_constant_orientation(angle; atol::Real = 1e-10)
    vals = Float64.(angle)
    isempty(vals) && return true
    ref = first(vals)
    return maximum(abs.(_orientation_difference.(vals, ref))) <= Float64(atol)
end

function _matern_fields_stationary(fields)
    angle_relevant = maximum(fields.ratio) >= 1.0 + 1e-10
    return _matern_constant_field(fields.σ) &&
        _matern_constant_field(fields.ρ_major) &&
        _matern_constant_field(fields.ρ_minor) &&
        _matern_constant_field(fields.ratio) &&
        (!angle_relevant || _matern_constant_orientation(fields.orientation))
end

function _matern_central_anchor(layer_or_domain, target_major::AbstractVector)
    domain, _ = _coerce_matern_domain(layer_or_domain)
    length(target_major) == number_of_cells(domain) || throw(ArgumentError("target_major must have one value per cell."))
    mask = _matern_interior_mask(layer_or_domain, target_major)
    candidates = findall(mask)
    isempty(candidates) && (candidates = collect(1:number_of_cells(domain)))
    pts = _matern_points(domain)
    center = vec(mean(pts[:, candidates], dims = 2))
    best = argmin([sum(abs2, pts[:, c] .- center) for c in candidates])
    return [candidates[best]]
end

function _matern_auto_anchors(layer_or_domain, target_major::AbstractVector; resolution::NTuple{2, Int})
    domain, layer = _coerce_matern_domain(layer_or_domain)
    mask = _matern_interior_mask(layer_or_domain, target_major)
    candidates = findall(mask)
    isempty(candidates) && (candidates = collect(1:number_of_cells(domain)))
    if isnothing(layer)
        pts = _matern_points(domain)
        nx_pts = min(resolution[1], max(length(candidates), 1))
        ny_pts = min(resolution[2], max(length(candidates), 1))
        xlo, xhi = extrema(pts[1, candidates])
        ylo, yhi = extrema(pts[2, candidates])
        xs = nx_pts == 1 ? [0.5*(xlo + xhi)] : range(xlo, xhi, length = nx_pts)
        ys = ny_pts == 1 ? [0.5*(ylo + yhi)] : range(ylo, yhi, length = ny_pts)
        anchors = Int[]
        for x in xs, y in ys
            best = argmin([(pts[1, c] - x)^2 + (pts[2, c] - y)^2 for c in candidates])
            push!(anchors, candidates[best])
        end
        return unique(anchors)
    else
        nx, ny = layer.parent_shape
        ii = Int[]
        jj = Int[]
        candidate_set = Set(candidates)
        for (cell_ix, lin) in enumerate(layer.active_parent_linear)
            cell_ix in candidate_set || continue
            i, j = mod1(lin, nx), fld(lin - 1, nx) + 1
            push!(ii, i)
            push!(jj, j)
        end
        xs = unique(sort(ii))
        ys = unique(sort(jj))
        nix = min(resolution[1], length(xs))
        njy = min(resolution[2], length(ys))
        ix = nix == 1 ? [cld(length(xs), 2)] : unique(round.(Int, collect(range(1, length(xs), length = nix))))
        jy = njy == 1 ? [cld(length(ys), 2)] : unique(round.(Int, collect(range(1, length(ys), length = njy))))
        target_pairs = [(xs[i], ys[j]) for i in ix for j in jy]
        anchors = Int[]
        for (it, jt) in target_pairs
            d2 = map(candidates) do c
                lin = layer.active_parent_linear[c]
                i, j = mod1(lin, nx), fld(lin - 1, nx) + 1
                (i - it)^2 + (j - jt)^2
            end
            push!(anchors, candidates[argmin(d2)])
        end
        return unique(anchors)
    end
end

function _matern_interior_mask(layer_or_domain, target_major::AbstractVector)
    domain, _ = _coerce_matern_domain(layer_or_domain)
    dist = _matern_boundary_distance(layer_or_domain)
    length(dist) == number_of_cells(domain) || throw(ArgumentError("Boundary-distance length does not match cell count."))
    return dist .>= target_major
end

function _matern_boundary_distance(domain::DataDomain)
    cc = Matrix{Float64}(domain[:cell_centroids][1:2, :])
    bc = Matrix{Float64}(domain[:boundary_centroids][1:2, :])
    nc = size(cc, 2)
    out = fill(Inf, nc)
    for c in 1:nc
        x = cc[:, c]
        dmin = Inf
        for f in axes(bc, 2)
            dmin = min(dmin, norm(bc[:, f] .- x))
        end
        out[c] = dmin
    end
    return out
end

function _matern_boundary_distance(layer::ReservoirLayerDomain)
    nx, ny = layer.parent_shape
    dx, dy = _matern_parent_spacing(layer)
    out = zeros(Float64, number_of_cells(layer.domain))
    for (cell, lin) in enumerate(layer.active_parent_linear)
        i, j = mod1(lin, nx), fld(lin - 1, nx) + 1
        west = 0
        ii = i
        while ii >= 1 && layer.active_mask[(j - 1)*nx + ii]
            west += 1
            ii -= 1
        end
        east = 0
        ii = i
        while ii <= nx && layer.active_mask[(j - 1)*nx + ii]
            east += 1
            ii += 1
        end
        south = 0
        jj = j
        while jj >= 1 && layer.active_mask[(jj - 1)*nx + i]
            south += 1
            jj -= 1
        end
        north = 0
        jj = j
        while jj <= ny && layer.active_mask[(jj - 1)*nx + i]
            north += 1
            jj += 1
        end
        out[cell] = min((west - 0.5)*dx, (east - 0.5)*dx, (south - 0.5)*dy, (north - 0.5)*dy)
    end
    return out
end

function _matern_anchor_controls_to_cells(domain::DataDomain, anchors::Vector{Int}, controls::AbstractMatrix)
    return (
        logtau = _anchor_field_to_cells(domain, anchors, view(controls, 1, :)),
        logkappa = _anchor_field_to_cells(domain, anchors, view(controls, 2, :)),
        angle = _anchor_field_to_cells(domain, anchors, view(controls, 3, :)),
        logratio = _anchor_field_to_cells(domain, anchors, view(controls, 4, :))
    )
end

_zero_matern_corrections(nc::Integer) = (
    logtau = zeros(Float64, nc),
    logkappa = zeros(Float64, nc),
    angle = zeros(Float64, nc),
    logratio = zeros(Float64, nc)
)

function _matern_precision_operator_from_data_file(
        data_path::AbstractString;
        layer::Integer = 1,
        target_variance,
        target_range,
        target_rotation = nothing,
        target_anisotropy = nothing,
        angle_unit::Symbol = :degree,
        diffusion_scheme::Symbol = :auto,
        calibrate::Bool = true,
        anchor_resolution::Tuple{<:Integer, <:Integer} = (2, 2),
        tolerances = _default_matern_tolerances(),
        maxiter::Integer = 100
    )
    setup = _matern_precision_api_setup(
        data_path;
        layer = layer,
        target_variance = target_variance,
        target_range = target_range,
        target_rotation = target_rotation,
        target_anisotropy = target_anisotropy,
        angle_unit = angle_unit,
        diffusion_scheme = diffusion_scheme
    )
    if calibrate
        cal = matern_calibrate(
            setup.layer,
            setup.prior,
            setup.θ;
            anchor_resolution = anchor_resolution,
            tolerances = tolerances,
            maxiter = maxiter
        )
        return matern_spde_operator(setup.layer, setup.prior, setup.θ; calibration = cal, strict = true), setup.layer
    else
        return matern_spde_operator(setup.layer, setup.prior, setup.θ; strict = false), setup.layer
    end
end

function _matern_precision_api_setup(
        data_path::AbstractString;
        layer::Integer,
        target_variance,
        target_range,
        target_rotation,
        target_anisotropy,
        angle_unit::Symbol,
        diffusion_scheme::Symbol
    )
    endswith(lowercase(data_path), ".data") || throw(ArgumentError("Expected `data_path` to point to a .DATA file, got $data_path."))
    angle_unit in (:degree, :radian) || throw(ArgumentError("angle_unit must be :degree or :radian, got $angle_unit."))

    layer_domain = extract_reservoir_layer(data_path; layer = layer)
    nc = number_of_cells(layer_domain.domain)
    variance_vals, variance_is_vector = _matern_api_cell_field(target_variance, nc; name = :target_variance, lower = 0.0, lower_open = true)
    range_vals, range_is_vector = _matern_api_cell_field(target_range, nc; name = :target_range, lower = 0.0, lower_open = true)
    rotation_vals, rotation_is_vector = _matern_api_cell_field(isnothing(target_rotation) ? 0.0 : target_rotation, nc; name = :target_rotation)
    ratio_vals, ratio_is_vector = _matern_api_cell_field(isnothing(target_anisotropy) ? 1.0 : target_anisotropy, nc; name = :target_anisotropy, lower = 1.0)
    angle_vals = _convert_angle.(rotation_vals, Ref(angle_unit))
    sd_vals = sqrt.(variance_vals)
    nonstationary = variance_is_vector || range_is_vector || rotation_is_vector || ratio_is_vector
    scheme = _matern_api_diffusion_scheme(diffusion_scheme, angle_vals, ratio_vals)

    if nonstationary
        basis = MaternBasis(
            spdiagm(0 => ones(Float64, nc));
            names = [Symbol("cell_", i) for i in 1:nc],
            provenance = "python-api/cellwise"
        )
        fit = fit_matern_fields(
            layer_domain,
            basis;
            range0 = exp(mean(log.(range_vals))),
            sd0 = exp(mean(log.(sd_vals))),
            angle0 = _matern_api_axial_mean(angle_vals),
            logratio0 = mean(log.(ratio_vals)),
            range = range_vals,
            sd = sd_vals,
            angle = angle_vals,
            ratio = ratio_vals,
            angle_unit = :radian
        )
        prior = MaternSPDE2Prior(
            nc;
            basis = fit.basis,
            range0 = fit.baselines.range0,
            sd0 = fit.baselines.sd0,
            angle0 = fit.baselines.angle0,
            logratio0 = fit.baselines.logratio0,
            diffusion_scheme = scheme
        )
        θ = fit.θ
        mode = :nonstationary
    else
        prior = MaternSPDE2Prior(
            nc;
            range0 = range_vals[1],
            sd0 = sd_vals[1],
            angle0 = angle_vals[1],
            logratio0 = log(ratio_vals[1]),
            diffusion_scheme = scheme
        )
        θ = NamedTuple()
        mode = :stationary
    end
    return (layer = layer_domain, prior = prior, θ = θ, mode = mode, diffusion_scheme = scheme)
end

function _matern_api_cell_field(value, nc::Integer; name::Symbol, lower = -Inf, lower_open::Bool = false)
    if value isa Real
        vals = fill(Float64(value), nc)
        is_vector = false
    elseif value isa AbstractArray || value isa Tuple
        length(value) == nc || throw(ArgumentError("Non-stationary `$name` must contain exactly one value per active cell. Expected $nc values, got $(length(value))."))
        vals = Float64.(vec(collect(value)))
        is_vector = true
    else
        throw(ArgumentError("`$name` must be a scalar or a vector with one value per active cell, got $(typeof(value))."))
    end
    any(!isfinite, vals) && throw(ArgumentError("`$name` must be finite on every active cell."))
    if lower_open
        any(x -> x <= lower, vals) && throw(ArgumentError("`$name` must be greater than $lower on every active cell."))
    else
        any(x -> x < lower, vals) && throw(ArgumentError("`$name` must be at least $lower on every active cell."))
    end
    return vals, is_vector
end

function _matern_api_diffusion_scheme(diffusion_scheme::Symbol, angle_vals::AbstractVector, ratio_vals::AbstractVector)
    if diffusion_scheme == :auto
        return _matern_api_requires_fvm9(angle_vals, ratio_vals) ? :fvm9 : :tpfa
    elseif diffusion_scheme in (:tpfa, :fvm9)
        return diffusion_scheme
    else
        throw(ArgumentError("diffusion_scheme must be :auto, :tpfa, or :fvm9, got $diffusion_scheme."))
    end
end

function _matern_api_requires_fvm9(angle_vals::AbstractVector, ratio_vals::AbstractVector)
    for (angle, ratio) in zip(angle_vals, ratio_vals)
        if abs(log(ratio)) > 1e-10 && abs(sin(2.0*angle)) > 1e-10
            return true
        end
    end
    return false
end

function _matern_api_axial_mean(angle_vals::AbstractVector)
    isempty(angle_vals) && return 0.0
    z = zero(ComplexF64)
    for angle in angle_vals
        z += cis(2.0*angle)
    end
    z /= length(angle_vals)
    abs(z) < 1e-12 && return 0.0
    return _wrap_orientation(0.5*Base.angle(z))
end

function _coerce_matern_domain(layer_or_domain)
    if layer_or_domain isa ReservoirLayerDomain
        return layer_or_domain.domain, layer_or_domain
    elseif layer_or_domain isa DataDomain
        return layer_or_domain, nothing
    else
        throw(ArgumentError("Expected a DataDomain or ReservoirLayerDomain, got $(typeof(layer_or_domain))."))
    end
end

function _matern_basis_matrix(basis, nc::Integer, ::Type{T}) where T
    if isnothing(basis)
        return zeros(T, nc, 0)
    elseif basis isa MaternBasis
        size(basis.matrix, 1) == nc || throw(ArgumentError("Basis has $(size(basis.matrix, 1)) rows, expected $nc."))
        M = basis.matrix
        return M isa SparseMatrixCSC ? SparseMatrixCSC{T, Int}(M) : Matrix{T}(M)
    else
        M = Matrix{T}(basis)
        size(M, 1) == nc || throw(ArgumentError("Basis has $(size(M, 1)) rows, expected $nc."))
        return M
    end
end

function _theta_block(theta::NamedTuple, name::Symbol, n::Integer, names::AbstractVector{Symbol})
    if haskey(theta, name)
        x = theta[name]
        if x isa AbstractDict
            out = zeros(Float64, n)
            allowed = Set(names)
            for (k, v) in pairs(x)
                key = Symbol(k)
                key in allowed || throw(ArgumentError("Unknown coefficient name $key in θ.$name."))
                out[findfirst(isequal(key), names)] = Float64(v)
            end
            return out
        else
            vals = collect(x)
            length(vals) == n || throw(ArgumentError("θ.$name has length $(length(vals)), expected $n."))
            return Float64.(vals)
        end
    else
        return zeros(Float64, n)
    end
end

_theta_block(theta::NamedTuple, name::Symbol, n::Integer) = _theta_block(theta, name, n, Symbol[])

function _normalize_matern_theta(prior::MaternSPDE2Prior, θ)
    θ isa NamedTuple || throw(ArgumentError("θ must be a named tuple with optional fields :range, :sd, :angle, :logratio."))
    names = isnothing(prior.basis) ? Symbol[] : prior.basis.names
    n = size(prior.basis_matrix, 2)
    return (
        range = _theta_block(θ, :range, n, names),
        sd = _theta_block(θ, :sd, n, names),
        angle = _theta_block(θ, :angle, n, names),
        logratio = _theta_block(θ, :logratio, n, names)
    )
end

function _zero_matern_theta(prior::MaternSPDE2Prior)
    n = size(prior.basis_matrix, 2)
    return (
        range = zeros(Float64, n),
        sd = zeros(Float64, n),
        angle = zeros(Float64, n),
        logratio = zeros(Float64, n)
    )
end

_flatten_theta(θ) = vcat(θ.range, θ.sd, θ.angle, θ.logratio)

function _canonicalize_orientation_logratio(angle::AbstractVector, logratio::AbstractVector)
    n = length(angle)
    length(logratio) == n || throw(ArgumentError("angle and logratio must have the same length."))
    angle_out = similar(Float64.(angle))
    logratio_out = similar(Float64.(logratio))
    @inbounds for i in 1:n
        ϕ = Float64(angle[i])
        r = Float64(logratio[i])
        if r < 0
            ϕ += π/2
            r = -r
        end
        angle_out[i] = _wrap_orientation(ϕ)
        logratio_out[i] = r
    end
    return angle_out, logratio_out
end

_orientation_difference(a::Real, b::Real) = _wrap_orientation(Float64(a) - Float64(b))

function _realized_covariance_orientation(pts::AbstractMatrix, anchor::Integer, corr::AbstractVector)
    x0 = pts[:, anchor]
    S = zeros(Float64, 2, 2)
    for j in axes(pts, 2)
        j == anchor && continue
        w = max(corr[j], 0.0)
        if isfinite(w) && w > 1e-6
            dx = pts[:, j] - x0
            S .+= w .* (dx * dx')
        end
    end
    if maximum(abs, S) < 1e-12
        return 0.0
    end
    F = eigen(Symmetric(S))
    vec = F.vectors[:, argmax(F.values)]
    return _wrap_orientation(atan(vec[2], vec[1]))
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

function _matern_diffusion_transmissibility(domain::DataDomain, prior::MaternSPDE2Prior, H_eff::AbstractMatrix)
    has_anisotropy = maximum(abs.(H_eff[2, :])) > 1e-12 || maximum(abs.(H_eff[1, :] .- 1.0)) > 1e-12 || maximum(abs.(H_eff[3, :] .- 1.0)) > 1e-12
    if prior.laplace_from == :geometry
        prior.version == :xyz || throw(ArgumentError("Anisotropic Matérn SPDE assembly requires version = :xyz."))
        d2 = deepcopy(domain)
        d2[:permeability] = _embed_horizontal_tensor(domain, H_eff)
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
        has_anisotropy && throw(ArgumentError("laplace_from = :unit does not support anisotropy. Use :geometry instead."))
        return ones(Float64, size(domain[:neighbors], 2))
    elseif prior.laplace_from == :flow
        has_anisotropy && throw(ArgumentError("laplace_from = :flow with anisotropy is not supported for MaternSPDE2Prior. Use :geometry."))
        return reservoir_transmissibility(domain; version = prior.version)
    else
        error("Unsupported laplace_from = $(prior.laplace_from)")
    end
end

function _validate_matern_diffusion_scheme(prior::MaternSPDE2Prior, layer, H_eff::AbstractMatrix, halo)
    offdiag = maximum(abs.(H_eff[2, :]))
    if prior.diffusion_scheme == :tpfa
        offdiag <= 1e-10 || throw(ArgumentError("diffusion_scheme = :tpfa cannot represent rotated anisotropy (nonzero H12). Use diffusion_scheme = :fvm9."))
    elseif prior.diffusion_scheme == :fvm9
        prior.laplace_from == :geometry || throw(ArgumentError("diffusion_scheme = :fvm9 requires laplace_from = :geometry, got $(prior.laplace_from)."))
        prior.version == :xyz || throw(ArgumentError("diffusion_scheme = :fvm9 requires version = :xyz, got $(prior.version)."))
        layer isa ReservoirLayerDomain || throw(ArgumentError("diffusion_scheme = :fvm9 requires a ReservoirLayerDomain from extract_reservoir_layer."))
        isnothing(halo) || throw(ArgumentError("diffusion_scheme = :fvm9 does not support MaternHaloSpec padding yet. Calibrate/use it with halo = nothing."))
    else
        error("Unsupported diffusion_scheme = $(prior.diffusion_scheme)")
    end
    return nothing
end

function _matern_parent_spacing(layer::ReservoirLayerDomain)
    function spacing(vals)
        u = sort(unique(Float64.(filter(isfinite, vals))))
        d = diff(u)
        d = filter(x -> x > 1e-10, d)
        isempty(d) ? 1.0 : median(d)
    end
    nx, ny = layer.parent_shape
    px = reshape(layer.parent_centroids[1, :], nx, ny)
    py = reshape(layer.parent_centroids[2, :], nx, ny)
    dx = spacing(vec(px))
    dy = spacing(vec(py))
    return dx, dy
end

function _matern_filter_matrix_fvm9(layer::ReservoirLayerDomain, κ_eff::AbstractVector, H_eff::AbstractMatrix)
    nc = number_of_cells(layer.domain)
    length(κ_eff) == nc || throw(ArgumentError("κ_eff must have one value per active cell."))
    size(H_eff, 1) == 3 && size(H_eff, 2) == nc || throw(ArgumentError("H_eff must be a 3 x nc compact tensor field."))

    nx, ny = layer.parent_shape
    dx, dy = _matern_parent_spacing(layer)
    volumes = Float64.(layer.domain[:volumes])
    thickness = volumes ./ max(dx*dy, eps(Float64))

    active = fill(0, nx*ny)
    for (cell, lin) in enumerate(layer.active_parent_linear)
        active[lin] = cell
    end

    cell_at(i, j) = (1 <= i <= nx && 1 <= j <= ny) ? active[(j - 1)*nx + i] : 0
    cell_or(default, i, j) = begin
        c = cell_at(i, j)
        c == 0 ? default : c
    end
    function add!(coeffs::Dict{Int, Float64}, col::Integer, val::Real)
        v = Float64(val)
        abs(v) <= 1e-14 && return nothing
        coeffs[Int(col)] = get(coeffs, Int(col), 0.0) + v
        return nothing
    end
    function add_terms!(coeffs, scale, terms)
        for (col, val) in terms
            add!(coeffs, col, scale*val)
        end
        return nothing
    end
    face_tensor(c, n) = 0.5 .* (view(H_eff, :, c) .+ view(H_eff, :, n))

    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, 9nc)
    sizehint!(J, 9nc)
    sizehint!(V, 9nc)

    for (row, lin) in enumerate(layer.active_parent_linear)
        i, j = mod1(lin, nx), fld(lin - 1, nx) + 1
        coeffs = Dict{Int, Float64}()

        east = cell_at(i + 1, j)
        if east != 0
            h = face_tensor(row, east)
            area = dy * 0.5*(thickness[row] + thickness[east])
            gx = ((east, 1.0/dx), (row, -1.0/dx))
            gy = (
                (cell_or(row, i, j + 1), 1.0/(4dy)),
                (cell_or(east, i + 1, j + 1), 1.0/(4dy)),
                (cell_or(row, i, j - 1), -1.0/(4dy)),
                (cell_or(east, i + 1, j - 1), -1.0/(4dy))
            )
            add_terms!(coeffs, area*h[1], gx)
            add_terms!(coeffs, area*h[2], gy)
        end

        west = cell_at(i - 1, j)
        if west != 0
            h = face_tensor(row, west)
            area = dy * 0.5*(thickness[row] + thickness[west])
            gx = ((row, 1.0/dx), (west, -1.0/dx))
            gy = (
                (cell_or(west, i - 1, j + 1), 1.0/(4dy)),
                (cell_or(row, i, j + 1), 1.0/(4dy)),
                (cell_or(west, i - 1, j - 1), -1.0/(4dy)),
                (cell_or(row, i, j - 1), -1.0/(4dy))
            )
            add_terms!(coeffs, -area*h[1], gx)
            add_terms!(coeffs, -area*h[2], gy)
        end

        north = cell_at(i, j + 1)
        if north != 0
            h = face_tensor(row, north)
            area = dx * 0.5*(thickness[row] + thickness[north])
            gx = (
                (cell_or(row, i + 1, j), 1.0/(4dx)),
                (cell_or(north, i + 1, j + 1), 1.0/(4dx)),
                (cell_or(row, i - 1, j), -1.0/(4dx)),
                (cell_or(north, i - 1, j + 1), -1.0/(4dx))
            )
            gy = ((north, 1.0/dy), (row, -1.0/dy))
            add_terms!(coeffs, area*h[2], gx)
            add_terms!(coeffs, area*h[3], gy)
        end

        south = cell_at(i, j - 1)
        if south != 0
            h = face_tensor(row, south)
            area = dx * 0.5*(thickness[row] + thickness[south])
            gx = (
                (cell_or(row, i + 1, j), 1.0/(4dx)),
                (cell_or(south, i + 1, j - 1), 1.0/(4dx)),
                (cell_or(row, i - 1, j), -1.0/(4dx)),
                (cell_or(south, i - 1, j - 1), -1.0/(4dx))
            )
            gy = ((row, 1.0/dy), (south, -1.0/dy))
            add_terms!(coeffs, -area*h[2], gx)
            add_terms!(coeffs, -area*h[3], gy)
        end

        for (col, val) in coeffs
            push!(I, row)
            push!(J, col)
            push!(V, -val)
        end
    end

    L_H = sparse(I, J, V, nc, nc)
    A = sparse(L_H + Diagonal(volumes .* κ_eff.^2))
    return (
        L_H = L_H,
        A = A,
        meta = (
            diffusion_scheme = :fvm9,
            parent_shape = layer.parent_shape,
            dx = dx,
            dy = dy
        )
    )
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

function _validate_matern_domain(domain::DataDomain, prior::MaternSPDE2Prior, layer = nothing)
    number_of_cells(domain) == prior.nc || throw(ArgumentError("Prior has nc = $(prior.nc), but domain has $(number_of_cells(domain)) cells."))
    D = size(domain[:cell_centroids], 1)
    D in (2, 3) || throw(ArgumentError("Only 2D or embedded single-layer 3D domains are supported, got dimension $D."))
    if D == 3 && isnothing(layer)
        dims = Tuple(grid_dims_ijk(physical_representation(domain)))
        nz = length(dims) == 3 ? dims[3] : 1
        nz == 1 || throw(ArgumentError("MaternSPDE2Prior only supports 2D domains or extracted single-layer 3D domains. Use extract_reservoir_layer first."))
    end
    return nothing
end

function _validate_matern_halo(domain::DataDomain, prior::MaternSPDE2Prior, halo::MaternHaloSpec)
    prior.laplace_from == :geometry || throw(ArgumentError("Halo padding currently requires laplace_from = :geometry, got $(prior.laplace_from)."))
    prior.version == :xyz || throw(ArgumentError("Halo padding currently requires version = :xyz, got $(prior.version)."))
    haskey(domain, :boundary_centroids) || throw(ArgumentError("Domain is missing :boundary_centroids required for halo padding."))
    haskey(domain, :boundary_areas) || throw(ArgumentError("Domain is missing :boundary_areas required for halo padding."))
    haskey(domain, :boundary_neighbors) || throw(ArgumentError("Domain is missing :boundary_neighbors required for halo padding."))
    haskey(domain, :boundary_normals) || throw(ArgumentError("Domain is missing :boundary_normals required for halo padding."))
    return nothing
end

function _matern_extend_fields(
        fields,
        source::AbstractVector{<:Integer};
        κ_eff,
        τ_eff,
        angle_eff,
        logratio_eff,
        ratio_eff,
        u_eff,
        v_eff,
        H_eff,
        logtau_correction,
        logkappa_correction,
        angle_correction,
        logratio_correction
    )
    return (
        θ = fields.θ,
        ρ = vcat(fields.ρ, fields.ρ[source]),
        σ = vcat(fields.σ, fields.σ[source]),
        angle = vcat(fields.angle, fields.angle[source]),
        orientation = vcat(fields.orientation, fields.orientation[source]),
        raw_logratio = vcat(fields.raw_logratio, fields.raw_logratio[source]),
        logratio = vcat(fields.logratio, fields.logratio[source]),
        ratio = vcat(fields.ratio, fields.ratio[source]),
        u = vcat(fields.u, fields.u[source]),
        v = vcat(fields.v, fields.v[source]),
        r = vcat(fields.r, fields.r[source]),
        H = hcat(fields.H, fields.H[:, source]),
        κ = vcat(fields.κ, fields.κ[source]),
        τ_nominal = vcat(fields.τ_nominal, fields.τ_nominal[source]),
        ρ_major = vcat(fields.ρ_major, fields.ρ_major[source]),
        ρ_minor = vcat(fields.ρ_minor, fields.ρ_minor[source]),
        κ_eff = vcat(κ_eff, κ_eff[source]),
        τ = vcat(τ_eff, τ_eff[source]),
        angle_eff = vcat(angle_eff, angle_eff[source]),
        logratio_eff = vcat(logratio_eff, logratio_eff[source]),
        ratio_eff = vcat(ratio_eff, ratio_eff[source]),
        u_eff = vcat(u_eff, u_eff[source]),
        v_eff = vcat(v_eff, v_eff[source]),
        H_eff = hcat(H_eff, H_eff[:, source]),
        logtau_correction = vcat(logtau_correction, logtau_correction[source]),
        logkappa_correction = vcat(logkappa_correction, logkappa_correction[source]),
        angle_correction = vcat(angle_correction, angle_correction[source]),
        logratio_correction = vcat(logratio_correction, logratio_correction[source])
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

function _build_matern_halo(domain::DataDomain, halo::MaternHaloSpec, fields, κ_eff, τ_eff, angle_eff, logratio_eff, ratio_eff, u_eff, v_eff, H_eff, corrections, transmissibilities)
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
    total_padding = _halo_total_padding(halo.total_padding, fields.ρ_major[boundary_cells])
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
    fields_ext = _matern_extend_fields(
        fields,
        halo_source;
        κ_eff = κ_eff,
        τ_eff = τ_eff,
        angle_eff = angle_eff,
        logratio_eff = logratio_eff,
        ratio_eff = ratio_eff,
        u_eff = u_eff,
        v_eff = v_eff,
        H_eff = H_eff,
        logtau_correction = corrections.logtau,
        logkappa_correction = corrections.logkappa,
        angle_correction = corrections.angle,
        logratio_correction = corrections.logratio
    )

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
    return _selected_qinv_diagonal_from_factor(F, size(Q, 1), idx; chunk_size = chunk_size)
end

function _selected_qinv_diagonal_from_factor(F, n::Integer, idx::AbstractVector{<:Integer}; chunk_size::Int = 32)
    vals = zeros(Float64, length(idx))
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
    return _selected_qinv_columns_from_factor(F, size(Q, 1), idx; chunk_size = chunk_size)
end

function _selected_qinv_columns_from_factor(F, n::Integer, idx::AbstractVector{<:Integer}; chunk_size::Int = 16)
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

function _content_hash(x)
    h = hash(typeof(x))
    if x isa AbstractArray
        h = hash(size(x), h)
        for v in x
            h = hash(v, h)
        end
        return h
    elseif x isa NamedTuple
        for (k, v) in pairs(x)
            h = hash(k, h)
            h = hash(_content_hash(v), h)
        end
        return h
    else
        return hash(x, h)
    end
end
