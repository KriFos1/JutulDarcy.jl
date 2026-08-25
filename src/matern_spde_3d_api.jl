function _matern_api_cell_field_3d(value, nc::Integer; name::Symbol, lower = -Inf, lower_open::Bool = false)
    isnothing(value) && throw(ArgumentError("$name must be provided."))
    vals = if value isa Real
        fill(Float64(value), nc)
    else
        v = Float64.(collect(value))
        length(v) == 1 ? fill(first(v), nc) : v
    end
    length(vals) == nc || throw(ArgumentError("$name has length $(length(vals)), expected 1 or $nc."))
    if lower_open
        all(x -> x > lower, vals) || throw(ArgumentError("$name must be > $lower."))
    else
        all(x -> x >= lower, vals) || throw(ArgumentError("$name must be >= $lower."))
    end
    any(!isfinite, vals) && throw(ArgumentError("$name entries must be finite."))
    return vals
end

function _matern_full_3d_domain_from_data_file(data_path::AbstractString)
    endswith(lowercase(data_path), ".data") || throw(ArgumentError("Expected `data_path` to point to a .DATA file, got $data_path."))
    return reservoir_domain(parse_data_file(data_path))
end

function _matern_3d_identity_basis(nc::Integer)
    return MaternBasis(spdiagm(0 => ones(Float64, nc)); names = [Symbol("cell_", i) for i in 1:nc], provenance = "3d-api/cellwise")
end

function _matern_precision_api_setup_3d(
        data_path::AbstractString;
        target_variance,
        target_range = nothing,
        target_range_major = target_range,
        target_range_minor = target_range,
        target_range_vertical = target_range,
        target_azimuth = 0.0,
        target_dip = 0.0,
        target_rake = 0.0,
        angle_unit::Symbol = :degree,
        diffusion_scheme::Symbol = :auto
    )
    angle_unit in (:degree, :radian) || throw(ArgumentError("angle_unit must be :degree or :radian, got $angle_unit."))
    domain = _matern_full_3d_domain_from_data_file(data_path)
    nc = number_of_cells(domain)
    variance_vals = _matern_api_cell_field_3d(target_variance, nc; name = :target_variance, lower = 0.0, lower_open = true)
    rx_raw = isnothing(target_range_major) ? target_range : target_range_major
    ry_raw = isnothing(target_range_minor) ? target_range : target_range_minor
    rz_raw = isnothing(target_range_vertical) ? target_range : target_range_vertical
    rx = _matern_api_cell_field_3d(rx_raw, nc; name = :target_range_major, lower = 0.0, lower_open = true)
    ry = _matern_api_cell_field_3d(ry_raw, nc; name = :target_range_minor, lower = 0.0, lower_open = true)
    rz = _matern_api_cell_field_3d(rz_raw, nc; name = :target_range_vertical, lower = 0.0, lower_open = true)
    az = _convert_angle.(_matern_api_cell_field_3d(target_azimuth, nc; name = :target_azimuth), Ref(angle_unit))
    dip = _convert_angle.(_matern_api_cell_field_3d(target_dip, nc; name = :target_dip), Ref(angle_unit))
    rake = _convert_angle.(_matern_api_cell_field_3d(target_rake, nc; name = :target_rake), Ref(angle_unit))

    ρ0 = exp(mean(log.((rx .* ry .* rz).^(1/3))))
    σ0 = exp(mean(log.(sqrt.(variance_vals))))
    basis = _matern_3d_identity_basis(nc)
    prior = MaternSPDE3Prior(
        nc;
        ρ0 = ρ0,
        σ0 = σ0,
        range_x_basis = basis,
        range_y_basis = basis,
        range_z_basis = basis,
        sd_basis = basis,
        azimuth_basis = basis,
        dip_basis = basis,
        rake_basis = basis,
        diffusion_scheme = diffusion_scheme
    )
    θ = (
        range_x = log.(rx ./ ρ0),
        range_y = log.(ry ./ ρ0),
        range_z = log.(rz ./ ρ0),
        sd = log.(sqrt.(variance_vals) ./ σ0),
        azimuth = az,
        dip = dip,
        rake = rake
    )
    return (domain = domain, prior = prior, θ = θ)
end

function matern_precision_from_data_file_3d(data_path::AbstractString; boundary::Symbol = :robin, kwargs...)
    setup = _matern_precision_api_setup_3d(data_path; kwargs...)
    op = matern_spde_operator(setup.domain, setup.prior, setup.θ; compensated = false, boundary = boundary)
    return sparse(op.Q)
end

function _matern_3d_grid_metadata(domain::DataDomain, n_total::Integer)
    mesh = physical_representation(domain)
    nc = number_of_cells(domain)
    ijk = [Tuple(cell_ijk(mesh, c)) for c in 1:nc]
    ni = maximum(first, ijk)
    nj = maximum(x -> x[2], ijk)
    nk = maximum(last, ijk)
    lin(i, j, k) = i + (j - 1)*ni + (k - 1)*ni*nj
    grid_mask = zeros(Int8, ni*nj*nk)
    active_linear = Vector{Int}(undef, nc)
    for (c, (i, j, k)) in enumerate(ijk)
        l = lin(i, j, k)
        active_linear[c] = l
        grid_mask[l] = Int8(1)
    end
    parent_centroids = fill(NaN, 3, ni*nj*nk)
    cc = Matrix{Float64}(domain[:cell_centroids])
    for c in 1:nc
        parent_centroids[:, active_linear[c]] .= cc[:, c]
    end
    domain_mask = falses(n_total)
    domain_mask[1:nc] .= true
    return (
        domain_mask = domain_mask,
        grid_mask = grid_mask,
        grid_ni = ni,
        grid_nj = nj,
        grid_nk = nk,
        active_parent_linear = active_linear,
        parent_centroids = parent_centroids
    )
end

function matern_precision_csc_from_data_file_3d(data_path::AbstractString; nugget::Union{Nothing, Real} = nothing, boundary::Symbol = :robin, kwargs...)
    setup = _matern_precision_api_setup_3d(data_path; kwargs...)
    op = matern_spde_operator(setup.domain, setup.prior, setup.θ; compensated = false, boundary = boundary)
    Q = sparse(op.Q)
    nug = isnothing(nugget) ? 1e-8 * mean(diag(Q)) : Float64(nugget)
    if nug > 0
        Q = Q + nug * sparse(I, size(Q)...)
    end
    meta = _matern_3d_grid_metadata(setup.domain, size(Q, 1))
    return (
        shape = size(Q),
        colptr = copy(Q.colptr),
        rowval = copy(Q.rowval),
        nzval = copy(Q.nzval),
        domain_mask = meta.domain_mask,
        interior_idx = collect(1:number_of_cells(setup.domain)),
        halo_idx = Int[],
        grid_mask = meta.grid_mask,
        grid_ni = meta.grid_ni,
        grid_nj = meta.grid_nj,
        grid_nk = meta.grid_nk,
        active_parent_linear = meta.active_parent_linear,
        parent_centroids = meta.parent_centroids,
        n_halo = 0,
        diffusion_scheme = op.diffusion_scheme,
        boundary = op.boundary
    )
end
