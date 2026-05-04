struct ReservoirLayerDomain{D<:DataDomain, T<:AbstractFloat}
    domain::D
    layer::Int
    parent_shape::NTuple{2, Int}
    active_parent_cells::Vector{Int}
    active_parent_linear::Vector{Int}
    active_parent_ijk::Vector{NTuple{3, Int}}
    parent_centroids::Matrix{T}
    active_mask::BitVector
end

struct MaternBasis{T<:AbstractFloat, M<:AbstractMatrix{T}}
    matrix::M
    names::Vector{Symbol}
    provenance::String
end

struct MaternBasisFit
    basis::MaternBasis
    baselines::NamedTuple
    θ::NamedTuple
    diagnostics::NamedTuple
end

function ReservoirLayerDomain(
        domain::DataDomain,
        layer::Integer,
        parent_shape::NTuple{2, Int},
        active_parent_cells::AbstractVector{<:Integer},
        active_parent_linear::AbstractVector{<:Integer},
        active_parent_ijk::AbstractVector{<:NTuple{3, <:Integer}},
        parent_centroids::AbstractMatrix{<:Real},
        active_mask::AbstractVector{Bool}
    )
    nx, ny = parent_shape
    length(active_parent_cells) == number_of_cells(domain) || throw(ArgumentError("active_parent_cells must have one entry per active cell."))
    length(active_parent_linear) == number_of_cells(domain) || throw(ArgumentError("active_parent_linear must have one entry per active cell."))
    length(active_parent_ijk) == number_of_cells(domain) || throw(ArgumentError("active_parent_ijk must have one entry per active cell."))
    length(active_mask) == nx*ny || throw(ArgumentError("active_mask must have length nx*ny = $(nx*ny), got $(length(active_mask))."))
    size(parent_centroids, 2) == nx*ny || throw(ArgumentError("parent_centroids must have nx*ny columns = $(nx*ny), got $(size(parent_centroids, 2))."))
    return ReservoirLayerDomain(
        domain,
        Int(layer),
        parent_shape,
        Int.(collect(active_parent_cells)),
        Int.(collect(active_parent_linear)),
        NTuple{3, Int}.(collect(active_parent_ijk)),
        Matrix{Float64}(parent_centroids),
        BitVector(active_mask)
    )
end

function MaternBasis(matrix::AbstractMatrix{<:Real}; names = nothing, provenance::AbstractString = "")
    size(matrix, 1) > 0 || throw(ArgumentError("MaternBasis must have at least one row."))
    size(matrix, 2) > 0 || throw(ArgumentError("MaternBasis must have at least one column. Use `nothing` for a stationary prior."))
    any(!isfinite, matrix) && throw(ArgumentError("MaternBasis entries must be finite."))
    M = matrix isa SparseMatrixCSC ? SparseMatrixCSC{Float64, Int}(matrix) : Matrix{Float64}(matrix)
    if isnothing(names)
        cols = [Symbol("basis_", i) for i in 1:size(M, 2)]
    else
        length(names) == size(M, 2) || throw(ArgumentError("Expected $(size(M, 2)) column names, got $(length(names))."))
        cols = Symbol.(collect(names))
    end
    length(unique(cols)) == length(cols) || throw(ArgumentError("MaternBasis column names must be unique."))
    _warn_matern_basis_collinearity(M, cols)
    return MaternBasis(M, cols, String(provenance))
end

function combine_matern_basis(bases::MaternBasis...)
    isempty(bases) && throw(ArgumentError("At least one MaternBasis is required."))
    nrows = size(first(bases).matrix, 1)
    all(size(b.matrix, 1) == nrows for b in bases) || throw(ArgumentError("All MaternBasis inputs must have the same number of rows."))
    names = reduce(vcat, (b.names for b in bases))
    length(unique(names)) == length(names) || throw(ArgumentError("Duplicate basis-column names are not allowed when combining MaternBasis objects."))
    matrices = map(b -> b.matrix, bases)
    use_sparse = any(m -> m isa SparseMatrixCSC, matrices)
    M = use_sparse ? sparse(hcat(matrices...)) : hcat(Matrix.(matrices)...)
    provenance = join(filter(x -> !isempty(x), [b.provenance for b in bases]), " + ")
    return MaternBasis(M; names = names, provenance = provenance)
end

function matern_identity_basis(layer::ReservoirLayerDomain)
    nc = number_of_cells(layer.domain)
    nc > 500 && @warn "Building an identity MaternBasis for $nc active cells. This creates one coefficient per cell and is usually impractical on larger layers."
    return MaternBasis(spdiagm(0 => ones(Float64, nc)); names = [Symbol("cell_", i) for i in 1:nc], provenance = "identity")
end

function matern_trend_basis(layer::ReservoirLayerDomain; axes = (:x, :y), coords::Symbol = :physical)
    coords in (:physical, :index) || throw(ArgumentError("coords must be :physical or :index, got $coords."))
    columns = AbstractVector{Float64}[]
    names = Symbol[]
    for axis in axes
        vals = if coords == :physical
            if axis == :x
                vec(layer.domain[:cell_centroids][1, :])
            elseif axis == :y
                vec(layer.domain[:cell_centroids][2, :])
            else
                throw(ArgumentError("Physical trend axes must be :x or :y, got $axis."))
            end
        else
            if axis == :i
                map(_parent_linear_to_ij(layer.parent_shape), layer.active_parent_linear) .|> first
            elseif axis == :j
                map(_parent_linear_to_ij(layer.parent_shape), layer.active_parent_linear) .|> last
            else
                throw(ArgumentError("Index trend axes must be :i or :j, got $axis."))
            end
        end
        normalized = _normalize_basis_column(vals)
        if isnothing(normalized)
            @warn "Skipping degenerate trend axis $axis for MaternBasis construction."
            continue
        end
        push!(columns, normalized)
        push!(names, Symbol(axis))
    end
    isempty(columns) && throw(ArgumentError("No non-degenerate trend columns were produced."))
    return MaternBasis(hcat(columns...); names = names, provenance = "trend/$coords")
end

function matern_raster_basis(layer::ReservoirLayerDomain, raster)
    vals = _coerce_layer_field(layer, raster; allow_scalar = false, name = :raster)
    normalized = _normalize_basis_column(vals)
    isnothing(normalized) && throw(ArgumentError("Raster basis column is degenerate on the active cells of this layer."))
    return MaternBasis(reshape(normalized, :, 1); names = [:raster], provenance = "raster")
end

function matern_region_basis(layer::ReservoirLayerDomain, region_map; reference = :auto)
    labels = _coerce_layer_region_map(layer, region_map)
    regs = sort(unique(labels))
    isempty(regs) && throw(ArgumentError("Region map did not contain any active-cell labels."))
    ref = if reference === :auto
        counts = Dict(r => count(==(r), labels) for r in regs)
        first(sort(collect(regs), by = r -> (-counts[r], string(r))))
    else
        reference
    end
    ref in regs || throw(ArgumentError("Reference region $ref is not present among the active-cell region labels."))
    keep = filter(!=(ref), regs)
    isempty(keep) && throw(ArgumentError("Region basis requires at least two distinct active-cell regions."))
    M = zeros(Float64, number_of_cells(layer.domain), length(keep))
    for (j, reg) in enumerate(keep)
        @inbounds for i in eachindex(labels)
            M[i, j] = labels[i] == reg ? 1.0 : 0.0
        end
    end
    return MaternBasis(M; names = Symbol.("region_" .* string.(keep)), provenance = "regions")
end

function fit_matern_fields(
        layer::ReservoirLayerDomain,
        basis::MaternBasis;
        range0,
        sd0,
        angle0 = 0.0,
        logratio0 = 0.0,
        range = nothing,
        sd = nothing,
        angle = nothing,
        ratio = nothing,
        major = nothing,
        angle_unit::Symbol = :radian
    )
    size(basis.matrix, 1) == number_of_cells(layer.domain) || throw(ArgumentError("Basis row count $(size(basis.matrix, 1)) does not match active cell count $(number_of_cells(layer.domain))."))
    range0 > 0 || throw(ArgumentError("range0 must be positive, got $range0."))
    sd0 > 0 || throw(ArgumentError("sd0 must be positive, got $sd0."))
    angle_unit in (:radian, :degree) || throw(ArgumentError("angle_unit must be :radian or :degree, got $angle_unit."))

    if !isnothing(major)
        isnothing(range) || throw(ArgumentError("Pass either `range` or `major`, not both."))
        isnothing(ratio) && throw(ArgumentError("`major` requires `ratio`."))
        ratio_vals = _coerce_layer_field(layer, ratio; name = :ratio)
        _validate_positive_ratio(ratio_vals)
        major_vals = _coerce_layer_field(layer, major; name = :major)
        any(x -> !isfinite(x) || x <= 0, major_vals) && throw(ArgumentError("major must be finite and positive on every active cell."))
        range = major_vals ./ sqrt.(ratio_vals)
        ratio = ratio_vals
    end

    baselines = (
        range0 = Float64(range0),
        sd0 = Float64(sd0),
        angle0 = _convert_angle(Float64(angle0), angle_unit),
        logratio0 = Float64(logratio0)
    )
    θ = (
        range = zeros(Float64, size(basis.matrix, 2)),
        sd = zeros(Float64, size(basis.matrix, 2)),
        angle = zeros(Float64, size(basis.matrix, 2)),
        logratio = zeros(Float64, size(basis.matrix, 2))
    )
    diagnostics = Dict{Symbol, NamedTuple}()

    if !isnothing(range)
        vals = _coerce_layer_field(layer, range; name = :range)
        any(x -> !isfinite(x) || x <= 0, vals) && throw(ArgumentError("range must be finite and positive on every active cell."))
        θ = merge(θ, (range = _fit_basis_coefficients(layer, basis, log.(vals) .- log(baselines.range0), :range, diagnostics),))
    end
    if !isnothing(sd)
        vals = _coerce_layer_field(layer, sd; name = :sd)
        any(x -> !isfinite(x) || x <= 0, vals) && throw(ArgumentError("sd must be finite and positive on every active cell."))
        θ = merge(θ, (sd = _fit_basis_coefficients(layer, basis, log.(vals) .- log(baselines.sd0), :sd, diagnostics),))
    end
    if !isnothing(angle)
        vals = _coerce_layer_field(layer, angle; name = :angle)
        vals = _convert_angle.(vals, Ref(angle_unit))
        residual = _wrap_orientation.(vals .- baselines.angle0)
        θ = merge(θ, (angle = _fit_basis_coefficients(layer, basis, residual, :angle, diagnostics),))
    end
    if !isnothing(ratio)
        vals = _coerce_layer_field(layer, ratio; name = :ratio)
        _validate_positive_ratio(vals)
        θ = merge(θ, (logratio = _fit_basis_coefficients(layer, basis, log.(vals) .- baselines.logratio0, :logratio, diagnostics),))
    end

    return MaternBasisFit(
        basis,
        baselines,
        θ,
        (fields = NamedTuple(diagnostics),)
    )
end

function extract_reservoir_layer(input; layer::Integer, remove_inactive::Bool = true)
    if input isa ReservoirLayerDomain
        input.layer == layer || throw(ArgumentError("ReservoirLayerDomain already represents layer $(input.layer), cannot extract layer $layer from it."))
        return input
    end
    domain = _resolve_layer_input_to_domain(input)
    return _extract_reservoir_layer_from_domain(domain, Int(layer); remove_inactive = remove_inactive)
end

function _resolve_layer_input_to_domain(input)
    if input isa DataDomain
        return input
    elseif input isa SimulationModel || input isa MultiModel || input isa JutulCase
        return reservoir_domain(input)
    elseif input isa AbstractString
        lower = lowercase(input)
        if endswith(lower, ".data")
            return reservoir_domain(parse_data_file(input))
        elseif endswith(lower, ".afi")
            x = GeoEnergyIO.IXParser.read_afi_file(input, convert = true, verbose = false)
            return reservoir_domain(GeoEnergyIO.IXParser.AFIInputFile(x))
        else
            throw(ArgumentError("Unsupported layer-input path $input. Expected a .DATA or .afi file."))
        end
    elseif input isa GeoEnergyIO.IXParser.AFIInputFile
        return reservoir_domain(input)
    elseif input isa AbstractDict
        if haskey(input, "IX") || haskey(input, :IX)
            return reservoir_domain(GeoEnergyIO.IXParser.AFIInputFile(input))
        else
            return reservoir_domain(input)
        end
    else
        throw(ArgumentError("Unsupported layer input of type $(typeof(input))."))
    end
end

function _extract_reservoir_layer_from_domain(domain::DataDomain, layer::Int; remove_inactive::Bool)
    mesh = physical_representation(domain)
    nx, ny, nz = _grid_dims_ijk3(mesh)
    1 <= layer <= nz || throw(ArgumentError("Layer $layer is outside 1:$nz."))

    D = size(domain[:cell_centroids], 1)
    global_map = hasproperty(mesh, :cell_map) ? collect(mesh.cell_map) : collect(1:number_of_cells(mesh))
    all_ijk = NTuple{3, Int}[tuple(_cell_ijk3(mesh, c)...) for c in 1:number_of_cells(mesh)]
    layer_cells = Int[]
    active_parent_linear = Int[]
    active_parent_ijk = NTuple{3, Int}[]
    active_parent_cells = Int[]
    for c in 1:number_of_cells(mesh)
        i, j, k = all_ijk[c]
        if k == layer
            push!(layer_cells, c)
            push!(active_parent_linear, (j - 1)*nx + i)
            push!(active_parent_ijk, (i, j, k))
            push!(active_parent_cells, global_map[c])
        end
    end
    isempty(layer_cells) && throw(ArgumentError("No cells were found in layer $layer."))

    if !remove_inactive
        @warn "`remove_inactive = false` cannot recreate cells missing from the input domain. Only cells present in the source domain are retained."
    end

    parent_centroids = fill(Float64(NaN), D, nx*ny)
    active_mask = falses(nx*ny)
    cc = Matrix{Float64}(domain[:cell_centroids])
    for (local_idx, lin) in enumerate(active_parent_linear)
        parent_centroids[:, lin] .= cc[:, layer_cells[local_idx]]
        active_mask[lin] = true
    end

    if D == 2 && layer != 1
        throw(ArgumentError("A 2D domain only supports layer = 1, got $layer."))
    end

    if D == 2
        return ReservoirLayerDomain(
            domain,
            1,
            (nx, ny),
            active_parent_cells,
            active_parent_linear,
            active_parent_ijk,
            parent_centroids,
            active_mask
        )
    end

    submesh = extract_submesh(mesh, layer_cells)
    poro = copy(domain[:porosity][layer_cells])
    perm = _subset_cell_parameter(domain[:permeability], layer_cells)
    local_neighbors = reservoir_domain(submesh, porosity = poro, permeability = perm)[:neighbors]
    local_nnc = _layer_local_nnc_from_domain(domain, layer_cells, submesh)
    local_mult = _layer_local_face_field(domain, layer_cells, local_neighbors, :transmissibility_multiplier; default = 1.0)
    local_override = _layer_local_face_field(domain, layer_cells, local_neighbors, :transmissibility_override; default = NaN)

    kwargs = Dict{Symbol, Any}(
        :porosity => poro,
        :permeability => perm
    )
    !ismissing(local_nnc) && (kwargs[:nnc] = local_nnc)
    if haskey(domain, :net_to_gross)
        kwargs[:net_to_gross] = copy(domain[:net_to_gross][layer_cells])
    end
    !ismissing(local_mult) && (kwargs[:transmissibility_multiplier] = local_mult)
    !ismissing(local_override) && (kwargs[:transmissibility_override] = local_override)
    layer_domain = reservoir_domain(submesh; kwargs...)

    return ReservoirLayerDomain(
        layer_domain,
        layer,
        (nx, ny),
        active_parent_cells,
        active_parent_linear,
        active_parent_ijk,
        parent_centroids,
        active_mask
    )
end

function _subset_cell_parameter(x, idx::AbstractVector{<:Integer})
    if x isa AbstractMatrix
        return copy(x[:, idx])
    elseif x isa AbstractVector
        return copy(x[idx])
    else
        return x
    end
end

function _layer_local_face_field(domain, layer_cells, local_neighbors, key; default)
    haskey(domain, key, Faces()) || return missing

    global_to_local = Dict(global_cell => local_cell for (local_cell, global_cell) in enumerate(layer_cells))
    full_neighbors = domain[:neighbors]
    num_nnc = haskey(domain, :nnc) ? length(domain[:nnc].cells) : 0
    nfull_geom = size(full_neighbors, 2) - num_nnc
    full_values = domain[key]

    pair_to_face = Dict{Tuple{Int, Int}, Int}()
    for f in 1:nfull_geom
        l = full_neighbors[1, f]
        r = full_neighbors[2, f]
        if haskey(global_to_local, l) && haskey(global_to_local, r)
            pair = l < r ? (l, r) : (r, l)
            pair_to_face[pair] = f
        end
    end

    out = fill(default, size(local_neighbors, 2))
    for f in axes(local_neighbors, 2)
        l = layer_cells[local_neighbors[1, f]]
        r = layer_cells[local_neighbors[2, f]]
        pair = l < r ? (l, r) : (r, l)
        if haskey(pair_to_face, pair)
            out[f] = full_values[pair_to_face[pair]]
        end
    end
    return out
end

function _layer_local_nnc_from_domain(domain, layer_cells, submesh)
    haskey(domain, :nnc) || return missing
    nnc_full = domain[:nnc]
    global_to_local = Dict(global_cell => local_cell for (local_cell, global_cell) in enumerate(layer_cells))

    neighbors = Tuple{Int, Int}[]
    trans_flow = Float64[]
    trans_thermal = Float64[]
    for (k, (l, r)) in enumerate(nnc_full.cells)
        if haskey(global_to_local, l) && haskey(global_to_local, r)
            push!(neighbors, (global_to_local[l], global_to_local[r]))
            push!(trans_flow, Float64(nnc_full.trans_flow[k]))
            push!(trans_thermal, Float64(nnc_full.trans_thermal[k]))
        end
    end
    isempty(neighbors) && return missing
    return setup_nnc_connections(submesh, neighbors, trans_flow, trans_thermal)
end

function _grid_dims_ijk3(mesh)
    dims = Tuple(grid_dims_ijk(mesh))
    if length(dims) == 2
        return (Int(dims[1]), Int(dims[2]), 1)
    elseif length(dims) == 3
        return (Int(dims[1]), Int(dims[2]), Int(dims[3]))
    else
        throw(ArgumentError("Structured IJK semantics are required for layer extraction."))
    end
end

function _cell_ijk3(mesh, cell)
    ijk = Tuple(cell_ijk(mesh, cell))
    if length(ijk) == 2
        return (Int(ijk[1]), Int(ijk[2]), 1)
    else
        return (Int(ijk[1]), Int(ijk[2]), Int(ijk[3]))
    end
end

_parent_linear_to_ij(parent_shape::NTuple{2, Int}) = lin -> (mod1(lin, parent_shape[1]), fld(lin - 1, parent_shape[1]) + 1)

function _normalize_basis_column(values::AbstractVector{<:Real})
    x = Float64.(collect(values))
    xc = x .- mean(x)
    s = maximum(abs, xc)
    if !(isfinite(s) && s > 1e-12)
        return nothing
    end
    return xc ./ s
end

function _warn_matern_basis_collinearity(M::AbstractMatrix, names::Vector{Symbol})
    n = size(M, 2)
    n <= 1 && return nothing
    norms = [norm(view(M, :, j)) for j in 1:n]
    for j in 1:n
        norms[j] > 1e-12 || @warn "MaternBasis column $(names[j]) is nearly zero."
    end
    for i in 1:(n - 1), j in (i + 1):n
        ni = norms[i]
        nj = norms[j]
        (ni > 1e-12 && nj > 1e-12) || continue
        corr = abs(dot(view(M, :, i), view(M, :, j)) / (ni*nj))
        corr > 1 - 1e-10 && @warn "MaternBasis columns $(names[i]) and $(names[j]) are nearly collinear."
    end
    return nothing
end

function _coerce_layer_field(layer::ReservoirLayerDomain, value; allow_scalar::Bool = true, name::Symbol = :field)
    nc = number_of_cells(layer.domain)
    if allow_scalar && value isa Real
        return fill(Float64(value), nc)
    elseif value isa AbstractVector
        if length(value) == nc
            vals = Float64.(collect(value))
        elseif length(value) == prod(layer.parent_shape)
            vals = _active_from_parent_layer_values(layer, value)
        else
            throw(ArgumentError("$name must be a scalar, an active-cell vector of length $nc, or a parent-layer vector of length $(prod(layer.parent_shape))."))
        end
    elseif value isa AbstractMatrix
        size(value) == reverse(layer.parent_shape) && return _coerce_layer_field(layer, vec(permutedims(value)); allow_scalar = false, name = name)
        size(value) == layer.parent_shape && return _coerce_layer_field(layer, vec(value); allow_scalar = false, name = name)
        throw(ArgumentError("$name raster must have size $(layer.parent_shape) or $(reverse(layer.parent_shape))."))
    else
        throw(ArgumentError("Unsupported $name input of type $(typeof(value))."))
    end
    any(x -> !(ismissing(x) || isfinite(x)), vals) && throw(ArgumentError("$name must be finite on all active cells."))
    return vals
end

function _active_from_parent_layer_values(layer::ReservoirLayerDomain, values)
    parent_vals = collect(values)
    length(parent_vals) == prod(layer.parent_shape) || throw(ArgumentError("Parent-layer input must have length $(prod(layer.parent_shape))."))
    vals = zeros(Float64, number_of_cells(layer.domain))
    for (i, lin) in enumerate(layer.active_parent_linear)
        v = parent_vals[lin]
        ismissing(v) && throw(ArgumentError("Active parent-layer position $lin is missing. Missing values are only allowed on inactive cells."))
        isfinite(v) || throw(ArgumentError("Active parent-layer position $lin is not finite. Missing values are only allowed on inactive cells."))
        vals[i] = Float64(v)
    end
    return vals
end

function _coerce_layer_region_map(layer::ReservoirLayerDomain, value)
    nc = number_of_cells(layer.domain)
    if value isa AbstractVector && length(value) == nc
        labels = collect(value)
    elseif value isa AbstractVector && length(value) == prod(layer.parent_shape)
        labels = [_require_region_value(value[lin], lin) for lin in layer.active_parent_linear]
    elseif value isa AbstractMatrix && size(value) == layer.parent_shape
        labels = _coerce_layer_region_map(layer, vec(value))
    elseif value isa AbstractMatrix && size(value) == reverse(layer.parent_shape)
        labels = _coerce_layer_region_map(layer, vec(permutedims(value)))
    else
        throw(ArgumentError("Region map must match the parent-layer shape $(layer.parent_shape) or the active-cell count $nc."))
    end
    any(ismissing, labels) && throw(ArgumentError("Region labels are missing on active cells."))
    return labels
end

_require_region_value(v, lin) = ismissing(v) ? throw(ArgumentError("Active region at parent-layer position $lin is missing.")) : v

function _validate_positive_ratio(vals)
    any(x -> !isfinite(x) || x < 1.0, vals) && throw(ArgumentError("anisotropy ratio must be finite and at least 1.0 on every active cell."))
end

function _convert_angle(x::Real, unit::Symbol)
    if unit == :radian
        return Float64(x)
    else
        return Float64(x) * (π/180.0)
    end
end

function _fit_basis_coefficients(layer::ReservoirLayerDomain, basis::MaternBasis, residual::AbstractVector, name::Symbol, diagnostics::Dict{Symbol, NamedTuple})
    B = basis.matrix
    vols = Float64.(layer.domain[:volumes])
    sqrtw = sqrt.(max.(vols, eps(Float64)))
    y = Float64.(collect(residual))
    A = Diagonal(sqrtw) * Matrix(B)
    rhs = sqrtw .* y
    coef = A \ rhs
    fit = Matrix(B) * coef
    res = fit .- y
    rel = norm(sqrtw .* res) / max(norm(rhs), eps(Float64))
    offset = sum(vols .* y) / max(sum(vols), eps(Float64))
    diagnostics[name] = (
        weighted_rel_l2 = rel,
        maxabs = maximum(abs.(res)),
        baseline_offset = offset
    )
    (rel > 0.05 || abs(offset) > 0.05) && @warn "MaternBasis fit for $name is a poor approximation with the supplied baseline/basis." diagnostics = diagnostics[name]
    return vec(Float64.(coef))
end

function _wrap_orientation(x::Real)
    y = Float64(x) + π/2
    y -= floor(y / π) * π
    return y - π/2
end
