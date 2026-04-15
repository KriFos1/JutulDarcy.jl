#!/usr/bin/env julia

using Jutul
using JutulDarcy
using Jutul: si_unit
using CairoMakie
using SparseArrays
using LinearAlgebra
using Random
using Statistics
using GeoEnergyIO

"""
    load_spe10_domain(; layers = 1:1, remove_cells = true, minporo = 0.01)

Read SPE10 and extract parent-grid metadata for handling inactive cells in
variogram plots.
"""
function load_spe10_domain(; layers = 1:1, remove_cells = true, minporo = 0.01)
    full_domain = JutulDarcy.SPE10.setup_reservoir(
        layers = layers,
        remove_cells = false,
        minporo = eps(Float64)
    )
    mesh = physical_representation(full_domain)
    nx, ny, nz = grid_dims_ijk(mesh)

    if remove_cells
        active_cells = findall(full_domain[:porosity] .>= minporo)
        domain = JutulDarcy.SPE10.setup_reservoir(layers = layers, remove_cells = true, minporo = minporo)
    else
        active_cells = collect(1:number_of_cells(mesh))
        domain = JutulDarcy.SPE10.setup_reservoir(layers = layers, remove_cells = false, minporo = minporo)
    end

    parent_ijk = map(cell -> cell_ijk(mesh, cell), active_cells)
    all_parent_ijk = map(cell -> cell_ijk(mesh, cell), 1:number_of_cells(mesh))

    return (
        domain = domain,
        parent_mesh = mesh,
        parent_ijk = parent_ijk,
        all_parent_ijk = all_parent_ijk,
        parent_centroids = full_domain[:cell_centroids],
        layer_cells = active_cells,
        nx = nx,
        ny = ny,
        nz = nz
    )
end

function _local_face_field_from_full_domain(full_domain, layer_cells, local_neighbors, key; default)
    haskey(full_domain, key, Faces()) || return missing

    global_to_local = Dict(global_cell => local_cell for (local_cell, global_cell) in enumerate(layer_cells))
    full_neighbors = full_domain[:neighbors]
    num_nnc = haskey(full_domain, :nnc) ? length(full_domain[:nnc].cells) : 0
    nfull_geom = size(full_neighbors, 2) - num_nnc
    full_values = full_domain[key]

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

function _local_nnc_from_full_domain(full_domain, layer_cells, submesh)
    haskey(full_domain, :nnc) || return missing
    nnc_full = full_domain[:nnc]
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
    return JutulDarcy.setup_nnc_connections(submesh, neighbors, trans_flow, trans_thermal)
end

"""
    load_norne_domain(; layer = 1)

Read Norne model and extract a single layer, returning a `Jutul.DataDomain`.
Returns the domain and parent mesh information for handling inactive cells.
"""
function load_norne_domain(; layer = 1)
    norne_dir = GeoEnergyIO.test_input_file_path("NORNE_NOHYST")
    data_pth = joinpath(norne_dir, "NORNE_NOHYST.DATA")
    data = parse_data_file(data_pth)
    
    # Get full reservoir domain
    full_domain = reservoir_domain(data)
    mesh = physical_representation(full_domain)
    
    # Extract cells from the specified layer
    nx, ny, nz = grid_dims_ijk(mesh)
    layer_cells = Int[]
    parent_ijk = Tuple{Int, Int, Int}[]  # Store parent IJK for each extracted cell
    
    for cell in 1:number_of_cells(mesh)
        i, j, k = cell_ijk(mesh, cell)
        if k == layer
            push!(layer_cells, cell)
            push!(parent_ijk, (i, j, k))
        end
    end
    
    # Extract submesh for this layer
    submesh = extract_submesh(mesh, layer_cells)
    
    # Extract properties for the layer
    poro = full_domain[:porosity][layer_cells]
    perm = full_domain[:permeability][:, layer_cells]

    base_domain = reservoir_domain(submesh, porosity = poro, permeability = perm)
    local_neighbors = base_domain[:neighbors]
    local_nnc = _local_nnc_from_full_domain(full_domain, layer_cells, submesh)
    local_mult = _local_face_field_from_full_domain(
        full_domain,
        layer_cells,
        local_neighbors,
        :transmissibility_multiplier;
        default = 1.0
    )
    local_override = _local_face_field_from_full_domain(
        full_domain,
        layer_cells,
        local_neighbors,
        :transmissibility_override;
        default = NaN
    )

    domain_kwargs = (
        porosity = poro,
        permeability = perm,
        nnc = local_nnc
    )
    if haskey(full_domain, :net_to_gross)
        domain_kwargs = merge(domain_kwargs, (net_to_gross = full_domain[:net_to_gross][layer_cells],))
    end
    if !ismissing(local_mult)
        domain_kwargs = merge(domain_kwargs, (transmissibility_multiplier = local_mult,))
    end
    if !ismissing(local_override)
        domain_kwargs = merge(domain_kwargs, (transmissibility_override = local_override,))
    end
    domain = reservoir_domain(submesh; domain_kwargs...)

    all_parent_ijk = [(i, j, k) for k in 1:nz for j in 1:ny for i in 1:nx]
    parent_centroids = fill(NaN, size(full_domain[:cell_centroids], 1), length(all_parent_ijk))
    for (local_cell, global_cell) in enumerate(mesh.cell_map)
        parent_centroids[:, global_cell] .= full_domain[:cell_centroids][:, local_cell]
    end

    # Return domain along with parent mesh info
    return (
        domain = domain,
        parent_mesh = mesh,
        parent_ijk = parent_ijk,
        all_parent_ijk = all_parent_ijk,
        parent_centroids = parent_centroids,
        layer_cells = layer_cells,
        nx = nx,
        ny = ny,
        nz = nz
    )
end

"""
    plot_matern_sample(domain, x; output_path = joinpath(@__DIR__, "matern_tpfa_spe10_sample.png"), title = "Matérn sample", colormap = :RdBu, figure_size = (1100, 450))

Plot the sampled Matérn field on the SPE10 layer and save it as a PNG.

Requires `CairoMakie` to be installed in the environment used to run this
example.
"""
function plot_matern_sample(
        domain,
        x;
        output_path = joinpath(@__DIR__, "matern_tpfa_spe10_sample.png"),
        title = "Matérn sample",
        colormap = :RdBu,
        figure_size = (1100, 450),
        robust_quantile = 0.98
    )
    nc = number_of_cells(domain)
    if length(x) > nc
        x = x[1:nc]
    end
    mesh = physical_representation(domain)
    fig = Figure(size = figure_size)
    if dim(mesh) == 3
        ax = Axis3(
            fig[1, 1],
            title = title,
            zreversed = true,
            azimuth = 0.0,
            elevation = 0.5π,
            perspectiveness = 0.0,
            aspect = (1.0, 1.0, 0.15)
        )
        hidezdecorations!(ax)
    else
        ax = Axis(fig[1, 1], title = title)
    end

    finite_values = x[isfinite.(x)]
    if isempty(finite_values)
        x_abs = 1.0
    else
        0 < robust_quantile <= 1 || throw(ArgumentError("robust_quantile must lie in (0, 1], got $robust_quantile."))
        abs_values = abs.(finite_values)
        if robust_quantile < 1
            x_abs = quantile(abs_values, robust_quantile)
        else
            x_abs = maximum(abs_values)
        end
        x_abs = max(x_abs, eps(Float64))
    end
    # Use a light-centered diverging scale so smooth structure is not hidden by a dark midpoint.
    plt = plot_cell_data!(ax, mesh, x, colormap = colormap, colorrange = (-x_abs, x_abs), shading = false)
    Colorbar(fig[1, 2], plt)

    save(output_path, fig)
    return (fig = fig, output_path = output_path)
end

"""
    plot_point_variogram(domain, idx, Σ; output_path = joinpath(@__DIR__, "matern_tpfa_spe10_variogram.png"), title = "Point variogram from subset covariance", figure_size = (900, 450), parent_mesh = missing, selected_cells_in_parent = missing)

Plot the pointwise semivariogram implied by the subset covariance matrix as a
function of lag in the y-direction, starting from the first cell in idx.
Assumes idx contains cells sorted along a direction from a reference point.

If parent_mesh and selected_cells_in_parent are provided, handles inactive cells
by inserting NaN values in the plot for gaps.
"""
function plot_point_variogram(
        domain,
        idx,
        Σ;
        output_path = joinpath(@__DIR__, "matern_tpfa_spe10_variogram.png"),
        title = "Point variogram in y-direction",
        figure_size = (900, 450),
        parent_mesh = missing,
        selected_cells_in_parent = missing
    )
    # Use the first cell in idx as the reference (should be the center cell)
    ref_cell = first(idx)
    ref_idx = 1
    
    # Get cell centroids
    cc = domain[:cell_centroids]
    ref_point = cc[:, ref_cell]
    
    # Get covariance row and variances for the reference cell
    cov_row = vec(Σ[ref_idx, :])
    vars = diag(Σ)
    ref_var = vars[ref_idx]
    
    # Compute variogram for all cells in idx
    lags = Float64[]
    γ_values = Float64[]
    
    for (i, cell) in enumerate(idx)
        # Calculate spatial lag distance
        cell_point = cc[:, cell]
        lag = norm(cell_point - ref_point)
        
        # Calculate semivariogram value
        γ = 0.5 * (ref_var + vars[i] - 2 * cov_row[i])
        
        push!(lags, lag)
        push!(γ_values, γ)
    end

    order = sortperm(lags)
    lags = lags[order]
    γ_values = γ_values[order]

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1],
        title = "$title (from grid center)",
        xlabel = "Lag distance [m]",
        ylabel = "Semivariogram"
    )
    
    # Use scatter + lines with NaN handling for gaps
    # Create segments that skip over NaN values
    if any(isnan.(γ_values))
        # Plot with scatter to show gaps
        valid_mask = .!isnan.(γ_values)
        scatter!(ax, lags[valid_mask], γ_values[valid_mask], 
                color = :black, markersize = 8)
        # Connect valid points with lines but break at NaN
        for i in 1:length(lags)-1
            if !isnan(γ_values[i]) && !isnan(γ_values[i+1])
                lines!(ax, [lags[i], lags[i+1]], [γ_values[i], γ_values[i+1]], 
                      color = :black, linewidth = 2)
            end
        end
    else
        # No gaps, use simple line plot
        lines!(ax, lags, γ_values, color = :black, linewidth = 2)
        scatter!(ax, lags, γ_values, color = :black, markersize = 6)
    end

    save(output_path, fig)
    return (fig = fig, output_path = output_path, reference_cell = ref_cell)
end

function fill_missing_lags(lags, positions)
    filled = copy(lags)
    known = findall(.!isnan.(filled))
    isempty(known) && return collect(positions .- first(positions))

    if length(known) >= 2
        first_known = known[1]
        second_known = known[2]
        slope = (filled[second_known] - filled[first_known]) / (positions[second_known] - positions[first_known])
        for i in (first_known - 1):-1:1
            filled[i] = filled[i + 1] - slope * (positions[i + 1] - positions[i])
        end

        last_known = known[end]
        penultimate_known = known[end - 1]
        slope = (filled[last_known] - filled[penultimate_known]) / (positions[last_known] - positions[penultimate_known])
        for i in (last_known + 1):length(filled)
            filled[i] = filled[i - 1] + slope * (positions[i] - positions[i - 1])
        end
    else
        filled .= filled[known[1]]
    end

    known = findall(.!isnan.(filled))
    for (left, right) in zip(known[1:end-1], known[2:end])
        right == left + 1 && continue
        slope = (filled[right] - filled[left]) / (positions[right] - positions[left])
        for i in (left + 1):(right - 1)
            filled[i] = filled[left] + slope * (positions[i] - positions[left])
        end
    end
    return filled
end

function plot_variogram_series(
        lags,
        γ_values;
        output_path = joinpath(@__DIR__, "matern_tpfa_spe10_variogram.png"),
        title = "Point variogram in y-direction",
        figure_size = (900, 450),
        reservoir_end_lag = nothing,
        halo_start_lag = nothing
    )
    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1],
        title = "$title (from grid center)",
        xlabel = "Lag distance [m]",
        ylabel = "Semivariogram"
    )

    curve_mask = isfinite.(lags) .& .!isnan.(γ_values)
    curve_lags = lags[curve_mask]
    if !isnothing(halo_start_lag) && isfinite(halo_start_lag) && !isempty(curve_lags) && maximum(curve_lags) > halo_start_lag
        vspan!(ax, halo_start_lag, maximum(curve_lags), color = (:gray85, 0.35))
        vlines!(ax, [halo_start_lag], color = :gray25, linestyle = :dot, linewidth = 2)
    end
    if !isnothing(reservoir_end_lag) && isfinite(reservoir_end_lag)
        vlines!(ax, [reservoir_end_lag], color = :firebrick, linestyle = :dash, linewidth = 2)
    end

    valid_mask = .!isnan.(γ_values)
    scatter!(ax, lags[valid_mask], γ_values[valid_mask], color = :black, markersize = 8)
    for i in 1:(length(lags) - 1)
        if !isnan(γ_values[i]) && !isnan(γ_values[i + 1])
            lines!(ax, [lags[i], lags[i + 1]], [γ_values[i], γ_values[i + 1]], color = :black, linewidth = 2)
        end
    end

    save(output_path, fig)
    return (fig = fig, output_path = output_path)
end

function _variogram_boundary_side(direction, forward)
    if direction == :y
        return forward ? :north : :south
    elseif direction == :x
        return forward ? :east : :west
    else
        error("Unsupported variogram direction $direction")
    end
end

function parent_grid_variogram(domain, Q, parent_info; direction = :y, forward = true, halo_meta = nothing)
    direction == :y || error("Only y-direction variograms are implemented.")

    active_ijk = collect(parent_info.ijk)
    isempty(active_ijk) && error("No active cells available for parent-grid variogram.")

    center_target = (
        round(Int, (parent_info.nx + 1) / 2),
        round(Int, (parent_info.ny + 1) / 2),
        active_ijk[1][3]
    )
    center_idx = argmin(map(active_ijk) do ijk
        (ijk[1] - center_target[1])^2 + (ijk[2] - center_target[2])^2
    end)
    i_ref, j_ref, k_ref = active_ijk[center_idx]

    active_lookup = Dict(ijk => i for (i, ijk) in enumerate(active_ijk))
    all_lookup = Dict(ijk => i for (i, ijk) in enumerate(parent_info.all_parent_ijk))
    parent_cc = parent_info.parent_centroids
    cc = domain[:cell_centroids]

    line_positions = collect(forward ? (j_ref:parent_info.ny) : (j_ref:-1:1))
    idx_variogram = Int[]
    for j in line_positions
        ijk_key = (i_ref, j, k_ref)
        haskey(active_lookup, ijk_key) || continue
        push!(idx_variogram, active_lookup[ijk_key])
    end
    isempty(idx_variogram) && error("Failed to identify active cells along the parent-grid variogram line.")

    reservoir_idx = copy(idx_variogram)
    halo_chain = Int[]
    if !isnothing(halo_meta)
        boundary_side = _variogram_boundary_side(direction, forward)
        boundary_cell = last(reservoir_idx)
        halo_chain = get(halo_meta.cell_side_to_chain, (boundary_cell, boundary_side), Int[])
    end

    idx_variogram = vcat(reservoir_idx, halo_chain)
    Σ = covariance_subset_from_precision(Q; idx = idx_variogram)
    subset_lookup = Dict(cell => i for (i, cell) in enumerate(idx_variogram))

    ref_point = cc[:, center_idx]
    ref_var = Σ[1, 1]
    lags = fill(NaN, length(line_positions))
    γ_values = fill(NaN, length(line_positions))

    for (pos, j) in enumerate(line_positions)
        ijk_key = (i_ref, j, k_ref)
        if haskey(active_lookup, ijk_key)
            cell_idx = active_lookup[ijk_key]
            subset_idx = subset_lookup[cell_idx]
            cell_point = cc[:, cell_idx]
            cov_val = Σ[1, subset_idx]
            cell_var = Σ[subset_idx, subset_idx]
            lags[pos] = norm(cell_point - ref_point)
            γ_values[pos] = 0.5 * (ref_var + cell_var - 2 * cov_val)
        elseif haskey(all_lookup, ijk_key)
            parent_cell = all_lookup[ijk_key]
            parent_point = parent_cc[:, parent_cell]
            if all(isfinite, parent_point)
                lags[pos] = norm(parent_point - ref_point)
            end
        end
    end

    inactive_count = count(isnan, γ_values)
    lags = fill_missing_lags(lags, Float64.(line_positions))
    reservoir_end_lag = norm(cc[:, last(reservoir_idx)] - ref_point)
    halo_start_lag = lags[end]
    halo_lag_shift = max(halo_start_lag - reservoir_end_lag, 0.0)

    if !isempty(halo_chain)
        ext_centroids = halo_meta.ext_centroids
        halo_lags = Float64[]
        halo_γ = Float64[]
        for cell in halo_chain
            subset_idx = subset_lookup[cell]
            cell_point = ext_centroids[:, cell]
            cov_val = Σ[1, subset_idx]
            cell_var = Σ[subset_idx, subset_idx]
            push!(halo_lags, norm(cell_point - ref_point) + halo_lag_shift)
            push!(halo_γ, 0.5 * (ref_var + cell_var - 2 * cov_val))
        end
        lags = vcat(lags, halo_lags)
        γ_values = vcat(γ_values, halo_γ)
    end

    return (
        idx_variogram = idx_variogram,
        Σ = Σ,
        lags = lags,
        γ_values = γ_values,
        reference_cell = center_idx,
        reference_ijk = (i_ref, j_ref, k_ref),
        inactive_count = inactive_count,
        reservoir_end_lag = reservoir_end_lag,
        halo_start_lag = halo_start_lag,
        halo_count = length(halo_chain)
    )
end

function sample_from_precision(Q::Symmetric; rng = Random.default_rng())
    ξ = randn(rng, size(Q, 1))
    return cholesky(Q) \ ξ
end

function covariance_subset_from_precision(Q::Symmetric; idx::AbstractVector{<:Integer})
    nc = size(Q, 1)
    m = length(idx)
    E = zeros(Float64, nc, m)
    for (k, j) in enumerate(idx)
        E[j, k] = 1.0
    end
    U = cholesky(Q) \ E
    return Symmetric(U[idx, :])
end

function normalized_basis_from_coord(domain, dim)
    x = collect(domain[:cell_centroids][dim, :])
    xc = x .- mean(x)
    s = maximum(abs, xc)
    if s < 1e-12
        return zeros(length(x), 1)
    end
    return reshape(xc ./ s, :, 1)
end

function lattice_anchors(domain; nx = 4, ny = 4)
    cc = domain[:cell_centroids]
    xs = range(minimum(cc[1, :]), maximum(cc[1, :]), length = nx)
    ys = range(minimum(cc[2, :]), maximum(cc[2, :]), length = ny)
    anchors = Int[]
    for x in xs, y in ys
        d2 = [(cc[1, i] - x)^2 + (cc[2, i] - y)^2 for i in axes(cc, 2)]
        push!(anchors, argmin(d2))
    end
    return unique(anchors)
end

function build_prior_spec(domain; mode = :stationary)
    nc = number_of_cells(domain)
    ρ0 = 25.0 * si_unit(:meter)
    σ0 = sqrt(3.0)
    Bx = normalized_basis_from_coord(domain, 1)
    By = normalized_basis_from_coord(domain, 2)
    Bc = ones(nc, 1)

    if mode == :stationary
        prior = MaternSPDE2Prior(nc; ρ0 = ρ0, σ0 = σ0)
        θ = NamedTuple()
    elseif mode == :anisotropic
        prior = MaternSPDE2Prior(nc; ρ0 = ρ0, σ0 = σ0, anisotropy_u_basis = Bc)
        θ = (anisotropy_u = [0.6],)
    elseif mode == :nonstationary
        prior = MaternSPDE2Prior(
            nc;
            ρ0 = 0.85*ρ0,
            σ0 = σ0,
            range_basis = Bx,
            sd_basis = By,
            anisotropy_u_basis = Bx,
            anisotropy_v_basis = By
        )
        θ = (
            range = [0.35],
            sd = [0.15],
            anisotropy_u = [0.45],
            anisotropy_v = [0.15]
        )
    else
        error("Unknown prior mode: $mode. Choose :stationary, :anisotropic or :nonstationary.")
    end
    return prior, θ
end

function plot_prior_diagnostics(variance_diag, range_diag; output_path, title = "Matérn diagnostics", figure_size = (1100, 450))
    n = length(variance_diag.anchors)
    ix = 1:n
    fig = Figure(size = figure_size)

    ax1 = Axis(fig[1, 1], title = "$title: Variance", xlabel = "Anchor index", ylabel = "Variance")
    lines!(ax1, ix, variance_diag.target, color = :black, linestyle = :dash, label = "Target")
    lines!(ax1, ix, variance_diag.realized, color = :dodgerblue, linewidth = 2, label = "Realized")
    scatter!(ax1, ix, variance_diag.realized, color = :dodgerblue, markersize = 8)
    axislegend(ax1, position = :rb)

    ax2 = Axis(fig[1, 2], title = "$title: Directional ranges", xlabel = "Anchor index", ylabel = "Range [m]")
    lines!(ax2, ix, range_diag.target_major, color = :black, linestyle = :dash, label = "Target major")
    lines!(ax2, ix, range_diag.target_minor, color = :gray40, linestyle = :dash, label = "Target minor")
    lines!(ax2, ix, range_diag.realized_major, color = :firebrick, linewidth = 2, label = "Realized major")
    lines!(ax2, ix, range_diag.realized_minor, color = :forestgreen, linewidth = 2, label = "Realized minor")
    scatter!(ax2, ix, range_diag.realized_major, color = :firebrick, markersize = 8)
    scatter!(ax2, ix, range_diag.realized_minor, color = :forestgreen, markersize = 8)
    axislegend(ax2, position = :rb)

    save(output_path, fig)
    return (fig = fig, output_path = output_path)
end

function main(
        ;
        model = "spe10",  # "spe10" or "norne"
        layer = 1,
        prior_mode = "stationary",
        compensate = false,
        edge_mode = :none,
        halo_layers = 3,
        halo_growth = 1.5,
        plot_output = nothing,
        variogram_output = nothing,
        diagnostics_output = nothing
    )
    # --- 1) Read + initialize grid ---
    parent_info = nothing
    if lowercase(model) == "spe10"
        result = load_spe10_domain(layers = layer:layer)
        domain = result.domain
        parent_info = (
            mesh = result.parent_mesh,
            ijk = result.parent_ijk,
            all_parent_ijk = result.all_parent_ijk,
            parent_centroids = result.parent_centroids,
            nx = result.nx,
            ny = result.ny,
            nz = result.nz,
            layer_cells = result.layer_cells
        )
        model_name = "SPE10"
    elseif lowercase(model) == "norne"
        result = load_norne_domain(layer = layer)
        domain = result.domain
        parent_info = (
            mesh = result.parent_mesh,
            ijk = result.parent_ijk,
            all_parent_ijk = result.all_parent_ijk,
            parent_centroids = result.parent_centroids,
            nx = result.nx,
            ny = result.ny,
            nz = result.nz,
            layer_cells = result.layer_cells
        )
        model_name = "Norne"
    else
        error("Unknown model: $model. Choose 'spe10' or 'norne'.")
    end
    
    # Set default output paths if not provided
    edge_mode_sym = edge_mode isa Symbol ? edge_mode : Symbol(lowercase(String(edge_mode)))
    edge_suffix = edge_mode_sym == :halo ? "_halo" : ""
    if isnothing(plot_output)
        plot_output = joinpath(@__DIR__, "matern_tpfa_$(lowercase(model))_layer$(layer)_sample$(edge_suffix).png")
    end
    if isnothing(variogram_output)
        variogram_output = joinpath(@__DIR__, "matern_tpfa_$(lowercase(model))_layer$(layer)_variogram$(edge_suffix).png")
    end
    if isnothing(diagnostics_output)
        diagnostics_output = joinpath(@__DIR__, "matern_tpfa_$(lowercase(model))_layer$(layer)_$(lowercase(prior_mode))_diagnostics$(edge_suffix).png")
    end

    prior_mode_sym = Symbol(lowercase(prior_mode))
    prior, θ = build_prior_spec(domain; mode = prior_mode_sym)
    anchors = lattice_anchors(domain)
    halo_spec = nothing
    compensation_label = :none
    if edge_mode_sym == :halo
        halo_spec = MaternHaloSpec(layers = halo_layers, growth = halo_growth)
    elseif edge_mode_sym != :none
        throw(ArgumentError("Unsupported edge_mode $edge_mode_sym. Use :none or :halo."))
    end
    if compensate
        if isnothing(halo_spec)
            matern_sd_compensation!(prior, domain; anchors = anchors)
            matern_range_compensation!(prior, domain; anchors = anchors, corr_level = 0.2)
            compensation_label = :local_sd_and_range
        else
            matern_sd_compensation!(prior, domain; anchors = :all, mode = :mean, halo = halo_spec)
            compensation_label = :mean_sd_with_halo
        end
    end

    op = matern_spde_operator(domain, prior, θ; compensated = compensate, halo = halo_spec)
    N = domain[:neighbors]
    T = op.transmissibilities
    V = domain[:volumes]
    nc = length(V)
    nf = size(N, 2)
    x = sample_from_precision(op.Q)
    if !isempty(op.halo_idx)
        x[op.halo_idx] .= NaN
    end
    plot_result = plot_matern_sample(domain, x, output_path = plot_output, title = "Matérn sample - $model_name Layer $layer")

    variogram_data = parent_grid_variogram(domain, op.Q, parent_info; halo_meta = op.halo_meta)
    idx_variogram = variogram_data.idx_variogram
    Σ = variogram_data.Σ
    variogram_plot_base = plot_variogram_series(
        variogram_data.lags,
        variogram_data.γ_values;
        output_path = variogram_output,
        title = "Variogram - $model_name Layer $layer",
        reservoir_end_lag = variogram_data.reservoir_end_lag,
        halo_start_lag = variogram_data.halo_start_lag
    )
    variogram_plot = (
        fig = variogram_plot_base.fig,
        output_path = variogram_plot_base.output_path,
        reference_cell = variogram_data.reference_cell
    )
    i_ref, j_ref, k_ref = variogram_data.reference_ijk
    println("Reference cell in parent grid: (i=$i_ref, j=$j_ref, k=$k_ref)")
    println("Variogram computed with $(length(variogram_data.lags)) positions ($(variogram_data.inactive_count) inactive, $(variogram_data.halo_count) halo)")

    variance_diag = matern_realized_variance(domain, prior, θ; anchors = anchors, compensated = compensate, halo = halo_spec)
    range_diag = matern_realized_ranges(domain, prior, θ; anchors = anchors, compensated = compensate, corr_level = 0.2, halo = halo_spec)
    diagnostics_plot = plot_prior_diagnostics(
        variance_diag,
        range_diag;
        output_path = diagnostics_output,
        title = "Matérn diagnostics - $(uppercasefirst(prior_mode))"
    )

    println("$model_name domain (layer $layer): nc=$nc, nf=$nf")
    println("Full SPDE system size: $(size(op.Q, 1)) cells ($(length(op.halo_idx)) halo)")
    println("TPFA stencil quantities: neighbors N (2×nf), transmissibilities T (nf), volumes V (nc)")
    println("Prior mode: $(prior_mode_sym), compensation = $compensation_label, edge_mode = $edge_mode_sym")
    finite_sample = x[isfinite.(x)]
    println("Sample x: mean=$(mean(finite_sample)) std=$(std(finite_sample))")
    println("Saved sample plot to $(plot_result.output_path)")
    println("Saved variogram plot to $(variogram_plot.output_path) (reference cell = $(variogram_plot.reference_cell))")
    println("Saved diagnostics plot to $(diagnostics_plot.output_path)")
    println("Computed covariance Σ for $(length(idx_variogram)) cells along the parent-grid y-direction, including halo if present")

    return (
        domain = domain,
        N = N,
        T = T,
        V = V,
        L_H = op.L_H,
        κ = op.fields.κ_eff,
        τ = op.fields.τ,
        Q = op.Q,
        halo_idx = op.halo_idx,
        interior_idx = op.interior_idx,
        halo_meta = op.halo_meta,
        prior = prior,
        θ = θ,
        x = x,
        Σ = Σ,
        idx_variogram = idx_variogram,
        plot_path = plot_result.output_path,
        variogram_path = variogram_plot.output_path,
        diagnostics_path = diagnostics_plot.output_path,
        variogram_reference_cell = variogram_plot.reference_cell
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command-line arguments or use defaults
    # Usage: julia matern_tpfa_spe10.jl [model] [layer] [prior_mode] [compensate] [edge_mode] [halo_layers] [halo_growth]
    model = length(ARGS) >= 1 ? ARGS[1] : "spe10"
    layer = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    prior_mode = length(ARGS) >= 3 ? ARGS[3] : "stationary"
    compensate = length(ARGS) >= 4 ? lowercase(ARGS[4]) in ("true", "1", "yes") : true
    edge_mode = length(ARGS) >= 5 ? ARGS[5] : "none"
    halo_layers = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
    halo_growth = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 1.5

    println("Running Matérn TPFA analysis for $model (layer $layer, mode $prior_mode, compensate=$compensate, edge_mode=$edge_mode)...")
    main(model = model, layer = layer, prior_mode = prior_mode, compensate = compensate, edge_mode = edge_mode, halo_layers = halo_layers, halo_growth = halo_growth)
end
