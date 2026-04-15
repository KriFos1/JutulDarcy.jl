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

function load_norne_domain_3d()
    norne_dir = GeoEnergyIO.test_input_file_path("NORNE_NOHYST")
    data_pth = joinpath(norne_dir, "NORNE_NOHYST.DATA")
    data = parse_data_file(data_pth)
    return reservoir_domain(data)
end

function sample_matern(op; rng = Random.default_rng())
    n = size(op.K, 1)
    ξ = randn(rng, n)
    sqrtC_ξ = sqrt.(op.C.diag) .* ξ
    y = op.K \ sqrtC_ξ
    τ_vec = op.fields.τ
    return y ./ τ_vec
end

function lattice_anchors_3d(domain; nx = 3, ny = 3, nz = 2)
    cc = domain[:cell_centroids]
    xs = range(minimum(cc[1, :]), maximum(cc[1, :]), length = nx)
    ys = range(minimum(cc[2, :]), maximum(cc[2, :]), length = ny)
    zs = range(minimum(cc[3, :]), maximum(cc[3, :]), length = nz)
    anchors = Int[]
    for x in xs, y in ys, z in zs
        d2 = [(cc[1, i] - x)^2 + (cc[2, i] - y)^2 + (cc[3, i] - z)^2 for i in axes(cc, 2)]
        push!(anchors, argmin(d2))
    end
    return unique(anchors)
end

function build_prior_spec_3d(domain; aniso_a = 0.0, aniso_b = 0.0)
    nc = number_of_cells(domain)
    ρ0 = 1500.0 * si_unit(:meter)
    σ0 = 1.0
    has_aniso = aniso_a != 0.0 || aniso_b != 0.0
    if has_aniso
        Ba = ones(Float64, nc, 1)
        Bb = ones(Float64, nc, 1)
        prior = MaternSPDE3Prior(nc; ρ0 = ρ0, σ0 = σ0, anisotropy_a_basis = Ba, anisotropy_b_basis = Bb)
        return prior, (anisotropy_a = [Float64(aniso_a)], anisotropy_b = [Float64(aniso_b)])
    else
        prior = MaternSPDE3Prior(nc; ρ0 = ρ0, σ0 = σ0)
        return prior, NamedTuple()
    end
end

function _slice_indices(mesh)
    ijk = map(cell -> cell_ijk(mesh, cell), 1:number_of_cells(mesh))
    is = sort(unique(first.(ijk)))
    js = sort(unique(map(x -> x[2], ijk)))
    ks = sort(unique(last.(ijk)))
    i_mid = is[cld(length(is), 2)]
    j_mid = js[cld(length(js), 2)]
    k_mid = ks[cld(length(ks), 2)]
    return (ijk = ijk, i_mid = i_mid, j_mid = j_mid, k_mid = k_mid)
end

function _cells_on_slice(mesh, ijk_cache, axis::Symbol, index::Int)
    cells = Int[]
    for (cell, ijk) in enumerate(ijk_cache)
        if axis == :i && ijk[1] == index
            push!(cells, cell)
        elseif axis == :j && ijk[2] == index
            push!(cells, cell)
        elseif axis == :k && ijk[3] == index
            push!(cells, cell)
        end
    end
    return cells
end

function _robust_abs_scale(x; robust_quantile = 0.98)
    finite_values = x[isfinite.(x)]
    isempty(finite_values) && return 1.0
    0 < robust_quantile <= 1 || throw(ArgumentError("robust_quantile must lie in (0, 1], got $robust_quantile."))
    abs_values = abs.(finite_values)
    x_abs = robust_quantile < 1 ? quantile(abs_values, robust_quantile) : maximum(abs_values)
    return max(x_abs, eps(Float64))
end

function _slice_axis(figpos, title; orientation::Symbol)
    if orientation == :top
        return Axis(figpos, title = title, xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect())
    elseif orientation == :xz
        return Axis(figpos, title = title, xlabel = "x [m]", ylabel = "z [m]", yreversed = true, aspect = DataAspect())
    elseif orientation == :yz
        return Axis(figpos, title = title, xlabel = "y [m]", ylabel = "z [m]", yreversed = true, aspect = DataAspect())
    else
        error("Unsupported slice orientation $orientation")
    end
end

function plot_sample_slices(
        domain,
        x;
        output_path = joinpath(@__DIR__, "matern_tpfa_norne_3d_sample_halo.png"),
        title = "3D Matérn sample",
        colormap = :RdBu,
        figure_size = (1400, 900),
        robust_quantile = 0.98
    )
    mesh = physical_representation(domain)
    nc = number_of_cells(domain)
    values = length(x) > nc ? x[1:nc] : copy(x)
    cc = Matrix{Float64}(domain[:cell_centroids])
    cache = _slice_indices(mesh)
    cells_k = _cells_on_slice(mesh, cache.ijk, :k, cache.k_mid)
    cells_j = _cells_on_slice(mesh, cache.ijk, :j, cache.j_mid)
    cells_i = _cells_on_slice(mesh, cache.ijk, :i, cache.i_mid)
    color_abs = _robust_abs_scale(values; robust_quantile = robust_quantile)

    fig = Figure(size = figure_size)
    axk = _slice_axis(fig[1, 1], "$title: K = $(cache.k_mid)", orientation = :top)
    axj = _slice_axis(fig[1, 2], "$title: J = $(cache.j_mid)", orientation = :xz)
    axi = _slice_axis(fig[2, 1], "$title: I = $(cache.i_mid)", orientation = :yz)

    plt = nothing
    for (ax, cells, dims) in ((axk, cells_k, (1, 2)), (axj, cells_j, (1, 3)), (axi, cells_i, (2, 3)))
        plt = scatter!(
            ax,
            cc[dims[1], cells],
            cc[dims[2], cells],
            color = values[cells],
            colormap = colormap,
            colorrange = (-color_abs, color_abs),
            markersize = 9
        )
    end
    Colorbar(fig[1:2, 3], plt)

    save(output_path, fig)
    return (fig = fig, output_path = output_path)
end

function plot_prior_diagnostics_3d(variance_diag, axis_diag; output_path, title = "3D Matérn diagnostics", figure_size = (1200, 450))
    n = length(variance_diag.anchors)
    ix = 1:n
    fig = Figure(size = figure_size)

    ax1 = Axis(fig[1, 1], title = "$title: Variance", xlabel = "Anchor index", ylabel = "Variance")
    lines!(ax1, ix, variance_diag.target, color = :black, linestyle = :dash, label = "Target")
    lines!(ax1, ix, variance_diag.realized, color = :dodgerblue, linewidth = 2, label = "Realized")
    scatter!(ax1, ix, variance_diag.realized, color = :dodgerblue, markersize = 8)
    axislegend(ax1, position = :rb)

    ax2 = Axis(fig[1, 2], title = "$title: Axis ranges", xlabel = "Anchor index", ylabel = "Range [m]")
    lines!(ax2, ix, axis_diag.target_x, color = :firebrick, linestyle = :dash, label = "Target x")
    lines!(ax2, ix, axis_diag.target_y, color = :forestgreen, linestyle = :dash, label = "Target y")
    lines!(ax2, ix, axis_diag.target_z, color = :darkorange, linestyle = :dash, label = "Target z")
    lines!(ax2, ix, axis_diag.realized_x, color = :firebrick, linewidth = 2, label = "Realized x")
    lines!(ax2, ix, axis_diag.realized_y, color = :forestgreen, linewidth = 2, label = "Realized y")
    lines!(ax2, ix, axis_diag.realized_z, color = :darkorange, linewidth = 2, label = "Realized z")
    scatter!(ax2, ix, axis_diag.realized_x, color = :firebrick, markersize = 8)
    scatter!(ax2, ix, axis_diag.realized_y, color = :forestgreen, markersize = 8)
    scatter!(ax2, ix, axis_diag.realized_z, color = :darkorange, markersize = 8)
    axislegend(ax2, position = :rb)

    save(output_path, fig)
    return (fig = fig, output_path = output_path)
end

function covariance_subset_from_precision(Q::Symmetric; idx::AbstractVector{<:Integer})
    nc = size(Q, 1)
    m = length(idx)
    E = zeros(Float64, nc, m)
    for (k, j) in enumerate(idx)
        E[j, k] = 1.0
    end
    F = JutulDarcy._factor_precision_matrix(Q)
    U = F \ E
    return Symmetric(U[idx, :])
end

"""
    variogram_line_3d(domain, op; direction = :j)

Compute the semivariogram along one IJK axis from the grid center towards
higher indices plus the halo. Returns lags, γ values, and marker positions
for the first inactive cell and halo start.
"""
function variogram_line_3d(domain, op; direction::Symbol = :j)
    mesh = physical_representation(domain)
    nc = number_of_cells(domain)
    cc = Matrix{Float64}(domain[:cell_centroids])

    # Build IJK index
    ijk = map(cell -> cell_ijk(mesh, cell), 1:nc)
    is = sort(unique(first.(ijk)))
    js = sort(unique(map(x -> x[2], ijk)))
    ks = sort(unique(last.(ijk)))
    nx, ny, nz = grid_dims_ijk(mesh)

    i_mid = is[cld(length(is), 2)]
    j_mid = js[cld(length(js), 2)]
    k_mid = ks[cld(length(ks), 2)]

    active_lookup = Dict(ijk[c] => c for c in 1:nc)

    # Walk from center towards max index in the chosen direction
    if direction == :i
        line_positions = collect(i_mid:nx)
        make_key = pos -> (pos, j_mid, k_mid)
    elseif direction == :j
        line_positions = collect(j_mid:ny)
        make_key = pos -> (i_mid, pos, k_mid)
    elseif direction == :k
        line_positions = collect(k_mid:nz)
        make_key = pos -> (i_mid, j_mid, pos)
    else
        error("direction must be :i, :j, or :k, got $direction")
    end

    # Collect active cells along the line
    idx_reservoir = Int[]
    for pos in line_positions
        key = make_key(pos)
        if haskey(active_lookup, key)
            push!(idx_reservoir, active_lookup[key])
        end
    end
    isempty(idx_reservoir) && error("No active cells found along $direction from center.")

    ref_cell = first(idx_reservoir)
    ref_point = cc[:, ref_cell]
    last_active = last(idx_reservoir)

    # Find halo chain from the last active cell along this direction
    halo_chain = Int[]
    halo_meta = op.halo_meta
    if !isnothing(halo_meta)
        # Find boundary face whose source cell is last_active and whose
        # outward direction aligns with the walk direction
        bn = Int.(vec(domain[:boundary_neighbors]))
        bc = Matrix{Float64}(domain[:boundary_centroids])
        face_to_chain = halo_meta.face_to_chain

        dir_vec = zeros(3)
        if direction == :i;     dir_vec[1] = 1.0
        elseif direction == :j; dir_vec[2] = 1.0
        else                    dir_vec[3] = 1.0
        end

        best_face = 0
        best_dot = -Inf
        for f in eachindex(bn)
            bn[f] == last_active || continue
            haskey(face_to_chain, f) || continue
            any(!isfinite, bc[:, f]) && continue
            outward = bc[:, f] .- cc[:, last_active]
            d = dot(outward, dir_vec)
            if d > best_dot
                best_dot = d
                best_face = f
            end
        end
        if best_face > 0
            halo_chain = face_to_chain[best_face]
        end
    end

    # Compute covariance subset for reservoir + halo cells along the line
    idx_all = vcat(idx_reservoir, halo_chain)
    Σ = covariance_subset_from_precision(op.Q; idx = idx_all)
    ref_var = Σ[1, 1]

    # Reservoir lags and γ
    lags = Float64[]
    γ_values = Float64[]
    first_inactive_lag = NaN

    active_set = Set(idx_reservoir)
    for (pos_idx, pos) in enumerate(line_positions)
        key = make_key(pos)
        if haskey(active_lookup, key)
            cell = active_lookup[key]
            cell_point = cc[:, cell]
            lag = norm(cell_point - ref_point)
            # Find position of this cell in idx_all
            subset_idx = findfirst(==(cell), idx_all)
            cov_val = Σ[1, subset_idx]
            cell_var = Σ[subset_idx, subset_idx]
            push!(lags, lag)
            push!(γ_values, 0.5 * (ref_var + cell_var - 2 * cov_val))
        else
            # Inactive cell: insert NaN gap
            if isnan(first_inactive_lag) && !isempty(lags)
                first_inactive_lag = lags[end]
            end
            push!(lags, NaN)
            push!(γ_values, NaN)
        end
    end

    # Reservoir end lag
    reservoir_end_lag = norm(cc[:, last_active] - ref_point)

    # Halo lags and γ
    halo_start_lag = NaN
    if !isempty(halo_chain)
        ext_centroids = halo_meta.ext_centroids
        for (k, cell) in enumerate(halo_chain)
            subset_idx = findfirst(==(cell), idx_all)
            cell_point = ext_centroids[:, cell]
            lag = norm(cell_point - ref_point)
            cov_val = Σ[1, subset_idx]
            cell_var = Σ[subset_idx, subset_idx]
            if k == 1
                halo_start_lag = lag
            end
            push!(lags, lag)
            push!(γ_values, 0.5 * (ref_var + cell_var - 2 * cov_val))
        end
    end

    return (
        lags = lags,
        γ = γ_values,
        reservoir_end_lag = reservoir_end_lag,
        halo_start_lag = halo_start_lag,
        first_inactive_lag = first_inactive_lag,
        direction = direction
    )
end

function plot_variogram_3d(
        domain,
        op;
        output_path = joinpath(@__DIR__, "matern_tpfa_norne_3d_variogram.png"),
        figure_size = (1500, 450)
    )
    vario_i = variogram_line_3d(domain, op; direction = :i)
    vario_j = variogram_line_3d(domain, op; direction = :j)
    vario_k = variogram_line_3d(domain, op; direction = :k)

    fig = Figure(size = figure_size)
    colors = (:firebrick, :forestgreen, :darkorange)
    labels = ("I-direction", "J-direction", "K-direction")

    for (col, vario, color, label) in zip(1:3, (vario_i, vario_j, vario_k), colors, labels)
        ax = Axis(
            fig[1, col],
            title = "Variogram: $label",
            xlabel = "Lag distance [m]",
            ylabel = "Semivariogram"
        )

        lags = vario.lags
        γ = vario.γ

        # Shade halo region
        valid_mask = isfinite.(lags) .& isfinite.(γ)
        if isfinite(vario.halo_start_lag) && any(valid_mask)
            max_lag = maximum(lags[valid_mask])
            if max_lag > vario.halo_start_lag
                vspan!(ax, vario.halo_start_lag, max_lag, color = (:gray85, 0.4))
            end
        end

        # Vertical lines
        if isfinite(vario.halo_start_lag)
            vlines!(ax, [vario.halo_start_lag], color = :gray25, linestyle = :dot, linewidth = 2, label = "Halo start")
        end
        if isfinite(vario.first_inactive_lag)
            vlines!(ax, [vario.first_inactive_lag], color = :purple, linestyle = :dashdot, linewidth = 2, label = "First inactive")
        end

        # Plot all variogram values
        scatter!(ax, lags[valid_mask], γ[valid_mask], color = color, markersize = 7)
        # Connect consecutive valid points with lines
        for k in 1:(length(lags) - 1)
            if isfinite(lags[k]) && isfinite(lags[k + 1]) && isfinite(γ[k]) && isfinite(γ[k + 1])
                lines!(ax, [lags[k], lags[k + 1]], [γ[k], γ[k + 1]], color = color, linewidth = 2)
            end
        end

        axislegend(ax, position = :rb)
    end

    save(output_path, fig)
    println("Saved variogram plot to $output_path")
    return (fig = fig, output_path = output_path)
end

function main(
        ;
        compensate = false,
        halo_layers = 3,
        halo_growth = 1.5,
        aniso_a = 0.0,
        aniso_b = 0.0,
        sample_output = nothing,
        diagnostics_output = nothing,
        variogram_output = nothing
    )
    domain = load_norne_domain_3d()
    prior, θ = build_prior_spec_3d(domain; aniso_a = aniso_a, aniso_b = aniso_b)
    halo_spec = MaternHaloSpec(layers = halo_layers, growth = halo_growth)
    anchors = lattice_anchors_3d(domain; nx = 3, ny = 3, nz = 2)

    if isnothing(sample_output)
        sample_output = joinpath(@__DIR__, "matern_tpfa_norne_3d_sample_halo.png")
    end
    if isnothing(diagnostics_output)
        diagnostics_output = joinpath(@__DIR__, "matern_tpfa_norne_3d_diagnostics_halo.png")
    end
    if isnothing(variogram_output)
        variogram_output = joinpath(@__DIR__, "matern_tpfa_norne_3d_variogram.png")
    end

    if compensate
        matern_sd_compensation!(prior, domain, θ; mode = :mean, anchors = anchors, halo = halo_spec)
    end

    op = matern_spde_operator(domain, prior, θ; compensated = compensate, halo = halo_spec)
    x = sample_matern(op)
    x_masked = copy(x)
    x_masked[op.halo_idx] .= NaN

    sample_plot = plot_sample_slices(domain, x_masked; output_path = sample_output, title = "Norne 3D Matérn sample")
    variance_diag = matern_realized_variance(domain, prior, θ; anchors = anchors, compensated = compensate, halo = halo_spec)
    axis_diag = matern_realized_axis_ranges(domain, prior, θ; anchors = anchors, compensated = compensate, halo = halo_spec, corr_level = 0.2)
    diagnostics_plot = plot_prior_diagnostics_3d(variance_diag, axis_diag; output_path = diagnostics_output, title = "Norne 3D Matérn diagnostics")
    variogram_plot = plot_variogram_3d(domain, op; output_path = variogram_output)

    has_aniso = aniso_a != 0.0 || aniso_b != 0.0
    finite_sample = x_masked[1:number_of_cells(domain)]
    finite_sample = finite_sample[isfinite.(finite_sample)]
    println("Norne 3D domain: nc=$(number_of_cells(domain))")
    println("Full SPDE system size: $(size(op.Q, 1)) cells ($(length(op.halo_idx)) halo)")
    println("Compensation: $(compensate ? "mean_sd" : "none")")
    if has_aniso
        fields = matern_parameter_fields(prior, θ)
        ρ0 = prior.ρ0
        println("Anisotropy: a=$aniso_a, b=$aniso_b")
        println("  Effective ranges: ρ_I=$(ρ0*exp(aniso_a/2)), ρ_J=$(ρ0*exp(aniso_b/2)), ρ_K=$(ρ0*exp(-(aniso_a+aniso_b)/2))")
    end
    println("Sample x: mean=$(mean(finite_sample)) std=$(std(finite_sample))")
    println("Saved sample slices to $(sample_plot.output_path)")
    println("Saved diagnostics plot to $(diagnostics_plot.output_path)")

    return (
        domain = domain,
        prior = prior,
        θ = θ,
        operator = op,
        sample = x_masked,
        anchors = anchors,
        sample_plot = sample_plot.output_path,
        diagnostics_plot = diagnostics_plot.output_path,
        variogram_plot = variogram_plot.output_path
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    compensate = length(ARGS) >= 1 ? lowercase(ARGS[1]) in ("true", "1", "yes", "y") : false
    halo_layers = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    halo_growth = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.5
    aniso_a = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.0
    aniso_b = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.0
    main(compensate = compensate, halo_layers = halo_layers, halo_growth = halo_growth, aniso_a = aniso_a, aniso_b = aniso_b)
end
