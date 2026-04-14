#!/usr/bin/env julia

using Jutul
using JutulDarcy
using Jutul: si_unit
using CairoMakie
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

function sample_from_precision(Q::Symmetric; rng = Random.default_rng())
    ξ = randn(rng, size(Q, 1))
    return cholesky(Q) \ ξ
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

function build_prior_spec_3d(domain)
    nc = number_of_cells(domain)
    ρ0 = 1500.0 * si_unit(:meter)
    σ0 = 1.0
    return MaternSPDE3Prior(nc; ρ0 = ρ0, σ0 = σ0)
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
        ax = Axis3(
            figpos,
            title = title,
            zreversed = true,
            azimuth = 0.0,
            elevation = 0.5π,
            perspectiveness = 0.0,
            aspect = (1.0, 1.0, 0.2)
        )
        hidezdecorations!(ax)
        return ax
    elseif orientation == :xz
        return Axis3(figpos, title = title, azimuth = 0.0, elevation = 0.0, perspectiveness = 0.0)
    elseif orientation == :yz
        return Axis3(figpos, title = title, azimuth = -0.5π, elevation = 0.0, perspectiveness = 0.0)
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
    for (ax, cells) in ((axk, cells_k), (axj, cells_j), (axi, cells_i))
        submesh = extract_submesh(mesh, cells)
        vals = values[cells]
        plt = plot_cell_data!(ax, submesh, vals, colormap = colormap, colorrange = (-color_abs, color_abs), shading = false)
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
    lines!(ax2, ix, axis_diag.target_x, color = :black, linestyle = :dash, label = "Target")
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

function main(
        ;
        compensate = false,
        halo_layers = 3,
        halo_growth = 1.5,
        sample_output = nothing,
        diagnostics_output = nothing
    )
    domain = load_norne_domain_3d()
    prior = build_prior_spec_3d(domain)
    halo_spec = MaternHaloSpec(layers = halo_layers, growth = halo_growth)
    anchors = lattice_anchors_3d(domain; nx = 3, ny = 3, nz = 2)

    if isnothing(sample_output)
        sample_output = joinpath(@__DIR__, "matern_tpfa_norne_3d_sample_halo.png")
    end
    if isnothing(diagnostics_output)
        diagnostics_output = joinpath(@__DIR__, "matern_tpfa_norne_3d_diagnostics_halo.png")
    end

    if compensate
        matern_sd_compensation!(prior, domain; mode = :mean, anchors = anchors, halo = halo_spec)
    end

    op = matern_spde_operator(domain, prior; compensated = compensate, halo = halo_spec)
    x = sample_from_precision(op.Q)
    x_masked = copy(x)
    x_masked[op.halo_idx] .= NaN

    sample_plot = plot_sample_slices(domain, x_masked; output_path = sample_output, title = "Norne 3D Matérn sample")
    variance_diag = matern_realized_variance(domain, prior; anchors = anchors, compensated = compensate, halo = halo_spec)
    axis_diag = matern_realized_axis_ranges(domain, prior; anchors = anchors, compensated = compensate, halo = halo_spec, corr_level = 0.2)
    diagnostics_plot = plot_prior_diagnostics_3d(variance_diag, axis_diag; output_path = diagnostics_output, title = "Norne 3D Matérn diagnostics")

    finite_sample = x_masked[1:number_of_cells(domain)]
    finite_sample = finite_sample[isfinite.(finite_sample)]
    println("Norne 3D domain: nc=$(number_of_cells(domain))")
    println("Full SPDE system size: $(size(op.Q, 1)) cells ($(length(op.halo_idx)) halo)")
    println("Compensation: $(compensate ? \"mean_sd\" : \"none\")")
    println("Sample x: mean=$(mean(finite_sample)) std=$(std(finite_sample))")
    println("Saved sample slices to $(sample_plot.output_path)")
    println("Saved diagnostics plot to $(diagnostics_plot.output_path)")

    return (
        domain = domain,
        prior = prior,
        operator = op,
        sample = x_masked,
        anchors = anchors,
        sample_plot = sample_plot.output_path,
        diagnostics_plot = diagnostics_plot.output_path
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    compensate = length(ARGS) >= 1 ? lowercase(ARGS[1]) in ("true", "1", "yes", "y") : false
    halo_layers = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    halo_growth = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.5
    main(compensate = compensate, halo_layers = halo_layers, halo_growth = halo_growth)
end
