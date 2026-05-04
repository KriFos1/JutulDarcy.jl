#!/usr/bin/env julia

using Jutul
using JutulDarcy
using CairoMakie
using LinearAlgebra
using Random
using Statistics
using GeoEnergyIO

function resolve_example_input(spec::AbstractString)
    lower = lowercase(spec)
    if lower == "spe10"
        return JutulDarcy.SPE10.setup_reservoir(remove_cells = true)
    elseif lower == "spe1"
        return GeoEnergyIO.test_input_file_path("SPE1", "SPE1.DATA")
    elseif lower == "norne"
        norne_dir = GeoEnergyIO.test_input_file_path("NORNE_NOHYST")
        return joinpath(norne_dir, "NORNE_NOHYST.DATA")
    else
        return spec
    end
end

function sample_from_precision(Q::Symmetric; rng = Random.default_rng())
    F = JutulDarcy._factor_precision_matrix(Q)
    ξ = randn(rng, size(Q, 1))
    y = F.U \ ξ
    x = similar(y)
    x[F.p] .= y
    return x
end

function layer_reference_range(layer::ReservoirLayerDomain)
    cc = layer.domain[:cell_centroids]
    lx = maximum(cc[1, :]) - minimum(cc[1, :])
    ly = maximum(cc[2, :]) - minimum(cc[2, :])
    return max(0.15 * min(lx, ly), 1e-6)
end

function build_stationary_prior(layer::ReservoirLayerDomain; ratio = 1.0, angle_deg = 0.0, diffusion_scheme::Symbol = :tpfa)
    prior = MaternSPDE2Prior(
        number_of_cells(layer.domain);
        range0 = layer_reference_range(layer),
        sd0 = sqrt(3.0),
        angle0 = deg2rad(angle_deg),
        logratio0 = log(ratio),
        diffusion_scheme = diffusion_scheme
    )
    natural_fields = (
        basis = nothing,
        major = fill(layer_reference_range(layer)*sqrt(ratio), number_of_cells(layer.domain)),
        ratio = fill(Float64(ratio), number_of_cells(layer.domain)),
        sd = fill(sqrt(3.0), number_of_cells(layer.domain)),
        angle_deg = fill(Float64(angle_deg), number_of_cells(layer.domain))
    )
    return prior, NamedTuple(), natural_fields
end

function build_nonstationary_prior(
        layer::ReservoirLayerDomain;
        anisotropic::Bool,
        rotated::Bool,
        diffusion_scheme::Symbol
    )
    basis = matern_trend_basis(layer; axes = (:x, :y))
    xcol = vec(basis.matrix[:, findfirst(isequal(:x), basis.names)])
    ycol = vec(basis.matrix[:, findfirst(isequal(:y), basis.names)])

    # Scale the nonstationary variation intensity by the grid resolution.
    # On coarse grids the calibration anchors are close together and extreme
    # spatial gradients in the target fields make the Gauss–Newton iteration
    # ill-conditioned.  A gentle taper keeps the problem well-posed.
    nc = number_of_cells(layer.domain)
    taper = clamp(nc / 5000, 0.25, 1.0)   # full strength ≥ 5000 cells

    major = layer_reference_range(layer) .* exp.(taper .* (0.08 .* xcol .- 0.04 .* ycol))
    #ratio = anisotropic ? 1.25 .+ taper .* 0.25 .* (xcol .+ 1.0) ./ 2.0 : ones(Float64, nc)
    ratio = fill(anisotropic ? 2.5 : 1.0, nc)
    sd = exp.(taper .* 0.04 .* ycol)
    #angle_deg = rotated ? 0.0 .+ taper .* (5.0 .* ycol .+ 5.0 .* xcol) : zeros(Float64, nc)
    y01 = (ycol .- minimum(ycol)) ./ (maximum(ycol) - minimum(ycol))
    angle_deg = rotated ? 179.0 .* y01 : zeros(Float64, nc)

    range_geom = major ./ sqrt.(ratio)

    fit = fit_matern_fields(
        layer,
        basis;
        range0 = exp(mean(log.(range_geom))),
        sd0 = exp(mean(log.(sd))),
        angle0 = rotated ? mean(angle_deg) : 0.0,
        logratio0 = mean(log.(ratio)),
        major = major,
        ratio = ratio,
        sd = sd,
        angle = angle_deg,
        angle_unit = :degree
    )

    prior = MaternSPDE2Prior(
        number_of_cells(layer.domain);
        basis = fit.basis,
        range0 = fit.baselines.range0,
        sd0 = fit.baselines.sd0,
        angle0 = fit.baselines.angle0,
        logratio0 = fit.baselines.logratio0,
        diffusion_scheme = diffusion_scheme
    )

    natural_fields = (
        basis = fit.basis,
        major = major,
        ratio = ratio,
        sd = sd,
        angle_deg = angle_deg
    )
    return prior, fit.θ, natural_fields
end

function build_prior(layer::ReservoirLayerDomain; mode::Symbol = :nonstationary)
    if mode == :stationary
        return build_stationary_prior(layer)
    elseif mode == :stationary_anisotropic
        return build_stationary_prior(layer; ratio = 2.0, angle_deg = 0.0, diffusion_scheme = :tpfa)
    elseif mode == :stationary_rotated
        return build_stationary_prior(layer; ratio = 2.0, angle_deg = 30.0, diffusion_scheme = :fvm9)
    elseif mode == :nonstationary_isotropic
        return build_nonstationary_prior(layer; anisotropic = false, rotated = false, diffusion_scheme = :tpfa)
    elseif mode == :nonstationary_anisotropic || mode == :nonstationary
        return build_nonstationary_prior(layer; anisotropic = true, rotated = false, diffusion_scheme = :tpfa)
    elseif mode == :nonstationary_rotated
        return build_nonstationary_prior(layer; anisotropic = true, rotated = true, diffusion_scheme = :fvm9)
    else
        throw(ArgumentError("Unsupported mode $mode. Use :stationary, :stationary_anisotropic, :stationary_rotated, :nonstationary_isotropic, :nonstationary_anisotropic, or :nonstationary_rotated."))
    end
end

function summarize_diagnostics(variance_diag, range_diag)
    geom_target = sqrt.(range_diag.target_major .* range_diag.target_minor)
    geom_realized = sqrt.(range_diag.realized_major .* range_diag.realized_minor)
    ratio_realized = range_diag.realized_major ./ range_diag.realized_minor
    angle_mask = range_diag.target_ratio .>= 1.05

    summary = (
        max_variance_error = maximum(abs.(variance_diag.realized ./ variance_diag.target .- 1.0)),
        max_geometric_range_error = maximum(abs.(geom_realized ./ geom_target .- 1.0)),
        max_axis_ratio_error = maximum(abs.(ratio_realized ./ range_diag.target_ratio .- 1.0)),
        max_angle_error_deg = any(angle_mask) ? maximum(abs.(rad2deg.(JutulDarcy._orientation_difference.(range_diag.realized_angle[angle_mask], range_diag.target_angle[angle_mask])))) : 0.0
    )
    return summary
end

function parent_xy(layer::ReservoirLayerDomain)
    nx, ny = layer.parent_shape
    px = reshape(layer.parent_centroids[1, :], nx, ny)
    py = reshape(layer.parent_centroids[2, :], nx, ny)
    return vec(px[:, 1]), vec(py[1, :])
end

function field_to_parent_raster(layer::ReservoirLayerDomain, values)
    nx, ny = layer.parent_shape
    raster = fill(NaN, nx * ny)
    raster[layer.active_parent_linear] .= Float64.(values)
    return reshape(raster, nx, ny)
end

"""
    axial_semivariogram(op, layer, centre_cell, dir)

Evaluate semivariogram along axis `dir` from `centre_cell` on a regular lag grid.
For each lag h, sample both points (centre ± h*dir), pick nearest active cells,
and average their semivariogram values. This yields a dense, continuous curve.
"""
function axial_semivariogram(op, layer, centre_cell, dir)
    nc = op.interior_idx[end]
    cc = layer.domain[:cell_centroids]
    pts = Matrix{Float64}(cc[1:2, 1:nc])
    covcol = JutulDarcy._selected_qinv_columns(op.Q, [centre_cell])[1:nc, 1]
    ref_var = covcol[centre_cell]
    # Accept legacy scalar direction selectors (1 => x, 2 => y) used in ad-hoc debugging.
    d = if dir isa Integer
        dir == 1 ? [1.0, 0.0] : dir == 2 ? [0.0, 1.0] : throw(ArgumentError("Scalar direction must be 1 or 2, got $dir"))
    else
        collect(Float64.(dir))
    end

    if length(d) != 2 || any(!isfinite, d) || norm(d) <= eps(Float64)
        d = [1.0, 0.0]
    end

    # Ensure unit direction.
    d ./= max(norm(d), eps(Float64))

    ref_pt = pts[:, centre_cell]
    delta = pts .- ref_pt
    along = d[1] .* delta[1, :] .+ d[2] .* delta[2, :]
    along_abs = abs.(along[isfinite.(along)])
    isempty(along_abs) && return [0.0], [0.0]
    max_lag = maximum(along_abs)
    if !isfinite(max_lag) || max_lag <= 0
        return [0.0], [0.0]
    end

    nx, ny = layer.parent_shape
    px = layer.parent_centroids[1, :]
    py = layer.parent_centroids[2, :]
    px_f = px[isfinite.(px)]
    py_f = py[isfinite.(py)]
    lx = isempty(px_f) ? 1.0 : (maximum(px_f) - minimum(px_f))
    ly = isempty(py_f) ? 1.0 : (maximum(py_f) - minimum(py_f))
    dx = nx > 1 ? lx / (nx - 1) : lx
    dy = ny > 1 ? ly / (ny - 1) : ly
    step_raw = sqrt(abs(dx * dy))
    step = (isfinite(step_raw) && step_raw > 0.0) ? step_raw : 1e-6
    lag_ratio = max_lag / step
    nsteps = (isfinite(lag_ratio) && lag_ratio > 0.0) ? max(2, Int(floor(lag_ratio))) : 2
    lags = collect(range(0.0, max_lag; length = nsteps + 1))
    gamma = zeros(Float64, length(lags))

    for (k, h) in enumerate(lags)
        if k == 1
            gamma[k] = 0.0
            continue
        end

        p_plus = ref_pt .+ h .* d
        p_minus = ref_pt .- h .* d

        i_plus = argmin((pts[1, :] .- p_plus[1]).^2 .+ (pts[2, :] .- p_plus[2]).^2)
        i_minus = argmin((pts[1, :] .- p_minus[1]).^2 .+ (pts[2, :] .- p_minus[2]).^2)

        c_plus = covcol[i_plus]
        c_minus = covcol[i_minus]

        γ_plus = isfinite(c_plus) ? max(ref_var - c_plus, 0.0) : NaN
        γ_minus = isfinite(c_minus) ? max(ref_var - c_minus, 0.0) : NaN
        γ = 0.5 * (γ_plus + γ_minus)
        gamma[k] = isfinite(γ) ? γ : gamma[k - 1]
    end

    return lags, gamma
end

"""Clip an infinite line (cx,cy) + t*(dx,dy) to the bounding box; returns xs,ys or nothing."""
function _clip_line_to_box(cx, cy, dx, dy, xmin, xmax, ymin, ymax)
    t_lo, t_hi = -Inf, Inf
    for (c, d, lo, hi) in ((cx, dx, xmin, xmax), (cy, dy, ymin, ymax))
        if abs(d) < 1e-12
            (lo <= c <= hi) || return nothing
        else
            t1, t2 = minmax((lo - c) / d, (hi - c) / d)
            t_lo = max(t_lo, t1)
            t_hi = min(t_hi, t2)
        end
    end
    t_lo > t_hi && return nothing
    return [cx + t_lo*dx, cx + t_hi*dx], [cy + t_lo*dy, cy + t_hi*dy]
end

function _short_axis_segment(cx, cy, dir, len)
    d = collect(Float64.(dir))
    nrm = norm(d)
    nrm > eps(Float64) || return nothing
    d ./= nrm
    half = 0.5*Float64(len)
    return [cx - half*d[1], cx + half*d[1]], [cy - half*d[2], cy + half*d[2]]
end

function plot_variation(layer, op, sample, variance_diag, range_diag, summary, corr_level)
    x, y = parent_xy(layer)
    sample_map = field_to_parent_raster(layer, sample)
    sample_scale = maximum(abs.(sample))
    sample_scale = isfinite(sample_scale) && sample_scale > 0 ? sample_scale : 1.0

    cc = layer.domain[:cell_centroids]
    nc = number_of_cells(layer.domain)
    variance_error = abs.(variance_diag.realized ./ variance_diag.target .- 1.0)
    geom_target   = sqrt.(range_diag.target_major  .* range_diag.target_minor)
    geom_realized  = sqrt.(range_diag.realized_major .* range_diag.realized_minor)
    geom_error    = abs.(geom_realized ./ geom_target .- 1.0)
    ratio_realized = range_diag.realized_major ./ range_diag.realized_minor
    ratio_error   = abs.(ratio_realized ./ range_diag.target_ratio .- 1.0)

    # ── Centre cell and its local principal directions ──────────────────────
    mx = mean(cc[1, 1:nc]);  my = mean(cc[2, 1:nc])
    centre_cell = argmin([(cc[1, i] - mx)^2 + (cc[2, i] - my)^2 for i in 1:nc])
    cx = cc[1, centre_cell];  cy = cc[2, centre_cell]

    anchor_dists = [(cc[1, a] - cx)^2 + (cc[2, a] - cy)^2 for a in range_diag.anchors]
    k_near = argmin(anchor_dists)
    dir_major      = range_diag.direction_major[:, k_near]
    dir_minor      = range_diag.direction_minor[:, k_near]
    rho_major_tgt  = range_diag.target_major[k_near]
    rho_minor_tgt  = range_diag.target_minor[k_near]
    rho_major_real = range_diag.realized_major[k_near]
    rho_minor_real = range_diag.realized_minor[k_near]

    # ── Variograms ───────────────────────────────────────────────────────────
    lags_maj, gamma_maj = axial_semivariogram(op, layer, centre_cell, dir_major)
    lags_min, gamma_min = axial_semivariogram(op, layer, centre_cell, dir_minor)
    sill = op.fields.σ[centre_cell]^2
    γplot_max = maximum(vcat([sill], gamma_maj[isfinite.(gamma_maj)], gamma_min[isfinite.(gamma_min)]))
    γplot_max = isfinite(γplot_max) && γplot_max > 0 ? γplot_max : 1.0

    # ── Bounding box for axis overlay ────────────────────────────────────────
    xmin, xmax = minimum(cc[1, 1:nc]), maximum(cc[1, 1:nc])
    ymin, ymax = minimum(cc[2, 1:nc]), maximum(cc[2, 1:nc])

    fig = Figure(size = (1800, 960))

    # ── Row 1: spatial maps ──────────────────────────────────────────────────
    ax_sample = Axis(fig[1, 1], title = "One sample on the active layer",
                     xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect())
    hm_sample = heatmap!(ax_sample, x, y, sample_map;
                         colormap = :balance, colorrange = (-sample_scale, sample_scale))
    # Overlay principal-axis lines through the centre
    for (dir, color, label) in ((dir_major, :steelblue, "major"), (dir_minor, :firebrick, "minor"))
        seg = _clip_line_to_box(cx, cy, dir[1], dir[2], xmin, xmax, ymin, ymax)
        if !isnothing(seg)
            lines!(ax_sample, seg[1], seg[2]; color = color, linewidth = 2, label = label)
        end
    end
    overlay_count = min(length(range_diag.anchors), 25)
    for k in 1:overlay_count
        a = range_diag.anchors[k]
        alen = 0.30*range_diag.target_major[k]
        target_seg = _short_axis_segment(cc[1, a], cc[2, a], range_diag.direction_major[:, k], alen)
        realized_dir = [cos(range_diag.realized_angle[k]), sin(range_diag.realized_angle[k])]
        realized_seg = _short_axis_segment(cc[1, a], cc[2, a], realized_dir, alen)
        !isnothing(target_seg) && lines!(ax_sample, target_seg[1], target_seg[2]; color = (:steelblue, 0.55), linewidth = 1.2)
        !isnothing(realized_seg) && lines!(ax_sample, realized_seg[1], realized_seg[2]; color = (:darkorange, 0.75), linestyle = :dash, linewidth = 1.2)
    end
    scatter!(ax_sample, [cx], [cy]; color = :black, marker = :xcross, markersize = 14, strokewidth = 2)
    axislegend(ax_sample; position = :lt, labelsize = 10)
    Colorbar(fig[1, 2], hm_sample, label = "sample value")

    ax_var = Axis(fig[1, 3], title = "Anchor variance relative error",
                  xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect())
    sc_var = scatter!(ax_var, cc[1, variance_diag.anchors], cc[2, variance_diag.anchors],
                      color = variance_error, colormap = :viridis, markersize = 14)
    Colorbar(fig[1, 4], sc_var, label = "|σ²_real / σ²_target - 1|")

    ax_range = Axis(fig[1, 5], title = "Anchor geometric-range relative error",
                    xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect())
    sc_range = scatter!(ax_range, cc[1, range_diag.anchors], cc[2, range_diag.anchors],
                        color = geom_error, colormap = :plasma, markersize = 14)
    Colorbar(fig[1, 6], sc_range, label = "|ρ_real / ρ_target - 1|")

    ax_text = Axis(fig[1, 7], title = "Calibration summary")
    hidedecorations!(ax_text)
    hidespines!(ax_text)
    summary_txt = join([
        "variance status = $(variance_diag.calibration_status)",
        "range status = $(range_diag.calibration_status)",
        "range corr level = $(round(corr_level, sigdigits = 4))",
        "max variance error = $(round(summary.max_variance_error, sigdigits = 4))",
        "max geom. range error = $(round(summary.max_geometric_range_error, sigdigits = 4))",
        "max axis-ratio error = $(round(summary.max_axis_ratio_error, sigdigits = 4))",
        "max angle error [deg] = $(round(summary.max_angle_error_deg, sigdigits = 4))",
        "mean axis-ratio error = $(round(mean(ratio_error), sigdigits = 4))"
    ], "\n")
    text!(ax_text, 0.02, 0.98, text = summary_txt, space = :relative, align = (:left, :top))

    # ── Row 2: variograms (line plots) ───────────────────────────────────────
    for (col_offset, lags, gamma, rho_tgt, rho_real, axis_label, lcolor) in (
            (1, lags_maj, gamma_maj, rho_major_tgt, rho_major_real, "major", :steelblue),
            (4, lags_min, gamma_min, rho_minor_tgt, rho_minor_real, "minor", :firebrick),
        )
        ax = Axis(fig[2, col_offset:col_offset+2],
                  title  = "Semivariogram – $(axis_label) axis (from model centre)",
                  xlabel = "lag [m]",
                  ylabel = "γ(h)")
        ylims!(ax, -0.02 * γplot_max, 1.05 * γplot_max)
        hlines!(ax, [sill]; color = (:grey50, 0.7), linestyle = :dash, linewidth = 1.5,
                label = "sill (σ²)")
        hlines!(ax, [sill * (1 - corr_level)]; color = (:black, 0.45), linestyle = :dot, linewidth = 1.5,
                label = "range level")
        lines!(ax, lags, gamma; color = lcolor, linewidth = 2.5,
               label = "variogram curve")
        vlines!(ax, [rho_tgt];  color = lcolor,     linestyle = :dash, linewidth = 2,
                label = "target ρ = $(round(rho_tgt, sigdigits = 4)) m")
        vlines!(ax, [rho_real]; color = :darkorange, linestyle = :dot,  linewidth = 2,
                label = "realized ρ = $(round(rho_real, sigdigits = 4)) m")
        axislegend(ax; position = :rb)
    end

    return fig
end

function main(;
        input = "spe1",
        layer = 1,
        mode = :stationary,
        halo_layers = 3,
        halo_growth = 1.5,
        output_path = nothing,
        rng = Random.default_rng()
    )
    source = resolve_example_input(input)
    layer_domain = extract_reservoir_layer(source; layer = layer)
    prior, θ, natural_fields = build_prior(layer_domain; mode = mode)
    halo = prior.diffusion_scheme == :fvm9 ? nothing : MaternHaloSpec(layers = halo_layers, growth = halo_growth, total_padding = :boundary_range)
    output_file = isnothing(output_path) ? joinpath(@__DIR__, "matern_$(input)_$(mode).png") : output_path

    # Adaptive calibration settings.
    # On very coarse grids (e.g. SPE1, 100 cells) the per-anchor Gauss–Newton
    # calibration can stagnate: the discretization error dominates the
    # corrections that per-anchor (τ, κ, angle, ratio) controls can provide.
    # We try calibration with adaptive settings and fall back to the
    # uncalibrated operator if it does not converge.  The nonstationary field
    # structure is still encoded in θ; calibration only fine-tunes the
    # marginal properties (variance, range, anisotropy ratio, orientation).
    nc_cal = number_of_cells(layer_domain.domain)
    if nc_cal < 500
        anchor_res = (2, 2)
        cal_maxiter = 40
        cal_tols = (variance = 0.15, geometric_range = 0.15, axis_ratio = 0.15, angle = 5π/180)
    else
        anchor_res = (2, 2)
        cal_maxiter = 20
        cal_tols = JutulDarcy._default_matern_tolerances()
    end

    calibration = nothing
    try
        calibration = matern_calibrate(layer_domain, prior, θ;
            halo = halo, maxiter = cal_maxiter,
            anchor_resolution = anchor_res, tolerances = cal_tols)
    catch err
        if err isa ArgumentError && occursin("failed to converge", string(err))
            try
                calibration = matern_calibrate(layer_domain, prior, θ;
                    halo = halo, maxiter = cal_maxiter,
                    anchor_resolution = (3, 3), tolerances = cal_tols)
            catch err_retry
                if err_retry isa ArgumentError && occursin("failed to converge", string(err_retry))
                    @warn "Calibration did not converge ($(nc_cal) cells, mode=$mode). " *
                          "Falling back to uncalibrated operator. The spatial structure " *
                          "from θ is preserved; only marginal-property fine-tuning is skipped."
                else
                    rethrow()
                end
            end
        else
            rethrow()
        end
    end

    op = matern_spde_operator(layer_domain, prior, θ;
        calibration = calibration, strict = false, halo = halo)

    nc = number_of_cells(layer_domain.domain)
    sample = sample_from_precision(op.Q; rng = rng)[1:nc]

    # Diagnostics: use calibration anchors if available, else a default set.
    diag_halo = isnothing(calibration) ? halo : nothing
    diag_anchors = isnothing(calibration) ? :all : calibration.anchors
    corr_level_val = isnothing(calibration) ? JutulDarcy._matern_practical_corr_level(prior.ν) : calibration.corr_level
    variance_diag = matern_realized_variance(layer_domain, prior, θ;
        anchors = diag_anchors, calibration = calibration, strict = false, halo = diag_halo)
    range_diag = matern_realized_ranges(layer_domain, prior, θ;
        anchors = diag_anchors, calibration = calibration, strict = false,
        corr_level = corr_level_val, halo = diag_halo)
    summary = summarize_diagnostics(variance_diag, range_diag)
    fig = plot_variation(layer_domain, op, sample, variance_diag, range_diag, summary, corr_level_val)
    save(output_file, fig)
    display(fig)

    println("Matérn SPDE example")
    println("  source: $input")
    println("  mode: $mode")
    println("  diffusion scheme: $(prior.diffusion_scheme)")
    println("  selected layer: $(layer_domain.layer)")
    println("  active cells: $nc")
    println("  basis columns: $(size(prior.basis_matrix, 2))")
    if !isnothing(calibration)
        if isnothing(calibration.halo)
            println("  halo layers: none")
        else
            println("  halo layers: $(calibration.halo.layers)")
        end
        println("  calibration anchors: $(length(calibration.anchors))")
        println("  calibration iterations: $(calibration.diagnostics.iterations)")
        println("  range correlation level: $(calibration.corr_level)")
    else
        println("  calibration: SKIPPED (did not converge)")
    end
    println("  sample mean/std: $(mean(sample)) / $(std(sample))")
    println("  max variance error: $(summary.max_variance_error)")
    println("  max geometric range error: $(summary.max_geometric_range_error)")
    println("  max axis-ratio error: $(summary.max_axis_ratio_error)")
    println("  max angle error [deg]: $(summary.max_angle_error_deg)")
    println("  figure saved to: $output_file")

    if !isnothing(natural_fields)
        println("  target major range: $(extrema(natural_fields.major))")
        println("  target anisotropy ratio: $(extrema(natural_fields.ratio))")
        println("  target rotation [deg]: $(extrema(natural_fields.angle_deg))")
    end

    return (
        layer = layer_domain,
        prior = prior,
        θ = θ,
        calibration = calibration,
        operator = op,
        sample = sample,
        variance = variance_diag,
        ranges = range_diag,
        summary = summary,
        natural_fields = natural_fields
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    input = length(ARGS) >= 1 ? ARGS[1] : "spe1"
    layer = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    mode = length(ARGS) >= 3 ? Symbol(lowercase(ARGS[3])) : :stationary
    halo_layers = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 3
    halo_growth = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.5
    main(input = input, layer = layer, mode = mode, halo_layers = halo_layers, halo_growth = halo_growth)
end
