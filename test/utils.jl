using Jutul, JutulDarcy, Test, LinearAlgebra, SparseArrays, Statistics

function nearest_cell_2d(domain, xref, yref)
    cc = domain[:cell_centroids]
    d2 = [(cc[1, i] - xref)^2 + (cc[2, i] - yref)^2 for i in axes(cc, 2)]
    return argmin(d2)
end

function nearest_cell_3d(domain, xref, yref, zref)
    cc = domain[:cell_centroids]
    d2 = [(cc[1, i] - xref)^2 + (cc[2, i] - yref)^2 + (cc[3, i] - zref)^2 for i in axes(cc, 2)]
    return argmin(d2)
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

g = CartesianMesh((1, 1, 1), (10, 3.14, 2.71828))
r = 0.2;

@testset "Peaceman well index" begin
    @testset "Full tensor" begin
        K = [1, 0.2, 3, 5, 7, 0.15]
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :x) ≈ 45.483053742963861
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :y) ≈ 4.887480697644804
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :z) ≈ 16.756382219371336
    end
    @testset "Diagonal" begin
        K = [0.2, 0.71, 0.314]
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :x) ≈ 28.026376954943252
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :y) ≈ 2.382332435163685
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :z) ≈ 2.890000782901316
    end
    @testset "Scalar" begin
        K = [3.14]
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :x) ≈ 1.848702389098454e+02
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :y) ≈ 31.263184953683595
        @test compute_peaceman_index(g, K, r, (1, 1, 1), :z) ≈ 26.90991906669857
    end
end


import JutulDarcy: current_phase_index
@testset "current_phase_index" begin
    depths = (1.0, 2.0) # W O G
    @test current_phase_index(0.1, depths, reverse = false) == 1
    @test current_phase_index(1.01, depths, reverse = false) == 2
    @test current_phase_index(1.5, depths, reverse = false) == 2
    @test current_phase_index(2.5, depths, reverse = false) == 3

    depths = (2.0, 1.0) # WO OG
    @test current_phase_index(0.1, depths, reverse = true) == 3 # Gas
    @test current_phase_index(1.01, depths, reverse = true) == 2 # Oil
    @test current_phase_index(1.5, depths, reverse = true) == 2 # Oil
    @test current_phase_index(2.5, depths, reverse = true) == 1 # Water
end

@testset "nnc" begin
    l = [1, 2, 3]
    r = [2, 3, 4]
    N = [l'; r']
    neighbors = [(l[i], r[i]) for i in eachindex(l)]
    T = [10.0, 5.0, 3.0]
    T_t = [0.1, 0.2, 0.3]

    m = UnstructuredMesh(CartesianMesh((5, 5, 1)))
    nnc1 = setup_nnc_connections(m, l, r, T, T_t)
    nnc2 = setup_nnc_connections(m, N, T, T_t)
    @test nnc1.cells == nnc2.cells
    @test nnc1.trans_flow == nnc2.trans_flow
    @test nnc1.trans_thermal == nnc2.trans_thermal

    nnc3 = setup_nnc_connections(m, neighbors, T, T_t)
    @test nnc1.cells == nnc3.cells
    @test nnc1.trans_flow == nnc3.trans_flow
    @test nnc1.trans_thermal == nnc3.trans_thermal

    nnc4 = setup_nnc_connections(m, l, r, T)
    @test nnc1.cells == nnc4.cells
    @test all(isequal(0.0), nnc4.trans_thermal)

    @test_throws "ArgumentError: Left neighbor index 26 at connection 1 exceeds number of cells 25" JutulDarcy.setup_nnc_connections(m, [26], [1], T, T_t)
    @test_throws "ArgumentError: Right neighbor index 0 at connection 1 must be positive" JutulDarcy.setup_nnc_connections(m, [1], [0], T, T_t)

    d0 = reservoir_domain(m)
    d = reservoir_domain(m, nnc = nnc1)

    @test number_of_cells(d) == number_of_cells(d)
    @test number_of_faces(d) == number_of_faces(d0) + 3
    @test number_of_half_faces(d) == number_of_half_faces(d0) + 6

    T_computed0 = JutulDarcy.reservoir_transmissibility(d0)
    T_computed = JutulDarcy.reservoir_transmissibility(d)
    @test length(T_computed) == length(T_computed0) + 3
    @test all(T_computed[1:end-3] .== T_computed0)
    @test T_computed[end-2] == 10.0
    @test T_computed[end-1] == 5.0
    @test T_computed[end] == 3.0

    Tt_computed0 = JutulDarcy.reservoir_conductivity(d0)
    Tt_computed = JutulDarcy.reservoir_conductivity(d)
    @test length(Tt_computed) == length(Tt_computed0) + 3
    @test all(Tt_computed[1:end-3] .== Tt_computed0)
    @test Tt_computed[end-2] == 0.1
    @test Tt_computed[end-1] == 0.2
    @test Tt_computed[end] == 0.3
end

@testset "tpfa laplacian" begin
    d = get_1d_reservoir(4, L = 4.0, perm = 1.0, poro = 0.2)
    q = tpfa_stencil_quantities(d, laplace_from = :geometry)
    L = Matrix(tpfa_laplacian(q.neighbors, q.transmissibilities, length(q.volumes)))

    expected = [
        1.0 -1.0  0.0  0.0
       -1.0  2.0 -1.0  0.0
        0.0 -1.0  2.0 -1.0
        0.0  0.0 -1.0  1.0
    ]

    @test q.transmissibilities ≈ ones(3)
    @test L ≈ expected
    @test vec(sum(L, dims = 2)) ≈ zeros(4)
    @test issymmetric(L)
    @test minimum(eigvals(Symmetric(L))) >= -1e-12
end

@testset "tpfa stencil variants" begin
    g = CartesianMesh((4, 1, 1), (4.0, 1.0, 1.0))
    d = reservoir_domain(g, permeability = [1.0 3.0 2.0 4.0], porosity = 0.2)

    q_geo = tpfa_stencil_quantities(d, laplace_from = :geometry)
    q_flow = tpfa_stencil_quantities(d, laplace_from = :flow)
    q_unit = tpfa_stencil_quantities(d, laplace_from = :unit)

    @test q_geo.neighbors == q_flow.neighbors
    @test q_geo.neighbors == q_unit.neighbors
    @test q_geo.volumes == q_flow.volumes
    @test q_geo.volumes == q_unit.volumes
    @test q_geo.transmissibilities ≈ ones(3)
    @test q_unit.transmissibilities == ones(3)
    @test any(abs.(q_flow.transmissibilities .- q_geo.transmissibilities) .> 1e-12)
end

@testset "tpfa transmissibility controls" begin
    g = CartesianMesh((3, 1, 1), (3.0, 1.0, 1.0))
    d0 = reservoir_domain(g, permeability = [1.0 2.0 4.0], porosity = 0.2)
    T0 = JutulDarcy.reservoir_transmissibility(d0)

    d = reservoir_domain(
        g,
        permeability = [1.0 2.0 4.0],
        porosity = 0.2,
        transmissibility_multiplier = [2.0, 0.5],
        transmissibility_override = [NaN, 7.0]
    )
    q = tpfa_stencil_quantities(d, laplace_from = :flow)
    expected = T0 .* [2.0, 0.5]
    expected[2] = 7.0

    @test q.transmissibilities ≈ expected
end

@testset "tpfa laplacian with nnc" begin
    g = CartesianMesh((2, 2, 1), (2.0, 2.0, 1.0))
    l = cell_index(g, (1, 1, 1))
    r = cell_index(g, (2, 2, 1))
    nnc = setup_nnc_connections(g, [l], [r], [7.0])
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2, nnc = nnc)

    q = tpfa_stencil_quantities(d, laplace_from = :flow)
    L = tpfa_laplacian(q.neighbors, q.transmissibilities, number_of_cells(d))

    @test L[l, r] ≈ -7.0
    @test L[r, l] ≈ -7.0
end

@testset "matern layer extraction and basis helpers" begin
    g = CartesianMesh((4, 4, 2), (4.0, 4.0, 2.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    layer = extract_reservoir_layer(d; layer = 1)

    @test number_of_cells(layer.domain) == 16
    @test layer.layer == 1
    @test layer.parent_shape == (4, 4)
    @test length(layer.active_parent_linear) == 16
    @test count(layer.active_mask) == 16

    trend = matern_trend_basis(layer)
    raster = matern_raster_basis(layer, reshape(1:16, 4, 4))
    region = matern_region_basis(layer, reshape([i <= 8 ? :west : :east for i in 1:16], 4, 4))
    identity_basis = matern_identity_basis(layer)
    combined = combine_matern_basis(trend, raster)

    @test size(trend.matrix) == (16, 2)
    @test trend.names == [:x, :y]
    @test size(raster.matrix) == (16, 1)
    @test raster.names == [:raster]
    @test size(region.matrix) == (16, 1)
    @test identity_basis.matrix isa SparseMatrixCSC
    @test combined.matrix isa Matrix
    @test combined.names == [:x, :y, :raster]

    range_field = collect(range(12.0, 18.0, length = 16))
    sd_field = collect(range(0.9, 1.1, length = 16))
    ratio_field = collect(range(1.05, 1.35, length = 16))
    angle_field_deg = collect(range(-10.0, 10.0, length = 16))
    fit = fit_matern_fields(
        layer,
        identity_basis;
        range0 = exp(mean(log.(range_field))),
        sd0 = exp(mean(log.(sd_field))),
        angle0 = 0.0,
        logratio0 = mean(log.(ratio_field)),
        range = range_field,
        sd = sd_field,
        angle = angle_field_deg,
        ratio = ratio_field,
        angle_unit = :degree
    )

    prior = MaternSPDE2Prior(
        number_of_cells(layer.domain);
        basis = fit.basis,
        range0 = fit.baselines.range0,
        sd0 = fit.baselines.sd0,
        angle0 = fit.baselines.angle0,
        logratio0 = fit.baselines.logratio0
    )
    fields = matern_parameter_fields(prior, fit.θ)

    @test fields.ρ ≈ range_field rtol = 1e-10
    @test fields.σ ≈ sd_field rtol = 1e-10
    @test fields.ratio ≈ ratio_field rtol = 1e-10
    @test fields.orientation ≈ deg2rad.(angle_field_deg) atol = 1e-10
end

@testset "matern spde stationary operator and fallback" begin
    g = CartesianMesh((4, 4), (4.0, 4.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    q = tpfa_stencil_quantities(d, laplace_from = :geometry)
    L = tpfa_laplacian(q.neighbors, q.transmissibilities, number_of_cells(d))

    prior = MaternSPDE2Prior(number_of_cells(d); range0 = 2.5, sd0 = 1.2)
    fields = matern_parameter_fields(prior)

    @test_throws ArgumentError matern_spde_operator(d, prior; strict = true)
    op = @test_logs (:warn, r"No MaternCalibration provided") matern_spde_operator(d, prior)

    @test Matrix(op.L_H) ≈ Matrix(L)
    @test op.fields.κ_eff ≈ fields.κ
    @test op.fields.τ ≈ fields.τ_nominal
    @test op.calibration_status == :missing_fallback
    @test issymmetric(Matrix(op.Q))
    @test minimum(eigvals(Symmetric(Matrix(op.Q)))) > 0
end

@testset "matern spde basis-driven anisotropy fields" begin
    g = CartesianMesh((15, 15, 2), (15.0, 15.0, 2.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    layer = extract_reservoir_layer(d; layer = 1)
    B = matern_trend_basis(layer)
    nc = number_of_cells(layer.domain)

    prior = MaternSPDE2Prior(
        nc;
        basis = B,
        range0 = 2.6,
        sd0 = 1.0,
        angle0 = 8*pi/180,
        logratio0 = log(1.15),
        diffusion_scheme = :fvm9
    )
    θ = (
        range = Dict(:x => 0.18, :y => -0.10),
        sd = Dict(:x => 0.05),
        angle = Dict(:x => 6*pi/180, :y => -3*pi/180),
        logratio = Dict(:x => 0.12, :y => 0.05)
    )

    fields = matern_parameter_fields(prior, θ)
    op = @test_logs (:warn, r"No MaternCalibration provided") matern_spde_operator(layer, prior, θ)

    @test maximum(fields.ρ) > minimum(fields.ρ)
    @test maximum(fields.σ) > minimum(fields.σ)
    @test maximum(fields.ratio) > minimum(fields.ratio)
    @test maximum(abs.(fields.orientation .- fields.orientation[1])) > 0.0
    @test all(fields.ratio .>= 1.0)
    @test op.calibration_status == :missing_fallback
    @test size(op.Q, 1) == nc
end

@testset "matern spde rotated anisotropy fvm9" begin
    g = CartesianMesh((15, 15, 2), (15.0, 15.0, 2.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    layer = extract_reservoir_layer(d; layer = 1)
    nc = number_of_cells(layer.domain)

    prior_tpfa = MaternSPDE2Prior(nc; range0 = 3.0, sd0 = 1.0, angle0 = pi/6, logratio0 = log(2.0))
    @test_throws ArgumentError matern_spde_operator(layer, prior_tpfa)

    prior = MaternSPDE2Prior(nc; range0 = 3.0, sd0 = 1.0, angle0 = pi/6, logratio0 = log(2.0), diffusion_scheme = :fvm9)
    cal = matern_calibrate(layer, prior; maxiter = 12, tolerances = (variance = 0.08, geometric_range = 0.08, axis_ratio = 0.08, angle = 3pi/180))
    op = matern_spde_operator(layer, prior; calibration = cal, strict = true)
    vd = matern_realized_variance(layer, prior; anchors = cal.anchors, calibration = cal, strict = true)
    rd = matern_realized_ranges(layer, prior; anchors = cal.anchors, calibration = cal, strict = true, corr_level = cal.corr_level)

    geom_target = sqrt.(rd.target_major .* rd.target_minor)
    geom_realized = sqrt.(rd.realized_major .* rd.realized_minor)
    ratio_realized = rd.realized_major ./ rd.realized_minor
    angle_error = abs.(JutulDarcy._orientation_difference.(rd.realized_angle, rd.target_angle))

    @test op.calibration_status == :applied
    @test isempty(op.halo_idx)
    @test maximum(abs.(vd.realized ./ vd.target .- 1.0)) <= cal.tolerances.variance + 1e-8
    @test maximum(abs.(geom_realized ./ geom_target .- 1.0)) <= cal.tolerances.geometric_range + 1e-8
    @test maximum(abs.(ratio_realized ./ rd.target_ratio .- 1.0)) <= cal.tolerances.axis_ratio + 1e-8
    @test maximum(angle_error) <= cal.tolerances.angle + 1e-8
end

@testset "matern spde fvm9 nonstationary rotated fields and anchors" begin
    g = CartesianMesh((15, 15, 2), (15.0, 15.0, 2.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    layer = extract_reservoir_layer(d; layer = 1)
    B = matern_trend_basis(layer)
    nc = number_of_cells(layer.domain)

    prior = MaternSPDE2Prior(
        nc;
        basis = B,
        range0 = 3.0,
        sd0 = 1.0,
        angle0 = pi/9,
        logratio0 = log(1.5),
        diffusion_scheme = :fvm9
    )
    θ = (
        range = Dict(:x => 0.10, :y => -0.05),
        sd = Dict(:x => 0.03, :y => -0.02),
        angle = Dict(:x => 0.08, :y => -0.04),
        logratio = Dict(:x => 0.06, :y => 0.02)
    )

    fields = matern_parameter_fields(prior, θ)
    anchors = JutulDarcy._matern_auto_anchors(layer, fields.ρ_major; resolution = (2, 2))
    op = @test_logs (:warn, r"No MaternCalibration provided") matern_spde_operator(layer, prior, θ)
    rd = matern_realized_ranges(layer, prior, θ; anchors = anchors)

    nx, ny = layer.parent_shape
    anchor_ij = map(a -> (mod1(layer.active_parent_linear[a], nx), fld(layer.active_parent_linear[a] - 1, nx) + 1), anchors)
    @test all(anchor_ij) do ij
        i, j = ij
        1 < i < nx && 1 < j < ny
    end
    @test maximum(fields.ratio) > minimum(fields.ratio)
    @test maximum(abs.(fields.orientation .- fields.orientation[1])) > 0.0
    @test maximum(abs.(JutulDarcy._orientation_difference.(rd.realized_angle, rd.target_angle))) < 10pi/180
    @test size(op.Q, 1) == nc
end

@testset "matern spde explicit calibration" begin
    g = CartesianMesh((15, 15, 2), (15.0, 15.0, 2.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    layer = extract_reservoir_layer(d; layer = 1)
    prior = MaternSPDE2Prior(number_of_cells(layer.domain); range0 = 2.5, sd0 = 1.0)

    cal = matern_calibrate(layer, prior)
    op = matern_spde_operator(layer, prior; calibration = cal, strict = true)
    vd = matern_realized_variance(layer, prior; anchors = cal.anchors, calibration = cal, strict = true)
    rd = matern_realized_ranges(layer, prior; anchors = cal.anchors, calibration = cal, strict = true, corr_level = cal.corr_level)

    geom_target = sqrt.(rd.target_major .* rd.target_minor)
    geom_realized = sqrt.(rd.realized_major .* rd.realized_minor)
    ratio_realized = rd.realized_major ./ rd.realized_minor

    @test op.calibration_status == :applied
    @test !isempty(op.halo_idx)
    @test cal.anchor_resolution == (1, 1)
    @test maximum(abs.(vd.realized ./ vd.target .- 1.0)) <= cal.tolerances.variance + 1e-8
    @test maximum(abs.(geom_realized ./ geom_target .- 1.0)) <= cal.tolerances.geometric_range + 1e-8
    @test maximum(abs.(ratio_realized ./ rd.target_ratio .- 1.0)) <= cal.tolerances.axis_ratio + 1e-8
    @test_throws ArgumentError matern_spde_operator(layer, prior; calibration = cal, strict = true, halo = MaternHaloSpec(layers = 2, growth = 1.0))
end

@testset "matern real-input smoke" begin
    pth = JutulDarcy.GeoEnergyIO.test_input_file_path("SPE1", "SPE1.DATA")
    layer = extract_reservoir_layer(pth; layer = 1)
    cc = layer.domain[:cell_centroids]
    lx = maximum(cc[1, :]) - minimum(cc[1, :])
    ly = maximum(cc[2, :]) - minimum(cc[2, :])
    range0 = 0.15 * min(lx, ly)

    prior = MaternSPDE2Prior(number_of_cells(layer.domain); range0 = range0, sd0 = 1.0)
    cal = matern_calibrate(layer, prior; anchor_resolution = (3, 3))
    op = matern_spde_operator(layer, prior; calibration = cal, strict = true)

    @test layer.layer == 1
    @test number_of_cells(layer.domain) > 0
    @test size(op.Q, 1) > number_of_cells(layer.domain)
    @test op.calibration_status == :applied
end

@testset "matern data-file precision api" begin
    pth = JutulDarcy.GeoEnergyIO.test_input_file_path("SPE1", "SPE1.DATA")
    layer = extract_reservoir_layer(pth; layer = 1)
    nc = number_of_cells(layer.domain)
    cc = layer.domain[:cell_centroids]
    lx = maximum(cc[1, :]) - minimum(cc[1, :])
    ly = maximum(cc[2, :]) - minimum(cc[2, :])
    range0 = 0.20 * min(lx, ly)

    Q = @test_logs (:warn, r"No MaternCalibration provided") matern_precision_from_data_file(
        pth;
        target_variance = 1.5,
        target_range = range0,
        calibrate = false
    )
    csc = @test_logs (:warn, r"No MaternCalibration provided") matern_precision_csc_from_data_file(
        pth;
        target_variance = 1.5,
        target_range = range0,
        calibrate = false
    )

    @test Q isa SparseMatrixCSC
    @test size(Q) == (nc, nc)
    @test csc.shape == (nc, nc)
    @test length(csc.colptr) == nc + 1
    @test length(csc.rowval) == length(csc.nzval) == nnz(Q)
    @test minimum(csc.rowval) >= 1
    @test length(csc.domain_mask) == size(Q, 1)
    @test count(csc.domain_mask) == nc
    @test csc.interior_idx == collect(1:nc)
    @test isempty(csc.halo_idx)

    csc_cal = matern_precision_csc_from_data_file(
        pth;
        target_variance = 1.0,
        target_range = range0
    )
    @test csc_cal.shape[1] > nc
    @test length(csc_cal.domain_mask) == csc_cal.shape[1]
    @test count(csc_cal.domain_mask) == nc
    @test all(csc_cal.domain_mask[csc_cal.interior_idx])
    @test all(.!csc_cal.domain_mask[csc_cal.halo_idx])

    @test_throws ArgumentError matern_precision_from_data_file(
        pth;
        target_variance = [1.0, 1.1],
        target_range = range0,
        calibrate = false
    )

    Qns = @test_logs (:warn, r"No MaternCalibration provided") matern_precision_from_data_file(
        pth;
        target_variance = fill(1.0, nc),
        target_range = fill(range0, nc),
        target_rotation = collect(range(0.0, stop = pi/6, length = nc)),
        target_anisotropy = 2.0,
        angle_unit = :radian,
        calibrate = false
    )
    @test Qns isa SparseMatrixCSC
    @test size(Qns) == (nc, nc)
end

@testset "matern spde 3d stationary operator" begin
    g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    q = tpfa_stencil_quantities(d, laplace_from = :geometry)
    L = tpfa_laplacian(q.neighbors, q.transmissibilities, number_of_cells(d))

    prior = MaternSPDE3Prior(number_of_cells(d); ρ0 = 2.5, σ0 = 1.1)
    fields = matern_parameter_fields(prior)
    op = matern_spde_operator(d, prior; compensated = false)

    @test Matrix(op.L_H) ≈ Matrix(L)
    @test op.fields.κ_eff ≈ fields.κ
    @test op.fields.τ ≈ fields.τ_nominal
    @test issymmetric(Matrix(op.Q))
    @test minimum(eigvals(Symmetric(Matrix(op.Q)))) > 0
end

@testset "matern spde 3d geometry ignores flow edits" begin
    g = CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0))
    d0 = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    nf = number_of_faces(d0)
    d1 = reservoir_domain(
        g,
        permeability = 1.0,
        porosity = 0.2,
        net_to_gross = fill(0.3, number_of_cells(d0)),
        transmissibility_multiplier = collect(range(0.2, 1.7, length = nf)),
        transmissibility_override = vcat(fill(NaN, nf - 1), 5.0)
    )

    prior = MaternSPDE3Prior(number_of_cells(d0); ρ0 = 2.0, σ0 = 1.0, laplace_from = :geometry)
    op0 = matern_spde_operator(d0, prior; compensated = false)
    op1 = matern_spde_operator(d1, prior; compensated = false)

    @test op0.transmissibilities ≈ op1.transmissibilities
    @test Matrix(op0.L_H) ≈ Matrix(op1.L_H)
end

@testset "matern spde 3d halo operator" begin
    g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    nc = number_of_cells(d)
    prior = MaternSPDE3Prior(nc; ρ0 = 2.5, σ0 = 1.0)
    halo = MaternHaloSpec(layers = 2, growth = 1.0)

    op0 = matern_spde_operator(d, prior; compensated = false)
    op = matern_spde_operator(d, prior; compensated = false, halo = halo)

    @test op.interior_idx == collect(1:nc)
    @test !isempty(op.halo_idx)
    @test size(op.Q, 1) == nc + length(op.halo_idx)
    @test size(op.halo_meta.ext_centroids, 2) == size(op.Q, 1)
    @test !isempty(op.halo_meta.boundary_adjacency)
    @test issymmetric(Matrix(op.Q))
    @test minimum(eigvals(Symmetric(Matrix(op.Q)))) > 0
    @test op.transmissibilities ≈ op0.transmissibilities

    halo_source = op.halo_meta.halo_source
    @test op.fields.κ_eff[op.halo_idx] ≈ op.fields.κ_eff[halo_source]
    @test op.fields.τ[op.halo_idx] ≈ op.fields.τ[halo_source]

    f1, f2, _ = first(op.halo_meta.boundary_adjacency)
    h1 = op.halo_meta.face_to_chain[f1][1]
    h2 = op.halo_meta.face_to_chain[f2][1]
    @test Matrix(op.L_H)[h1, h2] < 0
end

@testset "matern spde 3d sd compensation" begin
    g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    prior = MaternSPDE3Prior(number_of_cells(d); ρ0 = 2.5, σ0 = 1.0)
    halo = MaternHaloSpec(layers = 2, growth = 1.0)

    var0 = matern_realized_variance(d, prior; anchors = :all, compensated = false, halo = halo)
    err0 = abs(mean(log.(var0.realized ./ var0.target)))

    matern_sd_compensation!(prior, d; anchors = :all, mode = :mean, halo = halo)
    var1 = matern_realized_variance(d, prior; anchors = :all, compensated = true, halo = halo)
    err1 = abs(mean(log.(var1.realized ./ var1.target)))

    @test err1 <= err0 + 1e-8
end

@testset "matern spde 3d axis ranges" begin
    g = CartesianMesh((5, 5, 5), (5.0, 5.0, 5.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    prior = MaternSPDE3Prior(number_of_cells(d); ρ0 = 3.0, σ0 = 1.0)
    anchor = nearest_cell_3d(d, 2.5, 2.5, 2.5)
    diag = matern_realized_axis_ranges(d, prior; anchors = [anchor], compensated = false, corr_level = 0.2)

    rx = diag.realized_x[1]
    ry = diag.realized_y[1]
    rz = diag.realized_z[1]
    @test all(isfinite, (rx, ry, rz))
    @test maximum((rx / ry, ry / rz, rx / rz, ry / rx, rz / ry, rz / rx)) <= 1.5
end
