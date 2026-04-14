using Jutul, JutulDarcy, Test, LinearAlgebra, Statistics

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

@testset "matern spde stationary operator" begin
    g = CartesianMesh((4, 4), (4.0, 4.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    q = tpfa_stencil_quantities(d, laplace_from = :geometry)
    L = tpfa_laplacian(q.neighbors, q.transmissibilities, number_of_cells(d))

    prior = MaternSPDE2Prior(number_of_cells(d); ρ0 = 2.5, σ0 = 1.2)
    fields = matern_parameter_fields(prior)
    op = matern_spde_operator(d, prior; compensated = false)

    @test Matrix(op.L_H) ≈ Matrix(L)
    @test op.fields.κ_eff ≈ fields.κ
    @test op.fields.τ ≈ fields.τ_nominal
    @test issymmetric(Matrix(op.Q))
    @test minimum(eigvals(Symmetric(Matrix(op.Q)))) > 0
end

@testset "matern spde anisotropy and compensation" begin
    g = CartesianMesh((5, 5), (5.0, 5.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    nc = number_of_cells(d)
    cc = d[:cell_centroids]

    Bx = normalized_basis_from_coord(d, 1)
    Bconst = ones(nc, 1)

    prior = MaternSPDE2Prior(
        nc;
        ρ0 = 1.75,
        σ0 = 1.0,
        range_basis = Bx,
        anisotropy_u_basis = Bconst
    )

    θ = (
        range = [0.35],
        anisotropy_u = [0.55]
    )

    fields = matern_parameter_fields(prior, θ)
    detH = fields.H[1, :].*fields.H[3, :] .- fields.H[2, :].^2
    @test all(isapprox.(detH, 1.0; atol = 1e-10))
    @test all(fields.H[1, :] .> 0)
    @test all(fields.H[3, :] .> 0)

    anchor = nearest_cell_2d(d, 2.5, 2.5)
    ranges0 = matern_realized_ranges(d, prior, θ; anchors = [anchor], compensated = false)
    ratio_target = ranges0.target_major[1] / ranges0.target_minor[1]
    ratio_realized0 = ranges0.realized_major[1] / ranges0.realized_minor[1]
    @test ranges0.realized_major[1] > ranges0.realized_minor[1]

    var0 = matern_realized_variance(d, prior, θ; anchors = :all, compensated = false)
    sd_err0 = mean(abs.(log.(var0.realized ./ var0.target)))

    matern_sd_compensation!(prior, d; anchors = :all)
    var1 = matern_realized_variance(d, prior, θ; anchors = :all, compensated = true)
    sd_err1 = mean(abs.(log.(var1.realized ./ var1.target)))
    @test sd_err1 <= sd_err0 + 1e-8

    left_anchor = nearest_cell_2d(d, minimum(cc[1, :]) + 0.5, 2.5)
    right_anchor = nearest_cell_2d(d, maximum(cc[1, :]) - 0.5, 2.5)
    ranges_geom0 = matern_realized_ranges(d, prior, θ; anchors = [left_anchor, right_anchor], compensated = true)
    geom_err0 = mean(abs.(log.(sqrt.(ranges_geom0.realized_major .* ranges_geom0.realized_minor) ./ sqrt.(ranges_geom0.target_major .* ranges_geom0.target_minor))))
    ratio_err0 = abs(log(ratio_realized0 / ratio_target))

    matern_range_compensation!(prior, d; anchors = :all, corr_level = 0.2)
    ranges1 = matern_realized_ranges(d, prior, θ; anchors = [anchor], compensated = true)
    ratio_realized1 = ranges1.realized_major[1] / ranges1.realized_minor[1]
    ratio_err1 = abs(log(ratio_realized1 / (ranges1.target_major[1] / ranges1.target_minor[1])))
    @test ratio_err1 <= ratio_err0 + 1e-8

    ranges_geom1 = matern_realized_ranges(d, prior, θ; anchors = [left_anchor, right_anchor], compensated = true)
    geom_err1 = mean(abs.(log.(sqrt.(ranges_geom1.realized_major .* ranges_geom1.realized_minor) ./ sqrt.(ranges_geom1.target_major .* ranges_geom1.target_minor))))
    @test geom_err1 <= geom_err0 + 1e-8
end

@testset "matern spde halo operator" begin
    g = CartesianMesh((5, 5), (5.0, 5.0))
    d = reservoir_domain(g, permeability = 1.0, porosity = 0.2)
    nc = number_of_cells(d)
    prior = MaternSPDE2Prior(nc; ρ0 = 2.5, σ0 = 1.0)
    halo = MaternHaloSpec(layers = 2, growth = 1.0)

    op0 = matern_spde_operator(d, prior; compensated = false)
    op = matern_spde_operator(d, prior; compensated = false, halo = halo)

    @test op.interior_idx == collect(1:nc)
    @test !isempty(op.halo_idx)
    @test size(op.Q, 1) == nc + length(op.halo_idx)
    @test size(op.L_H, 1) == size(op.Q, 1)
    @test issymmetric(Matrix(op.Q))
    @test minimum(eigvals(Symmetric(Matrix(op.Q)))) > 0
    @test op.transmissibilities ≈ op0.transmissibilities

    halo_source = op.halo_meta.halo_source
    @test op.fields.κ_eff[op.halo_idx] ≈ op.fields.κ_eff[halo_source]
    @test op.fields.τ[op.halo_idx] ≈ op.fields.τ[halo_source]
    @test size(op.halo_meta.ext_centroids, 2) == size(op.Q, 1)

    side_faces = op.halo_meta.side_faces
    if !isempty(side_faces[:south]) && !isempty(side_faces[:west])
        south_face = first(side_faces[:south])
        west_face = first(side_faces[:west])
        south_halo = op.halo_meta.face_to_chain[south_face][1]
        west_halo = op.halo_meta.face_to_chain[west_face][1]
        @test Matrix(op.L_H)[south_halo, west_halo] < 0
    end

    center = nearest_cell_2d(d, 2.5, 2.5)
    west = nearest_cell_2d(d, minimum(d[:cell_centroids][1, :]) + 0.5, 2.5)
    var0 = matern_realized_variance(d, prior; anchors = [center, west], compensated = false)
    varh = matern_realized_variance(d, prior; anchors = [center, west], compensated = false, halo = halo)
    err0 = abs(log(var0.realized[2]/var0.realized[1]))
    errh = abs(log(varh.realized[2]/varh.realized[1]))
    @test errh <= err0 + 1e-8

    varh0 = matern_realized_variance(d, prior; anchors = :all, compensated = false, halo = halo)
    scale_err0 = abs(mean(log.(varh0.realized ./ varh0.target)))

    matern_sd_compensation!(prior, d; anchors = :all, mode = :mean, halo = halo)
    varh1 = matern_realized_variance(d, prior; anchors = :all, compensated = true, halo = halo)
    scale_err1 = abs(mean(log.(varh1.realized ./ varh1.target)))
    @test scale_err1 <= scale_err0 + 1e-8
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
