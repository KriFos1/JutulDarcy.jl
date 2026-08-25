"""Python helpers for Matérn SPDE precision matrices.

The implementation delegates model loading and SPDE assembly to the Julia
package in this repository, then converts Julia CSC arrays into a SciPy sparse
matrix.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
from scipy.io import savemat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve_triangular


_JL = None
_LOADED_PROJECT: str | None = None
_DEFAULT_BACKEND = "subprocess"

_JULIA_SUBPROCESS_CODE = r"""
using JutulDarcy
using MAT

input_path = ARGS[1]
output_path = ARGS[2]
data_path = ARGS[3]
layer = parse(Int, ARGS[4])
angle_unit = Symbol(ARGS[5])
diffusion_scheme = Symbol(ARGS[6])
calibrate = parse(Bool, ARGS[7])
anchor_resolution = (parse(Int, ARGS[8]), parse(Int, ARGS[9]))
maxiter = parse(Int, ARGS[10])
nugget_raw = ARGS[11]
nugget = nugget_raw == "nothing" ? nothing : parse(Float64, nugget_raw)

input = matread(input_path)

function mat_scalar(x)
    return x isa AbstractArray ? first(x) : x
end

function mat_target(input, key::String, has_key::String)
    has = Bool(mat_scalar(input[has_key]))
    has || return nothing
    raw = input[key]
    x = raw isa AbstractArray ? vec(Float64.(raw)) : Float64[Float64(raw)]
    return length(x) == 1 ? x[1] : x
end

function mat_tolerances(input)
    Bool(mat_scalar(input["has_tolerances"])) || return JutulDarcy._default_matern_tolerances()
    defaults = JutulDarcy._default_matern_tolerances()
    tol_v  = Bool(mat_scalar(input["has_tol_variance"]))         ? Float64(mat_scalar(input["tol_variance"]))         : defaults.variance
    tol_gr = Bool(mat_scalar(input["has_tol_geometric_range"]))  ? Float64(mat_scalar(input["tol_geometric_range"]))  : defaults.geometric_range
    tol_ar = Bool(mat_scalar(input["has_tol_axis_ratio"]))       ? Float64(mat_scalar(input["tol_axis_ratio"]))       : defaults.axis_ratio
    tol_a  = Bool(mat_scalar(input["has_tol_angle"]))            ? Float64(mat_scalar(input["tol_angle"]))            : defaults.angle
    return (variance = tol_v, geometric_range = tol_gr, axis_ratio = tol_ar, angle = tol_a)
end

data = JutulDarcy.matern_precision_csc_from_data_file(
    data_path;
    layer = layer,
    target_variance = mat_target(input, "target_variance", "has_target_variance"),
    target_range = mat_target(input, "target_range", "has_target_range"),
    target_rotation = mat_target(input, "target_rotation", "has_target_rotation"),
    target_anisotropy = mat_target(input, "target_anisotropy", "has_target_anisotropy"),
    angle_unit = angle_unit,
    diffusion_scheme = diffusion_scheme,
    calibrate = calibrate,
    anchor_resolution = anchor_resolution,
    maxiter = maxiter,
    tolerances = mat_tolerances(input),
    nugget = nugget
)

open(output_path, "w") do io
    write(io, Int64(data.shape[1]))
    write(io, Int64(data.shape[2]))
    write(io, Int64(length(data.nzval)))
    write(io, Int64.(data.colptr))
    write(io, Int64.(data.rowval))
    write(io, Float64.(data.nzval))
    write(io, UInt8.(data.domain_mask))
    write(io, Int64(data.grid_ni))
    write(io, Int64(data.grid_nj))
    write(io, Int64(data.n_halo))
    write(io, Int8.(data.grid_mask))
    write(io, Int64(size(data.parent_centroids, 1)))
    write(io, Float64.(data.parent_centroids))
end
"""

_JULIA_SUBPROCESS_CODE_3D = r"""
using JutulDarcy
using MAT

input_path = ARGS[1]
output_path = ARGS[2]
data_path = ARGS[3]
angle_unit = Symbol(ARGS[4])
diffusion_scheme = Symbol(ARGS[5])
boundary = Symbol(ARGS[6])
nugget_raw = ARGS[7]
nugget = nugget_raw == "nothing" ? nothing : parse(Float64, nugget_raw)

input = matread(input_path)

function mat_scalar(x)
    return x isa AbstractArray ? first(x) : x
end

function mat_target(input, key::String, has_key::String)
    has = Bool(mat_scalar(input[has_key]))
    has || return nothing
    raw = input[key]
    x = raw isa AbstractArray ? vec(Float64.(raw)) : Float64[Float64(raw)]
    return length(x) == 1 ? x[1] : x
end

data = JutulDarcy.matern_precision_csc_from_data_file_3d(
    data_path;
    target_variance = mat_target(input, "target_variance", "has_target_variance"),
    target_range = mat_target(input, "target_range", "has_target_range"),
    target_range_major = mat_target(input, "target_range_major", "has_target_range_major"),
    target_range_minor = mat_target(input, "target_range_minor", "has_target_range_minor"),
    target_range_vertical = mat_target(input, "target_range_vertical", "has_target_range_vertical"),
    target_azimuth = mat_target(input, "target_azimuth", "has_target_azimuth"),
    target_dip = mat_target(input, "target_dip", "has_target_dip"),
    target_rake = mat_target(input, "target_rake", "has_target_rake"),
    angle_unit = angle_unit,
    diffusion_scheme = diffusion_scheme,
    boundary = boundary,
    nugget = nugget
)

open(output_path, "w") do io
    write(io, Int64(data.shape[1]))
    write(io, Int64(data.shape[2]))
    write(io, Int64(length(data.nzval)))
    write(io, Int64.(data.colptr))
    write(io, Int64.(data.rowval))
    write(io, Float64.(data.nzval))
    write(io, UInt8.(data.domain_mask))
    write(io, Int64(data.grid_ni))
    write(io, Int64(data.grid_nj))
    write(io, Int64(data.grid_nk))
    write(io, Int64(data.n_halo))
    write(io, Int8.(data.grid_mask))
    write(io, Int64(size(data.parent_centroids, 1)))
    write(io, Float64.(data.parent_centroids))
end
"""


def matern_precision_from_data_file(
    data_path: str | Path,
    *,
    layer: int = 1,
    target_variance: Any,
    target_range: Any,
    target_rotation: Any = None,
    target_anisotropy: Any = None,
    angle_unit: str = "degree",
    diffusion_scheme: str = "auto",
    calibrate: bool = True,
    anchor_resolution: tuple[int, int] = (2, 2),
    maxiter: int = 100,
    tolerances: dict[str, float] | None = None,
    nugget: float | None = None,
    julia_project: str | Path | None = None,
    backend: str | None = None,
    return_mask: bool = False,
    return_geometry: bool = False,
) -> csc_matrix | tuple:
    """Build a Matérn SPDE precision matrix from an Eclipse ``.DATA`` file.

    Scalar target inputs produce a stationary prior. Any vector target input
    selects the non-stationary path, and every vector must contain exactly one
    value per active cell in the extracted layer.

    Parameters
    ----------
    data_path:
        Path to the input model ``.DATA`` file.
    layer:
        K-layer to extract. Defaults to 1.
    target_variance:
        Target marginal variance, scalar or active-cell vector.
    target_range:
        Target geometric Matérn range, scalar or active-cell vector.
    target_rotation:
        Optional anisotropy orientation. Scalars or active-cell vectors are
        interpreted using ``angle_unit``. Defaults to zero.
    target_anisotropy:
        Optional major/minor anisotropy ratio. Must be >= 1. Defaults to 1.
    angle_unit:
        ``"degree"`` or ``"radian"``.
    diffusion_scheme:
        ``"auto"``, ``"tpfa"``, or ``"fvm9"``. ``"auto"`` selects ``"fvm9"``
        for rotated anisotropy and ``"tpfa"`` otherwise.
    calibrate:
        If true, run the Julia calibration step before returning the precision.
    anchor_resolution:
        Anchor grid used by calibration for non-stationary inputs.
    maxiter:
        Maximum calibration iterations.
    tolerances:
        Optional dict with calibration convergence tolerances.  Accepted keys:
        ``variance``, ``geometric_range``, ``axis_ratio`` (relative, default
        0.05 each) and ``angle`` (radians, default ~0.035).  Only keys that
        are provided override the Julia defaults.
    nugget:
        Diagonal regularization added to Q before export.  ``None`` (default)
        uses an automatic nugget of ``1e-8 * mean(diag(Q))``.  Set to ``0``
        to disable.
    julia_project:
        Julia project to activate before loading JutulDarcy. Defaults to
        ``JUTULDARCY_JULIA_PROJECT`` when set, otherwise the repository root
        containing this Python file for editable installs.
    backend:
        ``"subprocess"`` or ``"juliacall"``. The default is controlled by
        ``JUTULDARCY_MATERN_BACKEND`` and otherwise uses ``"subprocess"``,
        which avoids shared-library conflicts from embedding Julia in Python.
    return_mask:
        If true, return ``(Q, mask)`` where ``mask`` is a 2-D parent-layer
        grid mask with ``1`` for active cells and ``0`` for inactive cells.
    return_geometry:
        If true, also return a geometry dictionary with parent-layer centroid
        arrays ``x`` and ``y`` of the same shape as ``mask``.  With
        ``return_mask=True`` the return value is ``(Q, mask, geometry)``.
    """

    backend = backend or os.environ.get("JUTULDARCY_MATERN_BACKEND", _DEFAULT_BACKEND)
    if backend == "subprocess":
        return _matern_precision_subprocess(
            data_path,
            layer=layer,
            target_variance=target_variance,
            target_range=target_range,
            target_rotation=target_rotation,
            target_anisotropy=target_anisotropy,
            angle_unit=angle_unit,
            diffusion_scheme=diffusion_scheme,
            calibrate=calibrate,
            anchor_resolution=anchor_resolution,
            maxiter=maxiter,
            tolerances=tolerances,
            nugget=nugget,
            julia_project=julia_project,
            return_mask=return_mask,
            return_geometry=return_geometry,
        )
    if backend != "juliacall":
        raise ValueError("backend must be 'subprocess' or 'juliacall'")

    jl = _julia(julia_project)
    jl_kwargs = dict(
        layer=int(layer),
        target_variance=_target_value(target_variance),
        target_range=_target_value(target_range),
        target_rotation=_target_value(target_rotation),
        target_anisotropy=_target_value(target_anisotropy),
        angle_unit=jl.Symbol(angle_unit),
        diffusion_scheme=jl.Symbol(diffusion_scheme),
        calibrate=bool(calibrate),
        anchor_resolution=(int(anchor_resolution[0]), int(anchor_resolution[1])),
        maxiter=int(maxiter),
    )
    if tolerances is not None:
        jl_tol = _tolerances_to_julia(jl, tolerances)
        jl_kwargs["tolerances"] = jl_tol
    if nugget is not None:
        jl_kwargs["nugget"] = float(nugget)
    data = jl.JutulDarcy.matern_precision_csc_from_data_file(
        str(data_path),
        **jl_kwargs,
    )
    return _result_from_julia(
        data,
        return_mask=return_mask,
        return_geometry=return_geometry,
    )


def matern_precision_from_data_file_3d(
    data_path: str | Path,
    *,
    target_variance: Any,
    target_range: Any | None = None,
    target_range_major: Any | None = None,
    target_range_minor: Any | None = None,
    target_range_vertical: Any | None = None,
    target_azimuth: Any = 0.0,
    target_dip: Any = 0.0,
    target_rake: Any = 0.0,
    angle_unit: str = "degree",
    diffusion_scheme: str = "auto",
    boundary: str = "robin",
    nugget: float | None = None,
    julia_project: str | Path | None = None,
    backend: str | None = None,
    return_mask: bool = False,
    return_geometry: bool = False,
) -> csc_matrix | tuple:
    """Build a full-3D Matérn SPDE precision matrix from a ``.DATA`` file.

    Ranges are principal practical ranges. If ``target_range`` is provided it
    is used for all three principal axes unless an axis-specific range is also
    provided. Angles use ``angle_unit`` and map to azimuth/dip/rake rotations.
    """
    backend = backend or os.environ.get("JUTULDARCY_MATERN_BACKEND", _DEFAULT_BACKEND)
    if backend == "subprocess":
        return _matern_precision_subprocess_3d(
            data_path,
            target_variance=target_variance,
            target_range=target_range,
            target_range_major=target_range_major,
            target_range_minor=target_range_minor,
            target_range_vertical=target_range_vertical,
            target_azimuth=target_azimuth,
            target_dip=target_dip,
            target_rake=target_rake,
            angle_unit=angle_unit,
            diffusion_scheme=diffusion_scheme,
            boundary=boundary,
            nugget=nugget,
            julia_project=julia_project,
            return_mask=return_mask,
            return_geometry=return_geometry,
        )
    if backend != "juliacall":
        raise ValueError("backend must be 'subprocess' or 'juliacall'")

    jl = _julia(julia_project)
    jl_kwargs = dict(
        target_variance=_target_value(target_variance),
        target_range=_target_value(target_range),
        target_range_major=_target_value(target_range_major),
        target_range_minor=_target_value(target_range_minor),
        target_range_vertical=_target_value(target_range_vertical),
        target_azimuth=_target_value(target_azimuth),
        target_dip=_target_value(target_dip),
        target_rake=_target_value(target_rake),
        angle_unit=jl.Symbol(angle_unit),
        diffusion_scheme=jl.Symbol(diffusion_scheme),
        boundary=jl.Symbol(boundary),
    )
    if nugget is not None:
        jl_kwargs["nugget"] = float(nugget)
    data = jl.JutulDarcy.matern_precision_csc_from_data_file_3d(str(data_path), **jl_kwargs)
    return _result_from_julia_3d(data, return_mask=return_mask, return_geometry=return_geometry)


def _matern_precision_subprocess(
    data_path: str | Path,
    *,
    layer: int,
    target_variance: Any,
    target_range: Any,
    target_rotation: Any,
    target_anisotropy: Any,
    angle_unit: str,
    diffusion_scheme: str,
    calibrate: bool,
    anchor_resolution: tuple[int, int],
    maxiter: int,
    tolerances: dict[str, float] | None,
    nugget: float | None,
    julia_project: str | Path | None,
    return_mask: bool,
    return_geometry: bool,
) -> csc_matrix | tuple:
    project = _resolve_project(julia_project)
    julia = os.environ.get("JULIA", "julia")
    with tempfile.TemporaryDirectory(prefix="jutuldarcy_matern_") as tmp:
        input_path = Path(tmp) / "input.mat"
        output_path = Path(tmp) / "output.cscbin"
        savemat(input_path, _subprocess_input_payload(
            target_variance=target_variance,
            target_range=target_range,
            target_rotation=target_rotation,
            target_anisotropy=target_anisotropy,
            tolerances=tolerances,
        ))
        cmd = [
            julia,
            f"--project={project}",
            "--compiled-modules=existing",
            "-e",
            _JULIA_SUBPROCESS_CODE,
            str(input_path),
            str(output_path),
            str(data_path),
            str(int(layer)),
            str(angle_unit),
            str(diffusion_scheme),
            str(bool(calibrate)).lower(),
            str(int(anchor_resolution[0])),
            str(int(anchor_resolution[1])),
            str(int(maxiter)),
            "nothing" if nugget is None else str(float(nugget)),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "Julia subprocess failed while building the Matérn precision "
                f"matrix.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return _scipy_csc_from_binary(
            output_path,
            return_mask=return_mask,
            return_geometry=return_geometry,
        )


def _matern_precision_subprocess_3d(
    data_path: str | Path,
    *,
    target_variance: Any,
    target_range: Any | None,
    target_range_major: Any | None,
    target_range_minor: Any | None,
    target_range_vertical: Any | None,
    target_azimuth: Any,
    target_dip: Any,
    target_rake: Any,
    angle_unit: str,
    diffusion_scheme: str,
    boundary: str,
    nugget: float | None,
    julia_project: str | Path | None,
    return_mask: bool,
    return_geometry: bool,
) -> csc_matrix | tuple:
    project = _resolve_project(julia_project)
    julia = os.environ.get("JULIA", "julia")
    with tempfile.TemporaryDirectory(prefix="jutuldarcy_matern3d_") as tmp:
        input_path = Path(tmp) / "input.mat"
        output_path = Path(tmp) / "output.cscbin"
        savemat(input_path, _subprocess_input_payload_3d(
            target_variance=target_variance,
            target_range=target_range,
            target_range_major=target_range_major,
            target_range_minor=target_range_minor,
            target_range_vertical=target_range_vertical,
            target_azimuth=target_azimuth,
            target_dip=target_dip,
            target_rake=target_rake,
        ))
        cmd = [
            julia,
            f"--project={project}",
            "--compiled-modules=existing",
            "-e",
            _JULIA_SUBPROCESS_CODE_3D,
            str(input_path),
            str(output_path),
            str(data_path),
            str(angle_unit),
            str(diffusion_scheme),
            str(boundary),
            "nothing" if nugget is None else str(float(nugget)),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "Julia subprocess failed while building the 3D Matérn precision "
                f"matrix.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return _scipy_csc_from_binary_3d(
            output_path,
            return_mask=return_mask,
            return_geometry=return_geometry,
        )


def _subprocess_input_payload(
    *,
    target_variance: Any,
    target_range: Any,
    target_rotation: Any,
    target_anisotropy: Any,
    tolerances: dict[str, float] | None,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    targets = dict(
        target_variance=target_variance,
        target_range=target_range,
        target_rotation=target_rotation,
        target_anisotropy=target_anisotropy,
    )
    for key, value in targets.items():
        has_key = f"has_{key}"
        payload[has_key] = np.array([[value is not None]], dtype=np.uint8)
        payload[key] = _target_array(0.0 if value is None else value)
    _TOLERANCE_KEYS = ("variance", "geometric_range", "axis_ratio", "angle")
    has_tol = tolerances is not None
    payload["has_tolerances"] = np.array([[has_tol]], dtype=np.uint8)
    for tk in _TOLERANCE_KEYS:
        payload[f"tol_{tk}"] = np.array(
            [[tolerances[tk] if has_tol and tk in tolerances else 0.0]]
        )
        payload[f"has_tol_{tk}"] = np.array(
            [[has_tol and tk in tolerances]], dtype=np.uint8
        )
    return payload


def _subprocess_input_payload_3d(**targets: Any) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for key, value in targets.items():
        has_key = f"has_{key}"
        payload[has_key] = np.array([[value is not None]], dtype=np.uint8)
        payload[key] = _target_array(0.0 if value is None else value)
    return payload


def _julia(julia_project: str | Path | None):
    global _JL, _LOADED_PROJECT
    project = _resolve_project(julia_project)
    project_key = str(project.resolve()) if project is not None else ""
    if _JL is None:
        try:
            from juliacall import Main as jl
        except ModuleNotFoundError as exc:
            raise ImportError(
                "matern_precision_from_data_file requires the Python package "
                "`juliacall`. Install it in the Python environment that calls "
                "this API."
            ) from exc
        _JL = jl
    jl = _JL
    if _LOADED_PROJECT != project_key:
        if project is not None:
            jl.seval("import Pkg")
            jl.Pkg.activate(str(project))
        jl.seval("using JutulDarcy")
        _LOADED_PROJECT = project_key
    return jl


def _resolve_project(julia_project: str | Path | None) -> Path:
    project = _default_project() if julia_project is None else Path(julia_project)
    if project is None:
        raise FileNotFoundError(
            "Could not find the Julia project. Pass julia_project=... or set "
            "JUTULDARCY_JULIA_PROJECT."
        )
    project = project.expanduser()
    if not (project / "Project.toml").is_file():
        raise FileNotFoundError(
            f"Julia project {project} does not contain Project.toml. "
            "Pass julia_project=... or set JUTULDARCY_JULIA_PROJECT."
        )
    return project


def _default_project() -> Path | None:
    env_project = os.environ.get("JUTULDARCY_JULIA_PROJECT")
    if env_project:
        return Path(env_project).expanduser()
    cwd_project = _find_project_upwards(Path.cwd())
    if cwd_project is not None:
        return cwd_project
    return _find_project_upwards(Path(__file__).resolve())


def _find_project_upwards(start: Path) -> Path | None:
    start = start if start.is_dir() else start.parent
    for candidate in (start, *start.parents):
        if (candidate / "Project.toml").is_file():
            return candidate
    return None


def _target_value(value: Any):
    if value is None:
        return None
    arr = _target_array(value)
    if arr.size == 1:
        return float(arr[0])
    return arr.tolist()


def _tolerances_to_julia(jl, tolerances: dict[str, float]):
    """Build a Julia NamedTuple for calibration tolerances."""
    defaults = dict(variance=0.05, geometric_range=0.05, axis_ratio=0.05, angle=2 * 3.141592653589793 / 180)
    merged = {k: tolerances.get(k, v) for k, v in defaults.items()}
    expr = (
        f"(variance = {merged['variance']}, "
        f"geometric_range = {merged['geometric_range']}, "
        f"axis_ratio = {merged['axis_ratio']}, "
        f"angle = {merged['angle']})"
    )
    return jl.seval(expr)


def _target_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=float)
    return np.asarray(arr, dtype=float).ravel()


def _field(obj, name: str):
    try:
        return getattr(obj, name)
    except AttributeError:
        return obj[name]


def _result_from_julia(data, *, return_mask: bool, return_geometry: bool) -> csc_matrix | tuple:
    Q = _scipy_csc_from_julia(data)
    mask = _grid_mask_from_julia(data) if return_mask or return_geometry else None
    geometry = _grid_geometry_from_julia(data) if return_geometry else None
    return _format_result(Q, mask, geometry, return_mask, return_geometry)


def _result_from_julia_3d(data, *, return_mask: bool, return_geometry: bool) -> csc_matrix | tuple:
    Q = _scipy_csc_from_julia(data)
    mask = _grid_mask_from_julia_3d(data) if return_mask or return_geometry else None
    geometry = _grid_geometry_from_julia_3d(data) if return_geometry else None
    return _format_result(Q, mask, geometry, return_mask, return_geometry)


def _format_result(
    Q: csc_matrix,
    mask: np.ndarray | None,
    geometry: dict[str, np.ndarray] | None,
    return_mask: bool,
    return_geometry: bool,
) -> csc_matrix | tuple:
    if return_mask and return_geometry:
        return Q, mask, geometry
    if return_mask:
        return Q, mask
    if return_geometry:
        return Q, geometry
    return Q


def _grid_mask_from_julia(data) -> np.ndarray:
    """Return a 2-D ``int8`` mask of shape ``(ni, nj)``.

    Values: 0 = inactive, 1 = active (interior), 2 = halo.
    Halo cells are synthetic boundary-padding cells that do not map to parent
    grid positions, so halo information is encoded in the ``n_halo`` field but
    does not appear in the 2-D mask (all grid entries are 0 or 1).
    """
    ni = int(np.asarray(_field(data, "grid_ni")).ravel()[0])
    nj = int(np.asarray(_field(data, "grid_nj")).ravel()[0])
    flat = np.asarray(_field(data, "grid_mask"), dtype=np.int8).ravel()
    return flat.copy().reshape((ni, nj), order="F")


def _grid_geometry_from_julia(data) -> dict[str, np.ndarray]:
    ni = int(np.asarray(_field(data, "grid_ni")).ravel()[0])
    nj = int(np.asarray(_field(data, "grid_nj")).ravel()[0])
    centroids = np.asarray(_field(data, "parent_centroids"), dtype=float)
    centroids = centroids.reshape((centroids.shape[0], ni * nj), order="F")
    geometry = {
        "centroids": centroids.copy(),
        "x": centroids[0, :].reshape((ni, nj), order="F").copy(),
        "y": centroids[1, :].reshape((ni, nj), order="F").copy(),
    }
    if centroids.shape[0] > 2:
        geometry["z"] = centroids[2, :].reshape((ni, nj), order="F").copy()
    return geometry


def _grid_mask_from_julia_3d(data) -> np.ndarray:
    ni = int(np.asarray(_field(data, "grid_ni")).ravel()[0])
    nj = int(np.asarray(_field(data, "grid_nj")).ravel()[0])
    nk = int(np.asarray(_field(data, "grid_nk")).ravel()[0])
    flat = np.asarray(_field(data, "grid_mask"), dtype=np.int8).ravel()
    return flat.copy().reshape((ni, nj, nk), order="F")


def _grid_geometry_from_julia_3d(data) -> dict[str, np.ndarray]:
    ni = int(np.asarray(_field(data, "grid_ni")).ravel()[0])
    nj = int(np.asarray(_field(data, "grid_nj")).ravel()[0])
    nk = int(np.asarray(_field(data, "grid_nk")).ravel()[0])
    centroids = np.asarray(_field(data, "parent_centroids"), dtype=float)
    centroids = centroids.reshape((centroids.shape[0], ni * nj * nk), order="F")
    return {
        "centroids": centroids.copy(),
        "x": centroids[0, :].reshape((ni, nj, nk), order="F").copy(),
        "y": centroids[1, :].reshape((ni, nj, nk), order="F").copy(),
        "z": centroids[2, :].reshape((ni, nj, nk), order="F").copy(),
    }


def _scipy_csc_from_julia(data) -> csc_matrix:
    shape = tuple(int(x) for x in np.asarray(_field(data, "shape")).ravel())
    colptr = np.asarray(_field(data, "colptr"), dtype=np.int64).ravel() - 1
    rowval = np.asarray(_field(data, "rowval"), dtype=np.int64).ravel() - 1
    nzval = np.asarray(_field(data, "nzval"), dtype=float).ravel()
    return csc_matrix((nzval.copy(), rowval.copy(), colptr.copy()), shape=shape)


def _scipy_csc_from_binary(
    path: Path,
    *,
    return_mask: bool,
    return_geometry: bool,
) -> csc_matrix | tuple:
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int64, count=3)
        if header.size != 3:
            raise RuntimeError(f"Invalid CSC output file {path}: missing header")
        nrow, ncol, nnz = (int(x) for x in header)
        colptr = np.fromfile(stream, dtype=np.int64, count=ncol + 1) - 1
        rowval = np.fromfile(stream, dtype=np.int64, count=nnz) - 1
        nzval = np.fromfile(stream, dtype=np.float64, count=nnz)
        _domain_mask = np.fromfile(stream, dtype=np.uint8, count=nrow)
        grid_header = np.fromfile(stream, dtype=np.int64, count=3)
        ni, nj, _n_halo = (int(x) for x in grid_header)
        grid_mask = np.fromfile(stream, dtype=np.int8, count=ni * nj)
        geom_header = np.fromfile(stream, dtype=np.int64, count=1)
        if geom_header.size == 1:
            ncoord = int(geom_header[0])
            parent_centroids = np.fromfile(stream, dtype=np.float64, count=ncoord * ni * nj)
        else:
            ncoord = 0
            parent_centroids = np.array([], dtype=float)
    if colptr.size != ncol + 1 or rowval.size != nnz or nzval.size != nnz:
        raise RuntimeError(f"Invalid CSC output file {path}: truncated arrays")
    if _domain_mask.size != nrow:
        raise RuntimeError(f"Invalid CSC output file {path}: missing domain mask")
    if grid_mask.size != ni * nj:
        raise RuntimeError(f"Invalid CSC output file {path}: missing grid mask")
    if return_geometry and parent_centroids.size != ncoord * ni * nj:
        raise RuntimeError(f"Invalid CSC output file {path}: missing parent centroids")
    Q = csc_matrix((nzval, rowval, colptr), shape=(nrow, ncol))
    mask = grid_mask.reshape((ni, nj), order="F") if return_mask or return_geometry else None
    geometry = None
    if return_geometry:
        centroids = parent_centroids.reshape((ncoord, ni * nj), order="F")
        geometry = {
            "centroids": centroids.copy(),
            "x": centroids[0, :].reshape((ni, nj), order="F").copy(),
            "y": centroids[1, :].reshape((ni, nj), order="F").copy(),
        }
        if ncoord > 2:
            geometry["z"] = centroids[2, :].reshape((ni, nj), order="F").copy()
    return _format_result(Q, mask, geometry, return_mask, return_geometry)


def _scipy_csc_from_binary_3d(
    path: Path,
    *,
    return_mask: bool,
    return_geometry: bool,
) -> csc_matrix | tuple:
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int64, count=3)
        if header.size != 3:
            raise RuntimeError(f"Invalid CSC output file {path}: missing header")
        nrow, ncol, nnz = (int(x) for x in header)
        colptr = np.fromfile(stream, dtype=np.int64, count=ncol + 1) - 1
        rowval = np.fromfile(stream, dtype=np.int64, count=nnz) - 1
        nzval = np.fromfile(stream, dtype=np.float64, count=nnz)
        _domain_mask = np.fromfile(stream, dtype=np.uint8, count=nrow)
        grid_header = np.fromfile(stream, dtype=np.int64, count=4)
        ni, nj, nk, _n_halo = (int(x) for x in grid_header)
        grid_mask = np.fromfile(stream, dtype=np.int8, count=ni * nj * nk)
        geom_header = np.fromfile(stream, dtype=np.int64, count=1)
        if geom_header.size == 1:
            ncoord = int(geom_header[0])
            parent_centroids = np.fromfile(stream, dtype=np.float64, count=ncoord * ni * nj * nk)
        else:
            ncoord = 0
            parent_centroids = np.array([], dtype=float)
    if colptr.size != ncol + 1 or rowval.size != nnz or nzval.size != nnz:
        raise RuntimeError(f"Invalid CSC output file {path}: truncated arrays")
    if _domain_mask.size != nrow:
        raise RuntimeError(f"Invalid CSC output file {path}: missing domain mask")
    if grid_mask.size != ni * nj * nk:
        raise RuntimeError(f"Invalid CSC output file {path}: missing 3D grid mask")
    if return_geometry and parent_centroids.size != ncoord * ni * nj * nk:
        raise RuntimeError(f"Invalid CSC output file {path}: missing parent centroids")
    Q = csc_matrix((nzval, rowval, colptr), shape=(nrow, ncol))
    mask = grid_mask.reshape((ni, nj, nk), order="F") if return_mask or return_geometry else None
    geometry = None
    if return_geometry:
        centroids = parent_centroids.reshape((ncoord, ni * nj * nk), order="F")
        geometry = {
            "centroids": centroids.copy(),
            "x": centroids[0, :].reshape((ni, nj, nk), order="F").copy(),
            "y": centroids[1, :].reshape((ni, nj, nk), order="F").copy(),
            "z": centroids[2, :].reshape((ni, nj, nk), order="F").copy(),
        }
    return _format_result(Q, mask, geometry, return_mask, return_geometry)


def realization_to_grid(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    fill: float = np.nan,
) -> np.ma.MaskedArray:
    """Map a Q-space realization onto the ``(ni, nj)`` parent grid.

    Parameters
    ----------
    x : array_like
        Realization vector of length ``>= n_active`` (e.g. from a Cholesky
        solve of Q).  Only the first ``n_active`` entries (interior cells) are
        used; trailing halo entries are discarded.
    mask : ndarray of int8, shape (ni, nj)
        Grid mask returned by ``matern_precision_from_data_file(...,
        return_mask=True)``.  Values: 0 = inactive, 1 = active.
    fill : float, optional
        Fill value for inactive cells (default ``np.nan``).

    Returns
    -------
    field : np.ma.MaskedArray, shape (ni, nj)
        Masked array where inactive cells are masked and filled with *fill*.
    """
    x = np.asarray(x).ravel()
    n_active = int((mask == 1).sum())
    if x.size < n_active:
        raise ValueError(
            f"x has {x.size} elements but the mask has {n_active} active cells"
        )
    field = np.full(mask.shape, fill, dtype=float)
    # Interior cells are stored in column-major (Fortran) order matching the
    # Julia mesh.  Iterate the flat mask in the same order to preserve the
    # cell-to-grid correspondence.
    flat = mask.ravel(order="F")
    field_flat = field.ravel(order="F")
    field_flat[flat == 1] = x[:n_active]
    field = field_flat.reshape(mask.shape, order="F")
    return np.ma.MaskedArray(field, mask=(mask != 1))


def realization_to_grid_3d(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    fill: float = np.nan,
) -> np.ma.MaskedArray:
    """Map a Q-space realization onto a full 3-D parent grid."""
    x = np.asarray(x).ravel()
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {mask_arr.shape}")
    n_active = int((mask_arr == 1).sum())
    if x.size < n_active:
        raise ValueError(
            f"x has {x.size} elements but the mask has {n_active} active cells"
        )
    field = np.full(mask_arr.shape, fill, dtype=float)
    flat = mask_arr.ravel(order="F")
    field_flat = field.ravel(order="F")
    field_flat[flat == 1] = x[:n_active]
    field = field_flat.reshape(mask_arr.shape, order="F")
    return np.ma.MaskedArray(field, mask=(mask_arr != 1))


def active_line_indices(
    mask: np.ndarray,
    start: tuple[int, int],
    stop: tuple[int, int],
    *,
    one_based: bool = True,
    geometry: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return active-cell vector indices along a straight grid line.

    Parameters
    ----------
    mask:
        Grid mask from ``matern_precision_from_data_file(...,
        return_mask=True)``.  Values equal to ``1`` are active cells.
    start, stop:
        Grid coordinates ``(i, j)`` for the line endpoints.  By default these
        are interpreted as one-based reservoir/Eclipse indices.  Set
        ``one_based=False`` for Python zero-based indices.
    one_based:
        Coordinate convention for input and returned ``coords``.
    geometry:
        Optional geometry returned by
        ``matern_precision_from_data_file(..., return_geometry=True)``.  When
        provided, ``distance`` is measured from cell-centroid coordinates
        instead of grid-cell units.

    Returns
    -------
    info : dict
        ``indices`` are zero-based vector indices into the active-cell block of
        ``Q``/``Qinv``.  ``coords`` are the corresponding active grid
        coordinates.  ``distance`` is measured in grid-cell units from
        ``start``.
    """
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask_arr.shape}")

    start0 = _normalize_grid_point(start, mask_arr.shape, one_based, "start")
    stop0 = _normalize_grid_point(stop, mask_arr.shape, one_based, "stop")
    index_grid = _active_vector_index_grid(mask_arr)

    indices: list[int] = []
    coords0: list[tuple[int, int]] = []
    for i, j in _bresenham_grid_line(start0, stop0):
        idx = int(index_grid[i, j])
        if idx < 0:
            continue
        indices.append(idx)
        coords0.append((i, j))

    if not indices:
        raise ValueError("The requested line does not pass through any active cells.")

    xvals, yvals, distances = _line_coordinates_and_distances(
        coords0,
        start0,
        mask_arr.shape,
        geometry,
    )
    offset = 1 if one_based else 0
    coords = np.array([(i + offset, j + offset) for i, j in coords0], dtype=int)
    out = {
        "indices": np.array(indices, dtype=int),
        "coords": coords,
        "distance": distances,
    }
    if xvals is not None and yvals is not None:
        out["x"] = xvals
        out["y"] = yvals
    return out


def dense_inverse_line_values(
    Qinv: np.ndarray,
    mask: np.ndarray,
    start: tuple[int, int],
    stop: tuple[int, int],
    *,
    mode: str = "profile",
    reference: tuple[int, int] | int | None = None,
    one_based: bool = True,
    geometry: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray | int | None]:
    """Extract dense inverse values on active cells along a grid line.

    ``mode="profile"`` returns ``Qinv[reference, line_indices]``.  If
    ``reference`` is omitted, the first active cell on the line is used.
    ``mode="diagonal"`` returns marginal variances along the line.
    ``mode="block"`` returns ``Qinv[line_indices, line_indices]``.
    """
    qinv = np.asarray(Qinv)
    if qinv.ndim != 2 or qinv.shape[0] != qinv.shape[1]:
        raise ValueError(f"Qinv must be a square dense matrix, got shape {qinv.shape}")

    line = active_line_indices(
        mask,
        start,
        stop,
        one_based=one_based,
        geometry=geometry,
    )
    indices = line["indices"]
    if int(indices.max()) >= qinv.shape[0]:
        raise ValueError(
            "Qinv is too small for the active-cell indices implied by mask: "
            f"max index {int(indices.max())}, Qinv size {qinv.shape[0]}"
        )

    mode_key = mode.lower()
    reference_index: int | None = None
    if mode_key == "profile":
        reference_index = (
            int(indices[0])
            if reference is None
            else _resolve_inverse_reference(reference, mask, one_based)
        )
        if reference_index >= qinv.shape[0]:
            raise ValueError(
                f"reference index {reference_index} is outside Qinv size {qinv.shape[0]}"
            )
        values = qinv[reference_index, indices]
    elif mode_key in ("diagonal", "variance"):
        values = np.diag(qinv)[indices]
    elif mode_key == "block":
        values = qinv[np.ix_(indices, indices)]
    else:
        raise ValueError("mode must be 'profile', 'diagonal', or 'block'")

    return {
        **line,
        "values": np.asarray(values),
        "reference_index": reference_index,
    }


def plot_dense_inverse_line(
    Qinv: np.ndarray,
    mask: np.ndarray,
    start: tuple[int, int],
    stop: tuple[int, int],
    *,
    mode: str = "profile",
    reference: tuple[int, int] | int | None = None,
    one_based: bool = True,
    geometry: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | None = None,
    ax=None,
    **plot_kwargs,
):
    """Plot dense inverse values along active cells on a straight grid line.

    Returns ``(ax, data)`` where ``data`` is the dictionary returned by
    :func:`dense_inverse_line_values`.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ImportError("plot_dense_inverse_line requires matplotlib") from exc

    data = dense_inverse_line_values(
        Qinv,
        mask,
        start,
        stop,
        mode=mode,
        reference=reference,
        one_based=one_based,
        geometry=geometry,
    )

    mode_key = mode.lower()
    if ax is None:
        _, ax = plt.subplots()

    if mode_key == "block":
        image_kwargs = {"origin": "lower", "aspect": "auto"}
        if geometry is not None:
            distances = np.asarray(data["distance"], dtype=float)
            image_kwargs["extent"] = (
                float(distances[0]),
                float(distances[-1]),
                float(distances[0]),
                float(distances[-1]),
            )
        image_kwargs.update(plot_kwargs)
        image = ax.imshow(data["values"], **image_kwargs)
        ax.figure.colorbar(image, ax=ax, label="Qinv value")
        unit = "physical units" if geometry is not None else "line cell"
        ax.set_xlabel(f"distance [{unit}]" if geometry is not None else unit)
        ax.set_ylabel(f"distance [{unit}]" if geometry is not None else unit)
    else:
        default_kwargs = {"marker": "o"}
        default_kwargs.update(plot_kwargs)
        ax.plot(data["distance"], data["values"], **default_kwargs)
        unit = "physical units" if geometry is not None else "grid cells"
        ax.set_xlabel(f"distance from start [{unit}]")
        ax.set_ylabel("Qinv value" if mode_key == "profile" else "marginal variance")
    ax.set_title(f"Dense inverse {mode_key} along active line")
    return ax, data


def sample_from_precision(
    Q: csc_matrix,
    mask: np.ndarray | None = None,
    *,
    n_samples: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray | np.ma.MaskedArray | list[np.ndarray | np.ma.MaskedArray]:
    """Draw GMRF samples from a sparse precision matrix.

    Uses ``sksparse.cholmod`` for a supernodal Cholesky factorization.  The
    factorization is computed once and reused for all samples.

    Parameters
    ----------
    Q : csc_matrix
        Precision matrix (with nugget already applied if obtained from
        ``matern_precision_from_data_file``).
    mask : ndarray of int8, shape (ni, nj), optional
        Grid mask from ``matern_precision_from_data_file(...,
        return_mask=True)``.  When provided, each sample is returned as a 2-D
        masked array via ``realization_to_grid``.
    n_samples : int
        Number of independent samples to draw.
    rng : numpy.random.Generator, optional
        Random number generator.  Defaults to ``np.random.default_rng()``.

    Returns
    -------
    samples : ndarray or MaskedArray (if n_samples == 1), or list thereof.
        If *mask* is ``None``, raw vectors of length ``Q.shape[0]``.
        If *mask* is given, 2-D masked arrays of shape ``(ni, nj)``.
    """
    try:
        from sksparse.cholmod import cho_factor
    except ModuleNotFoundError as exc:
        raise ImportError(
            "sample_from_precision requires scikit-sparse.  "
            "Install it with: pip install scikit-sparse"
        ) from exc

    if rng is None:
        rng = np.random.default_rng()

    n = Q.shape[1]
    z = rng.standard_normal((n, n_samples))
    factor = cho_factor(Q, lower=True)
    x = _sample_vectors_from_cholmod_factor(factor, z)
    results = []
    for i in range(n_samples):
        xi = x[:, i]
        if mask is not None:
            xi = realization_to_grid(xi, mask)
        results.append(xi)
    if n_samples == 1:
        return results[0]
    return results


def _sample_vectors_from_cholmod_factor(factor, z: np.ndarray) -> np.ndarray:
    """Draw samples from ``Q^-1`` using a CHOLMOD square-root factor.

    ``factor.solve(z)`` solves ``Q x = z`` and therefore has covariance
    ``Q^-2``.  For ``Q[p, p] = L D L.T``, a precision sample is obtained as
    ``x[p] = L.T \\ (D^-1/2 z)``.
    """
    rhs = np.asarray(z, dtype=float)
    d = np.asarray(factor.D.diagonal(), dtype=float).ravel()
    if d.size != rhs.shape[0]:
        raise RuntimeError(
            "CHOLMOD factor has an unexpected diagonal size: "
            f"{d.size} for matrix size {rhs.shape[0]}"
        )
    if np.any(d <= 0):
        raise RuntimeError("CHOLMOD factor has a non-positive diagonal entry")
    rhs = rhs / np.sqrt(d)[:, None]

    y = spsolve_triangular(factor.L.T.tocsc(), rhs, lower=False)
    x = np.empty_like(y)
    x[np.asarray(factor.perm, dtype=np.intp), :] = y
    return x


def _active_vector_index_grid(mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask) == 1
    out = np.full(active.shape, -1, dtype=int)
    flat_out = out.ravel(order="F")
    flat_active = active.ravel(order="F")
    flat_out[flat_active] = np.arange(int(flat_active.sum()), dtype=int)
    return flat_out.reshape(active.shape, order="F")


def _line_coordinates_and_distances(
    coords0: list[tuple[int, int]],
    start0: tuple[int, int],
    shape: tuple[int, int],
    geometry: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    if geometry is None:
        distances = np.array(
            [float(np.hypot(i - start0[0], j - start0[1])) for i, j in coords0],
            dtype=float,
        )
        return None, None, distances

    xgrid, ygrid = _geometry_xy_grids(geometry, shape)
    xvals = np.array([xgrid[i, j] for i, j in coords0], dtype=float)
    yvals = np.array([ygrid[i, j] for i, j in coords0], dtype=float)

    x0 = xgrid[start0]
    y0 = ygrid[start0]
    if not (np.isfinite(x0) and np.isfinite(y0)):
        x0 = xvals[0]
        y0 = yvals[0]
    distances = np.hypot(xvals - x0, yvals - y0)
    return xvals, yvals, distances


def _geometry_xy_grids(
    geometry: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(geometry, dict):
        if "x" not in geometry or "y" not in geometry:
            raise ValueError("geometry dict must contain 'x' and 'y' arrays")
        xgrid = np.asarray(geometry["x"], dtype=float)
        ygrid = np.asarray(geometry["y"], dtype=float)
    else:
        if len(geometry) != 2:
            raise ValueError("geometry tuple must be (xgrid, ygrid)")
        xgrid = np.asarray(geometry[0], dtype=float)
        ygrid = np.asarray(geometry[1], dtype=float)

    if xgrid.shape != shape or ygrid.shape != shape:
        raise ValueError(
            f"geometry x/y arrays must match mask shape {shape}, "
            f"got {xgrid.shape} and {ygrid.shape}"
        )
    return xgrid, ygrid


def _normalize_grid_point(
    point: tuple[int, int],
    shape: tuple[int, int],
    one_based: bool,
    name: str,
) -> tuple[int, int]:
    if len(point) != 2:
        raise ValueError(f"{name} must be an (i, j) pair")
    offset = 1 if one_based else 0
    i = int(point[0]) - offset
    j = int(point[1]) - offset
    ni, nj = shape
    if not (0 <= i < ni and 0 <= j < nj):
        convention = "one-based" if one_based else "zero-based"
        raise ValueError(
            f"{name}={point} is outside mask shape {shape} using {convention} indexing"
        )
    return i, j


def _bresenham_grid_line(
    start: tuple[int, int],
    stop: tuple[int, int],
) -> list[tuple[int, int]]:
    i0, j0 = start
    i1, j1 = stop
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    i, j = i0, j0
    out: list[tuple[int, int]] = []
    while True:
        out.append((i, j))
        if i == i1 and j == j1:
            return out
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj


def _resolve_inverse_reference(
    reference: tuple[int, int] | int,
    mask: np.ndarray,
    one_based: bool,
) -> int:
    if np.isscalar(reference):
        ref_idx = int(reference)
        if ref_idx < 0:
            raise ValueError("reference index must be non-negative")
        return ref_idx

    mask_arr = np.asarray(mask)
    point = _normalize_grid_point(reference, mask_arr.shape, one_based, "reference")
    idx = int(_active_vector_index_grid(mask_arr)[point])
    if idx < 0:
        raise ValueError(f"reference={reference} is not an active cell")
    return idx


__all__ = [
    "active_line_indices",
    "dense_inverse_line_values",
    "matern_precision_from_data_file",
    "matern_precision_from_data_file_3d",
    "plot_dense_inverse_line",
    "realization_to_grid",
    "realization_to_grid_3d",
    "sample_from_precision",
]
