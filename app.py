"""
Reconstructor 3D H&E para HDF5 piramidal usando PyVista (stpyvista).

Pipeline:
1) Detecta rutas piramidales con patron tXXXXX/sYY/N/cells.
2) Lee subvolumen de s00 (nuclei) y s01 (cyto) en modo lazy.
3) Aplica colorizacion H&E slice-a-slice con falseColor (colorization.py).
4) Renderiza en 3D con PyVista (ray casting volumetrico).
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.ndimage import zoom
import nest_asyncio
nest_asyncio.apply()

from colorization import falseColor

try:
    import pyvista as pv
except Exception as exc:  # pragma: no cover - control de import en runtime
    st.set_page_config(page_title="H&E 3D Pyramid Viewer", layout="wide")
    st.title("Reconstruccion 3D H&E")
    st.error(
        "No se pudo importar PyVista. Instala dependencias y reinicia la app."
    )
    st.code(
        "pip install pyvista",
        language="bash",
    )
    st.caption(f"Detalle de import: {exc}")
    st.stop()


MAX_RENDER_VOXELS_SAFE = 3_500_000
HE_CMAP = ["#fcfafa", "#f7dce8", "#e5b0cd", "#b278af", "#4e327a"]


st.set_page_config(page_title="H&E 3D Pyramid Viewer", layout="wide")
st.title("Reconstruccion 3D H&E desde HDF5 piramidal")
st.caption("PyVista + stpyvista | Canales: s00 (nuclei) y s01 (cyto)")


def _collect_datasets(h5_file: h5py.File) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(
                {
                    "name": name,
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )

    h5_file.visititems(visitor)
    return datasets


def _detect_pyramid(dataset_entries: list[dict[str, Any]]) -> dict[str, dict[str, dict[int, str]]]:
    """
    Detecta datasets con patron: tXXXXX/sYY/N/cells.
    Devuelve: {timepoint: {channel: {level: key}}}
    """
    pattern = re.compile(r"^(t\d+)/(s\d{2})/(\d+)/cells$")
    out: dict[str, dict[str, dict[int, str]]] = {}

    for entry in dataset_entries:
        match = pattern.match(entry["name"])
        if not match:
            continue

        timepoint, channel, level_str = match.groups()
        level = int(level_str)
        out.setdefault(timepoint, {}).setdefault(channel, {})[level] = entry["name"]

    return out


def _dataset_metadata(h5_file: h5py.File, key: str) -> dict[str, Any]:
    dset = h5_file[key]
    voxels = int(np.prod(dset.shape))
    item_size = int(np.dtype(dset.dtype).itemsize)
    return {
        "shape": tuple(dset.shape),
        "dtype": str(dset.dtype),
        "ndim": int(dset.ndim),
        "chunks": tuple(dset.chunks) if dset.chunks is not None else None,
        "compression": dset.compression,
        "bytes": int(voxels * item_size),
    }


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _estimate_shape_after_step(shape_3d: tuple[int, int, int], step: int) -> tuple[int, int, int]:
    return (
        max((shape_3d[0] + step - 1) // step, 1),
        max((shape_3d[1] + step - 1) // step, 1),
        max((shape_3d[2] + step - 1) // step, 1),
    )


def _normalize_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - mn) / (mx - mn)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _robust_positive_stats(vol: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vol, dtype=np.float32)
    positive = arr[arr > 0]
    if positive.size == 0:
        return {"p05": 0.0, "p50": 0.0, "p95": 1.0, "max": 1.0}

    p05, p50, p95 = np.percentile(positive, [5, 50, 95])
    return {
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "max": float(positive.max()),
    }


def _auto_falsecolor_params(
    nuclei_vol: np.ndarray,
    cyto_vol: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Estima parametros de falseColor segun intensidad real del subvolumen.

    Esto evita el problema de "todo blanco" cuando umbrales/norms fijos
    (pensados para otra escala) apagan completamente la señal.
    """
    nuc_stats = _robust_positive_stats(nuclei_vol)
    cyto_stats = _robust_positive_stats(cyto_vol)

    nuc_threshold = max(0, int(nuc_stats["p05"]))
    cyto_threshold = max(0, int(cyto_stats["p05"]))
    nuc_normfactor = max(1, int(nuc_stats["p95"] * 1.5))
    cyto_normfactor = max(1, int(cyto_stats["p95"] * 1.5))

    return nuc_threshold, cyto_threshold, nuc_normfactor, cyto_normfactor


def _normalize_scalar_robust(
    scalar_vol: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> tuple[np.ndarray, float, float]:
    """Normaliza escalar a [0,1] usando percentiles robustos."""
    arr = np.asarray(scalar_vol, dtype=np.float32)
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    lo_f = float(lo)
    hi_f = float(hi)

    if hi_f <= lo_f:
        return np.zeros_like(arr, dtype=np.float32), lo_f, hi_f

    out = (arr - lo_f) / (hi_f - lo_f)
    return np.clip(out, 0.0, 1.0).astype(np.float32), lo_f, hi_f


def _read_subset(
    dset: h5py.Dataset,
    z_start: int,
    z_stop: int,
    step: int,
    method: str,
    extra_indices: tuple[int, ...],
) -> np.ndarray:
    if dset.ndim < 3:
        raise ValueError("Dataset no volumetrico (ndim < 3).")

    extra_axes = dset.ndim - 3
    if len(extra_indices) != extra_axes:
        raise ValueError("Cantidad de indices extra inconsistente con dimensiones.")

    prefix = tuple(int(i) for i in extra_indices)

    if method == "slicing":
        selection = prefix + (
            slice(z_start, z_stop, step),
            slice(None, None, step),
            slice(None, None, step),
        )
        return np.ascontiguousarray(np.asarray(dset[selection]))

    # zoom: lee ROI completa y luego reescala
    selection = prefix + (slice(z_start, z_stop), slice(None), slice(None))
    subset = np.asarray(dset[selection], dtype=np.float32)
    factor = 1.0 / float(step)
    subset = zoom(subset, zoom=(factor, factor, factor), order=1, prefilter=False)
    return np.ascontiguousarray(subset)


def _false_color_volume(
    nuclei_vol: np.ndarray,
    cyto_vol: np.ndarray,
    nuc_threshold: int,
    cyto_threshold: int,
    nuc_normfactor: int,
    cyto_normfactor: int,
) -> np.ndarray:
    """Coloriza volumen Z,Y,X -> Z,Y,X,3 (uint8) usando falseColor por slice."""
    z_count, y_size, x_size = nuclei_vol.shape
    he_rgb = np.empty((z_count, y_size, x_size, 3), dtype=np.uint8)

    for i in range(z_count):
        nuc = nuclei_vol[i]
        cyto = cyto_vol[i]

        he_rgb[i] = falseColor(
            nuclei=nuc,
            cyto=cyto,
            nuc_threshold=nuc_threshold,
            cyto_threshold=cyto_threshold,
            nuc_normfactor=nuc_normfactor,
            cyto_normfactor=cyto_normfactor,
        )

    return he_rgb


def _he_scalar_from_rgb(he_rgb: np.ndarray) -> np.ndarray:
    """
    Deriva escalar desde RGB H&E para volumen en PyVista.
    Combina densidad optica y luminancia inversa para reforzar tejido.
    """
    rgb = np.clip(he_rgb.astype(np.float32) / 255.0, 1e-4, 1.0)

    # 1) Densidad optica total (resalta tincion)
    od = -np.log(rgb)
    stain = od[..., 0] + od[..., 1] + od[..., 2]

    # 2) Luminancia inversa (resalta nucleos oscuros)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    inv_lum = 1.0 - lum

    def robust_norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(arr, [1.0, 99.0])
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f <= lo_f:
            return np.zeros_like(arr, dtype=np.float32)
        out = (arr - lo_f) / (hi_f - lo_f)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    stain_n = robust_norm(stain)
    inv_lum_n = robust_norm(inv_lum)

    # Mezcla ponderada para mejorar visibilidad volumetrica.
    scalar = 0.75 * stain_n + 0.25 * inv_lum_n
    return np.ascontiguousarray(np.clip(scalar, 0.0, 1.0).astype(np.float32))


def _build_pyvista_plotter(
    he_scalar: np.ndarray,
    he_rgb: np.ndarray,
    isomin: float,
    spacing: tuple[float, float, float],
) -> pv.Plotter:
    """
    6 caras como quads texturizados con UV consistentes en aristas compartidas.

    Convención de ejes por cara:
      Z faces  → row=Y, col=X
      Y faces  → row=Z, col=X
      X faces  → row=Z, col=Y

    Verificación de arista ejemplo (X=0, Y=0, Z varía):
      Cara Y=0 col=0 → he_rgb[:,0,0]
      Cara X=0 col=0 → he_rgb[:,0,0]  ✓
    """
    z_dim, y_dim, x_dim = he_rgb.shape[:3]
    sz, sy, sx = spacing
    Lz, Ly, Lx = z_dim * sz, y_dim * sy, x_dim * sx

    plotter = pv.Plotter(window_size=[1100, 820], off_screen=True)
    plotter.set_background("#f5f6fb")

    # ─── UV fijos: siempre los mismos 4 puntos en el mismo orden de esquinas ───
    _FIXED_UV = np.array([
        [0.0, 1.0],   # esquina 0 → row=0,   col=0   (top-left)
        [1.0, 1.0],   # esquina 1 → row=0,   col=last(top-right)
        [1.0, 0.0],   # esquina 2 → row=last,col=last (bot-right)
        [0.0, 0.0],   # esquina 3 → row=last,col=0   (bot-left)
    ], dtype=np.float32)

    def _face_quad(
        img_rgb: np.ndarray,
        p_r0c0,   # mundo ↔ numpy (row=0,  col=0)
        p_r0cN,   # mundo ↔ numpy (row=0,  col=last)
        p_rNcN,   # mundo ↔ numpy (row=last,col=last)
        p_rNc0,   # mundo ↔ numpy (row=last,col=0)
    ) -> None:
        tex  = pv.Texture(np.ascontiguousarray(img_rgb, dtype=np.uint8))
        mesh = pv.PolyData()
        mesh.points = np.array([p_r0c0, p_r0cN, p_rNcN, p_rNc0], dtype=float)
        mesh.faces  = np.array([[4, 0, 1, 2, 3]])
        mesh.active_texture_coordinates = _FIXED_UV.copy()
        plotter.add_mesh(mesh, texture=tex, lighting=False, show_edges=False)

    def _pick_slice(arr: np.ndarray, idx: int, axis: int, margin: int = 4) -> np.ndarray:
        """
        Devuelve el slice en idx.
        Si es casi uniforme (std < 8 → fondo sin tejido),
        avanza `margin` posiciones hacia el interior del volumen.
        La cara aparecerá con tejido real y la discrepancia en la arista
        es imperceptible porque esa zona ya era casi blanca.
        """
        s = np.take(arr, idx, axis=axis)
        if float(np.std(s.astype(np.float32))) < 8.0:
            interior = int(np.clip(
                idx + margin if idx == 0 else idx - margin,
                0, arr.shape[axis] - 1
            ))
            s = np.take(arr, interior, axis=axis)
        return s

    # ── CARA Z=0 (base XY) ──────────────────────────────────────
    # Image shape: (Y, X, 3) — row=Y-axis, col=X-axis
    _face_quad(
        _pick_slice(he_rgb, 0, axis=0),
        p_r0c0=[0,   0,  0 ],   # Y=0,  X=0
        p_r0cN=[Lx,  0,  0 ],   # Y=0,  X=Lx
        p_rNcN=[Lx,  Ly, 0 ],   # Y=Ly, X=Lx
        p_rNc0=[0,   Ly, 0 ],   # Y=Ly, X=0
    )

    # ── CARA Z=Lz (tapa XY) ─────────────────────────────────────
    _face_quad(
        _pick_slice(he_rgb, -1, axis=0),
        p_r0c0=[0,   0,  Lz],
        p_r0cN=[Lx,  0,  Lz],
        p_rNcN=[Lx,  Ly, Lz],
        p_rNc0=[0,   Ly, Lz],
    )

    # ── CARA Y=0 (frente XZ) ────────────────────────────────────
    # Image shape: (Z, X, 3) — row=Z-axis, col=X-axis
    _face_quad(
        _pick_slice(he_rgb, 0, axis=1),
        p_r0c0=[0,  0, 0 ],   # Z=0,  X=0
        p_r0cN=[Lx, 0, 0 ],   # Z=0,  X=Lx
        p_rNcN=[Lx, 0, Lz],   # Z=Lz, X=Lx
        p_rNc0=[0,  0, Lz],   # Z=Lz, X=0
    )

    # ── CARA Y=Ly (trasera XZ) ──────────────────────────────────
    _face_quad(
        _pick_slice(he_rgb, -1, axis=1),
        p_r0c0=[0,  Ly, 0 ],
        p_r0cN=[Lx, Ly, 0 ],
        p_rNcN=[Lx, Ly, Lz],
        p_rNc0=[0,  Ly, Lz],
    )

    # ── CARA X=0 (izquierda YZ) ─────────────────────────────────
    # Image shape: (Z, Y, 3) — row=Z-axis, col=Y-axis
    _face_quad(
        _pick_slice(he_rgb, 0, axis=2),
        p_r0c0=[0, 0,  0 ],   # Z=0,  Y=0
        p_r0cN=[0, Ly, 0 ],   # Z=0,  Y=Ly
        p_rNcN=[0, Ly, Lz],   # Z=Lz, Y=Ly
        p_rNc0=[0, 0,  Lz],   # Z=Lz, Y=0
    )

    # ── CARA X=Lx (derecha YZ) ──────────────────────────────────
    _face_quad(
        _pick_slice(he_rgb, -1, axis=2),
        p_r0c0=[Lx, 0,  0 ],
        p_r0cN=[Lx, Ly, 0 ],
        p_rNcN=[Lx, Ly, Lz],
        p_rNc0=[Lx, 0,  Lz],
    )

    plotter.camera_position = "iso"
    plotter.reset_camera()
    return plotter

@st.cache_data(show_spinner=False)
def list_datasets_from_path(file_path: str) -> list[dict[str, Any]]:
    with h5py.File(file_path, "r") as h5_file:
        return _collect_datasets(h5_file)


@st.cache_data(show_spinner=False)
def list_datasets_from_bytes(file_bytes: bytes) -> list[dict[str, Any]]:
    with h5py.File(io.BytesIO(file_bytes), "r") as h5_file:
        return _collect_datasets(h5_file)


@st.cache_data(show_spinner=False)
def metadata_from_path(file_path: str, key: str) -> dict[str, Any]:
    with h5py.File(file_path, "r") as h5_file:
        return _dataset_metadata(h5_file, key)


@st.cache_data(show_spinner=False)
def metadata_from_bytes(file_bytes: bytes, key: str) -> dict[str, Any]:
    with h5py.File(io.BytesIO(file_bytes), "r") as h5_file:
        return _dataset_metadata(h5_file, key)


def _load_two_channels_from_path(
    file_path: str,
    cyto_key: str,
    nuclei_key: str,
    z_start: int,
    z_stop: int,
    step: int,
    method: str,
    extra_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(file_path, "r") as h5_file:
        cyto = _read_subset(h5_file[cyto_key], z_start, z_stop, step, method, extra_indices)
        nuclei = _read_subset(
            h5_file[nuclei_key], z_start, z_stop, step, method, extra_indices
        )
    return cyto, nuclei


def _load_two_channels_from_bytes(
    file_bytes: bytes,
    cyto_key: str,
    nuclei_key: str,
    z_start: int,
    z_stop: int,
    step: int,
    method: str,
    extra_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(io.BytesIO(file_bytes), "r") as h5_file:
        cyto = _read_subset(h5_file[cyto_key], z_start, z_stop, step, method, extra_indices)
        nuclei = _read_subset(
            h5_file[nuclei_key], z_start, z_stop, step, method, extra_indices
        )
    return cyto, nuclei


def _source_ui() -> tuple[str | None, bytes | None, str | None]:
    st.sidebar.header("Fuente HDF5")
    uploaded = st.sidebar.file_uploader("Sube archivo .h5/.hdf5", type=["h5", "hdf5"])
    local_path = st.sidebar.text_input(
        "O ruta local",
        value=r"C:\Users\Estudiante.DESKTOP-AAS1OV7\Downloads\data-f0.h5",
    )

    mode: str | None = None
    file_bytes: bytes | None = None
    path_str: str | None = None

    if uploaded is not None:
        mode = "uploaded"
        file_bytes = uploaded.getvalue()
        st.sidebar.success(f"Archivo cargado: {uploaded.name}")
        st.sidebar.warning(
            "Para 22GB usa ruta local. El uploader deja el archivo completo en memoria."
        )
    elif local_path.strip():
        path = Path(local_path.strip())
        if path.exists() and path.is_file():
            mode = "path"
            path_str = str(path)
            st.sidebar.success("Ruta valida")
        else:
            st.sidebar.error("Ruta invalida")

    return mode, file_bytes, path_str


mode, file_bytes, path_str = _source_ui()
if mode is None:
    st.info("Selecciona un archivo HDF5 por ruta local o uploader.")
    st.stop()


try:
    with st.spinner("Leyendo estructura del HDF5..."):
        if mode == "uploaded":
            entries = list_datasets_from_bytes(file_bytes or b"")
        else:
            entries = list_datasets_from_path(path_str or "")
except Exception as exc:
    st.error(f"No se pudo abrir el HDF5: {exc}")
    st.stop()

if not entries:
    st.error("No se detectaron datasets en el HDF5.")
    st.stop()


pyramid = _detect_pyramid(entries)
valid_timepoints: list[str] = []
for tp, channels in pyramid.items():
    if "s00" in channels and "s01" in channels:
        common = sorted(set(channels["s00"]).intersection(channels["s01"]))
        if common:
            valid_timepoints.append(tp)

if not valid_timepoints:
    st.error("No se encontraron rutas tXXXXX/s00/N/cells y tXXXXX/s01/N/cells compatibles.")
    st.stop()

st.sidebar.header("Seleccion piramidal")
selected_tp = st.sidebar.selectbox("Timepoint", sorted(valid_timepoints))
common_levels = sorted(
    set(pyramid[selected_tp]["s00"]).intersection(pyramid[selected_tp]["s01"])
)
default_level = 4 if 4 in common_levels else common_levels[0]
selected_level = st.sidebar.selectbox(
    "Nivel", common_levels, index=common_levels.index(default_level)
)

cyto_key = pyramid[selected_tp]["s01"][selected_level]
nuclei_key = pyramid[selected_tp]["s00"][selected_level]

st.sidebar.caption(f"cyto: {cyto_key}")
st.sidebar.caption(f"nuclei: {nuclei_key}")


try:
    if mode == "uploaded":
        cyto_meta = metadata_from_bytes(file_bytes or b"", cyto_key)
        nuclei_meta = metadata_from_bytes(file_bytes or b"", nuclei_key)
    else:
        cyto_meta = metadata_from_path(path_str or "", cyto_key)
        nuclei_meta = metadata_from_path(path_str or "", nuclei_key)
except Exception as exc:
    st.error(f"No se pudo leer metadata de canales: {exc}")
    st.stop()

if cyto_meta["ndim"] != nuclei_meta["ndim"] or cyto_meta["ndim"] < 3:
    st.error("Canales incompatibles: deben tener mismo ndim y al menos 3 dimensiones.")
    st.stop()

shape_c = cyto_meta["shape"]
shape_n = nuclei_meta["shape"]
vol_shape = (
    min(int(shape_c[-3]), int(shape_n[-3])),
    min(int(shape_c[-2]), int(shape_n[-2])),
    min(int(shape_c[-1]), int(shape_n[-1])),
)

st.caption(
    "Tamano teorico canales: "
    f"cyto={_human_bytes(int(cyto_meta['bytes']))}, "
    f"nuclei={_human_bytes(int(nuclei_meta['bytes']))}"
)


st.sidebar.markdown("### Carga subvolumen")
downsample_step = st.sidebar.select_slider(
    "Downsampling",
    options=[1, 2, 4],
    value=2,
    help="1 = sin downsampling, 2 y 4 para reducir carga.",
)

downsample_method = st.sidebar.radio(
    "Metodo",
    options=["slicing", "zoom"],
    index=0,
    help="Para maxima estabilidad usa slicing.",
)

z_max = vol_shape[0] - 1
if z_max <= 0:
    z_range = (0, 0)
else:
    default_end = min(z_max, 255)
    z_range = st.sidebar.slider("Rango Z", 0, z_max, (0, default_end))

z_start, z_end = int(z_range[0]), int(z_range[1])
z_stop = z_end + 1

extra_axes = cyto_meta["ndim"] - 3
extra_indices: list[int] = []
if extra_axes > 0:
    st.sidebar.markdown("### Ejes extra")
    for axis in range(extra_axes):
        axis_max = min(int(shape_c[axis]), int(shape_n[axis])) - 1
        val = st.sidebar.number_input(
            f"Indice eje {axis}",
            min_value=0,
            max_value=axis_max,
            value=0,
            step=1,
            key=f"extra_axis_{axis}",
        )
        extra_indices.append(int(val))

roi_shape = (z_stop - z_start, vol_shape[1], vol_shape[2])
est_shape = _estimate_shape_after_step(roi_shape, downsample_step)
est_voxels = int(np.prod(est_shape))
est_peak = int(est_voxels * (4 + 4 + 3 + 4))

st.sidebar.info(
    "Estimacion\n"
    f"shape={est_shape}\n"
    f"voxels={est_voxels:,}\n"
    f"RAM pico aprox={_human_bytes(est_peak)}"
)

allow_heavy = False
if est_voxels > MAX_RENDER_VOXELS_SAFE:
    st.sidebar.warning(
        "La seleccion es pesada para un PC modesto. Reduce Z o usa downsampling 2/4."
    )
    allow_heavy = st.sidebar.checkbox("Forzar carga pesada", value=False)
    if not allow_heavy:
        st.stop()


try:
    with st.spinner("Leyendo s00/s01 desde HDF5..."):
        if mode == "uploaded":
            cyto_vol, nuclei_vol = _load_two_channels_from_bytes(
                file_bytes=file_bytes or b"",
                cyto_key=cyto_key,
                nuclei_key=nuclei_key,
                z_start=z_start,
                z_stop=z_stop,
                step=downsample_step,
                method=downsample_method,
                extra_indices=tuple(extra_indices),
            )
        else:
            cyto_vol, nuclei_vol = _load_two_channels_from_path(
                file_path=path_str or "",
                cyto_key=cyto_key,
                nuclei_key=nuclei_key,
                z_start=z_start,
                z_stop=z_stop,
                step=downsample_step,
                method=downsample_method,
                extra_indices=tuple(extra_indices),
            )
except Exception as exc:
    st.error(f"Error al leer subvolumen de canales: {exc}")
    st.stop()


# Fuerza shape comun por seguridad
z_min = min(cyto_vol.shape[0], nuclei_vol.shape[0])
y_min = min(cyto_vol.shape[1], nuclei_vol.shape[1])
x_min = min(cyto_vol.shape[2], nuclei_vol.shape[2])
cyto_vol = np.ascontiguousarray(cyto_vol[:z_min, :y_min, :x_min])
nuclei_vol = np.ascontiguousarray(nuclei_vol[:z_min, :y_min, :x_min])

if not np.isfinite(cyto_vol).all() or not np.isfinite(nuclei_vol).all():
    st.warning("Se detectaron NaN/Inf en canales y se reemplazaron por 0.")
    cyto_vol = np.nan_to_num(cyto_vol, nan=0.0, posinf=0.0, neginf=0.0)
    nuclei_vol = np.nan_to_num(nuclei_vol, nan=0.0, posinf=0.0, neginf=0.0)


st.sidebar.markdown("### Colorizacion H&E")
auto_he = st.sidebar.checkbox("Auto-ajuste H&E", value=True)

auto_nuc_t, auto_cyto_t, auto_nuc_n, auto_cyto_n = _auto_falsecolor_params(
    nuclei_vol=nuclei_vol,
    cyto_vol=cyto_vol,
)

if auto_he:
    nuc_threshold = auto_nuc_t
    cyto_threshold = auto_cyto_t
    nuc_normfactor = auto_nuc_n
    cyto_normfactor = auto_cyto_n
    st.sidebar.caption(
        "Auto params\n"
        f"nuc_t={nuc_threshold}, cyto_t={cyto_threshold}\n"
        f"nuc_norm={nuc_normfactor}, cyto_norm={cyto_normfactor}"
    )
else:
    max_thr = int(max(50, np.percentile(np.concatenate([nuclei_vol.ravel(), cyto_vol.ravel()]), 99.9)))
    max_norm = int(max(1000, auto_nuc_n * 6, auto_cyto_n * 6))
    step_norm = max(1, max_norm // 300)

    nuc_threshold = st.sidebar.slider("Umbral nuclei", 0, max_thr, int(auto_nuc_t), 1)
    cyto_threshold = st.sidebar.slider("Umbral cyto", 0, max_thr, int(auto_cyto_t), 1)
    nuc_normfactor = st.sidebar.slider(
        "Norm nuclei", 1, max_norm, int(auto_nuc_n), step_norm
    )
    cyto_normfactor = st.sidebar.slider(
        "Norm cyto", 1, max_norm, int(auto_cyto_n), step_norm
    )

# CLAHE removed per user preference (kept disabled internally)


with st.spinner("Aplicando colorizacion FalseColor H&E..."):
    he_rgb = _false_color_volume(
        nuclei_vol=nuclei_vol,
        cyto_vol=cyto_vol,
        nuc_threshold=nuc_threshold,
        cyto_threshold=cyto_threshold,
        nuc_normfactor=nuc_normfactor,
        cyto_normfactor=cyto_normfactor,
    )

he_scalar = _he_scalar_from_rgb(he_rgb)
if not np.isfinite(he_scalar).all():
    st.warning("Se detectaron NaN/Inf en volumen H&E escalar y se reemplazaron por 0.")
    he_scalar = np.nan_to_num(he_scalar, nan=0.0, posinf=0.0, neginf=0.0)

signal_p99 = float(np.percentile(he_scalar, 99.0))
if signal_p99 < 0.03:
    st.warning(
        "Senal H&E muy baja detectada; se aplica ajuste automatico de rescate para evitar render blanco."
    )
    rescue_nuc_t = 0
    rescue_cyto_t = 0
    rescue_nuc_n = max(1, int(nuc_normfactor // 4))
    rescue_cyto_n = max(1, int(cyto_normfactor // 4))

    with st.spinner("Reintentando colorizacion con parametros de rescate..."):
        he_rgb = _false_color_volume(
            nuclei_vol=nuclei_vol,
            cyto_vol=cyto_vol,
            nuc_threshold=rescue_nuc_t,
            cyto_threshold=rescue_cyto_t,
            nuc_normfactor=rescue_nuc_n,
            cyto_normfactor=rescue_cyto_n,
        )
    he_scalar = _he_scalar_from_rgb(he_rgb)
    he_scalar = np.nan_to_num(he_scalar, nan=0.0, posinf=0.0, neginf=0.0)

he_scalar_norm, clip_low, clip_high = _normalize_scalar_robust(he_scalar, low_pct=1.0, high_pct=99.0)

# Suprime fondo residual para que el volumen no se pierda en "haze".
positive = he_scalar_norm[he_scalar_norm > 0]
if positive.size > 0:
    floor = float(np.percentile(positive, 5.0))
    he_scalar_norm = np.where(he_scalar_norm >= max(0.01, floor * 0.7), he_scalar_norm, 0.0)

vmin = float(he_scalar_norm.min())
vmax = float(he_scalar_norm.max())
if vmax <= vmin:
    st.error("El volumen resultante es constante; revisa parametros de colorizacion.")
    st.stop()

p10, p99 = np.percentile(he_scalar_norm, [10, 99])

st.sidebar.markdown("### Render 3D Médico (itkwidgets)")

# --- Opacidad y clipping ---
isomin = st.sidebar.slider(
    "Clip mínimo (isomin)", float(vmin), float(vmax), float(p10), 0.005,
    help="Voxeles por debajo de este valor son transparentes. El resto será totalmente opaco."
)

# --- Calidad de renderizado ---
# Modo de fusión fijado a 'composite' (no editable)
blend_mode = "composite"

# --- Spacing físico (anisotrópico) ---
st.sidebar.markdown("#### Spacing físico (micras/voxel)")
# Spacing fijado a 1.0 y no editable por el usuario
sz = st.sidebar.number_input("Spacing Z", min_value=0.1, max_value=50.0, value=1.0, step=0.1, disabled=True,
    help="Separación entre slices. Fijado a 1 µm/voxel.")
sy = st.sidebar.number_input("Spacing Y", min_value=0.1, max_value=10.0, value=1.0, step=0.1, disabled=True)
sx = st.sidebar.number_input("Spacing X", min_value=0.1, max_value=10.0, value=1.0, step=0.1, disabled=True)
voxel_spacing = (float(sz), float(sy), float(sx))


m1, m2, m3 = st.columns(3)
m1.metric("Shape subvolumen", str(tuple(he_scalar.shape)))
m2.metric("Voxels", f"{int(np.prod(he_scalar.shape)):,}")
m3.metric("RAM H&E RGB", _human_bytes(int(he_rgb.nbytes)))
st.caption(f"Normalizacion robusta H&E escalar: p1={clip_low:.4f}, p99={clip_high:.4f}")
active_ratio = float((he_scalar_norm > 0.02).mean() * 100.0)
st.caption(f"Voxels activos (>0.02): {active_ratio:.2f}%")

preview_idx = he_rgb.shape[0] // 2
c1, c2, c3 = st.columns(3)
c1.image(_normalize_uint8(cyto_vol[preview_idx]), caption="Canal cyto s01", use_container_width=True)
c2.image(_normalize_uint8(nuclei_vol[preview_idx]), caption="Canal nuclei s00", use_container_width=True)
c3.image(he_rgb[preview_idx], caption="FalseColor H&E", use_container_width=True)


with st.spinner("Renderizando volumen 3D con PyVista..."):
    plotter = _build_pyvista_plotter(
        he_scalar=he_scalar_norm,
        he_rgb=he_rgb,
        isomin=float(isomin),
        spacing=voxel_spacing,
    )

st.success("Renderizando con PyVista WebGL completado. Interactúa directamente en el panel.")
try:
    html_obj = plotter.export_html(filename=None)
    html_str = html_obj.read() if hasattr(html_obj, "read") else str(html_obj)
    components.html(html_str, height=840)
except Exception as exc:
    st.warning("No se pudo iniciar el WebGL embebido con stpyvista. Fallback a captura estática.")
    screenshot = plotter.screenshot(return_img=True)
    st.image(screenshot, caption="Vista estática", use_container_width=True)
    st.caption(f"Detalle técnico: {exc}")

st.markdown(
    """
### Configuracion recomendada (PC modesto)
- Fuente: ruta local.
- Nivel piramidal: 4 o superior.
- Downsampling: 2 (o 1 solo si Z es pequeno).
- Metodo: slicing.
- Rango Z inicial: 0-128 o 0-256.
"""
)
