"""
Visualizador 3D — OTLS + FalseColor H&E
Autores: Daniel Yaruro, Juan Mantilla
Fecha: Enero 2025
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import base64
from pathlib import Path
import time
import re
import cv2
import streamlit.components.v1 as components


# ============================================================
# FALSECOLOR ORIGINAL (Giacomelli et al., 2020)
# ============================================================

HE_COLOR_SETTINGS = {
    'nuclei': [0.17, 0.27, 0.105],
    'cyto':   [0.05, 1.0,  0.54],
}
BETA_DICT = {
    'K_nuclei': 0.08,
    'K_cyto':   0.012,
}


def preProcess(image, threshold=50, normfactor=None):
    image = image.astype(float)
    image -= threshold
    image[image < 0] = 0
    image = np.power(image, 0.85)
    if normfactor is None:
        foreground = image[image > threshold]
        if len(foreground) > 0:
            normfactor = np.mean(foreground) * 8
        else:
            normfactor = 1
    if normfactor == 0:
        normfactor = 1
    processed_image = image * (65535 / normfactor) * (255 / 65535)
    return processed_image


def falseColor(nuclei, cyto,
               nuc_threshold=50, cyto_threshold=50,
               nuc_normfactor=5000, cyto_normfactor=2000):
    constants_nuclei = HE_COLOR_SETTINGS['nuclei']
    k_nuclei = BETA_DICT['K_nuclei']
    constants_cyto = HE_COLOR_SETTINGS['cyto']
    k_cytoplasm = BETA_DICT['K_cyto']

    nuclei = preProcess(nuclei.astype(float), threshold=nuc_threshold,
                        normfactor=nuc_normfactor)
    cyto = preProcess(cyto.astype(float), threshold=cyto_threshold,
                      normfactor=cyto_normfactor)

    RGB_image = np.zeros((3, nuclei.shape[0], nuclei.shape[1]))
    for i in range(3):
        tmp_c = constants_cyto[i] * k_cytoplasm * cyto
        tmp_n = constants_nuclei[i] * k_nuclei * nuclei
        RGB_image[i] = 255 * np.multiply(np.exp(-tmp_c), np.exp(-tmp_n))

    RGB_image = np.moveaxis(RGB_image, 0, -1)
    return np.clip(RGB_image, 0, 255).astype(np.uint8)


def applyCLAHE(image, clip_limit=0.048, tile_size=(8, 8)):
    clahe = cv2.createCLAHE(tileGridSize=tile_size, clipLimit=clip_limit)
    image_16 = image.astype(np.uint16)
    equalized = clahe.apply(image_16)
    if equalized.max() > 0:
        final = (image_16.max()) * (equalized / equalized.max())
    else:
        final = equalized
    return final.astype(np.uint16)


def make_otls_rgb(nuc_img, cyto_img):
    """Crea imagen RGB de fluorescencia OTLS."""
    nuc_f = nuc_img.astype(float)
    cyto_f = cyto_img.astype(float)
    if nuc_f.max() > 0:
        nuc_8 = np.clip(nuc_f / nuc_f.max() * 255, 0, 255).astype(np.uint8)
    else:
        nuc_8 = np.zeros_like(nuc_img, dtype=np.uint8)
    if cyto_f.max() > 0:
        cyto_8 = np.clip(cyto_f / cyto_f.max() * 255, 0, 255).astype(np.uint8)
    else:
        cyto_8 = np.zeros_like(cyto_img, dtype=np.uint8)

    rgb = np.zeros((*nuc_8.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = cyto_8
    rgb[:, :, 1] = cyto_8 // 2
    rgb[:, :, 2] = nuc_8
    return rgb


class SuperResolution:
    @staticmethod
    def apply(img, scale=2, sharpen_strength=0.4):
        if scale not in (2, 4):
            raise ValueError("scale debe ser 2 o 4")
        h, w = img.shape[:2]
        upscaled = cv2.resize(img, (w * scale, h * scale),
                              interpolation=cv2.INTER_LANCZOS4)
        if sharpen_strength > 0:
            blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2.0)
            upscaled = cv2.addWeighted(
                upscaled, 1.0 + sharpen_strength,
                blurred, -sharpen_strength, 0
            )
            upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
        return upscaled


# ============================================================
# APP STREAMLIT
# ============================================================

st.set_page_config(
    page_title="OTLS FalseColor 3D",
    page_icon="",
    layout="wide"
)

st.title(" Visualización 3D — OTLS / FalseColor H&E")
st.markdown(
    "Dataset pca-bx-3dpathology · "
    "FalseColor-Python (Giacomelli et al., 2020) · "
    "Volume Rendering GPU"
)

# ==================== SIDEBAR ====================
st.sidebar.header(" Configuración")

st.sidebar.markdown("###  Dataset OTLS")
data_path = st.sidebar.text_input(
    "Ruta de los cortes:",
    r"C:\Users\ASUS STRIX\Documents\Trabajo de grado\H&E_Stained_Tiles\Raw_OTLS_Fluorescence_Slices",
)

st.sidebar.markdown("###  Tipo de imagen")
use_16bit = st.sidebar.radio(
    "Profundidad de bit:",
    ["16-bit (u16) — Mayor calidad", "8-bit (u8) — Preview"],
    index=0,
)
is_16bit = "16-bit" in use_16bit

st.sidebar.markdown("###  Modo de visualización")
view_mode = st.sidebar.radio(
    "Mostrar como:",
    ["FalseColor H&E", "OTLS Fluorescencia"],
    index=0,
    help=(
        "FalseColor H&E: colores rosa/púrpura (Beer-Lambert)\n"
        "OTLS Fluorescencia: canales originales (fondo negro)"
    )
)
is_falsecolor = view_mode == "FalseColor H&E"

st.sidebar.markdown("###  Cortes")
num_imagenes = st.sidebar.slider("Número de cortes:", 5, 50, 30, 5)

st.sidebar.markdown("###  Downsample")
downsample_factor = st.sidebar.select_slider(
    "Factor:", options=[1, 2, 4, 8], value=2,
    format_func=lambda x: f"{x}x"
)

if is_falsecolor:
    st.sidebar.markdown("### Parámetros FalseColor")
    st.sidebar.info(
        "**Constantes del paper:**\n"
        "- Hematoxilina: [0.17, 0.27, 0.105]\n"
        "- Eosina: [0.05, 1.0, 0.54]\n"
        "- K_nuclei: 0.08 · K_cyto: 0.012"
    )
    nuc_threshold = st.sidebar.slider("Umbral fondo nuclear:", 0, 500, 50, 10)
    cyto_threshold = st.sidebar.slider("Umbral fondo citoplasma:", 0, 500, 50, 10)
    nuc_normfactor = st.sidebar.slider("Norm nuclear:", 1000, 30000, 8500, 500)
    cyto_normfactor = st.sidebar.slider("Norm citoplasma:", 500, 15000, 2000, 250)
    use_clahe = st.sidebar.checkbox("CLAHE pre-colorización", value=False)
    clahe_clip = st.sidebar.slider(
        "Clip CLAHE:", 0.01, 0.2, 0.048, 0.005
    ) if use_clahe else 0.048
else:
    nuc_threshold = 50
    cyto_threshold = 50
    nuc_normfactor = 8500
    cyto_normfactor = 2000
    use_clahe = False
    clahe_clip = 0.048

st.sidebar.markdown("###  Super-Resolución")
enable_sr = st.sidebar.checkbox("Aplicar SR", value=False)
sr_scale = st.sidebar.selectbox("Factor SR:", [2, 4], index=0) if enable_sr else 1
sr_sharpen = st.sidebar.slider("Nitidez SR:", 0.0, 1.0, 0.4, 0.1) if enable_sr else 0.4

st.sidebar.markdown("###  Volumen 3D")
ray_steps = st.sidebar.slider("Pasos ray-march:", 50, 600, 250, 10)

if is_falsecolor:
    vol_density = st.sidebar.slider("Densidad:", 0.5, 8.0, 2.5, 0.5,
        help="Opacidad del tejido. Valores bajos (1.5–3.0) son más fieles a la imagen 2D")
    vol_bg_cutoff = st.sidebar.slider("Corte de fondo:", 0.01, 0.30, 0.03, 0.01,
        help="Umbral para ignorar fondo blanco. Bajar para mostrar tejido más pálido")
    vol_saturation = st.sidebar.slider("Saturación:", 0.5, 2.0, 1.0, 0.05,
        help="1.0 = idéntico a la imagen 2D. Subir ligeramente (1.1–1.3) para 3D")
else:
    vol_density = st.sidebar.slider("Densidad:", 0.5, 8.0, 3.0, 0.5)
    vol_bg_cutoff = st.sidebar.slider("Corte de fondo:", 0.001, 0.10, 0.02, 0.005)
    vol_saturation = st.sidebar.slider("Brillo:", 0.5, 5.0, 2.0, 0.1)

z_scale = st.sidebar.slider(
    "Escala Z (profundidad):", 0.5, 15.0, 5.0, 0.5,
    help="Amplifica la profundidad para que no se vea plano"
)
display_size = st.sidebar.slider("Tamaño (px):", 200, 1200, 600, 50)


# ==================== BUSCAR PARES ====================
def find_slice_pairs(folder: Path, use_u16: bool):
  valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
  suffix = "u16" if use_u16 else "u8"

  def extract_idx(text: str):
    match = re.search(r"(\d{4,})", text)
    return int(match.group(1)) if match else None

  def files_by_index(channel_folder: Path, require_suffix: bool):
    idx_map = {}
    for f in channel_folder.iterdir():
      if not f.is_file():
        continue
      name = f.name.lower()
      if "preview" in name or f.suffix.lower() not in valid_exts:
        continue
      if require_suffix and suffix not in name:
        continue
      idx = extract_idx(f.stem.lower()) or extract_idx(name)
      if idx is not None:
        idx_map[idx] = f
    return idx_map

  def find_flat_layout(require_suffix: bool):
    nuclei_files = {}
    for f in folder.iterdir():
      if not f.is_file():
        continue
      name = f.name.lower()
      if "nuclei" not in name or "preview" in name:
        continue
      if f.suffix.lower() not in valid_exts:
        continue
      if require_suffix and suffix not in name:
        continue
      idx = extract_idx(name)
      if idx is not None:
        nuclei_files[idx] = f

    pairs = []
    for idx in sorted(nuclei_files.keys()):
      nuc_path = nuclei_files[idx]
      cyto_name = re.sub(r"nuclei", "cyto", nuc_path.name, flags=re.IGNORECASE)
      cyto_path = folder / cyto_name
      if cyto_path.exists():
        pairs.append((nuc_path, cyto_path, idx))
    return pairs

  def find_split_layout(require_suffix: bool):
    subdirs = [d for d in folder.iterdir() if d.is_dir()]
    nuclei_dirs = [d for d in subdirs if "nuclei" in d.name.lower()]
    cyto_dirs = [d for d in subdirs if "cyto" in d.name.lower()]
    if not nuclei_dirs or not cyto_dirs:
      return []

    pairs = []
    for nuc_dir in sorted(nuclei_dirs, key=lambda p: p.name.lower()):
      expected_cyto_name = re.sub(
        r"nuclei", "cyto", nuc_dir.name, flags=re.IGNORECASE
      ).lower()
      cyto_dir = next(
        (d for d in cyto_dirs if d.name.lower() == expected_cyto_name),
        None,
      )
      if cyto_dir is None and len(cyto_dirs) == 1:
        cyto_dir = cyto_dirs[0]
      if cyto_dir is None:
        continue

      nuclei_files = files_by_index(nuc_dir, require_suffix)
      cyto_files = files_by_index(cyto_dir, require_suffix)
      common_idxs = sorted(set(nuclei_files.keys()) & set(cyto_files.keys()))
      for idx in common_idxs:
        pairs.append((nuclei_files[idx], cyto_files[idx], idx))
    return pairs

  # 1) Formato original: pares en la misma carpeta
  pairs = find_flat_layout(require_suffix=True)
  if pairs:
    return pairs

  # 2) Formato alternativo: subcarpetas separadas por canal
  pairs = find_split_layout(require_suffix=True)
  if pairs:
    return pairs

  # 3) Fallback: archivos sin etiqueta u16/u8 en nombre
  pairs = find_flat_layout(require_suffix=False)
  if pairs:
    return pairs

  pairs = find_split_layout(require_suffix=False)
  return pairs


# ==================== CARGA ====================
@st.cache_data(show_spinner=False)
def load_and_colorize(
    path_str, n, resize_factor, use_u16, do_falsecolor,
    nuc_thresh, cyto_thresh, nuc_norm, cyto_norm,
    clahe, clahe_clip_val,
    super_res, sr_scale_val, sr_sharpen_val,
):
    folder = Path(path_str)
    if not folder.exists():
        return None, None, f"Carpeta no encontrada: {folder}"

    pairs = find_slice_pairs(folder, use_u16)
    if not pairs:
      pairs = find_slice_pairs(folder, not use_u16)
      if not pairs:
        all_files = [
          str(f.relative_to(folder))
          for f in folder.rglob("*")
          if f.is_file()
          and f.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        ]
        return None, None, (
          "No se encontraron pares nuclei/cyto.\n"
          "Formato soportado: misma carpeta o subcarpetas nuclei/cyto.\n"
          f"Archivos ({len(all_files)}): "
          + ", ".join(all_files[:5])
        )

    pairs = pairs[:n]
    gray_list = []
    color_list = []
    errors = []

    for pair_idx, (nuc_path, cyto_path, idx) in enumerate(pairs):
        try:
            nuc_img = cv2.imread(str(nuc_path), cv2.IMREAD_UNCHANGED)
            cyto_img = cv2.imread(str(cyto_path), cv2.IMREAD_UNCHANGED)

            if nuc_img is None or cyto_img is None:
                errors.append(f"Corte {idx}: no se pudo leer")
                continue

            if nuc_img.ndim == 3:
                nuc_img = nuc_img[:, :, 0]
            if cyto_img.ndim == 3:
                cyto_img = cyto_img[:, :, 0]

            if nuc_img.size == 0 or cyto_img.size == 0:
                continue

            if nuc_img.shape != cyto_img.shape:
                min_h = min(nuc_img.shape[0], cyto_img.shape[0])
                min_w = min(nuc_img.shape[1], cyto_img.shape[1])
                nuc_img = nuc_img[:min_h, :min_w]
                cyto_img = cyto_img[:min_h, :min_w]

            if resize_factor > 1:
                h, w = nuc_img.shape
                new_h = max(h // resize_factor, 1)
                new_w = max(w // resize_factor, 1)
                nuc_img = cv2.resize(nuc_img, (new_w, new_h),
                                     interpolation=cv2.INTER_AREA)
                cyto_img = cv2.resize(cyto_img, (new_w, new_h),
                                      interpolation=cv2.INTER_AREA)

            if clahe:
                nuc_img = applyCLAHE(nuc_img.astype(np.uint16),
                                     clip_limit=clahe_clip_val)
                cyto_img = applyCLAHE(cyto_img.astype(np.uint16),
                                      clip_limit=clahe_clip_val)

            if do_falsecolor:
                colored = falseColor(
                    nuc_img, cyto_img,
                    nuc_threshold=nuc_thresh,
                    cyto_threshold=cyto_thresh,
                    nuc_normfactor=nuc_norm,
                    cyto_normfactor=cyto_norm,
                )
            else:
                colored = make_otls_rgb(nuc_img, cyto_img)

            if colored is None or colored.size == 0:
                continue

            gray = (
                0.2126 * colored[:, :, 0].astype(float) +
                0.7152 * colored[:, :, 1].astype(float) +
                0.0722 * colored[:, :, 2].astype(float)
            ).astype(np.uint8)

            gray_list.append(gray)
            color_list.append(colored)

        except Exception as e:
            errors.append(f"Corte {idx}: {type(e).__name__}: {str(e)}")
            continue

    if not gray_list:
        error_msg = f"No se procesaron imágenes. Pares: {len(pairs)}\n"
        if errors:
            for e in errors[:5]:
                error_msg += f"  • {e}\n"
        return None, None, error_msg

    stack_gray = np.stack(gray_list, axis=0).astype(np.uint8)
    stack_rgb = np.stack(color_list, axis=0).astype(np.uint8)

    if super_res and sr_scale_val > 1:
        sr_rgb, sr_gray = [], []
        for c, g in zip(stack_rgb, stack_gray):
            sr_rgb.append(SuperResolution.apply(c, sr_scale_val, sr_sharpen_val))
            sr_gray.append(SuperResolution.apply(g, sr_scale_val, 0.0))
        stack_rgb = np.stack(sr_rgb, axis=0)
        stack_gray = np.stack(sr_gray, axis=0)

    return stack_gray, stack_rgb, None


# ==================== RENDERIZADOR 3D ====================
def render_volume(stack_rgb, height, steps, density, bg_cutoff,
                  saturation, z_scale_val, is_fc):
    if stack_rgb is None:
        st.warning("No hay datos.")
        return

    z, y, x, c = stack_rgb.shape
    assert c == 3

    rgba = np.ones((z, y, x, 4), dtype=np.uint8) * 255
    rgba[..., :3] = np.ascontiguousarray(stack_rgb)
    b64 = base64.b64encode(
        np.ascontiguousarray(rgba).tobytes()
    ).decode("ascii")

    if is_fc:
        bg_color   = "0xf8f6f7"
        edge_color = "0xb0a0a8"
        mode_label = "FalseColor H&E"
        is_fc_js   = "1"
    else:
        bg_color   = "0x0a0a0f"
        edge_color = "0x223344"
        mode_label = "OTLS Fluorescencia"
        is_fc_js   = "0"

    html = (
'''<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;background:#'''
+ bg_color[2:]
+ ''';overflow:hidden;font-family:"Courier New",monospace}
#c{width:100%;height:100%;display:block}

/* ── Panel flotante ── */
#panel{
  position:fixed;right:14px;top:14px;z-index:100;
  width:224px;
  background:#1a1a2e;
  border:1px solid #3a3a5c;
  border-radius:10px;
  padding:12px 14px 14px;
  color:#d0cce8;
  font-size:11px;
  box-shadow:0 6px 24px rgba(0,0,0,0.55);
}
#panel h2{
  font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  margin-bottom:10px;color:#a090c8;
  border-bottom:1px solid #3a3a5c;
  padding-bottom:7px;
}
.row{display:flex;align-items:center;gap:6px;margin:4px 0}
.lbl{width:58px;color:#8880aa;flex-shrink:0;font-size:10px}
input[type=range]{
  flex:1;height:3px;
  -webkit-appearance:none;appearance:none;
  background:#3a3a5c;border-radius:2px;outline:none;cursor:pointer;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:12px;height:12px;border-radius:50%;
  background:#c084fc;border:2px solid #1a1a2e;cursor:pointer;
}
.val{width:32px;text-align:right;color:#6860a0;font-size:10px}
.section{
  margin-top:10px;padding-top:8px;
  border-top:1px solid #2e2e50;
  font-size:9px;letter-spacing:.12em;text-transform:uppercase;
  color:#6860a0;margin-bottom:5px;
}
.section.clip{color:#22d3b0;border-top-color:#1a3a34;}
.clip-bar{
  height:2px;border-radius:1px;margin:0 0 7px 0;
  background:linear-gradient(90deg,#22d3b0,#818cf8);
}
.lbl.x{color:#f87171}
.lbl.y{color:#4ade80}
.lbl.z{color:#60a5fa}
input.x::-webkit-slider-thumb{background:#f87171}
input.y::-webkit-slider-thumb{background:#4ade80}
input.z::-webkit-slider-thumb{background:#60a5fa}
#status{
  position:fixed;left:14px;top:14px;z-index:100;
  background:#1a1a2e;
  border:1px solid #3a3a5c;
  border-radius:8px;padding:6px 12px;color:#a090c8;font-size:11px;
}
#hints{
  position:fixed;left:14px;bottom:14px;z-index:100;
  color:#4a4870;font-size:10px;letter-spacing:.05em;
}
</style>
</head><body>
<canvas id="c"></canvas>
<div id="status">Iniciando...</div>
<div id="panel">
  <h2> ''' + mode_label + '''</h2>

  <div class="section">Volumen</div>
  <div class="row"><span class="lbl">Densidad</span>
    <input type="range" id="dens" min="0.2" max="10" step="0.1" value="%%DENS%%">
    <span class="val" id="vDens">%%DENS%%</span></div>
  <div class="row"><span class="lbl">Corte bg</span>
    <input type="range" id="cut" min="0.005" max="0.5" step="0.005" value="%%CUT%%">
    <span class="val" id="vCut">%%CUT%%</span></div>
  <div class="row"><span class="lbl">Saturac.</span>
    <input type="range" id="sat" min="0.5" max="2.5" step="0.05" value="%%SAT%%">
    <span class="val" id="vSat">%%SAT%%</span></div>
  <div class="row"><span class="lbl">Pasos</span>
    <input type="range" id="steps" min="50" max="600" step="10" value="%%STEPS%%">
    <span class="val" id="vSteps">%%STEPS%%</span></div>
  <div class="row"><span class="lbl">Luz</span>
    <input type="range" id="light" min="0" max="1" step="0.05" value="0.45">
    <span class="val" id="vLight">0.45</span></div>

  <div class="section clip">&#x2702; Corte XYZ</div>
  <div class="clip-bar"></div>
  <div class="row"><span class="lbl x">X min</span>
    <input type="range" class="x" id="xlo" min="0" max="1" step="0.01" value="0">
    <span class="val" id="vXlo">0.00</span></div>
  <div class="row"><span class="lbl x">X max</span>
    <input type="range" class="x" id="xhi" min="0" max="1" step="0.01" value="1">
    <span class="val" id="vXhi">1.00</span></div>
  <div class="row"><span class="lbl y">Y min</span>
    <input type="range" class="y" id="ylo" min="0" max="1" step="0.01" value="0">
    <span class="val" id="vYlo">0.00</span></div>
  <div class="row"><span class="lbl y">Y max</span>
    <input type="range" class="y" id="yhi" min="0" max="1" step="0.01" value="1">
    <span class="val" id="vYhi">1.00</span></div>
  <div class="row"><span class="lbl z">Z min</span>
    <input type="range" class="z" id="zlo" min="0" max="1" step="0.01" value="0">
    <span class="val" id="vZlo">0.00</span></div>
  <div class="row"><span class="lbl z">Z max</span>
    <input type="range" class="z" id="zhi" min="0" max="1" step="0.01" value="1">
    <span class="val" id="vZhi">1.00</span></div>
</div>
<div id="hints">
  Drag: rotar &nbsp;·&nbsp; Scroll: zoom &nbsp;·&nbsp; Clic der: pan
</div>

<script>
const W=%%W%%,H=%%H%%,D=%%D%%,ZS=%%ZS%%,IS_FC=%%IS_FC%%;

function setStatus(t){document.getElementById('status').textContent=t;}

function dec(b){
  const s=atob(b),a=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);
  return a;
}
async function ld(u){return new Promise((res,rej)=>{
  const s=document.createElement('script');
  s.src=u;s.onload=res;s.onerror=()=>rej(new Error('CDN fail: '+u));
  document.head.appendChild(s);
});}

setStatus('Decodificando textura...');
const raw=dec('%%DATA%%');

(async()=>{
try{
  setStatus('Cargando Three.js...');
  await ld('https://unpkg.com/three@0.146.0/build/three.min.js');
  await ld('https://unpkg.com/three@0.146.0/examples/js/controls/OrbitControls.js');

  const canvas = document.getElementById('c');
  const rn = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
  rn.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  rn.setSize(window.innerWidth, window.innerHeight);
  rn.setClearColor(''' + bg_color + ''');

  const gl = rn.getContext();
  if(!(gl instanceof WebGL2RenderingContext)){
    setStatus('ERROR: Requiere WebGL2'); return;
  }

  const sc = new THREE.Scene();
  const cm = new THREE.PerspectiveCamera(42, window.innerWidth/window.innerHeight, 0.01, 100);
  cm.position.set(0, 0, 2.8);

  const ct = new THREE.OrbitControls(cm, rn.domElement);
  ct.enableDamping=true; ct.dampingFactor=0.07;
  ct.minDistance=0.5; ct.maxDistance=8;

  // ── 3D Texture ──────────────────────────────────────────────────
  const T3 = THREE.Data3DTexture || THREE.DataTexture3D;
  const tx = new T3(raw, W, H, D);
  tx.format = THREE.RGBAFormat;
  tx.type   = THREE.UnsignedByteType;
  tx.unpackAlignment = 1;
  tx.magFilter = THREE.LinearFilter;
  tx.minFilter = THREE.LinearFilter;
  tx.wrapS = tx.wrapT = tx.wrapR = THREE.ClampToEdgeWrapping;
  tx.generateMipmaps = false;
  tx.needsUpdate = true;

  const aspY = H / W;
  const aspZ = (D / W) * ZS;
  const geo  = new THREE.BoxGeometry(1, aspY, aspZ);

  // ── Uniforms ─────────────────────────────────────────────────────
  const U = {
    u_vol:     {value: tx},
    u_invModel:{value: new THREE.Matrix4()},
    u_steps:   {value: %%STEPS%%},
    u_dens:    {value: %%DENS%%},
    u_cut:     {value: %%CUT%%},
    u_sat:     {value: %%SAT%%},
    u_light:   {value: 0.45},
    u_is_fc:   {value: IS_FC},
    // Clipping planes [0,1] en UV
    u_clip:    {value: new THREE.Vector3(0,0,0)},    // min X,Y,Z
    u_clipMax: {value: new THREE.Vector3(1,1,1)},    // max X,Y,Z
  };

  const mat = new THREE.ShaderMaterial({
    uniforms: U,
    vertexShader: `
      varying vec3 vLocalPos;
      void main(){
        vLocalPos = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      precision highp sampler3D;

      varying vec3 vLocalPos;

      uniform sampler3D u_vol;
      uniform mat4      u_invModel;
      uniform int       u_steps;
      uniform float     u_dens;
      uniform float     u_cut;
      uniform float     u_sat;
      uniform float     u_light;
      uniform int       u_is_fc;
      uniform vec3      u_clip;      // min UV clip
      uniform vec3      u_clipMax;   // max UV clip

      // ── AABB ray-box intersection en espacio local [-0.5,0.5] ──
      vec2 boxHit(vec3 ro, vec3 rd){
        vec3 iv   = 1.0 / rd;
        vec3 tmin = (-0.5 - ro) * iv;
        vec3 tmax = ( 0.5 - ro) * iv;
        vec3 t0   = min(tmin, tmax);
        vec3 t1   = max(tmin, tmax);
        return vec2(max(max(t0.x,t0.y),t0.z), min(min(t1.x,t1.y),t1.z));
      }

      // ── Pseudo-random jitter (hash) ──────────────────────────────
      // Elimina banding entre cortes con un offset aleatorio por rayo
      float hash(vec2 p){
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
      }

      // ── Gradiente central para normal volumétrica ────────────────
      vec3 gradient(vec3 uv, float eps){
        vec3 od_px = -log(max(texture(u_vol, uv+vec3(eps,0,0)).rgb, vec3(1e-4)));
        vec3 od_nx = -log(max(texture(u_vol, uv-vec3(eps,0,0)).rgb, vec3(1e-4)));
        vec3 od_py = -log(max(texture(u_vol, uv+vec3(0,eps,0)).rgb, vec3(1e-4)));
        vec3 od_ny = -log(max(texture(u_vol, uv-vec3(0,eps,0)).rgb, vec3(1e-4)));
        vec3 od_pz = -log(max(texture(u_vol, uv+vec3(0,0,eps)).rgb, vec3(1e-4)));
        vec3 od_nz = -log(max(texture(u_vol, uv-vec3(0,0,eps)).rgb, vec3(1e-4)));
        float gx = dot(od_px-od_nx, vec3(0.333));
        float gy = dot(od_py-od_ny, vec3(0.333));
        float gz = dot(od_pz-od_nz, vec3(0.333));
        return vec3(gx, gy, gz);
      }

      // ── Saturación preservando luminancia ───────────────────────
      vec3 adjustSat(vec3 c, float s){
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        return clamp(l + s * (c - l), 0.0, 1.0);
      }

      void main(){
        // Rayo en espacio LOCAL
        vec3 camL = (u_invModel * vec4(cameraPosition, 1.0)).xyz;
        vec3 rd   = normalize(vLocalPos - camL);

        vec2 b = boxHit(camL, rd);
        if(b.x >= b.y) discard;

        float tStart = max(b.x, 0.0);
        float tEnd   = b.y;
        float dt     = (tEnd - tStart) / float(u_steps);

        // ── Jitter: offset aleatorio por rayo para eliminar banding ─
        float jitter = hash(gl_FragCoord.xy) * dt;
        tStart += jitter;

        vec3  accColor = vec3(0.0);
        float accAlpha = 0.0;

        for(int i = 0; i < 600; i++){
          if(i >= u_steps || accAlpha >= 0.99) break;

          float t  = tStart + float(i) * dt;
          vec3  p  = camL + t * rd;
          vec3  uv = p + 0.5;    // [0,1]

          if(any(lessThan(uv, vec3(0.0))) || any(greaterThan(uv, vec3(1.0)))) continue;

          // ── Clipping planes XYZ ────────────────────────────────────
          if(any(lessThan(uv, u_clip)))    continue;
          if(any(greaterThan(uv, u_clipMax))) continue;

          vec3 rgb = texture(u_vol, uv).rgb;

          if(u_is_fc == 1){
            // ── H&E Beer-Lambert OD ───────────────────────────────
            vec3  od    = -log(max(rgb, vec3(1e-4)));
            float stain = od.r + od.g + od.b;

            float tissue = smoothstep(u_cut, u_cut * 4.5, stain);
            if(tissue < 0.001) continue;

            float a = clamp(tissue * stain * u_dens * dt * 6.0, 0.0, 0.95);

            // ── Iluminación volumétrica: Phong difuso ─────────────
            vec3  col = adjustSat(rgb, u_sat);
            if(u_light > 0.0){
              vec3  grad  = gradient(uv, 0.008);
              float gMag  = length(grad);
              if(gMag > 0.001){
                vec3  nrm   = normalize(grad);
                vec3  lDir  = normalize(vec3(0.6, 1.0, 0.8));   // luz fija
                float diff  = max(dot(nrm, lDir), 0.0);
                float amb   = 0.35;
                float lit   = amb + (1.0 - amb) * diff;
                // Blending suave: sin iluminación = imagen 2D, con = 3D shaded
                col = mix(col, col * lit, u_light);
              }
            }

            accColor += (1.0 - accAlpha) * col * a;
            accAlpha += (1.0 - accAlpha) * a;

          } else {
            // ── OTLS Fluorescencia ────────────────────────────────
            float lum    = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
            float tissue = smoothstep(u_cut * 0.5, u_cut * 3.0, lum);
            float a      = clamp(tissue * lum * u_dens * dt * 15.0, 0.0, 1.0);
            if(a > 0.001){
              vec3 col = clamp(rgb * u_sat, 0.0, 1.0);
              // Iluminación fluorescencia
              if(u_light > 0.0){
                vec3  od   = -log(max(rgb, vec3(1e-4)));
                vec3  grad = gradient(uv, 0.008);
                float gMag = length(grad);
                if(gMag > 0.001){
                  vec3  nrm  = normalize(grad);
                  vec3  lDir = normalize(vec3(0.6,1.0,0.8));
                  float lit  = 0.35 + 0.65 * max(dot(nrm, lDir), 0.0);
                  col = mix(col, col * lit, u_light);
                }
              }
              accColor += (1.0 - accAlpha) * col * a;
              accAlpha += (1.0 - accAlpha) * a;
            }
          }
        }

        vec3 bg = (u_is_fc == 1) ? vec3(1.0) : vec3(0.0);
        gl_FragColor = vec4(accColor + (1.0 - accAlpha) * bg, 1.0);
      }
    `,
    transparent: false,
    depthWrite:  true,
    side: THREE.FrontSide
  });

  const mesh  = new THREE.Mesh(geo, mat);
  sc.add(mesh);

  // Aristas del cubo (se actualizan con los clips)
  const edgeMat = new THREE.LineBasicMaterial({color:''' + edge_color + ''', opacity:0.5, transparent:true});
  const edgeMesh = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
  sc.add(edgeMesh);

  // ── Wiring UI controls ────────────────────────────────────────
  function wire(id, valId, uniform, decimals=2){
    const el = document.getElementById(id);
    const vl = document.getElementById(valId);
    el.addEventListener('input', ()=>{
      const v = parseFloat(el.value);
      U[uniform].value = v;
      vl.textContent = v.toFixed(decimals);
    });
  }
  wire('dens',  'vDens',  'u_dens',  1);
  wire('cut',   'vCut',   'u_cut',   3);
  wire('sat',   'vSat',   'u_sat',   2);
  wire('light', 'vLight', 'u_light', 2);

  // Steps is integer
  document.getElementById('steps').addEventListener('input', function(){
    U.u_steps.value = parseInt(this.value);
    document.getElementById('vSteps').textContent = this.value;
  });

  // Clip sliders
  function wireClip(){
    const ids = ['xlo','xhi','ylo','yhi','zlo','zhi'];
    ids.forEach(id=>{
      document.getElementById(id).addEventListener('input', updateClip);
    });
  }
  function updateClip(){
    const xlo = parseFloat(document.getElementById('xlo').value);
    const xhi = parseFloat(document.getElementById('xhi').value);
    const ylo = parseFloat(document.getElementById('ylo').value);
    const yhi = parseFloat(document.getElementById('yhi').value);
    const zlo = parseFloat(document.getElementById('zlo').value);
    const zhi = parseFloat(document.getElementById('zhi').value);
    U.u_clip.value.set(xlo, ylo, zlo);
    U.u_clipMax.value.set(xhi, yhi, zhi);
    document.getElementById('vXlo').textContent = xlo.toFixed(2);
    document.getElementById('vXhi').textContent = xhi.toFixed(2);
    document.getElementById('vYlo').textContent = ylo.toFixed(2);
    document.getElementById('vYhi').textContent = yhi.toFixed(2);
    document.getElementById('vZlo').textContent = zlo.toFixed(2);
    document.getElementById('vZhi').textContent = zhi.toFixed(2);
  }
  wireClip();

  // ── Render loop ───────────────────────────────────────────────
  function animate(){
    requestAnimationFrame(animate);
    ct.update();
    mat.uniforms.u_invModel.value.copy(mesh.matrixWorld).invert();
    rn.render(sc, cm);
  }
  setStatus("''' + mode_label + ''' - listo");
  animate();

  window.addEventListener('resize', ()=>{
    rn.setSize(window.innerWidth, window.innerHeight);
    cm.aspect = window.innerWidth / window.innerHeight;
    cm.updateProjectionMatrix();
  });

}catch(e){
  document.getElementById('status').textContent = 'Error: ' + e.message;
  console.error(e);
}
})();
</script>
</body></html>''')

    html = (html
        .replace("%%W%%", str(x))
        .replace("%%H%%", str(y))
        .replace("%%D%%", str(z))
        .replace("%%ZS%%", str(z_scale_val))
        .replace("%%DATA%%", b64)
        .replace("%%STEPS%%", str(steps))
        .replace("%%DENS%%", str(density))
        .replace("%%CUT%%", str(bg_cutoff))
        .replace("%%SAT%%", str(saturation))
        .replace("%%IS_FC%%", "1" if is_fc else "0")
    )
    components.html(html, height=height)


# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    " Volumen 3D", " Cortes", "▶ Animación", "ℹ Info"
])

with tab1:
    if "stack_rgb" in st.session_state and st.session_state["stack_rgb"] is not None:
        stack_gray = st.session_state["stack_gray"]
        stack_rgb = st.session_state["stack_rgb"]
        mode_label = st.session_state.get("mode_label", "FalseColor H&E")

        st.header(f"Volumen 3D — {mode_label}")
        st.subheader("Vista previa")
        prev_idx = st.slider(
            "Corte:", 0, stack_rgb.shape[0] - 1,
            stack_rgb.shape[0] // 2, key="prev3d"
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(
                stack_rgb[prev_idx], clamp=True, width=450,
                caption=f"Corte {prev_idx+1} — {mode_label}"
            )
        with c2:
            st.metric("Cortes", stack_rgb.shape[0])
            st.metric("Resolución",
                      f"{stack_rgb.shape[2]}×{stack_rgb.shape[1]}")
            st.metric("Escala Z", f"{z_scale:.1f}x")
            st.metric("Densidad", f"{vol_density:.1f}")
            st.metric("Modo", mode_label)

        st.markdown("---")
        if st.button(" Abrir Visor 3D", type="primary",
                     use_container_width=True):
            st.session_state["show_viewer"] = True

        if st.session_state.get("show_viewer", False):
            with st.spinner(f"Renderizando {mode_label}..."):
                render_volume(
                    stack_rgb, 800, ray_steps,
                    vol_density, vol_bg_cutoff, vol_saturation,
                    z_scale, is_falsecolor
                )

        st.info(
            f" {stack_rgb.shape[2]}×{stack_rgb.shape[1]}"
            f"×{stack_rgb.shape[0]} · "
            f"Z×{z_scale:.0f} · "
            f"{stack_rgb.nbytes / 1024**2:.1f} MB"
        )
    else:
        st.info("⬅ Configura y haz click en **Generar Visualización 3D**")
        st.markdown("""
        ### Dos modos de visualización

        | Modo | Color | Fondo | Shader |
        |---|---|---|---|
        | **FalseColor H&E** | Rosa + Púrpura | Blanco | Beer-Lambert OD |
        | **OTLS Fluorescencia** | Azul + Rojo | Negro | Luminancia |

        ### Escala Z
        Tus imágenes son ~1000px de ancho pero solo ~30 cortes de profundidad.
        El slider **Escala Z** (recomendado: 3–8) amplifica la profundidad
        para que el volumen no se vea como una lámina plana.

        ### Si el volumen se ve hueco
        - **Subir Densidad** a 3.0–5.0
        - **Bajar Corte de fondo** a 0.03–0.05
        - **Subir Pasos ray-march** a 250–400
        """)

with tab2:
    if "stack_rgb" in st.session_state and st.session_state["stack_rgb"] is not None:
        stack_gray = st.session_state["stack_gray"]
        stack_rgb = st.session_state["stack_rgb"]
        st.header("Explorador de Cortes")
        orient = st.radio(
            "Plano:", ["Axial (XY)", "Coronal (XZ)", "Sagital (YZ)"],
            horizontal=True
        )
        if orient == "Axial (XY)":
            mx = stack_rgb.shape[0] - 1
            idx = st.slider("Z:", 0, mx, mx // 2)
            img_c, img_g = stack_rgb[idx], stack_gray[idx]
            titulo = f"Axial Z={idx}"
        elif orient == "Coronal (XZ)":
            mx = stack_rgb.shape[1] - 1
            idx = st.slider("Y:", 0, mx, mx // 2)
            img_c = stack_rgb[:, idx, :]
            img_g = stack_gray[:, idx, :]
            titulo = f"Coronal Y={idx}"
        else:
            mx = stack_rgb.shape[2] - 1
            idx = st.slider("X:", 0, mx, mx // 2)
            img_c = stack_rgb[:, :, idx]
            img_g = stack_gray[:, :, idx]
            titulo = f"Sagital X={idx}"
        st.subheader(titulo)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(img_c, clamp=True, width=display_size)
        with c2:
            st.metric("Media", f"{float(img_g.mean()):.1f}")
            st.metric("Mín", f"{float(img_g.min()):.0f}")
            st.metric("Máx", f"{float(img_g.max()):.0f}")
            st.metric("Tamaño", f"{img_g.shape[1]}×{img_g.shape[0]}")
    else:
        st.info("⬅ Carga un stack primero.")

with tab3:
    if "stack_rgb" in st.session_state and st.session_state["stack_rgb"] is not None:
        stack_rgb = st.session_state["stack_rgb"]
        st.header("Animación")
        frame = st.slider("Frame:", 0, stack_rgb.shape[0] - 1, 0, key="anim")
        st.image(stack_rgb[frame], clamp=True,
                 caption=f"Corte {frame+1}/{stack_rgb.shape[0]}",
                 width=display_size)
        speed = st.slider("Velocidad (s/frame):", 0.02, 0.5, 0.08, 0.02)
        if st.button("▶ Reproducir"):
            ph = st.empty()
            pr = st.progress(0)
            tx = st.empty()
            for i in range(stack_rgb.shape[0]):
                tx.text(f"Corte {i+1}/{stack_rgb.shape[0]}")
                ph.image(stack_rgb[i], clamp=True, width=display_size,
                         caption=f"Corte {i+1}")
                pr.progress((i + 1) / stack_rgb.shape[0])
                time.sleep(speed)
            pr.empty()
            tx.empty()
            st.success(" Completado")
    else:
        st.info("⬅ Carga un stack primero.")

with tab4:
    st.header("Información")
    if "stack_rgb" in st.session_state and st.session_state["stack_rgb"] is not None:
        sg = st.session_state["stack_gray"]
        sr_data = st.session_state["stack_rgb"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cortes", sg.shape[0])
        c2.metric("Res", f"{sg.shape[2]}×{sg.shape[1]}")
        c3.metric("DS", f"{downsample_factor}x")
        c4.metric("Modo", st.session_state.get("mode_label", ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("Mem Gris", f"{sg.nbytes/1024**2:.1f} MB")
        c2.metric("Mem Color", f"{sr_data.nbytes/1024**2:.1f} MB")
        c3.metric("Escala Z", f"{z_scale}x")
        fig = go.Figure(go.Histogram(
            x=sg.flatten(), nbinsx=50, marker_color="#c06080"
        ))
        fig.update_layout(title="Intensidades", height=300)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown("""
    **Autores:** Daniel Yaruro, Juan Mantilla · **v6.1**

    ### Cambios v6.1
    - **Shader front-to-back** → volumen sólido (no hueco)
    - **smoothstep** → tejido tenue contribuye parcialmente
    - **Escala Z configurable** → volumen no se ve plano
    - **Modo OTLS / FalseColor** → selector en sidebar
    - **Densidad default 3.0** → más opaco por defecto

    ### Pipeline
    ```
    nuclei_u16.tif + cyto_u16.tif
      → cv2.imread uint16 → cv2.resize
      → [FalseColor] Beer-Lambert H&E
      → [OTLS] RGB fluorescencia directa
      → Shader 3D front-to-back + smoothstep + Z-scale
    ```

    ### Si el volumen se ve hueco/transparente
    1. Subir **Densidad** a 4.0–6.0
    2. Bajar **Corte de fondo** a 0.02–0.04
    3. Subir **Pasos** a 300–400
    4. Subir **Escala Z** a 5.0–10.0

    ### Ref
    Giacomelli et al. (2020) · Serafin/Liu Lab · Holzner et al. (2024)
    """)

# ==================== BOTÓN ====================
st.sidebar.markdown("---")
folder_check = Path(data_path)
if folder_check.exists():
    pairs_available = find_slice_pairs(folder_check, is_16bit)
    if not pairs_available:
        pairs_available = find_slice_pairs(folder_check, not is_16bit)
    if pairs_available:
        st.sidebar.success(f" {len(pairs_available)} pares encontrados")
    else:
        st.sidebar.error(" No se encontraron pares")
else:
    st.sidebar.warning("Carpeta no encontrada")

if st.sidebar.button(" Generar Visualización 3D", type="primary"):
    st.session_state["show_viewer"] = False
    with st.spinner(f"Cargando {num_imagenes} pares..."):
        stack_gray, stack_rgb, err = load_and_colorize(
            data_path, num_imagenes, downsample_factor, is_16bit,
            is_falsecolor,
            nuc_threshold, cyto_threshold, nuc_normfactor, cyto_normfactor,
            use_clahe, clahe_clip, enable_sr, sr_scale, sr_sharpen,
        )
    if err:
        st.error(err)
    else:
        st.session_state["stack_gray"] = stack_gray
        st.session_state["stack_rgb"] = stack_rgb
        st.session_state["mode_label"] = view_mode
        st.success(
            f" {stack_gray.shape[0]} cortes · "
            f"{stack_gray.shape[2]}×{stack_gray.shape[1]} px · "
            f"{view_mode}"
        )
        st.info(
            f" {(stack_gray.nbytes + stack_rgb.nbytes) / 1024**2:.1f} MB"
        )
        st.session_state["show_viewer"] = True
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    " **Valores para fidelidad H&E:**\n"
    "- Saturación: **1.0** (idéntico imagen)\n"
    "- Corte fondo: **0.02–0.04**\n"
    "- Densidad: **2.0–3.0**\n"
    "- Pasos: **250–400**\n"
    "- Escala Z: **3–8**"
)