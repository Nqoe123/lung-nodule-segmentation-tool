import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage import exposure
import tempfile
import SimpleITK as sitk
from collections import OrderedDict
from PIL import Image
import warnings
from datetime import datetime
import zipfile
import os
import gdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import shutil

warnings.filterwarnings('ignore')

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#151d30',
    'axes.facecolor': '#151d30',
    'text.color': '#f1f5f9',
    'axes.labelcolor': '#f1f5f9',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'font.family': 'sans-serif',
})

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LungVision AI | Clinical Nodule Segmentation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Clinical Dark Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #0ea5e9;
    --primary-dark: #0369a1;
    --primary-glow: rgba(14,165,233,0.25);
    --bg-deep: #060b14;
    --bg-main: #0b1120;
    --bg-card: #111827;
    --bg-elevated: #1a2332;
    --border: #1e2d4a;
    --border-light: #273755;
    --text-1: #f1f5f9;
    --text-2: #94a3b8;
    --text-3: #64748b;
    --green: #10b981;
    --red: #ef4444;
    --amber: #f59e0b;
    --cyan: #06b6d4;
}

*, *::before, *::after { box-sizing: border-box; }

.main .block-container {
    background: var(--bg-main);
    max-width: 1400px;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, var(--bg-deep) 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* ---- GLOBAL TEXT ---- */
.stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
.stMetric label, .stMetric div, .stMetric span,
.stMetricLabel, .stMetricValue,
.stSelectbox label, .stSelectbox div,
.stRadio label, .stRadio div,
.stTextInput label, .stTextInput div,
.stFileUploader label,
p, li, label, h1, h2, h3, h4, h5, h6, span {
    color: var(--text-1) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* ---- HEADER CARD ---- */
.header-card {
    background: linear-gradient(135deg, #0c2340 0%, #132e52 40%, #0c2340 100%);
    border: 1px solid var(--border-light);
    border-radius: 16px;
    padding: 1.75rem 2.5rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
}
.header-card::before {
    content: '';
    position: absolute;
    top: -80px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(14,165,233,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.header-card::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.header-card h1 {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0;
}
.header-card .tagline {
    color: var(--text-2) !important;
    font-size: 0.9rem;
    font-weight: 400;
    margin-top: 0.3rem;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(14,165,233,0.1);
    color: var(--primary) !important;
    padding: 0.25rem 0.85rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.75rem;
    border: 1px solid rgba(14,165,233,0.2);
}
.header-badge .dot {
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ---- LOGIN CARD ---- */
.login-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 70vh;
}
.login-card {
    background: linear-gradient(160deg, #111d32 0%, #0b1221 100%);
    border: 1px solid var(--border-light);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    width: 100%;
    max-width: 400px;
    box-shadow: 0 30px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(14,165,233,0.05);
}
.login-icon {
    text-align: center;
    font-size: 3.2rem;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 0 20px rgba(14,165,233,0.3));
}
.login-card h2 {
    text-align: center;
    font-weight: 700;
    font-size: 1.4rem !important;
    margin: 0;
}
.login-sub {
    text-align: center;
    color: var(--text-3) !important;
    font-size: 0.82rem;
    margin-top: 0.35rem;
    margin-bottom: 2rem;
}

/* ---- SECTION LABELS ---- */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-3) !important;
    margin-bottom: 0.75rem;
    padding-left: 0.15rem;
}

/* ---- METRIC CARDS ---- */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    padding: 1.2rem 1.4rem !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    transition: border-color 0.2s, transform 0.2s;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--primary) !important;
    transform: translateY(-2px);
}
div[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ---- BUTTONS ---- */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px var(--primary-glow) !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px var(--primary-glow) !important;
    filter: brightness(1.1);
}

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"] {
    border: none !important;
}
[data-testid="stFileUploader"] > section > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-light) !important;
    border-radius: 14px !important;
    padding: 2.25rem 1.5rem !important;
    transition: all 0.25s !important;
    cursor: pointer;
}
[data-testid="stFileUploader"] > section > div:hover {
    border-color: var(--primary) !important;
    background: var(--bg-elevated) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-2) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ---- RADIO GROUP ---- */
.stRadio [data-baseweb="radio-group"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.4rem !important;
    gap: 0.3rem !important;
    display: flex !important;
}
.stRadio [data-baseweb="radio"] {
    background: transparent !important;
    border-radius: 9px !important;
    padding: 0.55rem 1.1rem !important;
    flex: 1;
    text-align: center;
    transition: all 0.2s !important;
}
.stRadio [data-baseweb="radio"][aria-checked="true"] {
    background: rgba(14,165,233,0.12) !important;
    border: 1px solid rgba(14,165,233,0.35) !important;
    box-shadow: 0 0 12px rgba(14,165,233,0.1);
}

/* ---- DATAFRAME ---- */
.stDataFrame {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
}
.dataframe {
    background: var(--bg-card) !important;
    color: var(--text-1) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}
.dataframe th {
    background: var(--bg-elevated) !important;
    color: var(--text-2) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.8rem 1rem !important;
    border-bottom: 2px solid var(--border-light) !important;
}
.dataframe td {
    background: var(--bg-card) !important;
    color: var(--text-1) !important;
    padding: 0.65rem 1rem !important;
    border-bottom: 1px solid var(--border) !important;
}
.dataframe tr:hover td {
    background: var(--bg-elevated) !important;
}

/* ---- ALERTS ---- */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.88rem !important;
}
div[data-testid="stAlert"] { background: var(--bg-card) !important; }
.stAlert[data-baseweb="notification"][kind="success"] { border-left: 4px solid var(--green) !important; }
.stAlert[data-baseweb="notification"][kind="info"]    { border-left: 4px solid var(--primary) !important; }
.stAlert[data-baseweb="notification"][kind="warning"] { border-left: 4px solid var(--amber) !important; }
.stAlert[data-baseweb="notification"][kind="error"]   { border-left: 4px solid var(--red) !important; }

/* ---- TEXT INPUT ---- */
.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-1) !important;
    padding: 0.6rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
    outline: none !important;
}

/* ---- SELECT BOX ---- */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-1) !important;
}
.stSelectbox > div > div:hover { border-color: var(--primary) !important; }

/* ---- PROGRESS ---- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--cyan)) !important;
    border-radius: 10px !important;
}
.stProgress > div > div {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    height: 6px !important;
}

/* ---- DOWNLOAD BUTTON ---- */
.stDownloadButton > button {
    background: var(--bg-elevated) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    border-color: var(--primary) !important;
    background: rgba(14,165,233,0.1) !important;
}

/* ---- NODULE RESULT CARDS (HTML) ---- */
.nodule-result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
    transition: all 0.2s;
}
.nodule-result-card:hover {
    border-color: var(--border-light);
    transform: translateX(3px);
}
.nodule-result-card.routine  { border-left: 4px solid var(--green); }
.nodule-result-card.followup { border-left: 4px solid var(--amber); }
.nodule-result-card.urgent   { border-left: 4px solid var(--red); }

.nodule-id-badge {
    background: rgba(14,165,233,0.12);
    color: var(--primary);
    font-weight: 700;
    font-size: 0.8rem;
    padding: 0.35rem 0.7rem;
    border-radius: 8px;
    min-width: 50px;
    text-align: center;
}
.nodule-measures {
    flex: 1;
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}
.nodule-measure {
    display: flex;
    flex-direction: column;
}
.nodule-measure .val {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-1);
}
.nodule-measure .lbl {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-3);
    margin-top: 0.1rem;
}
.nodule-rec {
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    white-space: nowrap;
}
.nodule-rec.routine  { background: rgba(16,185,129,0.12); color: var(--green); }
.nodule-rec.followup { background: rgba(245,158,11,0.12); color: var(--amber); }
.nodule-rec.urgent   { background: rgba(239,68,68,0.12); color: var(--red); }

/* ---- FOOTER ---- */
.app-footer {
    text-align: center;
    color: var(--text-3) !important;
    font-size: 0.75rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

/* ---- HIDE DEFAULTS ---- */
#MainMenu, footer, header { visibility: hidden; }
.stSpinner > div { border-top-color: var(--primary) !important; }

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

/* ---- DIVIDER ---- */
.hr { height: 1px; background: var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE — U-Net
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear \
                  else nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x3.size(3) - x1.size(3) if False else x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

class MemoryEfficientUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        f = 2 if bilinear else 1
        self.inc  = DoubleConv(n_channels, 64)
        self.d1   = Down(64, 128)
        self.d2   = Down(128, 256)
        self.d3   = Down(256, 512)
        self.d4   = Down(512, 1024//f)
        self.u1   = Up(1024, 512//f, bilinear)
        self.u2   = Up(512, 256//f, bilinear)
        self.u3   = Up(256, 128//f, bilinear)
        self.u4   = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        return self.outc(self.u4(self.u3(self.u2(self.u1(x5,x4),x3),x2),x1))


# ============================================================
# MODEL LOADING
# ============================================================
GDRIVE_ID = "1PzCv2fJSr7e0QIfPGtLKLOL-9RSLdR2i"
MODEL_FN  = "best_model.pth"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_FN):
            url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
            gdown.download(url, MODEL_FN, quiet=True)
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        ckpt = torch.load(MODEL_FN, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt.state_dict()
        model.load_state_dict(OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in sd.items()))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ============================================================
# PROCESSING UTILITIES
# ============================================================
def segment_slice(model, img, threshold=0.3):
    """Return binary uint8 mask matching input shape."""
    shape = img.shape
    normed = img.astype(np.float32)
    if normed.max() > 1.0:
        normed = normed / 255.0
    normed = exposure.equalize_adapthist(normed)
    resized = resize(normed, (512, 512), preserve_range=True)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().numpy()
    mask = resize((prob > threshold).astype(np.float32), shape, order=0, preserve_range=True)
    return (mask > 0.5).astype(np.uint8)


def analyze_3d(mask_3d, spacing_zyx):
    """3D connected-component analysis with real-world measurements.
       spacing_zyx = (z_spacing, y_spacing, x_spacing) in mm."""
    sx, sy, sz = spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]
    voxel_vol = sx * sy * sz  # mm³
    labeled = label(mask_3d, connectivity=2)
    nodules = []
    for rp in regionprops(labeled):
        if rp.area < 10:
            continue
        vol_mm3 = rp.area * voxel_vol
        eq_diam  = 2.0 * (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1/3)
        bb = rp.bbox  # (z0,y0,x0,z1,y1,x1)
        ext_x = (bb[5]-bb[4]) * sx
        ext_y = (bb[3]-bb[2]) * sy
        ext_z = (bb[1]-bb[0]) * sz
        max_diam = max(ext_x, ext_y, ext_z)
        nodules.append(dict(
            id=len(nodules)+1,
            label_id=rp.label,
            volume_mm3=vol_mm3,
            eq_diameter_mm=eq_diam,
            max_diameter_mm=max_diam,
            num_voxels=rp.area,
            slice_range=(bb[0], bb[1]),
            num_slices=bb[1]-bb[0],
            centroid_zyx=rp.centroid,
        ))
    return labeled, nodules


def analyze_2d(mask, spacing_xy=None):
    """2D connected-component analysis. spacing_xy = (x_mm, y_mm)."""
    labeled = label(mask, connectivity=2)
    nodules = []
    for rp in regionprops(labeled):
        if rp.area < 10:
            continue
        area_px = rp.area
        diam_px = 2 * np.sqrt(area_px / np.pi)
        area_mm2 = area_px * spacing_xy[0] * spacing_xy[1] if spacing_xy else None
        diam_mm  = diam_px * spacing_xy[0] if spacing_xy else None
        nodules.append(dict(
            id=len(nodules)+1,
            area_px=area_px,
            diam_px=diam_px,
            area_mm2=area_mm2,
            diam_mm=diam_mm,
            centroid=(rp.centroid[0], rp.centroid[1]),
            mask=(labeled == rp.label).astype(np.float32),
        ))
    return nodules


def load_volume(zip_file):
    """Extract zip, find .mhd, read with SimpleITK. Returns (array, spacing_zyx, temp_dir)."""
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "upload.zip")
    with open(zpath, "wb") as f:
        f.write(zip_file.getbuffer())
    with zipfile.ZipFile(zpath, 'r') as zf:
        zf.extractall(tmp)
    mhd = None
    for root, _, files in os.walk(tmp):
        for fn in files:
            if fn.lower().endswith('.mhd'):
                mhd = os.path.join(root, fn)
                break
        if mhd:
            break
    if not mhd:
        shutil.rmtree(tmp, ignore_errors=True)
        return None, None, None
    img = sitk.ReadImage(mhd)
    arr = sitk.GetArrayFromImage(img)          # (Z, Y, X)
    sp  = img.GetSpacing()                     # (X, Y, Z) in mm
    sp_zyx = (sp[2], sp[1], sp[0])
    return arr, sp_zyx, tmp


def get_recommendation(diam_mm):
    if diam_mm < 5:
        return "Routine follow-up (12 mo)", "routine"
    elif diam_mm < 8:
        return "Short-term follow-up (3-6 mo)", "followup"
    else:
        return "Further evaluation recommended", "urgent"


# ============================================================
# VISUALIZATION
# ============================================================
def make_overlay(slice_img, mask_2d, alpha=0.45):
    """Red-tinted overlay of binary mask on grayscale slice."""
    n = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
    rgb = np.stack([n, n, n], axis=-1)
    m = mask_2d > 0.5
    rgb[m, 0] = np.clip(rgb[m, 0] + alpha, 0, 1)
    rgb[m, 1] = np.clip(rgb[m, 1] * 0.25, 0, 1)
    rgb[m, 2] = np.clip(rgb[m, 2] * 0.25, 0, 1)
    return rgb


def draw_slice_view(ax, slice_img, labeled_2d, nodules_info, title="", spacing_xy=None):
    """Draw a CT slice with nodule overlays and annotation circles."""
    mask_any = (labeled_2d > 0).astype(np.float32) if labeled_2d is not None else np.zeros_like(slice_img)
    overlay = make_overlay(slice_img, mask_any)
    ax.imshow(overlay, cmap='gray')

    for ninfo in nodules_info:
        if 'label_id' in ninfo:
            m2 = (labeled_2d == ninfo['label_id'])
        else:
            m2 = ninfo['mask'] > 0.5
        ys, xs = np.where(m2)
        if len(xs) == 0:
            continue
        cx, cy = np.mean(xs), np.mean(ys)
        rx = (np.max(xs) - np.min(xs)) / 2 + 6
        ry = (np.max(ys) - np.min(ys)) / 2 + 6
        r = max(rx, ry)
        circ = Circle((cx, cy), r, fill=False, edgecolor='#06b6d4', linewidth=2.2, linestyle='-')
        ax.add_patch(circ)

        if 'eq_diameter_mm' in ninfo:
            line1 = f"Nodule {ninfo['id']}"
            line2 = f"\u2300 {ninfo['eq_diameter_mm']:.1f} mm"
            line3 = f"V {ninfo['volume_mm3']:.0f} mm\u00b3"
        elif ninfo.get('diam_mm') is not None:
            line1 = f"Nodule {ninfo['id']}"
            line2 = f"\u2300 {ninfo['diam_mm']:.1f} mm"
            line3 = f"A {ninfo['area_mm2']:.1f} mm\u00b2"
        else:
            line1 = f"Nodule {ninfo['id']}"
            line2 = f"\u2300 {ninfo['diam_px']:.0f} px"
            line3 = f"A {ninfo['area_px']:.0f} px\u00b2"

        ax.annotate(
            f"{line1}\n{line2}\n{line3}",
            xy=(cx, cy), xytext=(cx + r + 12, cy - r),
            fontsize=9, fontweight='600', color='#f1f5f9',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#273755', alpha=0.92),
            arrowprops=dict(arrowstyle='-', color='#273755', lw=1.2),
        )

    ax.set_title(title, color='#f1f5f9', fontsize=13, fontweight='600', pad=10)
    ax.axis('off')
    return ax


# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    st.markdown("""
    <div class="login-card">
        <div class="login-icon">🫁</div>
        <h2>LungVision AI</h2>
        <p class="login-sub">Clinical Nodule Segmentation Platform</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Radiologist ID", placeholder="Enter your ID", key="login_user")
        pwd  = st.text_input("Password", type="password", placeholder="Enter password", key="login_pwd")
        submitted = st.form_submit_button("Sign In", use_container_width=True)
        if submitted:
            if user == "radiologist" and pwd == "hit500":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. Please verify your Radiologist ID and Password.")

    st.markdown('<p style="text-align:center;color:var(--text-3);font-size:0.75rem;margin-top:1.5rem;">Secure clinical access only</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================
def show_app(model):
    # ---- Header ----
    st.markdown("""
    <div class="header-card">
        <h1>🫁 LungVision AI</h1>
        <p class="tagline">Automatic Lung Nodule Detection &amp; Volumetric Analysis</p>
        <div class="header-badge"><span class="dot"></span> AI Model Ready</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0.5rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">
                <span style="font-size:1.3rem;">👨‍⚕️</span>
                <span style="font-weight:600;font-size:0.95rem;">Radiologist</span>
            </div>
            <span style="font-size:0.75rem;color:var(--green);font-weight:500;">● Active Session</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True, key="logout_btn"):
            st.session_state.clear()
            st.rerun()
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Model Info</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem;line-height:1.8;color:var(--text-2);">
            <div><strong style="color:var(--text-1);">Architecture</strong> &nbsp; U-Net (2D)</div>
            <div><strong style="color:var(--text-1);">Dice Score</strong> &nbsp; 0.74</div>
            <div><strong style="color:var(--text-1);">Training</strong> &nbsp; LUNA16</div>
            <div><strong style="color:var(--text-1);">Input</strong> &nbsp; 512 × 512</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Disclaimer</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem;line-height:1.6;color:var(--text-3);">
            This is a clinical decision support tool. All findings must be verified by a qualified radiologist. Do not use as sole basis for diagnosis.
        </div>
        """, unsafe_allow_html=True)

    # ---- Mode Selection ----
    st.markdown('<p class="section-label">Scan Type</p>', unsafe_allow_html=True)
    mode = st.radio("", ["Single CT Slice (PNG)", "CT Volume (MHD + RAW as ZIP)"], horizontal=True, label_visibility="collapsed")

    # Clear stale results when mode changes
    if st.session_state.get('_mode') != mode:
        st.session_state._mode = mode
        for k in ('_fkey', '_png_results', '_vol_loaded', '_vol_labeled', '_vol_nodules',
                   '_vol_array', '_vol_spacing', '_vol_tmp'):
            st.session_state.pop(k, None)

    # ================================================================
    # MODE 1 — SINGLE PNG SLICE
    # ================================================================
    if "PNG" in mode:
        st.markdown('<p class="section-label" style="margin-top:1rem;">Upload CT Slice</p>', unsafe_allow_html=True)
        upfile = st.file_uploader("Select a PNG or JPG file", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        # Optional spacing input
        sp_col1, sp_col2 = st.columns(2)
        with sp_col1:
            px_sp_x = st.number_input("Pixel spacing X (mm/px)", min_value=0.0, value=0.0, step=0.01, format="%.3f",
                                       help="Leave 0 to show pixel units only")
        with sp_col2:
            px_sp_y = st.number_input("Pixel spacing Y (mm/px)", min_value=0.0, value=0.0, step=0.01, format="%.3f",
                                       help="Leave 0 to show pixel units only")
        spacing_xy = (px_sp_x, px_sp_y) if px_sp_x > 0 and px_sp_y > 0 else None

        if upfile is not None:
            fkey = f"png_{upfile.name}_{upfile.size}_{px_sp_x}_{px_sp_y}"
            if st.session_state.get('_fkey') != fkey:
                st.session_state._fkey = fkey
                img_pil = Image.open(upfile).convert('L')
                img_arr = np.array(img_pil, dtype=np.float32)
                mask = segment_slice(model, img_arr)
                nodules = analyze_2d(mask, spacing_xy)
                st.session_state._png_results = dict(img=img_arr, mask=mask, nodules=nodules)

            res = st.session_state._png_results
            img_arr, mask, nodules = res['img'], res['mask'], res['nodules']

            # --- Display ---
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-label">Original Slice</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=100)
                ax.imshow(img_arr, cmap='gray')
                ax.set_title("CT Slice", color='#f1f5f9', fontsize=12, fontweight='600')
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with c2:
                st.markdown('<p class="section-label">Segmentation Overlay</p>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(5.5, 5.5), dpi=100)
                draw_slice_view(ax2, img_arr, label(mask), nodules,
                                title=f"{len(nodules)} Nodule(s) Detected", spacing_xy=spacing_xy)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            if nodules:
                st.markdown(f"## ✅ {len(nodules)} Nodule(s) Detected")

                # Nodule cards
                for n in nodules:
                    if n.get('diam_mm') is not None:
                        d = n['diam_mm']
                        rec_text, rec_cls = get_recommendation(d)
                        measures_html = f"""
                            <div class="nodule-measure"><span class="val">{n['diam_mm']:.1f} mm</span><span class="lbl">Diameter</span></div>
                            <div class="nodule-measure"><span class="val">{n['area_mm2']:.1f} mm²</span><span class="lbl">Area</span></div>
                        """
                    else:
                        d = n['diam_px']
                        rec_text, rec_cls = "N/A (no spacing)", "routine"
                        measures_html = f"""
                            <div class="nodule-measure"><span class="val">{n['diam_px']:.0f} px</span><span class="lbl">Diameter</span></div>
                            <div class="nodule-measure"><span class="val">{n['area_px']:.0f} px²</span><span class="lbl">Area</span></div>
                        """
                    st.markdown(f"""
                    <div class="nodule-result-card {rec_cls}">
                        <div class="nodule-id-badge">N{n['id']}</div>
                        <div class="nodule-measures">{measures_html}</div>
                        <div class="nodule-rec {rec_cls}">{rec_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Table
                rows = []
                for n in nodules:
                    r = {"Nodule": f"N{n['id']}"}
                    if n.get('diam_mm') is not None:
                        r["Diameter (mm)"] = f"{n['diam_mm']:.2f}"
                        r["Area (mm²)"] = f"{n['area_mm2']:.1f}"
                    else:
                        r["Diameter (px)"] = f"{n['diam_px']:.1f}"
                        r["Area (px²)"] = f"{n['area_px']:.0f}"
                    rows.append(r)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if spacing_xy is None:
                    st.info("💡 Enter pixel spacing above to see measurements in millimeters.")
            else:
                st.warning("No nodules detected in this slice.")

    # ================================================================
    # MODE 2 — CT VOLUME (ZIP)
    # ================================================================
    else:
        st.markdown('<p class="section-label" style="margin-top:1rem;">Upload CT Volume</p>', unsafe_allow_html=True)
        upzip = st.file_uploader("Select a ZIP file containing .mhd and .raw", type=["zip"], label_visibility="collapsed")

        if upzip is not None:
            fkey = f"vol_{upzip.name}_{upzip.size}"
            if st.session_state.get('_fkey') != fkey:
                st.session_state._fkey = fkey

                # Load volume
                with st.spinner("Loading volume metadata..."):
                    vol, sp_zyx, tmp = load_volume(upzip)

                if vol is None:
                    st.error("Could not read volume. Ensure the ZIP contains an .mhd file (with referenced .raw).")
                    return

                n_slices, h, w = vol.shape
                st.session_state._vol_array = vol
                st.session_state._vol_spacing = sp_zyx
                st.session_state._vol_tmp = tmp

                # Segment all slices with progress
                prog = st.progress(0, text="Initializing...")
                all_masks = []
                for i in range(n_slices):
                    sl = vol[i]
                    if sl.max() == 0:
                        all_masks.append(np.zeros((h, w), dtype=np.uint8))
                    else:
                        all_masks.append(segment_slice(model, sl))
                    prog.progress((i + 1) / n_slices, text=f"Analyzing slice {i+1} / {n_slices}")

                prog.progress(1.0, text="Computing 3D measurements...")
                mask_3d = np.stack(all_masks)
                labeled_3d, nodules_3d = analyze_3d(mask_3d, sp_zyx)
                st.session_state._vol_labeled = labeled_3d
                st.session_state._vol_nodules = nodules_3d
                st.session_state._vol_loaded = True
                prog.empty()

            # ---- DISPLAY RESULTS ----
            if st.session_state.get('_vol_loaded'):
                vol        = st.session_state._vol_array
                sp_zyx     = st.session_state._vol_spacing
                labeled_3d = st.session_state._vol_labeled
                nodules    = st.session_state._vol_nodules
                n_slices   = vol.shape[0]

                # --- Summary Metrics ---
                st.markdown(f"## ✅ {len(nodules)} Nodule(s) Detected  —  {n_slices} slices analyzed")

                if nodules:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Nodules", len(nodules))
                    mc2.metric("Avg \u2300", f"{np.mean([n['eq_diameter_mm'] for n in nodules]):.1f} mm")
                    mc3.metric("Max \u2300", f"{max(n['eq_diameter_mm'] for n in nodules):.1f} mm")
                    mc4.metric("Total Volume", f"{sum(n['volume_mm3'] for n in nodules):.0f} mm\u00b3")

                    # --- Nodule Cards ---
                    st.markdown('<p class="section-label" style="margin-top:1.5rem;">Nodule Findings</p>', unsafe_allow_html=True)
                    for n in nodules:
                        rec_text, rec_cls = get_recommendation(n['eq_diameter_mm'])
                        st.markdown(f"""
                        <div class="nodule-result-card {rec_cls}">
                            <div class="nodule-id-badge">N{n['id']}</div>
                            <div class="nodule-measures">
                                <div class="nodule-measure"><span class="val">{n['eq_diameter_mm']:.1f} mm</span><span class="lbl">Eq. Diameter</span></div>
                                <div class="nodule-measure"><span class="val">{n['max_diameter_mm']:.1f} mm</span><span class="lbl">Max Diameter</span></div>
                                <div class="nodule-measure"><span class="val">{n['volume_mm3']:.0f} mm³</span><span class="lbl">Volume</span></div>
                                <div class="nodule-measure"><span class="val">{n['slice_range'][0]}–{n['slice_range'][1]-1}</span><span class="lbl">Slice Range</span></div>
                                <div class="nodule-measure"><span class="val">{n['num_slices']}</span><span class="lbl">Slices</span></div>
                            </div>
                            <div class="nodule-rec {rec_cls}">{rec_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # --- Slice Viewer ---
                    st.markdown('<p class="section-label" style="margin-top:1.5rem;">Slice Viewer</p>', unsafe_allow_html=True)

                    # Build list of slices that contain nodules
                    nodule_slices = sorted(set(
                        s for n in nodules for s in range(n['slice_range'][0], n['slice_range'][1])
                    ))
                    if not nodule_slices:
                        nodule_slices = list(range(n_slices))

                    sel = st.selectbox(
                        "Select slice",
                        nodule_slices,
                        format_func=lambda s: f"Slice {s}  ({len(set(labeled_3d[s][labeled_3d[s]>0]))} nodule(s) visible)",
                    )

                    # Which nodules appear on this slice?
                    vis_nodules = [n for n in nodules if n['slice_range'][0] <= sel < n['slice_range'][1]]

                    vc1, vc2 = st.columns(2)
                    with vc1:
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
                        ax.imshow(vol[sel], cmap='gray')
                        ax.set_title(f"Slice {sel} — Original", color='#f1f5f9', fontsize=12, fontweight='600')
                        ax.axis('off')
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    with vc2:
                        fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=100)
                        draw_slice_view(ax2, vol[sel], labeled_3d[sel], vis_nodules,
                                        title=f"Slice {sel} — {len(vis_nodules)} Nodule(s)", spacing_xy=(sp_zyx[2], sp_zyx[1]))
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)

                    # --- Detailed Table ---
                    st.markdown('<p class="section-label" style="margin-top:1.5rem;">Detailed Measurements</p>', unsafe_allow_html=True)
                    rows = []
                    for n in nodules:
                        rec_text, _ = get_recommendation(n['eq_diameter_mm'])
                        rows.append({
                            "Nodule": f"N{n['id']}",
                            "Volume (mm³)": f"{n['volume_mm3']:.1f}",
                            "Eq. \u2300 (mm)": f"{n['eq_diameter_mm']:.2f}",
                            "Max \u2300 (mm)": f"{n['max_diameter_mm']:.2f}",
                            "Slices": f"{n['slice_range'][0]}–{n['slice_range'][1]-1}",
                            "# Slices": n['num_slices'],
                            "Voxels": n['num_voxels'],
                            "Recommendation": rec_text,
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    # --- Pixel Spacing Info ---
                    st.markdown(f"""
                    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1rem 1.25rem;
                                font-size:0.82rem;color:var(--text-2);margin-top:1rem;">
                        <strong style="color:var(--text-1);">Metadata Pixel Spacing:</strong>&nbsp;&nbsp;
                        X = {sp_zyx[2]:.4f} mm &nbsp;|&nbsp; Y = {sp_zyx[1]:.4f} mm &nbsp;|&nbsp; Z = {sp_zyx[0]:.4f} mm &nbsp;|&nbsp;
                        Voxel volume = {sp_zyx[0]*sp_zyx[1]*sp_zyx[2]:.6f} mm³
                    </div>
                    """, unsafe_allow_html=True)

                    # --- CSV Download ---
                    csv = pd.DataFrame(rows).to_csv(index=False)
                    st.download_button(
                        "📊 Download Results (CSV)",
                        csv,
                        f"lungvision_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        use_container_width=True,
                    )
                else:
                    st.info("✅ Analysis complete — no nodules detected in this volume.")

    # ---- Footer ----
    st.markdown("""
    <div class="app-footer">
        LungVision AI &nbsp;·&nbsp; Clinical Decision Support &nbsp;·&nbsp; Always verify with a qualified radiologist
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        show_login()
        return

    model = load_model()
    if model is None:
        st.stop()

    show_app(model)


if __name__ == "__main__":
    main()
