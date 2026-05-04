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
from matplotlib.patches import Circle
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
# CUSTOM CSS — Clinical Dark Theme (keeping your existing CSS)
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

.stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
.stMetric label, .stMetric div, .stMetric span,
.stSelectbox label, .stSelectbox div,
.stRadio label, .stRadio div,
.stTextInput label, .stTextInput div,
.stFileUploader label,
p, li, label, h1, h2, h3, h4, h5, h6, span {
    color: var(--text-1) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Header Card */
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
.header-card h1 {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    margin: 0;
}
.header-card .tagline {
    color: var(--text-2) !important;
    font-size: 0.9rem;
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
    box-shadow: 0 30px 80px rgba(0,0,0,0.5);
}
.login-icon {
    text-align: center;
    font-size: 3.2rem;
    margin-bottom: 0.75rem;
}
.login-card h2 {
    text-align: center;
    font-weight: 700;
    font-size: 1.4rem !important;
}
.login-sub {
    text-align: center;
    color: var(--text-3) !important;
    font-size: 0.82rem;
    margin-top: 0.35rem;
    margin-bottom: 2rem;
}

.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-3) !important;
    margin-bottom: 0.75rem;
    padding-left: 0.15rem;
}

div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    padding: 1.2rem 1.4rem !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 20px var(--primary-glow) !important;
}

[data-testid="stFileUploader"] > section > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-light) !important;
    border-radius: 14px !important;
    padding: 2.25rem 1.5rem !important;
}

.nodule-result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
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
}
.nodule-measure .lbl {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-3);
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

.app-footer {
    text-align: center;
    color: var(--text-3) !important;
    font-size: 0.75rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE — MemoryEfficientUNet (matches your training)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class MemoryEfficientUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(MemoryEfficientUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ============================================================
# MODEL LOADING — UPDATED ID
# ============================================================
GDRIVE_ID = "1lJOEoxPW3eUY3fdl5nuaI5V92T8Uwsyq"
MODEL_FN = "final_complete_model.pth"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_FN):
            with st.spinner("Downloading AI model from cloud..."):
                url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
                gdown.download(url, MODEL_FN, quiet=False)
        
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        checkpoint = torch.load(MODEL_FN, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel wrapping
        if state_dict and 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


# ============================================================
# PROCESSING UTILITIES
# ============================================================
def apply_lung_window(image):
    """Apply lung window: -1000 to 400 HU"""
    image = np.clip(image, -1000, 400)
    return ((image + 1000) / 1400).astype(np.float32)

def segment_slice(model, img, threshold=0.5):
    """Return binary uint8 mask matching input shape."""
    shape = img.shape
    normed = img.astype(np.float32)
    if normed.max() > 1.0:
        normed = normed / 255.0
    
    # Apply lung window
    normed = apply_lung_window(normed * 1400 - 1000) if normed.max() > 0.1 else normed
    
    resized = resize(normed, (128, 128), preserve_range=True)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().numpy()
    
    # Resize back to original dimensions
    mask = resize((prob > threshold).astype(np.float32), shape, order=0, preserve_range=True)
    return (mask > 0.5).astype(np.uint8)


def analyze_3d(mask_3d, spacing_zyx):
    """3D connected-component analysis with real-world measurements."""
    sx, sy, sz = spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]
    voxel_vol = sx * sy * sz
    labeled = label(mask_3d, connectivity=2)
    nodules = []
    for rp in regionprops(labeled):
        if rp.area < 10:
            continue
        vol_mm3 = rp.area * voxel_vol
        eq_diam = 2.0 * (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1/3)
        bb = rp.bbox
        ext_x = (bb[5]-bb[4]) * sx
        ext_y = (bb[3]-bb[2]) * sy
        ext_z = (bb[1]-bb[0]) * sz
        max_diam = max(ext_x, ext_y, ext_z)
        nodules.append({
            'id': len(nodules)+1,
            'label_id': rp.label,
            'volume_mm3': vol_mm3,
            'eq_diameter_mm': eq_diam,
            'max_diameter_mm': max_diam,
            'num_voxels': rp.area,
            'slice_range': (bb[0], bb[1]),
            'num_slices': bb[1]-bb[0],
            'centroid_zyx': rp.centroid,
        })
    return labeled, nodules


def analyze_2d(mask, spacing_xy=None):
    """2D connected-component analysis."""
    labeled = label(mask, connectivity=2)
    nodules = []
    for rp in regionprops(labeled):
        if rp.area < 10:
            continue
        area_px = rp.area
        diam_px = 2 * np.sqrt(area_px / np.pi)
        if spacing_xy:
            area_mm2 = area_px * spacing_xy[0] * spacing_xy[1]
            diam_mm = diam_px * spacing_xy[0]
        else:
            area_mm2 = None
            diam_mm = None
        nodules.append({
            'id': len(nodules)+1,
            'label_id': rp.label,
            'area_px': area_px,
            'diam_px': diam_px,
            'area_mm2': area_mm2,
            'diam_mm': diam_mm,
            'centroid': (rp.centroid[0], rp.centroid[1]),
            'mask': (labeled == rp.label).astype(np.float32),
        })
    return nodules


def load_volume(zip_file):
    """Extract zip, find .mhd, read with SimpleITK."""
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
    arr = sitk.GetArrayFromImage(img)
    sp = img.GetSpacing()
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
    """Draw a CT slice with nodule overlays."""
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
        circ = Circle((cx, cy), r, fill=False, edgecolor='#06b6d4', linewidth=2.2)
        ax.add_patch(circ)

        if 'eq_diameter_mm' in ninfo:
            label_text = f"N{ninfo['id']}\n\u2300 {ninfo['eq_diameter_mm']:.1f} mm\nV {ninfo['volume_mm3']:.0f} mm³"
        elif ninfo.get('diam_mm') is not None:
            label_text = f"N{ninfo['id']}\n\u2300 {ninfo['diam_mm']:.1f} mm\nA {ninfo['area_mm2']:.1f} mm²"
        else:
            label_text = f"N{ninfo['id']}\n\u2300 {ninfo['diam_px']:.0f} px\nA {ninfo['area_px']:.0f} px²"

        ax.annotate(
            label_text,
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
        pwd = st.text_input("Password", type="password", placeholder="Enter password", key="login_pwd")
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
    # Header
    st.markdown("""
    <div class="header-card">
        <h1>🫁 LungVision AI</h1>
        <p class="tagline">Automatic Lung Nodule Detection &amp; Volumetric Analysis</p>
        <div class="header-badge"><span class="dot"></span> AI Model Ready (Dice 0.636)</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
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
            <div><strong style="color:var(--text-1);">Architecture</strong> &nbsp; MemoryEfficientUNet</div>
            <div><strong style="color:var(--text-1);">Dice Score</strong> &nbsp; 0.636</div>
            <div><strong style="color:var(--text-1);">Training</strong> &nbsp; LUNA16 + QIN</div>
            <div><strong style="color:var(--text-1);">Patches</strong> &nbsp; 2,435</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Disclaimer</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem;line-height:1.6;color:var(--text-3);">
            This is a clinical decision support tool. All findings must be verified by a qualified radiologist. Do not use as sole basis for diagnosis.
        </div>
        """, unsafe_allow_html=True)

    # Mode Selection
    st.markdown('<p class="section-label">Scan Type</p>', unsafe_allow_html=True)
    mode = st.radio("", ["Single CT Slice (PNG)", "CT Volume (MHD + RAW as ZIP)"], horizontal=True, label_visibility="collapsed")

    # PNG Mode
    if "PNG" in mode:
        st.markdown('<p class="section-label" style="margin-top:1rem;">Upload CT Slice</p>', unsafe_allow_html=True)
        upfile = st.file_uploader("Select a PNG or JPG file", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        sp_col1, sp_col2 = st.columns(2)
        with sp_col1:
            px_sp_x = st.number_input("Pixel spacing X (mm/px)", min_value=0.0, value=0.0, step=0.01, format="%.3f")
        with sp_col2:
            px_sp_y = st.number_input("Pixel spacing Y (mm/px)", min_value=0.0, value=0.0, step=0.01, format="%.3f")
        spacing_xy = (px_sp_x, px_sp_y) if px_sp_x > 0 and px_sp_y > 0 else None

        if upfile is not None:
            img_pil = Image.open(upfile).convert('L')
            img_arr = np.array(img_pil, dtype=np.float32)
            mask = segment_slice(model, img_arr)
            nodules = analyze_2d(mask, spacing_xy)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-label">Original Slice</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=100)
                ax.imshow(img_arr, cmap='gray')
                ax.set_title("CT Slice", color='#f1f5f9', fontsize=12)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

            with c2:
                st.markdown('<p class="section-label">Segmentation Overlay</p>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(5.5, 5.5), dpi=100)
                draw_slice_view(ax2, img_arr, label(mask), nodules, title=f"{len(nodules)} Nodule(s) Detected", spacing_xy=spacing_xy)
                st.pyplot(fig2)
                plt.close(fig2)

            if nodules:
                st.markdown(f"## ✅ {len(nodules)} Nodule(s) Detected")
                for n in nodules:
                    if n.get('diam_mm') is not None:
                        rec_text, rec_cls = get_recommendation(n['diam_mm'])
                        measures_html = f"""
                            <div class="nodule-measure"><span class="val">{n['diam_mm']:.1f} mm</span><span class="lbl">Diameter</span></div>
                            <div class="nodule-measure"><span class="val">{n['area_mm2']:.1f} mm²</span><span class="lbl">Area</span></div>
                        """
                    else:
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
            else:
                st.info("No nodules detected in this slice.")

    # Volume Mode
    else:
        st.markdown('<p class="section-label" style="margin-top:1rem;">Upload CT Volume</p>', unsafe_allow_html=True)
        upzip = st.file_uploader("Select a ZIP file containing .mhd and .raw", type=["zip"], label_visibility="collapsed")

        if upzip is not None:
            with st.spinner("Loading volume..."):
                vol, sp_zyx, tmp = load_volume(upzip)

            if vol is None:
                st.error("Could not read volume. Ensure the ZIP contains an .mhd file.")
            else:
                n_slices, h, w = vol.shape
                prog = st.progress(0, text="Segmenting slices...")
                all_masks = []
                for i in range(n_slices):
                    sl = vol[i]
                    all_masks.append(segment_slice(model, sl))
                    prog.progress((i + 1) / n_slices, text=f"Analyzing slice {i+1} / {n_slices}")

                mask_3d = np.stack(all_masks)
                labeled_3d, nodules = analyze_3d(mask_3d, sp_zyx)
                prog.empty()
                shutil.rmtree(tmp, ignore_errors=True)

                st.markdown(f"## ✅ {len(nodules)} Nodule(s) Detected — {n_slices} slices analyzed")

                if nodules:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Nodules", len(nodules))
                    mc2.metric("Avg \u2300", f"{np.mean([n['eq_diameter_mm'] for n in nodules]):.1f} mm")
                    mc3.metric("Max \u2300", f"{max(n['eq_diameter_mm'] for n in nodules):.1f} mm")
                    mc4.metric("Total Volume", f"{sum(n['volume_mm3'] for n in nodules):.0f} mm³")

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
                            </div>
                            <div class="nodule-rec {rec_cls}">{rec_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Slice viewer
                    nodule_slices = sorted(set(s for n in nodules for s in range(n['slice_range'][0], n['slice_range'][1])))
                    sel = st.selectbox("Select slice", nodule_slices or list(range(n_slices)))
                    vis_nodules = [n for n in nodules if n['slice_range'][0] <= sel < n['slice_range'][1]]

                    vc1, vc2 = st.columns(2)
                    with vc1:
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
                        ax.imshow(vol[sel], cmap='gray')
                        ax.set_title(f"Slice {sel} — Original", color='#f1f5f9')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
                    with vc2:
                        fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=100)
                        draw_slice_view(ax2, vol[sel], labeled_3d[sel], vis_nodules, title=f"Slice {sel} — {len(vis_nodules)} Nodule(s)")
                        st.pyplot(fig2)
                        plt.close(fig2)

                    # Download
                    rows = [{"Nodule": f"N{n['id']}", "Volume (mm³)": f"{n['volume_mm3']:.1f}",
                             "Eq. Diameter (mm)": f"{n['eq_diameter_mm']:.2f}",
                             "Max Diameter (mm)": f"{n['max_diameter_mm']:.2f}",
                             "Slices": f"{n['slice_range'][0]}–{n['slice_range'][1]-1}"} for n in nodules]
                    csv = pd.DataFrame(rows).to_csv(index=False)
                    st.download_button("📊 Download Results (CSV)", csv, f"lungvision_results.csv", use_container_width=True)
                else:
                    st.info("No nodules detected in this volume.")

    # Footer
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
