import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.filters import sobel
import tempfile
import SimpleITK as sitk
from collections import OrderedDict
from PIL import Image
import warnings
from datetime import datetime
import zipfile
import os
import gdown
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import shutil
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LungVision AI | Nodule Segmentation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# RELAXED CT VALIDATION FUNCTIONS
# ============================================================
def is_valid_lung_ct(image_array):
    """
    Validate that the uploaded image is a genuine lung CT scan.
    Returns (is_valid, reason)
    """
    # Convert to 2D if needed
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 3:
            # Convert RGB to grayscale manually
            image_array = 0.2989 * image_array[:, :, 0] + 0.5870 * image_array[:, :, 1] + 0.1140 * image_array[:, :, 2]
        else:
            image_array = image_array[:, :, 0]
    
    h, w = image_array.shape
    
    # Check 1: Image dimensions
    if h < 100 or w < 100:
        return False, "Image too small for CT scan"
    
    # Check 2: Intensity distribution - relaxed
    img_min, img_max = image_array.min(), image_array.max()
    img_range = img_max - img_min
    
    # If image has very low contrast (almost uniform), reject
    if img_range < 10:
        return False, "Image has very low contrast - not a valid CT scan"
    
    # Check 3: Lung tissue has dark and light regions (relaxed)
    dark_threshold = np.percentile(image_array, 15)
    dark_regions = (image_array < dark_threshold).astype(np.uint8)
    dark_ratio = dark_regions.sum() / (h * w)
    
    # CT scans can have varying dark region ratios depending on slice position
    # Only flag extreme cases
    if dark_ratio < 0.01:
        return False, "No dark regions detected - likely not a lung CT"
    
    # Check 4: Edge characteristics - only flag extreme cases
    grad_x = sobel(image_array.astype(np.float32), axis=0)
    grad_y = sobel(image_array.astype(np.float32), axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mean = grad_mag.mean()
    
    # Photos have very high edge gradients ( > 0.5), CT scans are lower
    # This check is relaxed for CT scans (they can have sharper edges)
    if grad_mean > 0.8:
        return False, "Image has extremely sharp edges - appears to be a photograph"
    
    # Check 5: Edge density - only flag extreme cases
    edges = (grad_mag > 0.15).astype(np.uint8)
    edge_ratio = edges.sum() / (h * w)
    
    # Text/UI elements create many small edges, but lung CT can have some edges
    if edge_ratio > 0.35:
        return False, "Too many high-contrast edges - may contain text or UI elements"
    
    # All checks passed
    return True, "Valid lung CT detected"

def validate_zip_ct_volume(zip_file):
    """
    Validate that the ZIP contains a valid CT volume (relaxed)
    """
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "upload.zip")
    try:
        with open(zpath, "wb") as f:
            f.write(zip_file.getbuffer())
        
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(tmp)
        
        # Find MHD file
        mhd_file = None
        for root, _, files in os.walk(tmp):
            for fn in files:
                if fn.lower().endswith('.mhd'):
                    mhd_file = os.path.join(root, fn)
                    break
            if mhd_file:
                break
        
        if not mhd_file:
            shutil.rmtree(tmp, ignore_errors=True)
            return False, "No .mhd file found in ZIP"
        
        # Try to read the volume
        img = sitk.ReadImage(mhd_file)
        arr = sitk.GetArrayFromImage(img)
        
        if arr.ndim != 3:
            shutil.rmtree(tmp, ignore_errors=True)
            return False, "Not a 3D volume"
        
        if arr.shape[1] < 100 or arr.shape[2] < 100:
            shutil.rmtree(tmp, ignore_errors=True)
            return False, "Volume dimensions too small"
        
        # Quick validation on middle slice only (faster)
        mid_slice = arr.shape[0] // 2
        slice_img = arr[mid_slice]
        is_valid, reason = is_valid_lung_ct(slice_img)
        
        if not is_valid:
            shutil.rmtree(tmp, ignore_errors=True)
            return False, f"Volume validation failed: {reason}"
        
        shutil.rmtree(tmp, ignore_errors=True)
        return True, "Valid CT volume"
        
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return False, f"Error reading volume: {str(e)}"

# ============================================================
# CLEAN CSS - FIXED LOGIN CENTERING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #0ea5e9;
    --primary-dark: #0369a1;
    --bg-deep: #060b14;
    --bg-main: #0b1120;
    --bg-card: #111827;
    --border: #1e2d4a;
    --text-1: #f1f5f9;
    --text-2: #94a3b8;
    --text-3: #64748b;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
}

.main .block-container {
    background: var(--bg-main);
    max-width: 1400px;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, var(--bg-deep) 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* Login wrapper - FIXED CENTERING */
.login-wrapper {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    background: var(--bg-main);
    z-index: 999;
}

.login-card {
    background: linear-gradient(160deg, #111d32 0%, #0b1221 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    width: 100%;
    max-width: 380px;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
}

.login-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

.login-card h2 {
    font-size: 1.5rem !important;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: var(--text-1) !important;
}

.login-sub {
    color: var(--text-3) !important;
    font-size: 0.8rem;
    margin-bottom: 1.5rem;
}

/* Header */
.header-card {
    background: linear-gradient(135deg, #0c2340 0%, #132e52 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 2rem;
    margin-bottom: 1.5rem;
}

.header-card h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin: 0;
}

.header-card .tagline {
    color: var(--text-2) !important;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-3) !important;
    margin-bottom: 0.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
}

[data-testid="stFileUploader"] > section > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}

div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    padding: 0.8rem 1rem !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

.stRadio [data-baseweb="radio-group"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.3rem !important;
}

.nodule-result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}
.nodule-result-card.routine { border-left: 3px solid var(--green); }
.nodule-result-card.followup { border-left: 3px solid var(--amber); }
.nodule-result-card.urgent { border-left: 3px solid var(--red); }

.nodule-id-badge {
    background: rgba(14,165,233,0.12);
    color: var(--primary);
    font-weight: 700;
    font-size: 0.75rem;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    min-width: 45px;
    text-align: center;
}

.nodule-measures {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.nodule-measure {
    display: flex;
    flex-direction: column;
}
.nodule-measure .val {
    font-size: 1rem;
    font-weight: 700;
}
.nodule-measure .lbl {
    font-size: 0.6rem;
    text-transform: uppercase;
    color: var(--text-3);
}

.nodule-rec {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}
.nodule-rec.routine { background: rgba(16,185,129,0.12); color: var(--green); }
.nodule-rec.followup { background: rgba(245,158,11,0.12); color: var(--amber); }
.nodule-rec.urgent { background: rgba(239,68,68,0.12); color: var(--red); }

.app-footer {
    text-align: center;
    color: var(--text-3) !important;
    font-size: 0.7rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
}

.validation-error {
    background: rgba(239,68,68,0.1);
    border-left: 3px solid var(--red);
    padding: 0.75rem 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.validation-success {
    background: rgba(16,185,129,0.1);
    border-left: 3px solid var(--green);
    padding: 0.75rem 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE
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
# MODEL LOADING
# ============================================================
GDRIVE_ID = "1lJOEoxPW3eUY3fdl5nuaI5V92T8Uwsyq"
MODEL_FN = "final_complete_model.pth"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_FN):
            with st.spinner("Downloading AI model..."):
                url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
                gdown.download(url, MODEL_FN, quiet=False)
        
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        checkpoint = torch.load(MODEL_FN, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        if state_dict and 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


# ============================================================
# UTILITIES
# ============================================================
def apply_lung_window(image):
    image = np.clip(image, -1000, 400)
    return ((image + 1000) / 1400).astype(np.float32)

def segment_slice(model, img, threshold=0.5):
    shape = img.shape
    normed = img.astype(np.float32)
    if normed.max() > 1.0:
        normed = normed / 255.0
    
    normed = apply_lung_window(normed * 1400 - 1000) if normed.max() > 0.1 else normed
    
    resized = resize(normed, (128, 128), preserve_range=True)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().numpy()
    
    mask = resize((prob > threshold).astype(np.float32), shape, order=0, preserve_range=True)
    return (mask > 0.5).astype(np.uint8)


def analyze_3d(mask_3d, spacing_zyx):
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


def make_overlay(slice_img, mask_2d, alpha=0.45):
    n = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
    rgb = np.stack([n, n, n], axis=-1)
    m = mask_2d > 0.5
    rgb[m, 0] = np.clip(rgb[m, 0] + alpha, 0, 1)
    rgb[m, 1] = np.clip(rgb[m, 1] * 0.25, 0, 1)
    rgb[m, 2] = np.clip(rgb[m, 2] * 0.25, 0, 1)
    return rgb


def draw_slice_view(ax, slice_img, labeled_2d, nodules_info, title=""):
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
        circ = Circle((cx, cy), r, fill=False, edgecolor='#06b6d4', linewidth=2)
        ax.add_patch(circ)

        if 'eq_diameter_mm' in ninfo:
            label_text = f"N{ninfo['id']}\n{ninfo['eq_diameter_mm']:.1f}mm"
        elif ninfo.get('diam_mm') is not None:
            label_text = f"N{ninfo['id']}\n{ninfo['diam_mm']:.1f}mm"
        else:
            label_text = f"N{ninfo['id']}\n{ninfo['diam_px']:.0f}px"

        ax.annotate(
            label_text,
            xy=(cx, cy), xytext=(cx + r + 8, cy - r),
            fontsize=8, fontweight='600', color='#f1f5f9',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#273755', alpha=0.9),
        )

    ax.set_title(title, color='#f1f5f9', fontsize=11)
    ax.axis('off')


# ============================================================
# LOGIN PAGE - FIXED (no scroll)
# ============================================================
def show_login():
    st.markdown("""
    <div class="login-wrapper">
        <div class="login-card">
            <div class="login-icon">🫁</div>
            <h2>LungVision AI</h2>
            <p class="login-sub">Clinical Nodule Segmentation</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Use columns to center the form within the fixed wrapper
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            user = st.text_input("Radiologist ID", placeholder="Enter your ID")
            pwd = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            if submitted:
                if user == "radiologist" and pwd == "hit500":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")


# ============================================================
# MAIN APP
# ============================================================
def show_app(model):
    # Header
    st.markdown("""
    <div class="header-card">
        <h1>🫁 LungVision AI</h1>
        <p class="tagline">Automatic Lung Nodule Detection & Segmentation</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 👨‍⚕️ Radiologist")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.caption("Upload a lung CT scan to detect and segment nodules.")
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.caption("1. Select scan type below")
        st.caption("2. Upload PNG or ZIP file")
        st.caption("3. System validates it's a lung CT")
        st.caption("4. View detected nodules")
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.caption("Clinical decision support only. Verify all findings.")

    # Mode selection
    st.markdown('<p class="section-label">Scan Type</p>', unsafe_allow_html=True)
    mode = st.radio("", ["Single CT Slice", "CT Volume (MHD + RAW as ZIP)"], horizontal=True, label_visibility="collapsed")

    # ========== PNG MODE ==========
    if "Single" in mode:
        st.markdown('<p class="section-label">Upload CT Slice</p>', unsafe_allow_html=True)
        upfile = st.file_uploader("Select PNG or JPG", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if upfile is not None:
            img_pil = Image.open(upfile).convert('L')
            img_arr = np.array(img_pil, dtype=np.float32)
            
            # VALIDATE CT (relaxed)
            is_valid, reason = is_valid_lung_ct(img_arr)
            
            if not is_valid:
                st.markdown(f"""
                <div class="validation-error">
                    ⚠️ <strong>Invalid Input:</strong> {reason}<br>
                    Please upload a genuine lung CT scan.
                </div>
                """, unsafe_allow_html=True)
                return
            
            st.markdown("""
            <div class="validation-success">
                ✅ Valid lung CT detected. Processing...
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Analyzing..."):
                mask = segment_slice(model, img_arr)
                nodules = analyze_2d(mask, spacing_xy=None)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_arr, cmap='gray')
                ax.set_title("Original CT Slice")
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                draw_slice_view(ax2, img_arr, label(mask), nodules, title=f"{len(nodules)} Nodule(s)")
                st.pyplot(fig2)
                plt.close(fig2)

            if nodules:
                st.markdown(f"### ✅ {len(nodules)} Nodule(s) Detected")
                for n in nodules:
                    st.markdown(f"""
                    <div class="nodule-result-card routine">
                        <div class="nodule-id-badge">N{n['id']}</div>
                        <div class="nodule-measures">
                            <div class="nodule-measure"><span class="val">{n['diam_px']:.0f} px</span><span class="lbl">Diameter</span></div>
                            <div class="nodule-measure"><span class="val">{n['area_px']:.0f} px²</span><span class="lbl">Area</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No nodules detected in this slice.")

    # ========== VOLUME MODE ==========
    else:
        st.markdown('<p class="section-label">Upload CT Volume</p>', unsafe_allow_html=True)
        upzip = st.file_uploader("Select ZIP with .mhd and .raw files", type=["zip"], label_visibility="collapsed")

        if upzip is not None:
            # Validate ZIP (relaxed)
            with st.spinner("Validating CT volume..."):
                is_valid, reason = validate_zip_ct_volume(upzip)
            
            if not is_valid:
                st.markdown(f"""
                <div class="validation-error">
                    ⚠️ <strong>Invalid CT Volume:</strong> {reason}<br>
                    Please upload a valid lung CT volume in MHD/RAW format.
                </div>
                """, unsafe_allow_html=True)
                return
            
            st.markdown("""
            <div class="validation-success">
                ✅ Valid lung CT volume detected. Processing...
            </div>
            """, unsafe_allow_html=True)
            
            # Reset file pointer
            upzip.seek(0)
            
            with st.spinner("Loading volume..."):
                vol, sp_zyx, tmp = load_volume(upzip)

            if vol is None:
                st.error("Could not read volume.")
            else:
                n_slices = vol.shape[0]
                
                # Segment all slices
                prog = st.progress(0)
                status = st.empty()
                all_masks = []
                for i in range(n_slices):
                    status.text(f"Segmenting slice {i+1}/{n_slices}")
                    all_masks.append(segment_slice(model, vol[i]))
                    prog.progress((i + 1) / n_slices)
                
                mask_3d = np.stack(all_masks)
                labeled_3d, nodules = analyze_3d(mask_3d, sp_zyx)
                
                status.empty()
                prog.empty()
                shutil.rmtree(tmp, ignore_errors=True)

                st.markdown(f"### ✅ {len(nodules)} Nodule(s) Detected")

                if nodules:
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Nodules", len(nodules))
                    col2.metric("Avg Diameter", f"{np.mean([n['eq_diameter_mm'] for n in nodules]):.1f} mm")
                    col3.metric("Max Diameter", f"{max(n['eq_diameter_mm'] for n in nodules):.1f} mm")
                    col4.metric("Total Volume", f"{sum(n['volume_mm3'] for n in nodules):.0f} mm³")

                    # Nodule cards
                    for n in nodules:
                        rec_text = "Routine follow-up" if n['eq_diameter_mm'] < 5 else "Short-term follow-up" if n['eq_diameter_mm'] < 8 else "Further evaluation"
                        rec_class = "routine" if n['eq_diameter_mm'] < 5 else "followup" if n['eq_diameter_mm'] < 8 else "urgent"
                        
                        st.markdown(f"""
                        <div class="nodule-result-card {rec_class}">
                            <div class="nodule-id-badge">N{n['id']}</div>
                            <div class="nodule-measures">
                                <div class="nodule-measure"><span class="val">{n['eq_diameter_mm']:.1f} mm</span><span class="lbl">Diameter</span></div>
                                <div class="nodule-measure"><span class="val">{n['volume_mm3']:.0f} mm³</span><span class="lbl">Volume</span></div>
                                <div class="nodule-measure"><span class="val">Slices {n['slice_range'][0]}-{n['slice_range'][1]-1}</span><span class="lbl">Range</span></div>
                            </div>
                            <div class="nodule-rec {rec_class}">{rec_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Slice viewer
                    valid_slices = list(range(n_slices))
                    selected_slice = st.selectbox("View slice", valid_slices, format_func=lambda x: f"Slice {x}")
                    
                    if 0 <= selected_slice < n_slices:
                        visible_nodules = [n for n in nodules if n['slice_range'][0] <= selected_slice < n['slice_range'][1]]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig, ax = plt.subplots(figsize=(5, 5))
                            ax.imshow(vol[selected_slice], cmap='gray')
                            ax.set_title(f"Original - Slice {selected_slice}")
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col2:
                            fig2, ax2 = plt.subplots(figsize=(5, 5))
                            draw_slice_view(ax2, vol[selected_slice], labeled_3d[selected_slice], visible_nodules, title=f"Slice {selected_slice} - {len(visible_nodules)} Nodule(s)")
                            st.pyplot(fig2)
                            plt.close(fig2)

                    # Download results
                    rows = [{
                        "Nodule": f"N{n['id']}",
                        "Volume (mm³)": round(n['volume_mm3'], 1),
                        "Diameter (mm)": round(n['eq_diameter_mm'], 2),
                        "Max Diameter (mm)": round(n['max_diameter_mm'], 2),
                        "Slice Range": f"{n['slice_range'][0]}-{n['slice_range'][1]-1}",
                        "Number of Slices": n['num_slices']
                    } for n in nodules]
                    
                    csv = pd.DataFrame(rows).to_csv(index=False)
                    st.download_button(
                        "📊 Download Results (CSV)",
                        csv,
                        f"lungvision_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        use_container_width=True,
                    )
                else:
                    st.info("No nodules detected in this volume.")

    # Footer
    st.markdown("""
    <div class="app-footer">
        LungVision AI · Clinical Decision Support · Always verify with a qualified radiologist
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
