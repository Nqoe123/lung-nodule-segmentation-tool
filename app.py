import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.measure import label, regionprops
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
from scipy import ndimage

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LungVision AI | Nodule Segmentation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main .block-container {
    background: #0b1120;
    max-width: 1400px;
    padding: 1rem 1rem 2rem 1rem !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #060b14 100%) !important;
    border-right: 1px solid #1e2d4a !important;
}
.header-card {
    background: linear-gradient(135deg, #0c2340 0%, #132e52 100%);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 1.25rem 2rem;
    margin-bottom: 1.5rem;
}
.header-card h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin: 0;
    color: #f1f5f9;
}
.header-card .tagline {
    color: #94a3b8 !important;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b !important;
    margin-bottom: 0.5rem;
}
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0369a1) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] > section > div {
    background: #111827 !important;
    border: 2px dashed #1e2d4a !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}
div[data-testid="stMetric"] {
    background: #111827 !important;
    padding: 0.8rem 1rem !important;
    border-radius: 12px !important;
    border: 1px solid #1e2d4a !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    color: #f1f5f9 !important;
}
.nodule-result-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}
.nodule-result-card.routine { border-left: 3px solid #10b981; }
.nodule-result-card.followup { border-left: 3px solid #f59e0b; }
.nodule-result-card.urgent { border-left: 3px solid #ef4444; }
.nodule-id-badge {
    background: rgba(14,165,233,0.12);
    color: #0ea5e9;
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
    color: #f1f5f9;
}
.nodule-measure .lbl {
    font-size: 0.6rem;
    text-transform: uppercase;
    color: #64748b;
}
.nodule-rec {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}
.nodule-rec.routine { background: rgba(16,185,129,0.12); color: #10b981; }
.nodule-rec.followup { background: rgba(245,158,11,0.12); color: #f59e0b; }
.nodule-rec.urgent { background: rgba(239,68,68,0.12); color: #ef4444; }
.app-footer {
    text-align: center;
    color: #64748b !important;
    font-size: 0.7rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1e2d4a;
}
#MainMenu, footer, header { visibility: hidden; }
.warning-text {
    color: #f59e0b;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}
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
        return self.outc(x)


# ============================================================
# MODEL LOADING
# ============================================================
GDRIVE_ID = "1ZMXIzhxrvtEwXmbs1G2HrVRMl8-RkKc8"
MODEL_FN = "best_model.pth"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_FN):
            with st.spinner("Downloading model..."):
                url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
                gdown.download(url, MODEL_FN, quiet=False)
        
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        state_dict = torch.load(MODEL_FN, map_location='cpu')
        
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
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
# CT VALIDATION FUNCTIONS
# ============================================================
def is_valid_ct_slice(image):
    """Check if an image looks like a real CT slice"""
    # CT images typically have:
    # 1. High contrast (lung window range -1000 to 400 HU)
    # 2. Specific intensity distribution
    # 3. Non-uniform patterns (not solid beans/objects)
    
    img_min = image.min()
    img_max = image.max()
    img_range = img_max - img_min
    
    # CT images have large intensity range
    if img_range < 50:
        return False, "Image has very low contrast (not a CT scan)"
    
    # Check if image is too uniform (like a plate of beans)
    std_dev = np.std(image)
    if std_dev < 10:
        return False, "Image appears too uniform - not a valid CT slice"
    
    # Check if image has unrealistic patterns
    # Use edge detection - real CT has many edges
    from scipy import ndimage
    edges = ndimage.sobel(image)
    edge_density = np.abs(edges).mean()
    
    if edge_density < 5:
        return False, "Image lacks typical CT tissue texture"
    
    return True, "Valid CT slice"

def apply_lung_window(image):
    image = np.clip(image, -1000, 400)
    return ((image + 1000) / 1400).astype(np.float32)

# ============================================================
# SEGMENTATION FUNCTIONS
# ============================================================
PATCH_SIZE = 128

def segment_slice(model, img, threshold=0.5):
    shape = img.shape
    normed = img.astype(np.float32)
    
    # Normalize if needed
    if normed.max() > 1.0:
        normed = normed / 255.0
    
    # Apply lung window if it looks like CT (not already normalized)
    if normed.max() <= 1.0 and normed.min() >= 0:
        # Convert back to HU range approximation
        normed = normed * 1400 - 1000
        normed = apply_lung_window(normed)
    else:
        normed = apply_lung_window(normed)
    
    resized = resize(normed, (PATCH_SIZE, PATCH_SIZE), preserve_range=True)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().numpy()
    
    mask = resize((prob > threshold).astype(np.float32), shape, order=0, preserve_range=True)
    return (mask > 0.5).astype(np.uint8)

def analyze_3d_correct(mask_3d, spacing_zyx, volume_shape):
    """
    CORRECT 3D analysis - keeps nodules connected across slices
    """
    z_spacing, y_spacing, x_spacing = spacing_zyx
    voxel_volume_mm3 = x_spacing * y_spacing * z_spacing
    
    # Label connected components in 3D (keeps nodules spanning multiple slices connected)
    labeled_mask = label(mask_3d, connectivity=2)  # Full connectivity
    nodules = []
    
    for region in regionprops(labeled_mask):
        # Skip tiny regions (likely noise)
        if region.area < 10:
            continue
        
        # Calculate physical volume
        volume_mm3 = region.area * voxel_volume_mm3
        
        # Calculate equivalent spherical diameter
        diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1/3)
        
        # Get bounding box in voxel coordinates
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        
        # Ensure bounds are within volume
        min_z = max(0, min_z)
        max_z = min(volume_shape[0], max_z)
        min_y = max(0, min_y)
        max_y = min(volume_shape[1], max_y)
        min_x = max(0, min_x)
        max_x = min(volume_shape[2], max_x)
        
        # Slice range (actual slice indices, not scaled)
        slice_start = int(min_z)
        slice_end = int(max_z - 1)  # Convert exclusive to inclusive
        
        # Ensure slice_end is valid
        if slice_end < slice_start:
            slice_end = slice_start
        
        # Physical dimensions
        height_mm = (max_y - min_y) * y_spacing
        width_mm = (max_x - min_x) * x_spacing
        depth_mm = (max_z - min_z) * z_spacing
        
        nodules.append({
            'id': len(nodules) + 1,
            'volume_mm3': volume_mm3,
            'diameter_mm': diameter_mm,
            'height_mm': height_mm,
            'width_mm': width_mm,
            'depth_mm': depth_mm,
            'slice_start': slice_start,
            'slice_end': slice_end,
            'num_slices': slice_end - slice_start + 1,
            'centroid': region.centroid,
            'voxel_count': region.area
        })
    
    # Sort by slice_start for consistent display
    nodules.sort(key=lambda x: x['slice_start'])
    
    return labeled_mask, nodules

def analyze_2d_correct(mask_2d, spacing_xy=None, is_ct=True):
    """Correct 2D analysis for single slices with CT validation"""
    labeled = label(mask_2d, connectivity=2)
    nodules = []
    
    for region in regionprops(labeled):
        if region.area < 10:
            continue
        
        area_px = region.area
        diam_px = 2 * np.sqrt(area_px / np.pi)
        
        if spacing_xy and is_ct:
            x_spacing, y_spacing = spacing_xy
            area_mm2 = area_px * x_spacing * y_spacing
            diam_mm = diam_px * x_spacing
        else:
            area_mm2 = None
            diam_mm = None
        
        nodules.append({
            'id': len(nodules) + 1,
            'area_px': area_px,
            'diameter_px': diam_px,
            'area_mm2': area_mm2,
            'diameter_mm': diam_mm,
            'centroid': region.centroid,
            'mask': (labeled == region.label).astype(np.uint8)
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
    volume = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()  # (x, y, z) in mm
    spacing_zyx = (spacing[2], spacing[1], spacing[0])  # Convert to (z, y, x)
    
    return volume, spacing_zyx, tmp

def make_overlay(slice_img, mask_2d, alpha=0.45):
    norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
    rgb = np.stack([norm, norm, norm], axis=-1)
    mask_bool = mask_2d > 0.5
    rgb[mask_bool, 0] = np.clip(rgb[mask_bool, 0] + alpha, 0, 1)
    rgb[mask_bool, 1] = np.clip(rgb[mask_bool, 1] * 0.25, 0, 1)
    rgb[mask_bool, 2] = np.clip(rgb[mask_bool, 2] * 0.25, 0, 1)
    return rgb

def draw_slice(ax, slice_img, labeled_2d, nodules_info, title=""):
    overlay = make_overlay(slice_img, (labeled_2d > 0).astype(np.float32))
    ax.imshow(overlay)
    
    for nodule in nodules_info:
        # Get mask for this nodule in this slice
        if 'label_id' in nodule:
            mask = (labeled_2d == nodule['label_id'])
        elif 'mask' in nodule:
            mask = nodule['mask'] > 0.5
        else:
            continue
        
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        
        cx, cy = np.mean(xs), np.mean(ys)
        radius = max((np.max(xs) - np.min(xs)) / 2, (np.max(ys) - np.min(ys)) / 2) + 6
        
        circle = Circle((cx, cy), radius, fill=False, edgecolor='#06b6d4', linewidth=2)
        ax.add_patch(circle)
        
        label_text = f"N{nodule['id']}"
        if 'diameter_mm' in nodule and nodule['diameter_mm']:
            label_text += f"\n{nodule['diameter_mm']:.1f}mm"
        elif 'diameter_px' in nodule:
            label_text += f"\n{nodule['diameter_px']:.0f}px"
        
        ax.annotate(
            label_text,
            xy=(cx, cy), xytext=(cx + radius + 8, cy - radius),
            fontsize=8, fontweight='600', color='#f1f5f9',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#273755', alpha=0.9),
        )
    
    ax.set_title(title, color='#f1f5f9', fontsize=11)
    ax.axis('off')


# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown("""
    <style>#MainMenu{visibility:hidden;}header{visibility:hidden;}footer{visibility:hidden;}</style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background:linear-gradient(160deg,#111d32 0%,#0b1221 100%);border:1px solid #1e2d4a;border-radius:28px;padding:3rem 2.5rem;text-align:center;margin-top:15vh;">
            <div style="font-size:4rem;margin-bottom:1rem;">🫁</div>
            <h2 style="font-size:1.8rem;font-weight:700;margin-bottom:0.5rem;color:#f1f5f9;">LungVision AI</h2>
            <p style="color:#64748b;font-size:0.85rem;margin-bottom:2rem;">Clinical Nodule Segmentation</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Radiologist ID", placeholder="Enter your ID")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if username == "radiologist" and password == "hit500":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")


# ============================================================
# MAIN APP
# ============================================================
def show_app(model):
    st.markdown("""
    <div class="header-card">
        <h1>LungVision AI</h1>
        <p class="tagline">Automatic Lung Nodule Detection and Segmentation</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Radiologist Panel")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### Information")
        st.caption("Upload a CT scan to detect and segment lung nodules.")
        st.markdown("---")
        st.markdown("### Instructions")
        st.caption("1. Select scan type below")
        st.caption("2. For Volume: Upload ZIP with .mhd and .raw files")
        st.caption("3. For Single Slice: Upload a CT slice (PNG/JPG from DICOM)")
        st.markdown("---")
        st.markdown("### Disclaimer")
        st.caption("Clinical decision support only. Verify all findings.")

    st.markdown('<p class="section-label">Scan Type</p>', unsafe_allow_html=True)
    mode = st.radio("", ["Single CT Slice", "CT Volume (MHD + RAW as ZIP)"], horizontal=True, label_visibility="collapsed")

    # ========== SINGLE SLICE MODE ==========
    if "Single" in mode:
        st.markdown('<p class="section-label">Upload CT Slice</p>', unsafe_allow_html=True)
        st.info("Note: Please upload actual CT slices (DICOM converted to PNG/JPG). The model is trained on CT images only.")
        
        upfile = st.file_uploader("Select PNG or JPG", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if upfile:
            img = np.array(Image.open(upfile).convert('L'), dtype=np.float32)
            
            # Validate that this looks like a CT image
            is_ct, validation_msg = is_valid_ct_slice(img)
            
            if not is_ct:
                st.warning(f"⚠️ {validation_msg}")
                st.markdown('<p class="warning-text">The model may produce unreliable results. Please upload a real CT slice.</p>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing..."):
                mask = segment_slice(model, img)
                nodules = analyze_2d_correct(mask, is_ct=is_ct)
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img, cmap='gray')
                ax.set_title("Original CT Slice")
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                labeled = label(mask)
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                
                # Create nodule info for display
                display_nodules = []
                for n in nodules:
                    display_nodules.append({
                        'id': n['id'],
                        'label_id': n['id'],
                        'diameter_mm': n['diameter_mm'],
                        'diameter_px': n['diameter_px'],
                        'mask': n['mask']
                    })
                
                draw_slice(ax2, img, labeled, display_nodules, title=f"{len(nodules)} Nodule(s) Detected")
                st.pyplot(fig2)
                plt.close(fig2)
            
            if nodules:
                st.markdown(f"### {len(nodules)} Nodule(s) Detected")
                for n in nodules:
                    diam_display = f"{n['diameter_mm']:.1f} mm" if n['diameter_mm'] else f"{n['diameter_px']:.0f} px"
                    area_display = f"{n['area_mm2']:.1f} mm²" if n['area_mm2'] else f"{n['area_px']:.0f} px²"
                    st.markdown(f"""
                    <div class="nodule-result-card routine">
                        <div class="nodule-id-badge">N{n['id']}</div>
                        <div class="nodule-measures">
                            <div class="nodule-measure"><span class="val">{diam_display}</span><span class="lbl">Diameter</span></div>
                            <div class="nodule-measure"><span class="val">{area_display}</span><span class="lbl">Area</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No nodules detected in this slice.")

    # ========== VOLUME MODE ==========
    else:
        st.markdown('<p class="section-label">Upload CT Volume</p>', unsafe_allow_html=True)
        st.info("Upload a ZIP file containing .mhd and .raw files from a CT scan")
        
        upzip = st.file_uploader("Select ZIP with .mhd and .raw files", type=["zip"], label_visibility="collapsed")

        if upzip:
            with st.spinner("Loading volume..."):
                volume, spacing_zyx, temp_dir = load_volume(upzip)
            
            if volume is None:
                st.error("Could not read volume. Ensure ZIP contains .mhd and .raw files.")
            else:
                num_slices, height, width = volume.shape
                st.success(f"Volume loaded: {num_slices} slices | {height}x{width} pixels | Spacing: {spacing_zyx[2]:.3f}mm (x), {spacing_zyx[1]:.3f}mm (y), {spacing_zyx[0]:.3f}mm (z)")
                
                # Segment all slices
                prog = st.progress(0)
                status = st.empty()
                all_masks = []
                for i in range(num_slices):
                    status.text(f"Segmenting slice {i+1}/{num_slices}")
                    all_masks.append(segment_slice(model, volume[i]))
                    prog.progress((i + 1) / num_slices)
                
                mask_3d = np.stack(all_masks)
                labeled_3d, nodules = analyze_3d_correct(mask_3d, spacing_zyx, volume.shape)
                
                status.empty()
                prog.empty()
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                st.markdown(f"### {len(nodules)} Nodule(s) Detected")
                
                if nodules:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Nodules", len(nodules))
                    col2.metric("Avg Diameter", f"{np.mean([n['diameter_mm'] for n in nodules]):.1f} mm")
                    col3.metric("Largest", f"{max(n['diameter_mm'] for n in nodules):.1f} mm")
                    col4.metric("Total Volume", f"{sum(n['volume_mm3'] for n in nodules):.0f} mm³")
                    
                    for n in nodules:
                        rec = "Routine follow-up" if n['diameter_mm'] < 5 else "Short-term follow-up" if n['diameter_mm'] < 8 else "Further evaluation"
                        rec_class = "routine" if n['diameter_mm'] < 5 else "followup" if n['diameter_mm'] < 8 else "urgent"
                        
                        st.markdown(f"""
                        <div class="nodule-result-card {rec_class}">
                            <div class="nodule-id-badge">N{n['id']}</div>
                            <div class="nodule-measures">
                                <div class="nodule-measure"><span class="val">{n['diameter_mm']:.1f} mm</span><span class="lbl">Diameter</span></div>
                                <div class="nodule-measure"><span class="val">{n['volume_mm3']:.0f} mm³</span><span class="lbl">Volume</span></div>
                                <div class="nodule-measure"><span class="val">Slices {n['slice_start']}-{n['slice_end']}</span><span class="lbl">Range</span></div>
                            </div>
                            <div class="nodule-rec {rec_class}">{rec}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Slice viewer
                    slice_idx = st.selectbox("View slice", list(range(num_slices)), format_func=lambda x: f"Slice {x}")
                    
                    # Find nodules that include this slice
                    nodules_in_slice = [n for n in nodules if n['slice_start'] <= slice_idx <= n['slice_end']]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.imshow(volume[slice_idx], cmap='gray')
                        ax.set_title(f"Original - Slice {slice_idx}")
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(5, 5))
                        
                        # Get the labeled mask for this slice
                        slice_labeled = labeled_3d[slice_idx]
                        
                        # Create nodule info list for this slice
                        slice_nodule_info = []
                        for n in nodules_in_slice:
                            slice_nodule_info.append({
                                'id': n['id'],
                                'label_id': n['id'],
                                'diameter_mm': n['diameter_mm']
                            })
                        
                        draw_slice(ax2, volume[slice_idx], slice_labeled, slice_nodule_info, title=f"Slice {slice_idx} - {len(nodules_in_slice)} Nodule(s)")
                        st.pyplot(fig2)
                        plt.close(fig2)
                    
                    # Download CSV
                    rows = [{
                        "Nodule": f"N{n['id']}",
                        "Volume (mm³)": round(n['volume_mm3'], 1),
                        "Diameter (mm)": round(n['diameter_mm'], 2),
                        "Height (mm)": round(n['height_mm'], 2),
                        "Width (mm)": round(n['width_mm'], 2),
                        "Depth (mm)": round(n['depth_mm'], 2),
                        "Slice Range": f"{n['slice_start']}-{n['slice_end']}",
                        "Number of Slices": n['num_slices']
                    } for n in nodules]
                    
                    csv = pd.DataFrame(rows).to_csv(index=False)
                    st.download_button("Download Results (CSV)", csv, f"lungvision_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", use_container_width=True)
                else:
                    st.info("No nodules detected in this volume.")

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
