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
# UTILITIES
# ============================================================
PATCH_SIZE = 128

def apply_lung_window(image):
    image = np.clip(image, -1000, 400)
    return ((image + 1000) / 1400).astype(np.float32)

def segment_slice(model, img, threshold=0.5):
    shape = img.shape
    normed = img.astype(np.float32)
    if normed.max() > 1.0:
        normed = normed / 255.0
    
    normed = apply_lung_window(normed * 1400 - 1000) if normed.max() > 0.1 else normed
    
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
    
    # Label connected components in 3D
    labeled_mask = label(mask_3d, connectivity=2)
    nodules = []
    
    for region in regionprops(labeled_mask):
        if region.area < 10:
            continue
        
        volume_mm3 = region.area * voxel_volume_mm3
        diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1/3)
        
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        
        min_z = max(0, min_z)
        max_z = min(volume_shape[0], max_z)
        slice_start = int(min_z)
        slice_end = int(max_z - 1)
        
        if slice_end < slice_start:
            slice_end = slice_start
        
        nodules.append({
            'id': len(nodules) + 1,
            'label_id': region.label,
            'volume_mm3': volume_mm3,
            'diameter_mm': diameter_mm,
            'slice_start': slice_start,
            'slice_end': slice_end,
            'num_slices': slice_end - slice_start + 1,
            'centroid': region.centroid,
            'bbox': region.bbox
        })
    
    nodules.sort(key=lambda x: x['slice_start'])
    return labeled_mask, nodules

def analyze_2d_correct(mask_2d):
    labeled = label(mask_2d, connectivity=2)
    nodules = []
    
    for region in regionprops(labeled):
        if region.area < 10:
            continue
        
        area_px = region.area
        diam_px = 2 * np.sqrt(area_px / np.pi)
        
        nodules.append({
            'id': len(nodules) + 1,
            'label_id': region.label,
            'area_px': area_px,
            'diameter_px': diam_px,
            'centroid': region.centroid
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
    spacing = img.GetSpacing()
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    
    return volume, spacing_zyx, tmp

def make_overlay(slice_img, mask_2d, alpha=0.45):
    norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
    rgb = np.stack([norm, norm, norm], axis=-1)
    mask_bool = mask_2d > 0.5
    rgb[mask_bool, 0] = np.clip(rgb[mask_bool, 0] + alpha, 0, 1)
    rgb[mask_bool, 1] = np.clip(rgb[mask_bool, 1] * 0.25, 0, 1)
    rgb[mask_bool, 2] = np.clip(rgb[mask_bool, 2] * 0.25, 0, 1)
    return rgb

def draw_slice_with_nodules(ax, slice_img, labeled_mask, nodules_for_this_slice):
    """Draw slice with nodule outlines - FIXED VERSION"""
    # Create overlay with all masks
    overlay = make_overlay(slice_img, (labeled_mask > 0).astype(np.float32))
    ax.imshow(overlay)
    
    # Draw each nodule that appears in this slice
    for nodule in nodules_for_this_slice:
        # Get the mask for this specific nodule in this slice
        nodule_mask = (labeled_mask == nodule['label_id'])
        
        if not np.any(nodule_mask):
            continue
        
        # Get contour coordinates
        ys, xs = np.where(nodule_mask)
        if len(xs) == 0:
            continue
        
        # Calculate center
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # Calculate radius (use bounding box size)
        radius = max((np.max(xs) - np.min(xs)) / 2, (np.max(ys) - np.min(ys)) / 2) + 5
        
        # Draw circle
        circle = Circle((cx, cy), radius, fill=False, edgecolor='#06b6d4', linewidth=2)
        ax.add_patch(circle)
        
        # Add label
        if 'diameter_mm' in nodule:
            label_text = f"N{nodule['id']}\n{nodule['diameter_mm']:.1f}mm"
        elif 'diameter_px' in nodule:
            label_text = f"N{nodule['id']}\n{nodule['diameter_px']:.0f}px"
        else:
            label_text = f"N{nodule['id']}"
        
        ax.annotate(
            label_text,
            xy=(cx, cy), xytext=(cx + radius + 8, cy - radius),
            fontsize=9, fontweight='600', color='#f1f5f9',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#273755', alpha=0.9),
        )
    
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
        st.caption("2. Upload PNG/JPG for single slice or ZIP with .mhd/.raw for volume")
        st.caption("3. View detected nodules")

    st.markdown('<p class="section-label">Scan Type</p>', unsafe_allow_html=True)
    mode = st.radio("", ["Single CT Slice", "CT Volume (MHD + RAW as ZIP)"], horizontal=True, label_visibility="collapsed")

    # ========== SINGLE SLICE MODE ==========
    if "Single" in mode:
        st.markdown('<p class="section-label">Upload CT Slice</p>', unsafe_allow_html=True)
        upfile = st.file_uploader("Select PNG or JPG", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if upfile:
            img = np.array(Image.open(upfile).convert('L'), dtype=np.float32)
            
            with st.spinner("Analyzing..."):
                mask = segment_slice(model, img)
                nodules = analyze_2d_correct(mask)
                labeled_mask = label(mask)
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, cmap='gray')
                ax.set_title("Original CT Slice", color='#f1f5f9', fontsize=12)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                
                # Pass nodules with their label_ids
                nodules_for_display = []
                for n in nodules:
                    nodules_for_display.append({
                        'id': n['id'],
                        'label_id': n['label_id'],
                        'diameter_px': n['diameter_px']
                    })
                
                draw_slice_with_nodules(ax2, img, labeled_mask, nodules_for_display)
                ax2.set_title(f"{len(nodules)} Nodule(s) Detected", color='#f1f5f9', fontsize=12)
                st.pyplot(fig2)
                plt.close(fig2)
            
            if nodules:
                st.markdown(f"### {len(nodules)} Nodule(s) Detected")
                for n in nodules:
                    st.markdown(f"""
                    <div class="nodule-result-card routine">
                        <div class="nodule-id-badge">N{n['id']}</div>
                        <div class="nodule-measures">
                            <div class="nodule-measure"><span class="val">{n['diameter_px']:.0f} px</span><span class="lbl">Diameter</span></div>
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

        if upzip:
            with st.spinner("Loading volume..."):
                volume, spacing_zyx, temp_dir = load_volume(upzip)
            
            if volume is None:
                st.error("Could not read volume. Ensure ZIP contains .mhd and .raw files.")
            else:
                num_slices, height, width = volume.shape
                st.info(f"Volume: {num_slices} slices | {height}x{width} pixels | Spacing: X={spacing_zyx[2]:.3f}mm, Y={spacing_zyx[1]:.3f}mm, Z={spacing_zyx[0]:.3f}mm")
                
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
                    
                    # Display nodule cards
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
                    
                    # Find which nodules appear in this slice
                    nodules_in_this_slice = []
                    for n in nodules:
                        if n['slice_start'] <= slice_idx <= n['slice_end']:
                            nodules_in_this_slice.append({
                                'id': n['id'],
                                'label_id': n['label_id'],
                                'diameter_mm': n['diameter_mm']
                            })
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(volume[slice_idx], cmap='gray')
                        ax.set_title(f"Original - Slice {slice_idx}", color='#f1f5f9', fontsize=12)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(6, 6))
                        slice_labeled = labeled_3d[slice_idx]
                        draw_slice_with_nodules(ax2, volume[slice_idx], slice_labeled, nodules_in_this_slice)
                        ax2.set_title(f"Slice {slice_idx} - {len(nodules_in_this_slice)} Nodule(s)", color='#f1f5f9', fontsize=12)
                        st.pyplot(fig2)
                        plt.close(fig2)
                    
                    # Download CSV
                    rows = [{
                        "Nodule": f"N{n['id']}",
                        "Volume (mm³)": round(n['volume_mm3'], 1),
                        "Diameter (mm)": round(n['diameter_mm'], 2),
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
