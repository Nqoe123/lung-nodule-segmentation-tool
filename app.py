import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing
import tempfile
import SimpleITK as sitk
from collections import OrderedDict
from PIL import Image
import warnings
from datetime import datetime
import zipfile
import os
import matplotlib.pyplot as plt
import shutil

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG & GLOBAL CSS
# ============================================================
st.set_page_config(
    page_title="LungVision AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    #MainMenu, header, footer { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .glass-panel {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3 { color: #f8fafc; font-weight: 700; }
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::before {
        content: '';
        display: block;
        width: 20px;
        height: 2px;
        background: #38bdf8;
        border-radius: 2px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(14, 165, 233, 0.6); }
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.5);
        border: 2px dashed rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    .nodule-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .nodule-row:last-child { border-bottom: none; }
    .metric-box {
        background: rgba(0,0,0,0.2);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
        flex: 1;
    }
    .metric-val { font-size: 1.1rem; font-weight: 700; color: #f1f5f9; }
    .metric-lbl { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; }
    .badge { padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .badge-urgent { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-follow { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-routine { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
    .login-mode .main {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 100vh;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .login-mode [data-testid="stSidebar"] { display: none; }
    .login-mode header { display: none; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE (SAME AS TRAINING)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
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
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class MemoryEfficientUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
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
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)


# ============================================================
# MODEL LOADING
# ============================================================
MODEL_FN = "best_model.pth"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_FN):
            st.error(f"Model file '{MODEL_FN}' not found.")
            return None
        
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        state_dict = torch.load(MODEL_FN, map_location='cpu')
        
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        if state_dict and len(state_dict) > 0 and 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        st.success("Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None


# ============================================================
# CORRECT INFERENCE - PROCESS ENTIRE SLICE IN PATCHES
# ============================================================
PATCH_SIZE = 128
STRIDE = 64  # 50% overlap to avoid missing nodules at patch boundaries

def apply_lung_window(image):
    image = np.clip(image, -1000, 400)
    return ((image + 1000) / 1400).astype(np.float32)

def segment_slice_patch_based(model, img, threshold=0.7):
    """
    Process the entire slice by sliding a 128x128 window.
    This matches the training patch extraction method.
    """
    h, w = img.shape
    normalized = img.astype(np.float32)
    
    # Apply lung window
    normalized = apply_lung_window(normalized)
    
    # Create output mask
    mask = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    # Slide window across the image
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            # Extract patch
            patch = normalized[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Ensure patch is exactly 128x128
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                continue
            
            # Prepare for model
            patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                prob = torch.sigmoid(model(patch_tensor)).squeeze().numpy()
            
            # Add to mask (average overlapping regions)
            mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob
            count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
    
    # Handle edges - process remaining patches
    # Right edge
    if w % STRIDE != 0:
        x_start = max(0, w - PATCH_SIZE)
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            patch = normalized[y:y+PATCH_SIZE, x_start:x_start+PATCH_SIZE]
            if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    prob = torch.sigmoid(model(patch_tensor)).squeeze().numpy()
                mask[y:y+PATCH_SIZE, x_start:x_start+PATCH_SIZE] += prob
                count[y:y+PATCH_SIZE, x_start:x_start+PATCH_SIZE] += 1
    
    # Bottom edge
    if h % STRIDE != 0:
        y_start = max(0, h - PATCH_SIZE)
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = normalized[y_start:y_start+PATCH_SIZE, x:x+PATCH_SIZE]
            if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    prob = torch.sigmoid(model(patch_tensor)).squeeze().numpy()
                mask[y_start:y_start+PATCH_SIZE, x:x+PATCH_SIZE] += prob
                count[y_start:y_start+PATCH_SIZE, x:x+PATCH_SIZE] += 1
    
    # Bottom-right corner
    if w % STRIDE != 0 and h % STRIDE != 0:
        x_start = max(0, w - PATCH_SIZE)
        y_start = max(0, h - PATCH_SIZE)
        patch = normalized[y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE]
        if patch.shape == (PATCH_SIZE, PATCH_SIZE):
            patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(model(patch_tensor)).squeeze().numpy()
            mask[y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE] += prob
            count[y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE] += 1
    
    # Average overlapping regions
    count = np.maximum(count, 1)  # Avoid division by zero
    mask = mask / count
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Morphological closing to fill small holes
    binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3))).astype(np.uint8)
    
    return binary_mask, mask

def analyze_3d_connected(mask_3d, spacing_zyx, volume_shape):
    z_spacing, y_spacing, x_spacing = spacing_zyx
    voxel_volume_mm3 = x_spacing * y_spacing * z_spacing
    
    labeled_mask = label(mask_3d, connectivity=2)
    nodules = []
    
    for region in regionprops(labeled_mask):
        if region.area < 30:
            continue
        
        volume_mm3 = region.area * voxel_volume_mm3
        diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1/3)
        
        if diameter_mm < 2.0 or diameter_mm > 30.0:
            continue
        
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        min_z = max(0, min_z)
        max_z = min(volume_shape[0], max_z)
        
        nodules.append({
            'id': len(nodules) + 1,
            'label_id': region.label,
            'volume_mm3': volume_mm3,
            'diameter_mm': diameter_mm,
            'slice_start': int(min_z),
            'slice_end': int(max_z - 1),
            'num_slices': int(max_z - min_z),
            'centroid': region.centroid,
            'area': region.area
        })
    
    nodules.sort(key=lambda x: x['slice_start'])
    return labeled_mask, nodules

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

def create_overlay_image(slice_img, mask, alpha=0.6):
    # Normalize slice for display
    slice_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
    slice_uint8 = (slice_norm * 255).astype(np.uint8)
    
    # Create RGB
    rgb = np.stack([slice_uint8, slice_uint8, slice_uint8], axis=-1).astype(np.float32)
    
    # Apply red overlay on mask
    mask_bool = mask > 0
    rgb[mask_bool, 0] = np.clip(rgb[mask_bool, 0] * 0.3 + 200, 0, 255)
    rgb[mask_bool, 1] = np.clip(rgb[mask_bool, 1] * 0.2, 0, 255)
    rgb[mask_bool, 2] = np.clip(rgb[mask_bool, 2] * 0.2, 0, 255)
    
    return rgb.astype(np.uint8)


# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown('<div class="login-mode">', unsafe_allow_html=True)
    st.markdown('<div style="height: 15vh;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;"></div>
            <h1 style="margin-bottom: 0.5rem; font-size: 1.8rem;">LungVision AI</h1>
            <p style="color: #94a3b8; margin-bottom: 2rem;">Secure Clinical Access Portal</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Radiologist ID", placeholder="ID", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="......", label_visibility="collapsed")
            submit = st.form_submit_button("Authenticate", use_container_width=True)
            
            if submit:
                if username == "radiologist" and password == "hit500":
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Credentials")

        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# MAIN APP
# ============================================================
def show_app(model):
    st.markdown(f"""
    <div class="glass-panel" style="margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0; font-size: 1.5rem;">LungVision <span style="color:#38bdf8">AI</span></h1>
            <div style="color:#94a3b8; font-size: 0.85rem;">User: {st.session_state.get('username', 'Guest')} | System Online</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="glass-panel"><h3>Control Panel</h3></div>', unsafe_allow_html=True)
        if st.button("Terminate Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.info("Upload CT volume (MHD + RAW as ZIP)")

    st.markdown('<div class="section-label">Clinical CT Upload</div>', unsafe_allow_html=True)
    
    upzip = st.file_uploader(
        "Upload CT Volume (ZIP with .mhd and .raw files)", 
        type=["zip"], 
        label_visibility="collapsed"
    )

    if upzip:
        with st.spinner("Loading CT volume..."):
            volume, spacing_zyx, temp_dir = load_volume(upzip)
        
        if volume is None:
            st.error("Invalid CT volume. Ensure ZIP contains .mhd and .raw files.")
        else:
            num_slices = volume.shape[0]
            st.success(f"CT Loaded: {num_slices} slices")
            
            with st.expander("Scan Parameters", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Pixel Spacing (X)", f"{spacing_zyx[2]:.3f} mm")
                col2.metric("Pixel Spacing (Y)", f"{spacing_zyx[1]:.3f} mm")
                col3.metric("Slice Thickness (Z)", f"{spacing_zyx[0]:.3f} mm")
            
            prog = st.progress(0)
            status = st.empty()
            all_masks = []
            
            for i in range(num_slices):
                status.text(f"Analyzing slice {i+1}/{num_slices}...")
                binary_mask, _ = segment_slice_patch_based(model, volume[i])
                all_masks.append(binary_mask)
                prog.progress((i + 1) / num_slices)
            
            mask_3d = np.stack(all_masks)
            labeled_3d, nodules = analyze_3d_connected(mask_3d, spacing_zyx, volume.shape)
            
            status.empty()
            prog.empty()
            shutil.rmtree(temp_dir, ignore_errors=True)

            if nodules:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Nodules", len(nodules))
                c2.metric("Avg Diameter", f"{np.mean([n['diameter_mm'] for n in nodules]):.1f} mm")
                c3.metric("Largest Nodule", f"{max(n['diameter_mm'] for n in nodules):.1f} mm")
                c4.metric("Total Volume", f"{sum(n['volume_mm3'] for n in nodules):.0f} mm³")

                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown("### Clinical Findings")
                
                for n in nodules:
                    if n['diameter_mm'] < 5:
                        risk = "routine"
                        risk_label = "Routine follow-up"
                        rec = "Annual CT recommended"
                    elif n['diameter_mm'] < 8:
                        risk = "followup"
                        risk_label = "Short-term follow-up"
                        rec = "Repeat CT in 6-12 months"
                    else:
                        risk = "urgent"
                        risk_label = "Further evaluation"
                        rec = "Pulmonology consultation recommended"
                    
                    badge_class = f"badge-{risk}"
                    
                    st.markdown(f"""
                    <div class="nodule-row">
                        <div style="min-width:60px; font-weight:bold; color:#f1f5f9;">Nodule {n['id']}</div>
                        <div class="metric-box">
                            <div class="metric-val">{n['diameter_mm']:.1f} mm</div>
                            <div class="metric-lbl">Diameter</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-val">{n['volume_mm3']:.0f}</div>
                            <div class="metric-lbl">Volume</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-val">Slices {n['slice_start']}-{n['slice_end']}</div>
                            <div class="metric-lbl">Location</div>
                        </div>
                        <span class="badge {badge_class}">{risk_label}</span>
                    </div>
                    <div style="margin-left: 60px; margin-bottom: 10px; font-size: 0.7rem; color: #94a3b8;">
                        Recommendation: {rec}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-label">Slice Review</div>', unsafe_allow_html=True)
                slice_idx = st.slider("Select slice to review", 0, num_slices - 1, num_slices // 2)
                
                # Get the mask for this slice
                slice_mask = mask_3d[slice_idx]
                
                # Create overlay
                overlay_img = create_overlay_image(volume[slice_idx], slice_mask)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0b1120')
                    img_display = apply_lung_window(volume[slice_idx])
                    ax.imshow(img_display, cmap='gray')
                    ax.set_title(f"Slice {slice_idx} - Original CT", color='#f1f5f9', fontsize=12)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='#0b1120')
                    ax2.imshow(overlay_img)
                    ax2.set_title(f"Slice {slice_idx} - AI Detection", color='#f1f5f9', fontsize=12)
                    ax2.axis('off')
                    st.pyplot(fig2)
                    plt.close(fig2)
                
                df = pd.DataFrame([{
                    "Nodule ID": n['id'],
                    "Diameter (mm)": round(n['diameter_mm'], 2),
                    "Volume (mm³)": round(n['volume_mm3'], 2),
                    "Slice Range": f"{n['slice_start']}-{n['slice_end']}",
                    "Number of Slices": n['num_slices'],
                    "Recommendation": "Routine follow-up" if n['diameter_mm'] < 5 else "Short-term follow-up" if n['diameter_mm'] < 8 else "Further evaluation"
                } for n in nodules])
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Export Report (CSV)", csv, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
            else:
                st.info("No nodules detected.")


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_login()
    else:
        model = load_model()
        if model is not None:
            show_app(model)
        else:
            st.stop()

if __name__ == "__main__":
    main()
