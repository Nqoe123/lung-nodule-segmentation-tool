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
import gdown
import plotly.graph_objects as go
import shutil

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG & GLOBAL CSS
# ============================================================
st.set_page_config(
    page_title="LungVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced Styling
st.markdown("""
<style>
    /* Global Reset */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    
    /* Hide default streamlit elements */
    #MainMenu, header, footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Glassmorphism Cards */
    .glass-panel {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }

    /* Typography */
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

    /* Login Page Specific Fixes */
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

    /* Inputs & Buttons */
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

    /* Nodule Cards */
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
    
    /* Badges */
    .badge { padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .badge-urgent { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-follow { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-routine { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE (UNet)
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
        if bilinear: self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else: self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
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
GDRIVE_ID = "1ZMXIzhxrvtEwXmbs1G2HrVRMl8-RkKc8"
MODEL_FN = "best_model.pth"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_FN):
            with st.spinner("Initializing AI Core..."):
                url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
                gdown.download(url, MODEL_FN, quiet=False)
        
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        state_dict = torch.load(MODEL_FN, map_location='cpu')
        
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        
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
        st.error(f"System Alert: {str(e)}")
        return None


# ============================================================
# LOGIC & UTILITIES
# ============================================================
PATCH_SIZE = 128

def apply_lung_window(image):
    image = np.clip(image, -1000, 400)
    return ((image + 1000) / 1400).astype(np.float32)

def segment_slice(model, img, threshold=0.5):
    shape = img.shape
    normed = img.astype(np.float32)
    if normed.max() > 1.0: normed = normed / 255.0
    normed = apply_lung_window(normed * 1400 - 1000) if normed.max() > 0.1 else normed
    
    resized = resize(normed, (PATCH_SIZE, PATCH_SIZE), preserve_range=True)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().numpy()
    
    mask = resize((prob > threshold).astype(np.float32), shape, order=0, preserve_range=True)
    return (mask > 0.5).astype(np.uint8)

def analyze_3d_connected(mask_3d, spacing_zyx, volume_shape):
    z_spacing, y_spacing, x_spacing = spacing_zyx
    voxel_volume_mm3 = x_spacing * y_spacing * z_spacing
    
    # Bridge gaps in Z-direction
    mask_3d_closed = binary_closing(mask_3d, structure=np.ones((3, 1, 1))).astype(np.uint8)
    
    labeled_mask = label(mask_3d_closed, connectivity=2)
    nodules = []
    
    for region in regionprops(labeled_mask):
        if region.area < 10: continue
        
        volume_mm3 = region.area * voxel_volume_mm3
        diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1/3)
        
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
            'centroid': region.centroid
        })
    
    nodules.sort(key=lambda x: x['slice_start'])
    return labeled_mask, nodules

def analyze_2d(mask_2d):
    labeled = label(mask_2d, connectivity=2)
    nodules = []
    for region in regionprops(labeled):
        if region.area < 10: continue
        area_px = region.area
        diam_px = 2 * np.sqrt(area_px / np.pi)
        nodules.append({'id': len(nodules) + 1, 'label_id': region.label, 'diameter_px': diam_px, 'area_px': area_px})
    return nodules

def load_volume(zip_file):
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "upload.zip")
    with open(zpath, "wb") as f: f.write(zip_file.getbuffer())
    with zipfile.ZipFile(zpath, 'r') as zf: zf.extractall(tmp)
    
    mhd = None
    for root, _, files in os.walk(tmp):
        for fn in files:
            if fn.lower().endswith('.mhd'): mhd = os.path.join(root, fn); break
        if mhd: break
    
    if not mhd: return None, None, None
    
    img = sitk.ReadImage(mhd)
    volume = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return volume, (spacing[2], spacing[1], spacing[0]), tmp

def create_rgba_mask(mask_2d):
    """Creates an RGBA overlay array for the mask."""
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Red color, 40% opacity for masked areas
    mask_bool = mask_2d > 0
    rgba[mask_bool, 0] = 255  # R
    rgba[mask_bool, 3] = 100  # A (Transparency)
    
    return rgba

def create_plotly_viz(img, mask, nodules, title):
    """
    Robust Visualization using Heatmap + Image Overlay.
    This avoids the go.Image crash with raw data.
    """
    fig = go.Figure()
    
    # 1. Background CT Slice (Heatmap is more stable than go.Image for raw data)
    fig.add_trace(go.Heatmap(
        z=img,
        colorscale='Greys',
        showscale=False,
        zmin=img.min(), zmax=img.max()
    ))
    
    # 2. Segmentation Mask (RGBA Image Overlay)
    rgba_mask = create_rgba_mask(mask)
    fig.add_trace(go.Image(
        z=rgba_mask,
        hoverinfo='skip',
        name='Segmentation'
    ))

    # 3. Annotations
    for n in nodules:
        centroid = n.get('centroid')
        if centroid is None: continue
        
        # Handle 2D (y,x) or 3D (z,y,x) centroids
        if len(centroid) == 2:
            cy, cx = centroid[0], centroid[1]
        else:
            cy, cx = centroid[1], centroid[2]
        
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=cx - 10, y0=cy - 10, x1=cx + 10, y1=cy + 10,
            line_color="#06b6d4", line_width=2
        )
        
        fig.add_annotation(
            x=cx + 15, y=cy,
            text=f"N{n['id']}",
            showarrow=False,
            font=dict(color="#06b6d4", size=12, family="Arial Black"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="#06b6d4",
            borderwidth=1,
            borderpad=4
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color='#f1f5f9', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
        xaxis=dict(showgrid=False, visible=False, scaleanchor="y"), # Keep aspect ratio
        yaxis=dict(showgrid=False, visible=False),
        hovermode=False
    )
    return fig


# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    # Inject class to hide sidebar/margins via CSS
    st.markdown('<div class="login-mode">', unsafe_allow_html=True)
    
    # Use Native Vertical Centering
    col1, col2, col3 = st.columns([1, 2, 1], vertical_alignment="center")
    
    with col2:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🫁</div>
            <h1 style="margin-bottom: 0.5rem; font-size: 1.8rem;">LungVision AI</h1>
            <p style="color: #94a3b8; margin-bottom: 2rem;">Secure Clinical Access Portal</p>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Radiologist ID", placeholder="ID", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="••••••••", label_visibility="collapsed")
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
    # Header
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
        st.info("Upload a CT scan to begin analysis.")

    # Main Mode Selection
    st.markdown('<div class="section-label">Acquisition Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Single Slice Analysis", "Full Volume Scan (3D)"], horizontal=True, label_visibility="collapsed")

    # ================= SINGLE SLICE =================
    if "Single" in mode:
        st.markdown('<div class="section-label">Data Input</div>', unsafe_allow_html=True)
        upfile = st.file_uploader("Upload CT Slice (PNG/JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if upfile:
            img = np.array(Image.open(upfile).convert('L'), dtype=np.float32)
            
            with st.spinner("Processing neural network inference..."):
                mask = segment_slice(model, img)
                nodules = analyze_2d(mask)
                labeled_mask = label(mask)

            col1, col2 = st.columns(2)
            
            with col1:
                fig_orig = create_plotly_viz(img, np.zeros_like(mask), [], "Source Image")
                st.plotly_chart(fig_orig, use_container_width=True)

            with col2:
                fig_anal = create_plotly_viz(img, labeled_mask > 0, nodules, f"Analysis: {len(nodules)} Detected")
                st.plotly_chart(fig_anal, use_container_width=True)

            if nodules:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown(f"### Detected Findings ({len(nodules)})")
                for n in nodules:
                    st.markdown(f"""
                    <div class="nodule-row">
                        <div style="font-weight:700; color:#38bdf8; font-size:1.1rem;">N{n['id']}</div>
                        <div class="metric-box">
                            <div class="metric-val">{n['diameter_px']:.0f}</div>
                            <div class="metric-lbl">Diameter (px)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-val">{n['area_px']:.0f}</div>
                            <div class="metric-lbl">Area (px²)</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No significant findings detected.")

    # ================= VOLUME MODE =================
    else:
        st.markdown('<div class="section-label">Data Input</div>', unsafe_allow_html=True)
        upzip = st.file_uploader("Upload Volume (ZIP containing .mhd/.raw)", type=["zip"], label_visibility="collapsed")

        if upzip:
            with st.spinner("Loading DICOM Series..."):
                volume, spacing_zyx, temp_dir = load_volume(upzip)
            
            if volume is None:
                st.error("Error reading ZIP archive. Please check file structure.")
            else:
                num_slices = volume.shape[0]
                st.success(f"Volume Loaded: {num_slices} slices | Spacing Z={spacing_zyx[0]:.2f}mm")
                
                # Progress Bar
                prog = st.progress(0)
                status = st.empty()
                all_masks = []
                
                for i in range(num_slices):
                    status.text(f"Scanning slice {i+1}/{num_slices}...")
                    all_masks.append(segment_slice(model, volume[i]))
                    prog.progress((i + 1) / num_slices)
                
                mask_3d = np.stack(all_masks)
                labeled_3d, nodules = analyze_3d_connected(mask_3d, spacing_zyx, volume.shape)
                
                status.empty(); prog.empty()
                shutil.rmtree(temp_dir, ignore_errors=True)

                if nodules:
                    # Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f'<div class="glass-panel" style="text-align:center"><div class="metric-val">{len(nodules)}</div><div class="metric-lbl">Total Nodules</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="glass-panel" style="text-align:center"><div class="metric-val">{np.mean([n["diameter_mm"] for n in nodules]):.1f}mm</div><div class="metric-lbl">Avg Size</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="glass-panel" style="text-align:center"><div class="metric-val">{max(n["diameter_mm"] for n in nodules):.1f}mm</div><div class="metric-lbl">Max Size</div></div>', unsafe_allow_html=True)
                    c4.markdown(f'<div class="glass-panel" style="text-align:center"><div class="metric-val">{sum(n["num_slices"] for n in nodules)}</div><div class="metric-lbl">Affected Slices</div></div>', unsafe_allow_html=True)

                    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                    st.markdown('### Nodule Report')
                    
                    for n in nodules:
                        risk = "routine" if n['diameter_mm'] < 5 else "followup" if n['diameter_mm'] < 8 else "urgent"
                        risk_label = "Routine" if risk == "routine" else "Short-term Follow-up" if risk == "followup" else "Specialist Consult"
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
                                <div class="metric-lbl">Vol (mm³)</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-val">{n['slice_start']}-{n['slice_end']}</div>
                                <div class="metric-lbl">Range</div>
                            </div>
                            <span class="badge {badge_class}">{risk_label}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Viewer
                    st.markdown('<div class="section-label">Slice Navigator</div>', unsafe_allow_html=True)
                    slice_idx = st.slider("Select Slice Index", 0, num_slices-1, num_slices//2)
                    
                    nodules_in_slice = []
                    slice_mask = labeled_3d[slice_idx]
                    unique_labels = np.unique(slice_mask)
                    for ul in unique_labels:
                        if ul == 0: continue
                        n_data = next((n for n in nodules if n['label_id'] == ul), None)
                        if n_data:
                            nodules_in_slice.append(n_data)

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_raw = create_plotly_viz(volume[slice_idx], np.zeros_like(slice_mask), [], f"Raw Slice {slice_idx}")
                        st.plotly_chart(fig_raw, use_container_width=True)

                    with col2:
                        fig_anal = create_plotly_viz(volume[slice_idx], slice_mask > 0, nodules_in_slice, f"Segmented: {len(nodules_in_slice)} Nodules")
                        st.plotly_chart(fig_anal, use_container_width=True)
                    
                    # CSV
                    df = pd.DataFrame([{
                        "ID": n['id'], "Vol_mm3": round(n['volume_mm3'],2), "Diam_mm": round(n['diameter_mm'],2), 
                        "Z_Range": f"{n['slice_start']}-{n['slice_end']}", "Slices": n['num_slices']
                    } for n in nodules])
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Export Clinical Report", csv, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)

                else:
                    st.info("Analysis complete. No nodules detected in the volume.")

# ============================================================
# ENTRY POINT
# ============================================================
def main():
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_login()
    else:
        model = load_model()
        if model: show_app(model)
        else: st.stop()

if __name__ == "__main__":
    main()
