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
from PIL import Image, ImageDraw
import warnings
from datetime import datetime
import zipfile
import os
import gdown
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
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
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
# MODEL ARCHITECTURE
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
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
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
        st.error(f"System Alert: {str(e)}")
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

def analyze_3d_connected(mask_3d, spacing_zyx, volume_shape):
    z_spacing, y_spacing, x_spacing = spacing_zyx
    voxel_volume_mm3 = x_spacing * y_spacing * z_spacing
    
    mask_3d_closed = binary_closing(mask_3d, structure=np.ones((3, 1, 1))).astype(np.uint8)
    labeled_mask = label(mask_3d_closed, connectivity=2)
    nodules = []
    
    for region in regionprops(labeled_mask):
        if region.area < 10:
            continue
        
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
        return None, None, None
    
    img = sitk.ReadImage(mhd)
    volume = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    
    return volume, spacing_zyx, tmp

def create_overlay_image_pil(slice_img, nodules_in_slice):
    """Create overlay image using PIL (no matplotlib)"""
    # Normalize to 0-255
    img_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    
    # Convert to RGB PIL image
    img_pil = Image.fromarray(img_uint8).convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    
    for nodule in nodules_in_slice:
        if 'mask_in_slice' not in nodule:
            continue
        
        nodule_mask = nodule['mask_in_slice']
        ys, xs = np.where(nodule_mask)
        if len(xs) == 0:
            continue
        
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        radius = int(max((np.max(xs) - np.min(xs)) / 2, (np.max(ys) - np.min(ys)) / 2)) + 8
        
        # Draw circle
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(6, 182, 212), width=2)
        
        # Draw text
        label_text = f"N{nodule['id']}\n{nodule['diameter_mm']:.1f}mm"
        draw.text((cx + radius + 5, cy - radius), label_text, fill=(241, 245, 249))
    
    return img_pil


# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown('<div class="login-mode">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1], vertical_alignment="center")
    
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
        st.info("Upload CT volume (MHD + RAW as ZIP) for automated nodule detection.")

    st.markdown('<div class="section-label">Clinical CT Upload</div>', unsafe_allow_html=True)
    
    upzip = st.file_uploader(
        "Upload CT Volume (ZIP with .mhd and .raw files)", 
        type=["zip"], 
        label_visibility="collapsed"
    )

    if upzip:
        with st.spinner("Loading CT volume from DICOM..."):
            volume, spacing_zyx, temp_dir = load_volume(upzip)
        
        if volume is None:
            st.error("Invalid CT volume. Ensure ZIP contains .mhd and .raw files.")
        else:
            num_slices = volume.shape[0]
            st.success(f"CT Loaded: {num_slices} slices")
            
            with st.expander("Scan Parameters (from DICOM metadata)", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Pixel Spacing (X)", f"{spacing_zyx[2]:.3f} mm")
                col2.metric("Pixel Spacing (Y)", f"{spacing_zyx[1]:.3f} mm")
                col3.metric("Slice Thickness (Z)", f"{spacing_zyx[0]:.3f} mm")
            
            prog = st.progress(0)
            status = st.empty()
            all_masks = []
            
            for i in range(num_slices):
                status.text(f"Analyzing slice {i+1}/{num_slices}...")
                all_masks.append(segment_slice(model, volume[i]))
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
                            <div class="metric-lbl">Volume (mm³)</div>
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
                
                nodules_in_slice = []
                slice_labeled = labeled_3d[slice_idx]
                unique_labels = np.unique(slice_labeled)
                for ul in unique_labels:
                    if ul == 0:
                        continue
                    n_data = next((n for n in nodules if n['label_id'] == ul), None)
                    if n_data:
                        nodule_mask_in_slice = (slice_labeled == ul)
                        n_data_copy = n_data.copy()
                        n_data_copy['mask_in_slice'] = nodule_mask_in_slice
                        nodules_in_slice.append(n_data_copy)

                col1, col2 = st.columns(2)
                
                with col1:
                    img_display = apply_lung_window(volume[slice_idx])
                    st.image(img_display, caption=f"Slice {slice_idx} - Original", use_container_width=True, clamp=True)

                with col2:
                    overlay_img = create_overlay_image_pil(volume[slice_idx], nodules_in_slice)
                    st.image(overlay_img, caption=f"Slice {slice_idx} - {len(nodules_in_slice)} Nodule(s) Detected", use_container_width=True)
                
                # Export clinical report
                df = pd.DataFrame([{
                    "Nodule ID": n['id'],
                    "Diameter (mm)": round(n['diameter_mm'], 2),
                    "Volume (mm³)": round(n['volume_mm3'], 2),
                    "Slice Range": f"{n['slice_start']}-{n['slice_end']}",
                    "Number of Slices": n['num_slices'],
                    "Clinical Recommendation": "Routine follow-up" if n['diameter_mm'] < 5 else "Short-term follow-up" if n['diameter_mm'] < 8 else "Further evaluation"
                } for n in nodules])
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Export Clinical Report (CSV)", 
                    csv, 
                    f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                    "text/csv", 
                    use_container_width=True
                )
            else:
                st.info("No nodules detected in this CT volume.")


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
        if model:
            show_app(model)
        else:
            st.stop()

if __name__ == "__main__":
    main()
