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

warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="LungVision AI | Nodule Detection",
    page_icon="🫁",
    layout="wide"
)

# ========== ALL TEXT WHITE ==========
st.markdown("""
<style>
    /* Force ALL text to be white */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
    .stMetric label, .stMetric div, .stMetric span,
    .stDataFrame, .stDataFrame div, .stDataFrame span,
    .stSelectbox label, .stSelectbox div,
    p, li, div, span, label, h1, h2, h3, h4, h5, h6,
    .stAlert, .stAlert p, .stAlert div,
    .stButton label,
    .stTextInput label,
    .stFileUploader label,
    .stRadio label,
    .stSuccess, .stInfo, .stWarning, .stError {
        color: #FFFFFF !important;
    }
    
    /* Header styling */
    .clinical-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Metric cards with dark background */
    div[data-testid="stMetric"] {
        background-color: #1a202c;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a5568;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2c5282;
        color: white !important;
        font-weight: 500;
        border: 1px solid #4a5568;
    }
    .stButton > button:hover {
        background-color: #1e3a5f;
        color: white !important;
    }
    
    /* Dataframe - dark background */
    .stDataFrame {
        background-color: #1a202c;
    }
    .dataframe {
        background-color: #1a202c !important;
        color: white !important;
    }
    .dataframe th {
        background-color: #2d3748 !important;
        color: white !important;
    }
    .dataframe td {
        background-color: #1a202c !important;
        color: white !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: #2d3748 !important;
        border-left: 4px solid #4299e1 !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #22543d !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #1a365d !important;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #744210 !important;
    }
    
    /* Error boxes */
    .stError {
        background-color: #742a2a !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #1a202c;
        padding: 10px;
        border-radius: 8px;
    }
    
    /* Selectbox */
    .stSelectbox > div {
        background-color: #1a202c;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: #1a202c;
        padding: 10px;
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div {
        background-color: #2d3748;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1633t36 {
        background-color: #0f1419;
    }
    
    /* Main background */
    .main {
        background-color: #0a0e12;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== MODEL ARCHITECTURE ==========
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
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
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

# ========== LOAD MODEL ==========
GOOGLE_DRIVE_FILE_ID = "1PzCv2fJSr7e0QIfPGtLKLOL-9RSLdR2i"
MODEL_FILENAME = "best_model.pth"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("Loading AI model..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_FILENAME, quiet=True)
        
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint.state_dict()
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model error: {str(e)}")
        return None

# ========== SEGMENTATION ==========
def segment_nodule(model, image_array):
    try:
        original_shape = image_array.shape
        
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        resized = resize(image_array, (512, 512), preserve_range=True)
        input_tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            mask = (probs > 0.5).float().squeeze().numpy()
        
        mask_resized = resize(mask, original_shape, order=0, preserve_range=True)
        
        labeled = label(mask_resized > 0.5)
        nodules = []
        
        for region in regionprops(labeled):
            if region.area >= 20:
                nodules.append({
                    'id': len(nodules) + 1,
                    'area_pixels': region.area,
                    'diameter_pixels': 2 * np.sqrt(region.area / np.pi),
                    'centroid_x': region.centroid[1],
                    'centroid_y': region.centroid[0],
                    'mask': (labeled == region.label).astype(np.float32)
                })
        
        return nodules, mask_resized
    except Exception as e:
        return [], None

def load_volume(zip_file):
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "upload.zip")
    
    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    mhd_file = None
    raw_file = None
    
    for file in os.listdir(temp_dir):
        if file.endswith('.mhd'):
            mhd_file = os.path.join(temp_dir, file)
        elif file.endswith('.raw'):
            raw_file = os.path.join(temp_dir, file)
    
    if mhd_file and raw_file:
        img = sitk.ReadImage(mhd_file)
        volume = sitk.GetArrayFromImage(img)
        spacing = img.GetSpacing()
        return volume, spacing, temp_dir
    
    return None, None, None

def create_overlay(image_array, nodules):
    img_norm = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    overlay = np.stack([img_norm] * 3, axis=-1)
    
    for nodule in nodules:
        mask_bool = nodule['mask'] > 0.5
        overlay[mask_bool, 0] = 1.0
        overlay[mask_bool, 1] = 0.0
        overlay[mask_bool, 2] = 0.0
    
    return overlay

# ========== MAIN APP ==========
def main():
    st.markdown("""
    <div class="clinical-header">
        <h1>🫁 LungVision AI</h1>
        <p>Automatic Lung Nodule Detection</p>
        <small>Upload → Click Detect → Get Results</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Login
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 Login")
            username = st.text_input("Radiologist ID")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username == "radiologist" and password == "hit500":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        return
    
    with st.sidebar:
        st.success("👤 Radiologist - Active")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
        st.markdown("---")
        st.markdown("### Model Information")
        st.markdown("- **Architecture:** U-Net")
        st.markdown("- **Dice Score:** 0.74")
        st.markdown("- **Training Data:** LUNA16")
        st.markdown("- **Input Size:** 512×512")
    
    model = load_model()
    if model is None:
        st.stop()
    
    st.markdown("### 📤 Upload Study")
    
    upload_type = st.radio("", ["Single CT Slice", "Full CT Volume"], horizontal=True)
    
    if upload_type == "Single CT Slice":
        uploaded_file = st.file_uploader("Choose CT image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('L')
            image_array = np.array(image, dtype=np.float32)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original CT Slice")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(image_array, cmap='gray')
                ax.set_title("CT Slice", color='white')
                ax.axis('off')
                st.pyplot(fig)
            
            if st.button("🔍 Detect Nodules", type="primary"):
                with st.spinner("Analyzing..."):
                    nodules, _ = segment_nodule(model, image_array)
                
                if nodules:
                    with col2:
                        st.markdown("#### Detection Results")
                        overlay = create_overlay(image_array, nodules)
                        fig2, ax2 = plt.subplots(figsize=(5, 5))
                        ax2.imshow(overlay)
                        ax2.set_title(f"{len(nodules)} Nodule(s) Detected", color='white')
                        ax2.axis('off')
                        st.pyplot(fig2)
                    
                    st.markdown(f"## ✅ {len(nodules)} Nodule(s) Detected")
                    
                    for n in nodules:
                        with st.container():
                            st.markdown(f"### Nodule {n['id']}")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**Area:** {n['area_pixels']:.0f} pixels²")
                                st.markdown(f"**Diameter:** {n['diameter_pixels']:.1f} pixels")
                            with col_b:
                                st.markdown(f"**Location (X,Y):** ({n['centroid_x']:.0f}, {n['centroid_y']:.0f})")
                            
                            if n['area_pixels'] < 100:
                                st.info("📌 **Recommendation:** Routine follow-up in 12 months")
                            elif n['area_pixels'] < 300:
                                st.warning("⚠️ **Recommendation:** Short-term follow-up in 6 months")
                            else:
                                st.error("🚨 **Recommendation:** Urgent consultation recommended")
                            st.markdown("---")
                else:
                    st.info("✓ No nodules detected in this slice")
    
    else:  # Full CT Volume
        uploaded_zip = st.file_uploader("Choose CT volume (ZIP with .mhd/.raw)", type=["zip"])
        
        if uploaded_zip:
            if st.button("🔍 Analyze Full Volume", type="primary"):
                with st.spinner("Loading volume..."):
                    volume, spacing, temp_dir = load_volume(uploaded_zip)
                
                if volume is not None:
                    st.success(f"✅ Volume loaded: {volume.shape[0]} slices, {volume.shape[1]}×{volume.shape[2]} pixels")
                    if spacing:
                        st.info(f"📏 Pixel spacing: {spacing[0]:.2f} mm")
                    
                    all_nodules = []
                    slices_with_nodules = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(volume.shape[0]):
                        status_text.text(f"Analyzing slice {i+1}/{volume.shape[0]}...")
                        nodules, _ = segment_nodule(model, volume[i])
                        
                        if nodules:
                            slices_with_nodules[i] = {
                                'image': volume[i].copy(),
                                'nodules': nodules
                            }
                            for n in nodules:
                                if spacing:
                                    diam_mm = n['diameter_pixels'] * spacing[0]
                                    area_mm2 = n['area_pixels'] * (spacing[0] ** 2)
                                else:
                                    diam_mm = n['diameter_pixels']
                                    area_mm2 = n['area_pixels']
                                all_nodules.append({
                                    'Slice': i,
                                    'Area (px²)': n['area_pixels'],
                                    'Area (mm²)': f"{area_mm2:.1f}",
                                    'Diameter (mm)': f"{diam_mm:.1f}"
                                })
                        progress_bar.progress((i + 1) / volume.shape[0])
                    
                    status_text.text("Analysis complete!")
                    
                    if all_nodules:
                        st.markdown(f"## ✅ {len(all_nodules)} Nodule(s) Detected Across {len(slices_with_nodules)} Slices")
                        
                        # Summary Statistics
                        st.markdown("### 📊 Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Nodules", len(all_nodules))
                        col2.metric("Average Area", f"{np.mean([n['Area (px²)'] for n in all_nodules]):.0f} px²")
                        if spacing:
                            col3.metric("Largest Nodule", f"{max([float(n['Diameter (mm)']) for n in all_nodules]):.1f} mm")
                        
                        # Visual Results
                        if slices_with_nodules:
                            st.markdown("### 🔍 Visual Results")
                            
                            slice_options = sorted(slices_with_nodules.keys())
                            selected_slice = st.selectbox(
                                "Select slice to view:",
                                slice_options,
                                format_func=lambda x: f"Slice {x} ({len(slices_with_nodules[x]['nodules'])} nodules)"
                            )
                            
                            data = slices_with_nodules[selected_slice]
                            overlay = create_overlay(data['image'], data['nodules'])
                            
                            fig, ax = plt.subplots(figsize=(8, 8))
                            ax.imshow(overlay)
                            ax.set_title(f"Slice {selected_slice} - {len(data['nodules'])} Nodule(s)", color='white')
                            ax.axis('off')
                            st.pyplot(fig)
                            
                            # Show details for nodules in this slice
                            st.markdown(f"**Nodules in Slice {selected_slice}:**")
                            for n in data['nodules']:
                                if spacing:
                                    diam = n['diameter_pixels'] * spacing[0]
                                    area = n['area_pixels'] * (spacing[0] ** 2)
                                    st.markdown(f"- **Nodule {n['id']}:** {area:.1f} mm², {diam:.1f} mm at ({n['centroid_x']:.0f}, {n['centroid_y']:.0f})")
                                else:
                                    st.markdown(f"- **Nodule {n['id']}:** {n['area_pixels']:.0f} px² at ({n['centroid_x']:.0f}, {n['centroid_y']:.0f})")
                        
                        # Complete Results Table
                        st.markdown("### 📋 Complete Results")
                        df = pd.DataFrame(all_nodules)
                        st.dataframe(df, use_container_width=True)
                        
                        # Clinical Recommendations
                        st.markdown("### 🩺 Clinical Recommendations")
                        
                        small = [n for n in all_nodules if n['Area (px²)'] < 100]
                        medium = [n for n in all_nodules if 100 <= n['Area (px²)'] < 300]
                        large = [n for n in all_nodules if n['Area (px²)'] >= 300]
                        
                        if small:
                            st.info(f"📌 **Small Nodules ({len(small)} found)**\n\nSize < 100 pixels²\n\n**Recommendation:** Routine follow-up in 12 months")
                        if medium:
                            st.warning(f"⚠️ **Medium Nodules ({len(medium)} found)**\n\nSize 100-300 pixels²\n\n**Recommendation:** Short-term follow-up in 6 months")
                        if large:
                            st.error(f"🚨 **Large Nodules ({len(large)} found)**\n\nSize > 300 pixels²\n\n**Recommendation:** Urgent consultation recommended")
                        
                        # Download Options
                        st.markdown("### 📥 Download Results")
                        csv = df.to_csv(index=False)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button("📊 Download CSV", csv, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                        
                        # Generate report
                        report = f"""LUNG NODULE DETECTION REPORT
{'='*60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Nodules: {len(all_nodules)}
Slices with Nodules: {len(slices_with_nodules)}

DETAILS:
"""
                        for n in all_nodules:
                            report += f"\nSlice {n['Slice']}: {n['Area (px²)']} px², {n['Diameter (mm)']} mm"

                        with col2:
                            st.download_button("📄 Download Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    else:
                        st.info("✓ No nodules detected in this volume")
                    
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    st.error("Could not load volume. Make sure ZIP contains .mhd and .raw files")
    
    st.markdown("---")
    st.markdown("<center>LungVision AI | Clinical Decision Support System | Always verify with radiologist</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
