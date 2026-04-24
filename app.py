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

warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="LungVision AI | Nodule Detection",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>
    .clinical-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #2c5282;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #2c5282;
        color: white;
        font-weight: 500;
        width: 100%;
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

# ========== LOAD GOOD MODEL (0.74 DICE) ==========
GOOGLE_DRIVE_FILE_ID = "1PzCv2fJSr7e0QIfPGtLKLOL-9RSLdR2i"
MODEL_FILENAME = "best_model.pth"

@st.cache_resource
def load_model():
    """Load the good model (0.74 Dice) from Google Drive"""
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("Loading AI model..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_FILENAME, quiet=True)
        
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
        
        # Handle checkpoint format
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint.state_dict()
        
        # Remove 'module.' prefix if present
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        st.success("✅ Model loaded (Dice: 0.74)")
        return model
        
    except Exception as e:
        st.error(f"Model error: {str(e)}")
        return None

# ========== AUTOMATIC SEGMENTATION ==========
def segment_nodule(model, image_array):
    """Fully automatic segmentation - no settings needed"""
    try:
        original_shape = image_array.shape
        
        # Normalize to [0,1]
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Resize to 512x512 for model
        resized = resize(image_array, (512, 512), preserve_range=True)
        
        # Model inference
        input_tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            # Fixed threshold at 0.5 for good model
            mask = (probs > 0.5).float().squeeze().numpy()
        
        # Resize mask back
        mask_resized = resize(mask, original_shape, order=0, preserve_range=True)
        
        # Find nodules
        labeled = label(mask_resized > 0.5)
        nodules = []
        
        for region in regionprops(labeled):
            if region.area >= 20:  # Minimum nodule size
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
        st.error(f"Segmentation error: {str(e)}")
        return [], None

def load_volume(zip_file):
    """Load CT volume from ZIP"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        mhd_file = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.mhd')), None)
        raw_file = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.raw')), None)
        
        if mhd_file and raw_file:
            img = sitk.ReadImage(mhd_file)
            volume = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            return volume, spacing
    
    return None, None

# ========== MAIN APP ==========
def main():
    st.markdown("""
    <div class="clinical-header">
        <h1>🫁 LungVision AI</h1>
        <p>Automatic Lung Nodule Detection</p>
        <small>Upload → Click Detect → Get Results</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple login
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("🔐 Login")
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
        st.success("👤 Radiologist")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
        st.markdown("---")
        st.markdown("### Model: U-Net (Dice 0.74)")
        st.markdown("### Training: LUNA16")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Upload section
    st.markdown("### 📤 Upload & Detect")
    
    upload_type = st.radio("", ["Single CT Slice", "Full CT Volume"], horizontal=True)
    
    if upload_type == "Single CT Slice":
        uploaded_file = st.file_uploader("Choose CT image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            # Load image
            image = Image.open(uploaded_file).convert('L')
            image_array = np.array(image, dtype=np.float32)
            
            # Show image
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image_array, cmap='gray')
                ax.set_title("Original CT Slice")
                ax.axis('off')
                st.pyplot(fig)
            
            # ONE BUTTON - fully automatic
            if st.button("🔍 Detect Nodules", type="primary"):
                with st.spinner("Analyzing..."):
                    nodules, mask = segment_nodule(model, image_array)
                
                if nodules:
                    # Show result with overlay
                    with col2:
                        # Create overlay
                        img_norm = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
                        overlay = np.stack([img_norm] * 3, axis=-1)
                        for n in nodules:
                            mask_bool = n['mask'] > 0.5
                            overlay[mask_bool, 0] = 1.0  # Red
                            overlay[mask_bool, 1] = 0.0
                            overlay[mask_bool, 2] = 0.0
                        
                        fig2, ax2 = plt.subplots(figsize=(6, 6))
                        ax2.imshow(overlay)
                        ax2.set_title(f"✅ {len(nodules)} Nodule(s) Detected")
                        ax2.axis('off')
                        st.pyplot(fig2)
                    
                    # Results table
                    st.markdown(f'<div class="success-box">✅ <strong>{len(nodules)} Nodule(s) Detected</strong></div>', unsafe_allow_html=True)
                    
                    results = []
                    for n in nodules:
                        results.append({
                            "Nodule": n['id'],
                            "Area (px²)": f"{n['area_pixels']:.0f}",
                            "Diameter (px)": f"{n['diameter_pixels']:.1f}",
                            "Location": f"({n['centroid_x']:.0f}, {n['centroid_y']:.0f})"
                        })
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 🩺 Recommendations")
                    for n in nodules:
                        if n['area_pixels'] < 100:
                            st.markdown(f'<div class="info-box">📌 Nodule {n["id"]}: Small - Routine follow-up in 12 months</div>', unsafe_allow_html=True)
                        elif n['area_pixels'] < 300:
                            st.markdown(f'<div class="warning-box">⚠️ Nodule {n["id"]}: Medium - Follow-up in 6 months</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="warning-box">🚨 Nodule {n["id"]}: Large - Urgent consultation</div>', unsafe_allow_html=True)
                    
                    # Download report
                    report = f"""LUNG NODULE REPORT
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Nodules: {len(nodules)}

"""
                    for n in nodules:
                        report += f"Nodule {n['id']}: {n['area_pixels']:.0f} px² at ({n['centroid_x']:.0f}, {n['centroid_y']:.0f})\n"
                    
                    st.download_button("📄 Download Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    
                else:
                    st.markdown('<div class="info-box">✓ No nodules detected in this slice</div>', unsafe_allow_html=True)
                    
                    # Debug info
                    with st.expander("Debug Info"):
                        st.write(f"Image shape: {image_array.shape}")
                        st.write(f"Image range: [{image_array.min():.2f}, {image_array.max():.2f}]")
                        if mask is not None:
                            st.write(f"Max probability: {mask.max():.4f}")
                            fig3, ax3 = plt.subplots(figsize=(4, 4))
                            ax3.imshow(mask, cmap='hot')
                            ax3.set_title("Model Activation Map")
                            ax3.axis('off')
                            st.pyplot(fig3)
    
    else:  # Full CT Volume
        uploaded_zip = st.file_uploader("Choose CT volume (ZIP with .mhd/.raw)", type=["zip"])
        
        if uploaded_zip:
            if st.button("🔍 Analyze Full Volume", type="primary"):
                with st.spinner("Loading and analyzing volume..."):
                    volume, spacing = load_volume(uploaded_zip)
                
                if volume is not None:
                    st.success(f"Volume: {volume.shape[0]} slices, {volume.shape[1]}×{volume.shape[2]}")
                    
                    all_nodules = []
                    progress = st.progress(0)
                    
                    for i in range(volume.shape[0]):
                        nodules, _ = segment_nodule(model, volume[i])
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
                                'Area (mm²)': f"{area_mm2:.1f}" if spacing else "N/A",
                                'Diameter (mm)': f"{diam_mm:.1f}" if spacing else "N/A"
                            })
                        progress.progress((i + 1) / volume.shape[0])
                    
                    if all_nodules:
                        st.markdown(f'<div class="success-box">✅ <strong>{len(all_nodules)} Nodules Detected</strong></div>', unsafe_allow_html=True)
                        df = pd.DataFrame(all_nodules)
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False)
                        st.download_button("📊 Download Results", csv, f"volume_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    else:
                        st.info("✓ No nodules detected in this volume")
                else:
                    st.error("Could not load volume. Make sure ZIP contains .mhd and .raw files")
    
    st.markdown("---")
    st.markdown("<center><small>LungVision AI | Clinical Decision Support | Always verify with radiologist</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
