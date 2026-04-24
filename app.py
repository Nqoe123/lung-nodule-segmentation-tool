import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import gdown
from skimage.transform import resize
from skimage.measure import label, regionprops
import tempfile
import SimpleITK as sitk
from collections import OrderedDict
from PIL import Image
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="LungVision AI | Clinical Nodule Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== PROFESSIONAL MEDICAL CSS ==========
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f7f9fc;
    }
    
    /* Header styling */
    .clinical-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .clinical-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .clinical-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    .clinical-header small {
        font-size: 0.8rem;
        opacity: 0.75;
    }
    
    /* Card styling */
    .clinical-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Metric card */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #2c5282;
    }
    
    /* Info box */
    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #2c5282;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2c5282;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1e3a5f;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #fafbfc;
    }
    
    /* Footer */
    .clinical-footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
        color: #718096;
        font-size: 0.8rem;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e0, transparent);
        margin: 1.5rem 0;
    }
    
    /* Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background-color: #f7f9fc;
        padding: 8px;
        text-align: left;
        border-bottom: 2px solid #e2e8f0;
    }
    td {
        padding: 8px;
        border-bottom: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ========== GOOGLE DRIVE SETUP ==========
GOOGLE_DRIVE_FILE_ID = "1PzCv2fJSr7e0QIfPGtLKLOL-9RSLdR2i"
MODEL_FILENAME = "best_model(1).pth"

@st.cache_resource
def download_and_load_model():
    """Download model from Google Drive and load it"""
    try:
        # Download if not exists
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("Downloading AI model..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_FILENAME, quiet=True)
        
        # Load model
        model = UNet()
        checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove DataParallel wrapper if present
        if 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, True
        
    except Exception as e:
        st.error(f"Model Error: {str(e)}")
        return None, False

# ========== U-NET ARCHITECTURE ==========
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
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
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

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
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
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

# ========== IMAGE PROCESSING FUNCTIONS ==========
def normalize_ct_image(image_array):
    """Apply lung window normalization [-1000, 400] HU"""
    # If image is 0-255 range (typical PNG)
    if image_array.max() > 1.0:
        image_hu = (image_array / 255.0) * 1400 - 1000
    else:
        image_hu = image_array * 1400 - 1000
    
    image_hu = np.clip(image_hu, -1000, 400)
    normalized = (image_hu + 1000) / 1400
    return normalized

def segment_nodule(model, image_array):
    """Run segmentation on a single image"""
    # Normalize
    normalized = normalize_ct_image(image_array)
    
    # Resize to 512x512
    resized = resize(normalized, (512, 512), preserve_range=True)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
    
    # Run model
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        mask = (probs > 0.5).float().squeeze().numpy()
    
    # Resize back
    mask_original = resize(mask, image_array.shape[:2], order=0, preserve_range=True)
    
    # Find connected components
    labeled_mask = label(mask_original > 0.5)
    
    detections = []
    if labeled_mask.max() > 0:
        props = regionprops(labeled_mask)
        for region in props:
            if region.area >= 20:
                detections.append({
                    'area': region.area,
                    'mask': (labeled_mask == region.label).astype(np.float32),
                    'bbox': region.bbox,
                    'centroid': region.centroid
                })
    
    return detections, mask_original

# ========== MAIN UI ==========
def main():
    # Header
    st.markdown("""
    <div class="clinical-header">
        <h1>🫁 LungVision AI</h1>
        <p>Clinical Lung Nodule Detection System</p>
        <small>For Radiologist Use Only | Powered by Deep Learning</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Login check
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
            st.subheader("🔐 Clinical Access")
            username = st.text_input("Radiologist ID")
            password = st.text_input("Password", type="password")
            
            if st.button("Authenticate"):
                if username == "radiologist" and password == "hit500":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access Denied. Invalid credentials.")
            st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 Session")
        st.success("**Radiologist** (Active)")
        
        if st.button("End Session"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🧠 Model Specifications")
        
        model_info = {
            "Architecture": "U-Net (Efficient)",
            "Parameters": "3.4M",
            "Training Data": "LUNA16",
            "Best Dice": "0.6399",
            "Input Size": "512×512"
        }
        
        for key, val in model_info.items():
            st.markdown(f"**{key}:** {val}")
        
        st.markdown("---")
        st.markdown("### ⚠️ Clinical Note")
        st.info("This system is designed for **lung CT scans only**. Results should always be verified by a qualified radiologist.")
    
    # Main content
    st.markdown("### 📤 Image Upload")
    
    # Upload type selection
    upload_type = st.radio(
        "Select examination type:",
        ["Single CT Slice (Fast Analysis)", "Full CT Scan (3D Volume)"],
        horizontal=True
    )
    
    if upload_type == "Single CT Slice (Fast Analysis)":
        st.markdown('<div class="info-box">📌 <strong>Fast Analysis:</strong> Upload a single CT slice for rapid nodule detection. Results in seconds.</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select CT Image",
            type=["png", "jpg", "jpeg"],
            help="Upload a lung CT slice in PNG, JPG, or JPEG format"
        )
        
        if uploaded_file:
            # Load and display image
            image = Image.open(uploaded_file)
            if image.mode != 'L':
                image = image.convert('L')
            image_array = np.array(image, dtype=np.float32)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
                st.image(image_array, caption="Original CT Slice", use_container_width=True)
                st.markdown(f"*Dimensions: {image_array.shape[1]}×{image_array.shape[0]}*")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Image Size", f"{image_array.shape[1]}×{image_array.shape[0]}")
                st.metric("Format", uploaded_file.type.upper())
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Load model
            with st.spinner("Loading AI Model..."):
                model, success = download_and_load_model()
            
            if success and st.button("🔍 Run Analysis", type="primary"):
                with st.spinner("AI Analyzing..."):
                    detections, mask = segment_nodule(model, image_array)
                
                if detections:
                    st.markdown(f'<div class="success-box">✅ <strong>{len(detections)} nodule(s) detected</strong></div>', unsafe_allow_html=True)
                    
                    for idx, d in enumerate(detections):
                        st.markdown(f'<div class="clinical-card">', unsafe_allow_html=True)
                        st.markdown(f"#### Nodule {idx+1}")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            # Create overlay
                            img_norm = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
                            overlay = np.stack([img_norm] * 3, axis=-1)
                            overlay[:, :, 0] = np.where(d['mask'] > 0.5, 1.0, overlay[:, :, 0])
                            overlay[:, :, 1] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 1])
                            overlay[:, :, 2] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 2])
                            st.image(overlay, use_container_width=True)
                        
                        with col_b:
                            st.metric("Area", f"{d['area']:.0f} pixels²")
                            
                            if d['area'] < 100:
                                st.info("📌 **Size:** Small\n**Recommendation:** Routine monitoring")
                            elif d['area'] < 300:
                                st.warning("⚠️ **Size:** Medium\n**Recommendation:** Short-term follow-up")
                            else:
                                st.error("🚨 **Size:** Large\n**Recommendation:** Urgent consultation")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Generate report
                    report = f"""LUNG NODULE DETECTION REPORT
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image Dimensions: {image_array.shape[1]}×{image_array.shape[0]}
Nodules Detected: {len(detections)}

DETAILS:
"""
                    for idx, d in enumerate(detections):
                        report += f"""
Nodule {idx+1}:
  - Area: {d['area']:.0f} pixels²
  - Location (x,y): ({d['centroid'][1]:.0f}, {d['centroid'][0]:.0f})
  - Recommendation: {'Routine monitoring' if d['area'] < 100 else 'Short-term follow-up' if d['area'] < 300 else 'Urgent consultation'}
"""
                    
                    st.download_button("📊 Download Clinical Report", report, f"nodule_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    
                else:
                    st.markdown('<div class="info-box">✓ No nodules detected in this slice</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="info-box">📌 <strong>Full CT Analysis:</strong> Upload complete MHD+RAW files for volumetric analysis. Note: Larger files take longer to upload.</div>', unsafe_allow_html=True)
        
        st.info("Full CT scan support requires MHD and RAW files. Upload both files for 3D volumetric analysis.")
        
        uploaded_mhd = st.file_uploader("Select .mhd file", type=["mhd"])
        uploaded_raw = st.file_uploader("Select .raw file", type=["raw"])
        
        if uploaded_mhd and uploaded_raw:
            st.success("Both files selected. Ready for analysis.")
            
            if st.button("🔍 Analyze Full Scan", type="primary"):
                st.info("Full 3D analysis is available in the enterprise version. Contact support for details.")
    
    # Footer
    st.markdown("""
    <div class="clinical-footer">
        <p>LungVision AI | Clinical Decision Support System</p>
        <p>Always verify AI results with clinical expertise | HIT500 Capstone Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
