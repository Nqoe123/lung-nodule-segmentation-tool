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
import zipfile

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
        
        # Load model - using the same architecture as training
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        # Remove DataParallel wrapper if present
        if isinstance(state_dict, dict) and len(state_dict) > 0:
            first_key = list(state_dict.keys())[0]
            if first_key.startswith('module.'):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.'
                    new_state_dict[name] = v
                state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, True
        
    except Exception as e:
        st.error(f"Model Error: {str(e)}")
        st.error("Make sure the model file is the correct format from your training.")
        return None, False

# ========== MEMORY EFFICIENT U-NET (MATCHING YOUR TRAINING) ==========
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
        logits = self.outc(x)
        return logits

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
    
    # Resize to 512x512 (model expects 512x512)
    resized = resize(normalized, (512, 512), preserve_range=True)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
    
    # Run model
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        mask = (probs > 0.5).float().squeeze().numpy()
    
    # Resize back to original dimensions
    mask_original = resize(mask, image_array.shape[:2], order=0, preserve_range=True)
    
    # Find connected components
    labeled_mask = label(mask_original > 0.5)
    
    detections = []
    if labeled_mask.max() > 0:
        props = regionprops(labeled_mask)
        for region in props:
            if region.area >= 20:  # Minimum nodule size
                detections.append({
                    'area': region.area,
                    'mask': (labeled_mask == region.label).astype(np.float32),
                    'bbox': region.bbox,
                    'centroid': region.centroid,
                    'min_intensity': region.min_intensity if hasattr(region, 'min_intensity') else None
                })
    
    return detections, mask_original

def load_mhd_raw(mhd_bytes, raw_bytes):
    """Load and parse MHD + RAW files"""
    with tempfile.NamedTemporaryFile(suffix='.mhd', delete=False) as mhd_file:
        mhd_file.write(mhd_bytes)
        mhd_path = mhd_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as raw_file:
        raw_file.write(raw_bytes)
        raw_path = raw_file.name
    
    try:
        # Update MHD file to point to correct RAW path
        with open(mhd_path, 'r') as f:
            content = f.read()
        
        # Replace the RAW filename in MHD
        import re
        content = re.sub(r'ElementDataFile = .*\.raw', f'ElementDataFile = {raw_path}', content)
        
        with open(mhd_path, 'w') as f:
            f.write(content)
        
        # Load with SimpleITK
        img = sitk.ReadImage(mhd_path)
        array = sitk.GetArrayFromImage(img)
        
        return array, img
        
    except Exception as e:
        st.error(f"Error loading MHD/RAW: {e}")
        return None, None
    finally:
        # Clean up temp files
        try:
            os.unlink(mhd_path)
            os.unlink(raw_path)
        except:
            pass

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
            "Architecture": "Memory Efficient U-Net",
            "Initial Channels": "64",
            "Training Data": "LUNA16",
            "Best Dice": "0.70",
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
            type=["png", "jpg", "jpeg", "dcm"],
            help="Upload a lung CT slice in PNG, JPG, JPEG, or DICOM format"
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
                            
                            st.metric("Centroid (x,y)", f"({d['centroid'][1]:.0f}, {d['centroid'][0]:.0f})")
                        
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
        st.markdown('<div class="info-box">📌 <strong>Full CT Analysis:</strong> Upload a ZIP file containing both MHD and RAW files for volumetric analysis.</div>', unsafe_allow_html=True)
        
        # Single file upload for ZIP containing both MHD and RAW
        uploaded_zip = st.file_uploader(
            "Select ZIP file containing .mhd and .raw files",
            type=["zip"],
            help="Upload a ZIP archive that contains both the .mhd and .raw files together"
        )
        
        if uploaded_zip:
            # Extract ZIP
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                
                # Extract
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Find MHD and RAW files
                mhd_file = None
                raw_file = None
                
                for file in os.listdir(tmpdir):
                    if file.endswith('.mhd'):
                        mhd_file = os.path.join(tmpdir, file)
                    elif file.endswith('.raw'):
                        raw_file = os.path.join(tmpdir, file)
                
                if mhd_file and raw_file:
                    st.success(f"Found: {os.path.basename(mhd_file)} and {os.path.basename(raw_file)}")
                    
                    # Load and display info
                    img = sitk.ReadImage(mhd_file)
                    array = sitk.GetArrayFromImage(img)
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Volume Dimensions", f"{array.shape[0]} × {array.shape[1]} × {array.shape[2]}")
                    st.metric("Spacing", f"{img.GetSpacing()[0]:.2f}mm")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display first and last slice for preview
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(array[array.shape[0]//2], caption="Middle Slice Preview", use_container_width=True)
                    
                    if st.button("🔍 Analyze Full Volume", type="primary"):
                        st.info("Full 3D volumetric analysis processing...")
                        
                        # Process each slice
                        detections_all = []
                        model, success = download_and_load_model()
                        
                        if success:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i in range(array.shape[0]):
                                status_text.text(f"Processing slice {i+1}/{array.shape[0]}")
                                slice_img = array[i]
                                
                                # Normalize
                                slice_normalized = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
                                
                                # Segment
                                detections, _ = segment_nodule(model, slice_normalized)
                                
                                if detections:
                                    for d in detections:
                                        detections_all.append({
                                            'slice': i,
                                            'area': d['area'],
                                            'centroid': d['centroid']
                                        })
                                
                                progress_bar.progress((i + 1) / array.shape[0])
                            
                            status_text.text("Analysis complete!")
                            
                            # Show results
                            if detections_all:
                                st.markdown(f'<div class="success-box">✅ <strong>{len(detections_all)} nodule(s) detected across {array.shape[0]} slices</strong></div>', unsafe_allow_html=True)
                                
                                # Group by slice
                                import pandas as pd
                                df = pd.DataFrame(detections_all)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.markdown('<div class="info-box">✓ No nodules detected in this volume</div>', unsafe_allow_html=True)
                else:
                    st.error("ZIP file must contain both .mhd and .raw files")
    
    # Footer
    st.markdown("""
    <div class="clinical-footer">
        <p>LungVision AI | Clinical Decision Support System</p>
        <p>Always verify AI results with clinical expertise | HIT500 Capstone Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
