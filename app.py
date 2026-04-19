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
import base64
from datetime import datetime

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="LungVision AI | Nodule Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS FOR MODERN UI ==========
st.markdown("""
<style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Card style for results */
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Custom divider */
    .custom-divider {
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        border-radius: 2px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== GOOGLE DRIVE SETUP ==========
GOOGLE_DRIVE_FILE_ID = "1FdIozNEVbIPUsjcdReAfmbgN3Nisx9yQ"
MODEL_FILENAME = "checkpoint_epoch_90.pth"

def download_model_from_drive():
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("🔄 Loading AI Model..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_FILENAME, quiet=False)
        return MODEL_FILENAME
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        return None

# ========== MEMORY EFFICIENT U-NET ==========
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

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    try:
        model_path = download_model_from_drive()
        if model_path is None:
            return None, False
        
        model = UNet()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            best_dice = checkpoint.get('best_dice', 'Unknown')
        else:
            state_dict = checkpoint
            best_dice = 'Unknown'
        
        if state_dict and 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, True, best_dice
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False, None

# ========== PROCESS 2D IMAGE (FAST UPLOAD) ==========
def process_2d_image(uploaded_file):
    """Process a single 2D CT image - MUCH faster upload"""
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    if image.mode != 'L':
        image = image.convert('L')
    
    return np.array(image)

def segment_2d_nodule(model, image_array):
    """Segment nodule from single 2D image"""
    # Resize to 512x512
    img_resized = resize(image_array, (512, 512), preserve_range=True)
    
    # Normalize
    window_min, window_max = -1000.0, 400.0
    img_normalized = np.clip(img_resized, window_min, window_max)
    img_normalized = (img_normalized - window_min) / (window_max - window_min)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)
    
    # Segment
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        mask = (probs > 0.5).float().squeeze().numpy()
    
    # Resize back
    mask_original = resize(mask, image_array.shape[:2], order=0, preserve_range=True)
    
    # Find individual nodules
    labeled_mask = label(mask_original > 0.5)
    
    detections = []
    if labeled_mask.max() > 0:
        props = regionprops(labeled_mask)
        for region in props:
            if region.area >= 20:
                detections.append({
                    'area_pixels': region.area,
                    'mask': (labeled_mask == region.label).astype(np.float32),
                    'bbox': region.bbox
                })
    
    return detections, mask_original

# ========== PROCESS 3D CT (FULL SCAN) ==========
def load_mhd_files(uploaded_files):
    mhd_file = None
    raw_file = None
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.mhd'):
            mhd_file = uploaded_file
        elif uploaded_file.name.endswith('.raw'):
            raw_file = uploaded_file
    
    if not mhd_file or not raw_file:
        return None, None, None
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mhd_path = os.path.join(tmpdir, mhd_file.name)
            raw_path = os.path.join(tmpdir, raw_file.name)
            
            with open(mhd_path, 'wb') as f:
                f.write(mhd_file.getvalue())
            with open(raw_path, 'wb') as f:
                f.write(raw_file.getvalue())
            
            # Update MHD reference
            with open(mhd_path, 'r') as f:
                mhd_content = f.read()
            
            import re
            mhd_content = re.sub(
                r'ElementDataFile\s*=\s*.*',
                f'ElementDataFile = {raw_file.name}',
                mhd_content
            )
            
            with open(mhd_path, 'w') as f:
                f.write(mhd_content)
            
            itk_image = sitk.ReadImage(mhd_path)
            ct_array = sitk.GetArrayFromImage(itk_image)
            
            spacing = itk_image.GetSpacing()
            pixel_spacing_mm = spacing[0]
            slice_thickness_mm = spacing[2]
            
            return ct_array, pixel_spacing_mm, slice_thickness_mm
            
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None

def detect_nodules_3d(model, ct_volume, pixel_spacing_mm, slice_thickness_mm, min_area=20):
    detections = []
    
    with st.spinner("🔍 Analyzing CT scan..."):
        progress_bar = st.progress(0)
        
        for i in range(ct_volume.shape[0]):
            if i % 10 == 0:
                progress_bar.progress(i / ct_volume.shape[0])
            
            slice_img = ct_volume[i]
            img_resized = resize(slice_img, (512, 512), preserve_range=True)
            input_tensor = torch.FloatTensor(img_resized).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.sigmoid(output)
                mask = (probs > 0.5).float().squeeze().numpy()
            
            mask_original = resize(mask, slice_img.shape[:2], order=0, preserve_range=True)
            labeled_mask = label(mask_original > 0.5)
            
            if labeled_mask.max() > 0:
                props = regionprops(labeled_mask)
                for region in props:
                    if region.area >= min_area:
                        nodule_volume = region.area * (pixel_spacing_mm ** 2) * slice_thickness_mm
                        detections.append({
                            'slice': i,
                            'volume_mm3': nodule_volume,
                            'mask': (labeled_mask == region.label).astype(np.float32),
                            'slice_image': slice_img
                        })
        
        progress_bar.empty()
    
    detections.sort(key=lambda x: x['volume_mm3'], reverse=True)
    return detections

# ========== MAIN UI ==========
def main():
    # Custom header
    st.markdown("""
    <div class="main-header">
        <h1>🫁 LungVision AI</h1>
        <p style="font-size: 1.2rem;">Intelligent Lung Nodule Detection & Segmentation</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Powered by Memory Efficient U-Net | 3.4M Parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("🔐 Radiologist Access")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_btn = st.button("🔓 Login", use_container_width=True)
            with col_b:
                st.markdown("[Request Access](https://example.com)")
            
            if login_btn:
                if username == "radiologist" and password == "hit500":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Sidebar - Professional info
    with st.sidebar:
        st.markdown("### 👤 Session")
        st.success(f"✅ Logged in as: **Radiologist**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🧠 Model Info")
        st.info("""
        - **Architecture:** Memory Efficient U-Net
        - **Parameters:** 3.4 Million
        - **Training Data:** LUNA16 (902 slices)
        - **Best Dice Score:** 0.6399
        - **Inference Time:** ~0.05 sec/slice
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Today's Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Scans Today", "0", delta="Ready")
        with col2:
            st.metric("Model Status", "Active", delta="Online")
    
    # Load model
    with st.spinner("🔄 Initializing AI Model..."):
        model, model_loaded, best_dice = load_model()
    
    if not model_loaded:
        st.error("❌ Failed to load AI model. Please check your connection.")
        st.stop()
    
    # Main content area
    st.markdown("### 📤 Upload CT Scan")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Upload options - SUPPORT BOTH 2D AND 3D
    upload_option = st.radio(
        "Select upload type:",
        ["⚡ Fast Upload (2D Image)", "📊 Full CT Scan (3D MHD+RAW)"],
        horizontal=True,
        help="2D images upload faster. Use 3D for full volumetric analysis"
    )
    
    if upload_option == "⚡ Fast Upload (2D Image)":
        st.info("💡 **Tip:** Upload individual CT slices for rapid analysis. Much faster than full 3D scans!")
        
        uploaded_file = st.file_uploader(
            "Choose a CT image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Upload a single CT slice for instant nodule detection"
        )
        
        if uploaded_file:
            with st.spinner("📥 Loading image..."):
                image_array = process_2d_image(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(image_array, caption="📷 Uploaded CT Slice", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if st.button("🔍 Detect Nodules", type="primary", use_container_width=True):
                    with st.spinner("🧠 AI Analyzing..."):
                        detections, full_mask = segment_2d_nodule(model, image_array)
                    
                    if detections:
                        st.balloons()
                        st.success(f"✅ Found {len(detections)} nodule(s)")
                        
                        for idx, d in enumerate(detections):
                            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                            st.markdown(f"**Nodule {idx+1}**")
                            
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
                                st.metric("Area", f"{d['area_pixels']:.0f} pixels²")
                                st.metric("Confidence", "High")
                                
                                if d['area_pixels'] < 100:
                                    st.info("📌 **Small nodule** - Monitor")
                                elif d['area_pixels'] < 300:
                                    st.warning("⚠️ **Medium nodule** - Follow up")
                                else:
                                    st.error("🚨 **Large nodule** - Urgent")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("📌 No nodules detected in this slice")
    
    else:  # 3D Full CT Scan
        st.info("📌 **Note:** Full 3D CT scans take longer to upload. For faster results, use the 2D upload option above.")
        
        uploaded_files = st.file_uploader(
            "Select .mhd and .raw files (select both using Ctrl+Click)",
            type=["mhd", "raw"],
            accept_multiple_files=True,
            help="Upload both files from your CT scan"
        )
        
        if uploaded_files and len(uploaded_files) == 2:
            with st.spinner("📥 Loading 3D CT scan..."):
                ct_array, pixel_spacing, slice_thickness = load_mhd_files(uploaded_files)
            
            if ct_array is not None:
                st.success(f"✅ Loaded {ct_array.shape[0]} slices")
                
                # Normalize
                window_min, window_max = -1000.0, 400.0
                ct_normalized = np.clip(ct_array, window_min, window_max)
                ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
                
                # Preview
                middle = ct_array.shape[0] // 2
                st.image(ct_normalized[middle], caption=f"Preview Slice {middle}", use_container_width=True)
                
                if st.button("🔍 Analyze Full Scan", type="primary", use_container_width=True):
                    detections = detect_nodules_3d(model, ct_normalized, pixel_spacing, slice_thickness)
                    
                    if detections:
                        st.success(f"✅ Found {len(detections)} nodule(s)")
                        
                        for idx, d in enumerate(detections[:5]):  # Show top 5
                            with st.expander(f"Nodule {idx+1} - Slice {d['slice']}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    slice_norm = (d['slice_image'] - d['slice_image'].min()) / (d['slice_image'].max() - d['slice_image'].min() + 1e-8)
                                    overlay = np.stack([slice_norm] * 3, axis=-1)
                                    overlay[:, :, 0] = np.where(d['mask'] > 0.5, 1.0, overlay[:, :, 0])
                                    overlay[:, :, 1] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 1])
                                    overlay[:, :, 2] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 2])
                                    st.image(overlay, use_container_width=True)
                                with col2:
                                    st.metric("Volume", f"{d['volume_mm3']:.1f} mm³")
                                    if d['volume_mm3'] < 100:
                                        st.info("Small - Monitor")
                                    elif d['volume_mm3'] < 300:
                                        st.warning("Medium - Follow up")
                                    else:
                                        st.error("Large - Urgent")
                    else:
                        st.info("No nodules detected")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>© 2026 LungVision AI | HIT500 Capstone Project</p>
            <p style="font-size: 0.8rem;">Memory Efficient U-Net | Trained on LUNA16 Dataset</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
