import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.exposure import equalize_adapthist
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
    page_title="LungVision AI | Clinical Nodule Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CSS ==========
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
    .clinical-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #2c5282;
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
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        width: 100%;
    }
    .clinical-footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
        color: #718096;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== MODEL ARCHITECTURE (EXACTLY FROM YOUR TRAINING CODE) ==========
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

# ========== LOAD MODEL FROM GOOGLE DRIVE ==========
GOOGLE_DRIVE_FILE_ID = "1PzCv2fJSr7e0QIfPGtLKLOL-9RSLdR2i"
MODEL_FILENAME = "best_model.pth"

@st.cache_resource
def load_model_from_drive():
    """Download model from Google Drive and load it"""
    try:
        # Download from Google Drive if file doesn't exist
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("Downloading AI model from Google Drive..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_FILENAME, quiet=False)
                st.success("Model downloaded successfully!")
        
        # Create model instance (EXACT architecture from training)
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        
        # Load the checkpoint
        checkpoint = torch.load(MODEL_FILENAME, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()
        
        # Remove 'module.' prefix if present (from DataParallel training)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # remove 'module.'
            else:
                name = k
            new_state_dict[name] = v
        
        # Load state dict
        model.load_state_dict(new_state_dict)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please make sure the Google Drive file ID is correct and the file is accessible.")
        return None

# ========== CT IMAGE PROCESSING FUNCTIONS ==========

def apply_lung_window(image_array):
    """
    Apply lung window settings for display
    Lung window: width 1500 HU, level -600 HU (shows -1350 to +150 HU)
    """
    # Normalize to 0-1 range for display
    min_hu = -1350
    max_hu = 150
    
    # Clip to lung window range
    clipped = np.clip(image_array, min_hu, max_hu)
    
    # Normalize to 0-1 for display
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    
    return normalized

def preprocess_for_model(image_array):
    """
    Preprocess CT image for model input
    Your model was trained on CT images normalized to [0,1] range
    """
    # Your training normalization: (image - min) / (max - min)
    # Assuming input is already in HU or 0-255 range
    
    # Convert to float if needed
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float32)
    
    # Normalize to [0, 1] range
    if image_array.max() > 1.0:
        # Scale to 0-1
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    
    return image_array

def load_ct_image(uploaded_file):
    """Load and prepare CT image for processing"""
    try:
        # Open image
        image = Image.open(uploaded_file)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Display info
        st.write(f"Image range: [{image_array.min():.1f}, {image_array.max():.1f}]")
        st.write(f"Image dtype: {image_array.dtype}")
        
        return image_array, True
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None, False

def segment_nodule(model, image_array):
    """Run segmentation on a single CT slice"""
    try:
        # Store original shape
        original_shape = image_array.shape
        
        # Preprocess for model
        model_input = preprocess_for_model(image_array.copy())
        
        # Resize to 512x512 (model input size)
        resized = resize(model_input, (512, 512), preserve_range=True)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
        
        # Run model
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            mask = (probs > 0.5).float().squeeze().numpy()
        
        # Resize mask back to original size
        mask_resized = resize(mask, original_shape, order=0, preserve_range=True)
        
        # Find connected components (nodules)
        labeled = label(mask_resized > 0.5)
        nodules = []
        
        for region in regionprops(labeled):
            # Filter out tiny regions (noise)
            if region.area >= 20:
                # Calculate diameter from area (assuming circular)
                diameter = 2 * np.sqrt(region.area / np.pi)
                
                nodules.append({
                    'id': len(nodules) + 1,
                    'area_pixels': region.area,
                    'diameter_pixels': diameter,
                    'centroid_x': region.centroid[1],
                    'centroid_y': region.centroid[0],
                    'min_row': region.bbox[0],
                    'max_row': region.bbox[2],
                    'min_col': region.bbox[1],
                    'max_col': region.bbox[3],
                    'mask': (labeled == region.label).astype(np.float32)
                })
        
        return nodules, mask_resized
    
    except Exception as e:
        st.error(f"Segmentation error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return [], None

def create_overlay(image_array, mask, nodules):
    """Create RGB overlay with detected nodules highlighted"""
    # Normalize image for display
    img_norm = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    
    # Create RGB overlay
    overlay = np.stack([img_norm] * 3, axis=-1)
    
    # Highlight each nodule
    for nodule in nodules:
        mask_bool = nodule['mask'] > 0.5
        overlay[mask_bool, 0] = 1.0  # Red channel
        overlay[mask_bool, 1] = 0.0  # Green channel
        overlay[mask_bool, 2] = 0.0  # Blue channel
    
    return overlay

def load_ct_volume(zip_file):
    """Load MHD/RAW CT volume from ZIP file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        
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
            # Read the volume
            img = sitk.ReadImage(mhd_file)
            volume = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            return volume, spacing
    
    return None, None

# ========== MAIN APP ==========
def main():
    # Header
    st.markdown("""
    <div class="clinical-header">
        <h1>🫁 LungVision AI</h1>
        <p>Clinical Lung Nodule Detection System</p>
        <small>For Radiologist Use Only</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple login
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.subheader("🔐 Clinical Access")
                username = st.text_input("Radiologist ID")
                password = st.text_input("Password", type="password")
                
                if st.button("Login"):
                    if username == "radiologist" and password == "hit500":
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        return
    
    # Sidebar with logout
    with st.sidebar:
        st.success(f"👤 Radiologist - Active")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Model Information")
        st.markdown("- **Architecture:** Memory Efficient U-Net")
        st.markdown("- **Training Data:** LUNA16")
        st.markdown("- **Input Size:** 512×512")
        st.markdown("- **Initial Channels:** 64")
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("1. Upload a CT slice or volume")
        st.markdown("2. Click 'Detect Nodules'")
        st.markdown("3. Review measurements and recommendations")
        st.markdown("4. Download clinical report")
    
    # Load model from Google Drive
    with st.spinner("Loading AI model..."):
        model = load_model_from_drive()
    
    if model is None:
        st.stop()
    
    # Main content
    st.markdown("### 📤 Upload Study")
    
    # Upload options
    upload_type = st.radio(
        "Select examination type:",
        ["Single CT Slice", "Full CT Volume (ZIP with MHD/RAW)"],
        horizontal=True
    )
    
    if upload_type == "Single CT Slice":
        uploaded_file = st.file_uploader(
            "Upload CT Slice",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Upload a single lung CT slice"
        )
        
        if uploaded_file:
            # Load image
            image_array, success = load_ct_image(uploaded_file)
            
            if success and image_array is not None:
                # Apply lung window for better display
                display_img = apply_lung_window(image_array)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(display_img, caption="Original CT Slice (Lung Window)", use_container_width=True, clamp=True)
                    st.caption(f"Dimensions: {image_array.shape[1]}×{image_array.shape[0]} pixels")
                    st.caption(f"Intensity range: [{image_array.min():.1f}, {image_array.max():.1f}]")
                
                # Analyze button
                if st.button("🔍 Detect Nodules", type="primary"):
                    with st.spinner("Segmenting lung nodules..."):
                        nodules, mask = segment_nodule(model, image_array)
                    
                    if nodules:
                        # Create overlay
                        overlay = create_overlay(image_array, mask, nodules)
                        
                        with col2:
                            st.image(overlay, caption=f"{len(nodules)} Nodule(s) Detected", use_container_width=True, clamp=True)
                        
                        # Display results table
                        st.markdown("## 📊 Nodule Measurements")
                        
                        results_data = []
                        for nodule in nodules:
                            results_data.append({
                                "Nodule #": nodule['id'],
                                "Area (pixels²)": f"{nodule['area_pixels']:.0f}",
                                "Diameter (pixels)": f"{nodule['diameter_pixels']:.1f}",
                                "Centroid X": f"{nodule['centroid_x']:.0f}",
                                "Centroid Y": f"{nodule['centroid_y']:.0f}"
                            })
                        
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Visualize mask
                        st.markdown("## 🔬 Segmentation Mask")
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].imshow(display_img, cmap='gray')
                        ax[0].set_title('Original')
                        ax[0].axis('off')
                        ax[1].imshow(mask, cmap='hot')
                        ax[1].set_title('Nodule Probability Mask')
                        ax[1].axis('off')
                        st.pyplot(fig)
                        
                        # Clinical recommendations
                        st.markdown("## 🩺 Clinical Recommendations")
                        
                        for nodule in nodules:
                            if nodule['area_pixels'] < 100:
                                st.markdown(f"""
                                <div class="info-box">
                                    <strong>Nodule {nodule['id']}:</strong> Small nodule ({nodule['area_pixels']:.0f} pixels²)<br>
                                    📌 Recommendation: Routine follow-up in 12 months
                                </div>
                                """, unsafe_allow_html=True)
                            elif nodule['area_pixels'] < 300:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <strong>Nodule {nodule['id']}:</strong> Medium nodule ({nodule['area_pixels']:.0f} pixels²)<br>
                                    ⚠️ Recommendation: Short-term follow-up in 6 months
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <strong>Nodule {nodule['id']}:</strong> Large nodule ({nodule['area_pixels']:.0f} pixels²)<br>
                                    🚨 Recommendation: Urgent consultation recommended
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Download report
                        report_text = f"""LUNG NODULE DETECTION REPORT
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image: {uploaded_file.name}
Dimensions: {image_array.shape[1]}×{image_array.shape[0]} pixels
Nodules Detected: {len(nodules)}

{'='*50}
MEASUREMENTS:
{'='*50}

"""
                        for nodule in nodules:
                            report_text += f"""
Nodule #{nodule['id']}:
  - Area: {nodule['area_pixels']:.0f} pixels²
  - Diameter: {nodule['diameter_pixels']:.1f} pixels
  - Location (X, Y): ({nodule['centroid_x']:.0f}, {nodule['centroid_y']:.0f})
  - Bounding Box: Rows {nodule['min_row']}-{nodule['max_row']}, Cols {nodule['min_col']}-{nodule['max_col']}

"""
                        
                        report_text += f"""
{'='*50}
CLINICAL RECOMMENDATIONS:
{'='*50}
"""
                        for nodule in nodules:
                            if nodule['area_pixels'] < 100:
                                report_text += f"\nNodule #{nodule['id']}: Routine follow-up in 12 months"
                            elif nodule['area_pixels'] < 300:
                                report_text += f"\nNodule #{nodule['id']}: Short-term follow-up in 6 months"
                            else:
                                report_text += f"\nNodule #{nodule['id']}: Urgent consultation recommended"
                        
                        st.download_button(
                            "📄 Download Clinical Report",
                            report_text,
                            f"lung_nodule_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            use_container_width=True
                        )
                        
                    else:
                        st.info("✓ No nodules detected in this CT slice")
                        # Show the mask anyway
                        _, mask = segment_nodule(model, image_array)
                        if mask is not None and mask.max() > 0:
                            st.image(mask, caption="Segmentation Mask (No significant nodules found)", use_container_width=True)
    
    else:  # Full CT Volume
        uploaded_zip = st.file_uploader(
            "Upload CT Volume (ZIP with MHD/RAW files)",
            type=["zip"],
            help="Upload a ZIP file containing both .mhd and .raw files"
        )
        
        if uploaded_zip:
            with st.spinner("Loading CT volume..."):
                volume, spacing = load_ct_volume(uploaded_zip)
            
            if volume is not None:
                st.success(f"✅ Volume loaded successfully!")
                
                # Display volume info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Slices", volume.shape[0])
                with col2:
                    st.metric("Dimensions", f"{volume.shape[1]}×{volume.shape[2]}")
                with col3:
                    if spacing:
                        st.metric("Pixel Spacing", f"{spacing[0]:.3f} mm")
                
                # Preview middle slice with lung window
                mid_slice = volume[volume.shape[0] // 2]
                display_slice = apply_lung_window(mid_slice)
                st.image(display_slice, caption=f"Preview - Middle Slice ({volume.shape[0]//2})", use_container_width=True, clamp=True)
                
                # Slice range selector
                st.markdown("### Analysis Range")
                slice_range = st.slider(
                    "Select slice range to analyze",
                    0, volume.shape[0] - 1,
                    (0, min(100, volume.shape[0] - 1))
                )
                
                if st.button("🔍 Analyze Full Volume", type="primary"):
                    all_nodules = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_slices = slice_range[1] - slice_range[0] + 1
                    
                    for i in range(slice_range[0], slice_range[1] + 1):
                        status_text.text(f"Analyzing slice {i+1} of {volume.shape[0]}...")
                        
                        slice_img = volume[i]
                        
                        # Segment
                        nodules, _ = segment_nodule(model, slice_img)
                        
                        for nodule in nodules:
                            # Convert to physical units if spacing available
                            if spacing:
                                area_mm2 = nodule['area_pixels'] * (spacing[0] ** 2)
                                diameter_mm = nodule['diameter_pixels'] * spacing[0]
                            else:
                                area_mm2 = nodule['area_pixels']
                                diameter_mm = nodule['diameter_pixels']
                            
                            all_nodules.append({
                                'Slice': i,
                                'Area (pixels²)': f"{nodule['area_pixels']:.0f}",
                                'Diameter (pixels)': f"{nodule['diameter_pixels']:.1f}",
                                'Area (mm²)': f"{area_mm2:.1f}" if spacing else "N/A",
                                'Diameter (mm)': f"{diameter_mm:.1f}" if spacing else "N/A",
                                'Centroid X': f"{nodule['centroid_x']:.0f}",
                                'Centroid Y': f"{nodule['centroid_y']:.0f}"
                            })
                        
                        progress_bar.progress((i - slice_range[0] + 1) / total_slices)
                    
                    status_text.text("Analysis complete!")
                    
                    if all_nodules:
                        st.markdown(f'<div class="success-box">✅ <strong>{len(all_nodules)} nodules detected</strong> across {total_slices} slices</div>', unsafe_allow_html=True)
                        
                        # Display results
                        df = pd.DataFrame(all_nodules)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary stats
                        st.markdown("### 📊 Summary Statistics")
                        
                        # Extract numeric values for stats
                        areas = [float(n['Area (pixels²)']) for n in all_nodules]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Nodules", len(all_nodules))
                        with col2:
                            st.metric("Avg Area", f"{np.mean(areas):.0f} pixels²")
                        with col3:
                            st.metric("Largest Nodule", f"{np.max(areas):.0f} pixels²")
                        
                        # Download CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📊 Download Results CSV",
                            csv,
                            f"volume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            use_container_width=True
                        )
                    else:
                        st.info("✓ No nodules detected in this CT volume")
            else:
                st.error("❌ Failed to load volume. Please ensure your ZIP contains valid MHD and RAW files.")
    
    # Footer
    st.markdown("""
    <div class="clinical-footer">
        <p>LungVision AI | Clinical Decision Support System | HIT500 Capstone Project</p>
        <p>⚠️ Always verify AI results with clinical expertise</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
