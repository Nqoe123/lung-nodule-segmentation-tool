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
import io

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
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("Downloading AI model from Google Drive..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_FILENAME, quiet=False)
        
        model = MemoryEfficientUNet(n_channels=1, n_classes=1)
        checkpoint = torch.load(MODEL_FILENAME, map_location=torch.device('cpu'))
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
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
        st.error(f"Error loading model: {str(e)}")
        return None

# ========== IMAGE PROCESSING ==========

def load_and_display_image(uploaded_file):
    """Load image and return properly scaled for display"""
    try:
        image = Image.open(uploaded_file).convert('L')
        image_array = np.array(image, dtype=np.float32)
        
        # For display: stretch contrast to full range
        display_img = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
        
        return image_array, display_img
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None, None

def preprocess_for_model(image_array):
    """Normalize image for model input - exactly as trained"""
    # Normalize to [0, 1] range
    img_norm = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    return img_norm

def segment_nodule(model, image_array):
    """Run segmentation on a single image"""
    try:
        original_shape = image_array.shape
        
        # Preprocess
        model_input = preprocess_for_model(image_array)
        
        # Resize to 512x512
        resized = resize(model_input, (512, 512), preserve_range=True)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
        
        # Run model
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            mask = (probs > 0.5).float().squeeze().numpy()
        
        # Resize mask back
        mask_resized = resize(mask, original_shape, order=0, preserve_range=True)
        
        # Find nodules
        labeled = label(mask_resized > 0.5)
        nodules = []
        
        for region in regionprops(labeled):
            if region.area >= 20:
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
        return [], None

def create_overlay_with_circles(image_array, nodules, spacing=None):
    """Create RGB overlay with red circles highlighting nodules"""
    # Normalize image for display
    img_norm = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_norm, cmap='gray')
    
    # Add circles for each nodule
    for nodule in nodules:
        # Calculate radius in pixels
        radius = nodule['diameter_pixels'] / 2
        
        # Create circle
        circle = Circle(
            (nodule['centroid_x'], nodule['centroid_y']), 
            radius, 
            fill=False, 
            edgecolor='red', 
            linewidth=3,
            label=f"Nodule {nodule['id']}"
        )
        ax.add_patch(circle)
        
        # Add label
        ax.text(
            nodule['centroid_x'] + radius/2, 
            nodule['centroid_y'] - radius/2, 
            f"Nodule {nodule['id']}", 
            color='red',
            fontsize=12,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7)
        )
    
    ax.set_title(f"Detected Nodules ({len(nodules)} found)")
    ax.axis('off')
    
    return fig

def load_volume_fast(zip_file):
    """Load MHD/RAW volume with progress indication"""
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "upload.zip")
    
    # Save uploaded zip
    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find files
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
        
        # Clean up temp directory later
        return volume, spacing, temp_dir
    
    return None, None, None

# ========== MAIN APP ==========
def main():
    st.markdown("""
    <div class="clinical-header">
        <h1>🫁 LungVision AI</h1>
        <p>Clinical Lung Nodule Detection System</p>
        <small>For Radiologist Use Only</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Login
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
    
    with st.sidebar:
        st.success("👤 Radiologist - Active")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown("- **Architecture:** Memory Efficient U-Net")
        st.markdown("- **Training:** LUNA16")
        st.markdown("- **Input Size:** 512×512")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model_from_drive()
    
    if model is None:
        st.stop()
    
    st.markdown("### 📤 Upload Study")
    
    upload_type = st.radio(
        "Select examination type:",
        ["Single CT Slice", "Full CT Volume"],
        horizontal=True
    )
    
    if upload_type == "Single CT Slice":
        uploaded_file = st.file_uploader(
            "Upload CT Slice (PNG/JPG)",
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded_file:
            # Load image
            original_img, display_img = load_and_display_image(uploaded_file)
            
            if original_img is not None:
                # Show image
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(display_img, cmap='gray')
                ax.set_title(f"CT Slice - {original_img.shape[1]}×{original_img.shape[0]}")
                ax.axis('off')
                st.pyplot(fig)
                
                # Analyze button
                if st.button("🔍 Detect Nodules", type="primary"):
                    with st.spinner("Analyzing..."):
                        nodules, mask = segment_nodule(model, original_img)
                    
                    if nodules:
                        st.markdown(f'<div class="success-box">✅ <strong>{len(nodules)} Nodule(s) Detected</strong></div>', unsafe_allow_html=True)
                        
                        # Show overlay with circles
                        overlay_fig = create_overlay_with_circles(original_img, nodules)
                        st.pyplot(overlay_fig)
                        
                        # Results table
                        st.markdown("### 📊 Measurements")
                        results = []
                        for n in nodules:
                            results.append({
                                "Nodule": n['id'],
                                "Area (pixels²)": f"{n['area_pixels']:.0f}",
                                "Diameter (pixels)": f"{n['diameter_pixels']:.1f}",
                                "X": f"{n['centroid_x']:.0f}",
                                "Y": f"{n['centroid_y']:.0f}"
                            })
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                        
                        # Clinical Recommendations
                        st.markdown("### 🩺 Clinical Recommendations")
                        for n in nodules:
                            if n['area_pixels'] < 100:
                                st.markdown(f"""
                                <div class="info-box">
                                    <strong>Nodule {n['id']}</strong> - Size: {n['area_pixels']:.0f} pixels²<br>
                                    📌 <strong>Recommendation:</strong> Routine follow-up in 12 months<br>
                                    <small>Low risk, standard monitoring protocol</small>
                                </div>
                                """, unsafe_allow_html=True)
                            elif n['area_pixels'] < 300:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <strong>Nodule {n['id']}</strong> - Size: {n['area_pixels']:.0f} pixels²<br>
                                    ⚠️ <strong>Recommendation:</strong> Short-term follow-up in 6 months<br>
                                    <small>Moderate risk, closer monitoring required</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <strong>Nodule {n['id']}</strong> - Size: {n['area_pixels']:.0f} pixels²<br>
                                    🚨 <strong>Recommendation:</strong> Urgent consultation recommended<br>
                                    <small>High risk, immediate clinical evaluation needed</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Download report
                        report = f"""LUNG NODULE DETECTION REPORT
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image: {uploaded_file.name}
Nodules Detected: {len(nodules)}

MEASUREMENTS:
"""
                        for n in nodules:
                            report += f"""
Nodule {n['id']}:
  - Area: {n['area_pixels']:.0f} pixels²
  - Diameter: {n['diameter_pixels']:.1f} pixels
  - Location: ({n['centroid_x']:.0f}, {n['centroid_y']:.0f})

"""
                        
                        report += "\nRECOMMENDATIONS:\n"
                        for n in nodules:
                            if n['area_pixels'] < 100:
                                report += f"Nodule {n['id']}: Routine follow-up in 12 months\n"
                            elif n['area_pixels'] < 300:
                                report += f"Nodule {n['id']}: Short-term follow-up in 6 months\n"
                            else:
                                report += f"Nodule {n['id']}: Urgent consultation recommended\n"
                        
                        st.download_button("📄 Download Clinical Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    else:
                        st.info("✓ No nodules detected in this slice")
    
    else:  # Full CT Volume
        st.info("💡 Upload a ZIP file containing .mhd and .raw files")
        
        uploaded_zip = st.file_uploader(
            "Upload CT Volume (ZIP)",
            type=["zip"]
        )
        
        if uploaded_zip:
            with st.spinner("Loading volume (this may take a moment)..."):
                volume, spacing, temp_dir = load_volume_fast(uploaded_zip)
            
            if volume is not None:
                st.success(f"✅ Volume loaded: {volume.shape[0]} slices, {volume.shape[1]}×{volume.shape[2]}")
                if spacing:
                    st.info(f"Pixel spacing: {spacing[0]:.2f} mm")
                
                # Show preview
                mid = volume.shape[0] // 2
                preview = volume[mid]
                preview_norm = (preview - preview.min()) / (preview.max() - preview.min() + 1e-8)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(preview_norm, cmap='gray')
                ax.set_title(f"Preview - Slice {mid}")
                ax.axis('off')
                st.pyplot(fig)
                
                # Simple range selector
                col1, col2 = st.columns(2)
                with col1:
                    slice_start = st.number_input("Start slice", 0, volume.shape[0]-1, 0)
                with col2:
                    slice_end = st.number_input("End slice", 0, volume.shape[0]-1, min(100, volume.shape[0]-1))
                
                if st.button("🔍 Analyze Volume", type="primary"):
                    all_nodules = []
                    nodules_by_slice = {}
                    progress = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(slice_start, slice_end + 1):
                        status_text.text(f"Analyzing slice {i+1}/{volume.shape[0]}...")
                        nodules, _ = segment_nodule(model, volume[i])
                        
                        if nodules:
                            nodules_by_slice[i] = nodules
                            for n in nodules:
                                if spacing:
                                    diam_mm = n['diameter_pixels'] * spacing[0]
                                    area_mm2 = n['area_pixels'] * (spacing[0] ** 2)
                                else:
                                    diam_mm = n['diameter_pixels']
                                    area_mm2 = n['area_pixels']
                                
                                all_nodules.append({
                                    'Slice': i,
                                    'Nodule ID': n['id'],
                                    'Area (pixels²)': n['area_pixels'],
                                    'Area (mm²)': f"{area_mm2:.1f}" if spacing else "N/A",
                                    'Diameter (pixels)': f"{n['diameter_pixels']:.1f}",
                                    'Diameter (mm)': f"{diam_mm:.1f}" if spacing else "N/A",
                                    'Centroid X': f"{n['centroid_x']:.0f}",
                                    'Centroid Y': f"{n['centroid_y']:.0f}"
                                })
                        
                        progress.progress((i - slice_start + 1) / (slice_end - slice_start + 1))
                    
                    status_text.text("Analysis complete!")
                    
                    if all_nodules:
                        st.markdown(f'<div class="success-box">✅ <strong>{len(all_nodules)} Nodule(s) Detected</strong> across {slice_end - slice_start + 1} slices</div>', unsafe_allow_html=True)
                        
                        # Show overlay for slices with nodules
                        st.markdown("### 🔍 Visual Results - Slices with Nodules")
                        
                        # Select which slice to view
                        slice_with_nodules = list(nodules_by_slice.keys())
                        if slice_with_nodules:
                            selected_slice = st.selectbox(
                                "Select slice to view nodule overlay:",
                                slice_with_nodules,
                                format_func=lambda x: f"Slice {x}"
                            )
                            
                            # Show overlay for selected slice
                            slice_img = volume[selected_slice]
                            nodules_in_slice = nodules_by_slice[selected_slice]
                            
                            overlay_fig = create_overlay_with_circles(slice_img, nodules_in_slice, spacing)
                            st.pyplot(overlay_fig)
                        
                        # Results table
                        st.markdown("### 📊 Detailed Measurements")
                        df = pd.DataFrame(all_nodules)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("### 📈 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Nodules", len(all_nodules))
                        with col2:
                            if spacing:
                                areas_mm = [float(n['Area (mm²)']) for n in all_nodules if n['Area (mm²)'] != "N/A"]
                                if areas_mm:
                                    st.metric("Average Area", f"{np.mean(areas_mm):.1f} mm²")
                                else:
                                    st.metric("Average Area", f"{np.mean([n['Area (pixels²)'] for n in all_nodules]):.0f} pixels²")
                            else:
                                st.metric("Average Area", f"{np.mean([n['Area (pixels²)'] for n in all_nodules]):.0f} pixels²")
                        with col3:
                            if spacing:
                                diameters = [float(n['Diameter (mm)']) for n in all_nodules if n['Diameter (mm)'] != "N/A"]
                                if diameters:
                                    st.metric("Avg Diameter", f"{np.mean(diameters):.1f} mm")
                                else:
                                    st.metric("Avg Diameter", f"{np.mean([float(n['Diameter (pixels)']) for n in all_nodules]):.1f} pixels")
                            else:
                                st.metric("Avg Diameter", f"{np.mean([float(n['Diameter (pixels)']) for n in all_nodules]):.1f} pixels")
                        with col4:
                            if spacing:
                                largest = max([float(n['Diameter (mm)']) for n in all_nodules if n['Diameter (mm)'] != "N/A"])
                                st.metric("Largest Nodule", f"{largest:.1f} mm")
                            else:
                                largest = max([float(n['Diameter (pixels)']) for n in all_nodules])
                                st.metric("Largest Nodule", f"{largest:.1f} pixels")
                        
                        # Clinical Recommendations Summary
                        st.markdown("### 🩺 Clinical Summary & Recommendations")
                        
                        # Categorize nodules by size
                        small = [n for n in all_nodules if n['Area (pixels²)'] < 100]
                        medium = [n for n in all_nodules if 100 <= n['Area (pixels²)'] < 300]
                        large = [n for n in all_nodules if n['Area (pixels²)'] >= 300]
                        
                        if small:
                            st.markdown(f"""
                            <div class="info-box">
                                <strong>📌 Small Nodules ({len(small)} found):</strong><br>
                                Size &lt; 100 pixels²<br>
                                <strong>Recommendation:</strong> Routine follow-up in 12 months
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if medium:
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>⚠️ Medium Nodules ({len(medium)} found):</strong><br>
                                Size 100-300 pixels²<br>
                                <strong>Recommendation:</strong> Short-term follow-up in 6 months
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if large:
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>🚨 Large Nodules ({len(large)} found):</strong><br>
                                Size &gt; 300 pixels²<br>
                                <strong>Recommendation:</strong> Urgent consultation recommended
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Download CSV
                        csv = df.to_csv(index=False)
                        st.download_button("📊 Download Full Results (CSV)", csv, f"volume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                        
                        # Download Clinical Report
                        report = f"""LUNG NODULE DETECTION REPORT - FULL VOLUME
{'='*60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Volume Size: {volume.shape[0]} slices, {volume.shape[1]}×{volume.shape[2]} pixels
Pixel Spacing: {spacing[0]:.2f} mm (if available)
Slices Analyzed: {slice_start} to {slice_end}
Total Nodules Detected: {len(all_nodules)}

{'='*60}
DETAILED FINDINGS:
{'='*60}

"""
                        for n in all_nodules:
                            report += f"""
Slice {n['Slice']}, Nodule {n['Nodule ID']}:
  - Area: {n['Area (pixels²)']} pixels² {f'({n["Area (mm²)"]} mm²)' if n['Area (mm²)'] != "N/A" else ''}
  - Diameter: {n['Diameter (pixels)']} pixels {f'({n["Diameter (mm)"]} mm)' if n['Diameter (mm)'] != "N/A" else ''}
  - Location: ({n['Centroid X']}, {n['Centroid Y']})

"""
                        
                        report += f"""
{'='*60}
CLINICAL RECOMMENDATIONS:
{'='*60}

"""
                        if small:
                            report += f"\nSmall Nodules ({len(small)}): Routine follow-up in 12 months\n"
                        if medium:
                            report += f"\nMedium Nodules ({len(medium)}): Short-term follow-up in 6 months\n"
                        if large:
                            report += f"\nLarge Nodules ({len(large)}): Urgent consultation recommended\n"
                        
                        report += """
{'='*60}
DISCLAIMER:
This is an AI-assisted detection system. All findings should be verified 
by a qualified radiologist before clinical decision making.
{'='*60}
"""
                        
                        st.download_button("📄 Download Clinical Report (TXT)", report, f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                        
                    else:
                        st.info("✓ No nodules detected in this volume")
                
                # Cleanup temp directory
                if temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                st.error("Could not load volume. Make sure ZIP contains .mhd and .raw files")
    
    st.markdown("""
    <div class="clinical-footer">
        <p>LungVision AI | HIT500 Capstone Project | Always verify with clinical expertise</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
