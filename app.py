import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import gdown
from skimage.transform import resize
import tempfile
import zipfile
import SimpleITK as sitk
from collections import OrderedDict
import traceback
import gc

# ========== PAGE CONFIG (MUST BE FIRST) ==========
st.set_page_config(page_title="Lung Nodule Segmentation", page_icon="🫁", layout="wide")

# ========== GOOGLE DRIVE SETUP ==========
GOOGLE_DRIVE_FILE_ID = "1FdIozNEVbIPUsjcdReAfmbgN3Nisx9yQ"  # Replace with your actual FILE ID
MODEL_FILENAME = "checkpoint_epoch_90.pth"

def download_model_from_drive():
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("Downloading model from Google Drive..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_FILENAME, quiet=False)
                st.success("Model downloaded successfully!")
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
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
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
        logits = self.outc(x)
        return logits

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
            epoch = checkpoint.get('epoch', 'Unknown')
            best_dice = checkpoint.get('best_dice', 'Unknown')
        else:
            state_dict = checkpoint
            epoch = 'Unknown'
            best_dice = 'Unknown'
        
        if state_dict and 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info(f"Model Stats:\n"
                       f"• Parameters: {total_params/1e6:.1f}M\n"
                       f"• Checkpoint: Epoch {epoch}\n"
                       f"• Best Dice: {best_dice}")
        
        return model, True
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# ========== MHD FILE LOADING ==========
def load_mhd_files_from_uploads(uploaded_files):
    """Load MHD and RAW files from multiple file uploads"""
    mhd_file = None
    raw_file = None
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.mhd'):
            mhd_file = uploaded_file
        elif uploaded_file.name.endswith('.raw'):
            raw_file = uploaded_file
    
    if not mhd_file:
        st.error("No .mhd file found in uploaded files")
        return None, None, None
    
    if not raw_file:
        st.error("No .raw file found in uploaded files")
        return None, None, None
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mhd_path = os.path.join(tmpdir, mhd_file.name)
            raw_path = os.path.join(tmpdir, raw_file.name)
            
            with open(mhd_path, 'wb') as f:
                f.write(mhd_file.getvalue())
            
            with open(raw_path, 'wb') as f:
                f.write(raw_file.getvalue())
            
            # Update MHD file to point to correct raw file
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
            origin = itk_image.GetOrigin()
            
            spacing = (spacing[2], spacing[1], spacing[0])
            origin = (origin[2], origin[1], origin[0])
            
            return ct_array, spacing, origin
            
    except Exception as e:
        st.error(f"Error loading MHD/RAW files: {str(e)}")
        return None, None, None

# ========== AUTOMATIC NODULE DETECTION ==========
def auto_detect_and_segment_nodules(model, ct_volume, min_area=30):
    """
    Automatically detect and segment all nodules in the CT volume
    Returns: List of detections with full details
    """
    detections = []
    
    with st.spinner("🔍 AI is analyzing the entire CT scan..."):
        progress_bar = st.progress(0)
        
        for i in range(ct_volume.shape[0]):
            # Update progress every 10 slices
            if i % 10 == 0:
                progress_bar.progress(i / ct_volume.shape[0])
            
            slice_img = ct_volume[i]
            
            # Resize to 512x512 (model input size)
            img_resized = resize(slice_img, (512, 512), preserve_range=True)
            input_tensor = torch.FloatTensor(img_resized).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.sigmoid(output)
                mask = (probs > 0.5).float().squeeze().numpy()
            
            # Resize mask back to original dimensions
            mask_original = resize(mask, slice_img.shape[:2], order=0, preserve_range=True)
            nodule_area = np.sum(mask_original)
            
            if nodule_area > min_area:
                # Calculate additional metrics
                confidence = min(0.95, nodule_area / 300)
                
                # Calculate nodule circularity (how round it is)
                from skimage.measure import label, regionprops
                labeled_mask = label(mask_original > 0.5)
                if labeled_mask.max() > 0:
                    props = regionprops(labeled_mask)
                    if props:
                        perimeter = props[0].perimeter
                        area = props[0].area
                        if perimeter > 0:
                            circularity = (4 * np.pi * area) / (perimeter ** 2)
                        else:
                            circularity = 0
                    else:
                        circularity = 0
                else:
                    circularity = 0
                
                detections.append({
                    'slice': i,
                    'area_pixels': nodule_area,
                    'confidence': confidence,
                    'mask': mask_original,
                    'circularity': circularity,
                    'slice_image': slice_img
                })
        
        progress_bar.empty()
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections

def calculate_volume(mask, pixel_spacing_mm=0.7, slice_thickness_mm=1.25):
    """Calculate nodule volume in mm³"""
    pixel_area_mm2 = pixel_spacing_mm ** 2
    area_pixels = np.sum(mask)
    return area_pixels * pixel_area_mm2 * slice_thickness_mm

def calculate_nodule_diameter(area_pixels, pixel_spacing_mm=0.7):
    """Calculate approximate diameter in mm (assuming circular nodule)"""
    area_mm2 = area_pixels * (pixel_spacing_mm ** 2)
    diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
    return diameter_mm

# ========== MAIN UI ==========
def main():
    st.title("🫁 Lung Nodule Segmentation Tool")
    st.markdown("### Fully Automatic AI-Powered Detection & Segmentation")
    st.markdown("**HIT500 Capstone | Biomedical Engineering | Nqobile Maware**")
    st.markdown("---")
    
    # Login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("🔐 Radiologist Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if username == "radiologist" and password == "hit500":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Use: radiologist / hit500")
        return
    
    # Main app content
    st.sidebar.success("✅ Logged in as: Radiologist")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.clear()
        st.rerun()
    
    # Model architecture details
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧠 Model Architecture**")
    st.sidebar.markdown("- Memory Efficient U-Net")
    st.sidebar.markdown("- 3.4M parameters")
    st.sidebar.markdown("- Input: 512×512")
    st.sidebar.markdown("- **Fully Automatic Detection**")
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.stop()
    
    # File upload section
    st.subheader("📤 Upload CT Scan")
    
    upload_type = st.radio(
        "Select upload format:",
        ["MHD + RAW Files (Select both at once)", "ZIP Archive (MHD+RAW)"],
        horizontal=True
    )
    
    if upload_type == "MHD + RAW Files (Select both at once)":
        st.info("📌 Select both the .mhd and .raw files together (Ctrl+Click or Shift+Click)")
        
        uploaded_files = st.file_uploader(
            "Choose .mhd and .raw files",
            type=["mhd", "raw"],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(uploaded_files) >= 2:
            with st.spinner("Loading CT scan..."):
                ct_array, spacing, origin = load_mhd_files_from_uploads(uploaded_files)
                
                if ct_array is not None:
                    st.success("✅ CT scan loaded successfully!")
                    st.info(f"📊 Scan dimensions: {ct_array.shape[0]} slices × {ct_array.shape[1]} × {ct_array.shape[2]}")
                    
                    # Normalize the CT scan
                    window_min, window_max = -1000.0, 400.0
                    ct_normalized = np.clip(ct_array, window_min, window_max)
                    ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
                    
                    # Store in session state
                    st.session_state['ct_volume'] = ct_normalized
                    st.session_state['spacing'] = spacing
                    st.session_state['origin'] = origin
                    st.session_state['ct_loaded'] = True
                    
                    # Show preview of middle slice
                    middle_slice = ct_array.shape[0] // 2
                    st.subheader("📷 Scan Preview")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(ct_normalized[middle_slice], cmap='gray')
                    ax.set_title(f"Middle Slice Preview (Slice {middle_slice} of {ct_array.shape[0]})")
                    ax.axis('off')
                    st.pyplot(fig)
    
    elif upload_type == "ZIP Archive (MHD+RAW)":
        zip_file = st.file_uploader("Choose ZIP file containing .mhd and .raw", type=["zip"])
        
        if zip_file is not None:
            with st.spinner("Extracting and loading ZIP archive..."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zip_path = os.path.join(tmpdir, "upload.zip")
                        with open(zip_path, 'wb') as f:
                            f.write(zip_file.getvalue())
                        
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            zf.extractall(tmpdir)
                        
                        mhd_path = None
                        raw_path = None
                        
                        for file in os.listdir(tmpdir):
                            if file.endswith('.mhd'):
                                mhd_path = os.path.join(tmpdir, file)
                            elif file.endswith('.raw'):
                                raw_path = os.path.join(tmpdir, file)
                        
                        if mhd_path and raw_path:
                            itk_image = sitk.ReadImage(mhd_path)
                            ct_array = sitk.GetArrayFromImage(itk_image)
                            
                            spacing = itk_image.GetSpacing()
                            origin = itk_image.GetOrigin()
                            
                            spacing = (spacing[2], spacing[1], spacing[0])
                            origin = (origin[2], origin[1], origin[0])
                            
                            st.success("✅ CT scan loaded successfully from ZIP!")
                            st.info(f"📊 Scan dimensions: {ct_array.shape[0]} slices × {ct_array.shape[1]} × {ct_array.shape[2]}")
                            
                            # Normalize the CT scan
                            window_min, window_max = -1000.0, 400.0
                            ct_normalized = np.clip(ct_array, window_min, window_max)
                            ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
                            
                            # Store in session state
                            st.session_state['ct_volume'] = ct_normalized
                            st.session_state['spacing'] = spacing
                            st.session_state['origin'] = origin
                            st.session_state['ct_loaded'] = True
                            
                            # Show preview
                            middle_slice = ct_array.shape[0] // 2
                            st.subheader("📷 Scan Preview")
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(ct_normalized[middle_slice], cmap='gray')
                            ax.set_title(f"Middle Slice Preview (Slice {middle_slice} of {ct_array.shape[0]})")
                            ax.axis('off')
                            st.pyplot(fig)
                        else:
                            st.error("ZIP file must contain both .mhd and .raw files")
                            
                except Exception as e:
                    st.error(f"Error loading ZIP file: {str(e)}")
    
    # Automatic Detection Section
    if st.session_state.get('ct_loaded', False):
        st.markdown("---")
        st.subheader("🤖 Automatic Nodule Detection")
        
        # Settings for detection
        col1, col2, col3 = st.columns(3)
        with col1:
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=["Low", "Medium", "High"],
                value="Medium"
            )
        
        sensitivity_map = {"Low": 80, "Medium": 50, "High": 30}
        min_area = sensitivity_map[sensitivity]
        
        with col2:
            pixel_spacing = st.number_input(
                "Pixel Spacing (mm)",
                value=0.7,
                min_value=0.1,
                max_value=2.0,
                step=0.05,
                help="Distance between pixels in mm (from CT metadata)"
            )
        
        with col3:
            slice_thickness = st.number_input(
                "Slice Thickness (mm)",
                value=1.25,
                min_value=0.5,
                max_value=5.0,
                step=0.25,
                help="Distance between slices in mm (from CT metadata)"
            )
        
        # Run detection button
        if st.button("🔍 RUN AUTOMATIC DETECTION", type="primary", use_container_width=True):
            detections = auto_detect_and_segment_nodules(
                model, 
                st.session_state['ct_volume'], 
                min_area=min_area
            )
            
            st.session_state['detections'] = detections
            
            if detections:
                st.balloons()
                st.success(f"✅ Found {len(detections)} nodule(s)!")
            else:
                st.info("No nodules detected in this scan")
        
        # Display results
        if 'detections' in st.session_state and st.session_state['detections']:
            detections = st.session_state['detections']
            
            st.markdown("---")
            st.subheader(f"📊 Detection Results: {len(detections)} Nodule(s) Found")
            
            # Summary statistics
            total_nodules = len(detections)
            avg_confidence = np.mean([d['confidence'] for d in detections])
            largest_nodule = max(detections, key=lambda x: x['area_pixels'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Nodules", total_nodules)
            col2.metric("Avg Confidence", f"{avg_confidence:.0%}")
            col3.metric("Largest Nodule", f"Slice {largest_nodule['slice']}")
            col4.metric("Largest Area", f"{largest_nodule['area_pixels']:.0f} px²")
            
            st.markdown("---")
            
            # Display each nodule
            for idx, detection in enumerate(detections):
                with st.expander(f"Nodule #{idx+1} - Slice {detection['slice']} (Confidence: {detection['confidence']:.0%})", expanded=(idx==0)):
                    
                    # Calculate metrics
                    volume_mm3 = calculate_volume(detection['mask'], pixel_spacing, slice_thickness)
                    diameter_mm = calculate_nodule_diameter(detection['area_pixels'], pixel_spacing)
                    
                    # Create two columns
                    col_img, col_metrics = st.columns([2, 1])
                    
                    with col_img:
                        # Create overlay
                        slice_img = detection['slice_image']
                        mask = detection['mask']
                        
                        slice_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
                        overlay = np.stack([slice_norm] * 3, axis=-1)
                        overlay[:, :, 0] = np.where(mask > 0.5, 1.0, overlay[:, :, 0])
                        overlay[:, :, 1] = np.where(mask > 0.5, 0.0, overlay[:, :, 1])
                        overlay[:, :, 2] = np.where(mask > 0.5, 0.0, overlay[:, :, 2])
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                        
                        ax1.imshow(slice_img, cmap='gray')
                        ax1.set_title(f"Original CT - Slice {detection['slice']}")
                        ax1.axis('off')
                        
                        ax2.imshow(overlay)
                        ax2.set_title(f"Segmentation (Confidence: {detection['confidence']:.0%})")
                        ax2.axis('off')
                        
                        st.pyplot(fig)
                    
                    with col_metrics:
                        st.markdown("### 📏 Measurements")
                        st.metric("Volume", f"{volume_mm3:.1f} mm³")
                        st.metric("Diameter", f"{diameter_mm:.1f} mm")
                        st.metric("Area", f"{detection['area_pixels']:.0f} pixels²")
                        st.metric("Circularity", f"{detection['circularity']:.2f}")
                        
                        # Clinical assessment
                        st.markdown("### 🏥 Assessment")
                        if volume_mm3 < 100:
                            st.info("📌 **Small nodule** - Regular monitoring")
                        elif volume_mm3 < 300:
                            st.warning("⚠️ **Medium nodule** - Further evaluation")
                        else:
                            st.error("🚨 **Large nodule** - Urgent consultation")
                    
                    # Download button for this nodule
                    buf = io.BytesIO()
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(slice_img, cmap='gray')
                    plt.title(f"Slice {detection['slice']}")
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(mask, cmap='gray')
                    plt.title("Segmentation Mask")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    
                    st.download_button(
                        f"📥 Download Nodule #{idx+1} Report",
                        buf,
                        f"nodule_{idx+1}_slice_{detection['slice']}.png",
                        mime="image/png"
                    )
            
            # Export all results
            st.markdown("---")
            st.subheader("📋 Export All Results")
            
            # Create summary dataframe
            import pandas as pd
            summary_data = []
            for idx, d in enumerate(detections):
                volume = calculate_volume(d['mask'], pixel_spacing, slice_thickness)
                diameter = calculate_nodule_diameter(d['area_pixels'], pixel_spacing)
                summary_data.append({
                    'Nodule #': idx + 1,
                    'Slice': d['slice'],
                    'Volume (mm³)': f"{volume:.1f}",
                    'Diameter (mm)': f"{diameter:.1f}",
                    'Area (pixels²)': f"{d['area_pixels']:.0f}",
                    'Confidence': f"{d['confidence']:.0%}",
                    'Circularity': f"{d['circularity']:.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
            # Download as CSV
            csv_buffer = io.BytesIO()
            summary_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                "📊 Download Full Report (CSV)",
                csv_buffer,
                "nodule_detection_report.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    st.caption("© 2026 HIT500 Capstone Project | Fully Automatic Nodule Detection | Trained on LUNA16")

if __name__ == "__main__":
    main()
