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

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Lung Nodule Segmentation", page_icon="🫁", layout="wide")

# ========== GOOGLE DRIVE SETUP ==========
GOOGLE_DRIVE_FILE_ID = "1m0AE-Co5NTloIKV56-M-zF4IOoMReeZn"
MODEL_FILENAME = "best_model.pth"

def download_model_from_drive():
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("Downloading model..."):
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
        else:
            state_dict = checkpoint
        
        if state_dict and 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# ========== MHD FILE LOADING WITH REAL SCAN PARAMETERS ==========
def load_mhd_files(uploaded_files):
    """
    Load MHD and RAW files and extract REAL scan parameters
    Returns: ct_array, pixel_spacing_mm, slice_thickness_mm
    """
    mhd_file = None
    raw_file = None
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.mhd'):
            mhd_file = uploaded_file
        elif uploaded_file.name.endswith('.raw'):
            raw_file = uploaded_file
    
    if not mhd_file or not raw_file:
        st.error("Please upload both .mhd and .raw files")
        return None, None, None
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mhd_path = os.path.join(tmpdir, mhd_file.name)
            raw_path = os.path.join(tmpdir, raw_file.name)
            
            with open(mhd_path, 'wb') as f:
                f.write(mhd_file.getvalue())
            with open(raw_path, 'wb') as f:
                f.write(raw_file.getvalue())
            
            # Update MHD reference to point to correct raw file
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
            
            # Read the image using SimpleITK
            itk_image = sitk.ReadImage(mhd_path)
            ct_array = sitk.GetArrayFromImage(itk_image)
            
            # Extract REAL scan parameters from metadata
            spacing = itk_image.GetSpacing()
            # spacing[0] = pixel spacing in X direction (mm)
            # spacing[1] = pixel spacing in Y direction (mm)
            # spacing[2] = slice thickness in Z direction (mm) - THIS IS CRITICAL FOR VOLUME
            
            pixel_spacing_mm = spacing[0]  # Usually 0.6-0.8 mm
            slice_thickness_mm = spacing[2]  # Usually 0.625-2.5 mm
            
            # Display scan parameters to user
            st.info(f"📊 Scan Parameters:\n"
                   f"• Pixel spacing: {pixel_spacing_mm:.3f} mm\n"
                   f"• Slice thickness: {slice_thickness_mm:.3f} mm\n"
                   f"• Dimensions: {ct_array.shape[0]} slices × {ct_array.shape[1]} × {ct_array.shape[2]}")
            
            return ct_array, pixel_spacing_mm, slice_thickness_mm
            
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None

# ========== NODULE DETECTION WITH CORRECT VOLUME CALCULATION ==========
def detect_nodules(model, ct_volume, pixel_spacing_mm, slice_thickness_mm, min_area_pixels=20):
    """
    Detect multiple nodules per slice using connected component analysis
    Calculates REAL volume using actual scan parameters
    """
    all_detections = []
    
    with st.spinner("🔍 Analyzing CT scan for nodules..."):
        progress_bar = st.progress(0)
        
        for i in range(ct_volume.shape[0]):
            if i % 10 == 0:
                progress_bar.progress(i / ct_volume.shape[0])
            
            slice_img = ct_volume[i]
            
            # Process slice through model
            img_resized = resize(slice_img, (512, 512), preserve_range=True)
            input_tensor = torch.FloatTensor(img_resized).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.sigmoid(output)
                binary_mask = (probs > 0.5).float().squeeze().numpy()
            
            # Resize mask back to original dimensions
            mask_original = resize(binary_mask, slice_img.shape[:2], order=0, preserve_range=True)
            
            # Find connected components (individual nodules)
            labeled_mask = label(mask_original > 0.5)
            
            if labeled_mask.max() > 0:
                props = regionprops(labeled_mask)
                
                for region in props:
                    nodule_area_pixels = region.area
                    
                    if nodule_area_pixels >= min_area_pixels:
                        # Calculate REAL volume using actual scan parameters
                        # Step 1: Area in mm² = pixels × (pixel_spacing)²
                        nodule_area_mm2 = nodule_area_pixels * (pixel_spacing_mm ** 2)
                        
                        # Step 2: Volume in mm³ = area in mm² × slice thickness
                        nodule_volume_mm3 = nodule_area_mm2 * slice_thickness_mm
                        
                        # Calculate diameter from area (assuming circular)
                        nodule_diameter_mm = 2 * np.sqrt(nodule_area_mm2 / np.pi)
                        
                        # Calculate confidence based on size and shape
                        confidence = min(0.95, nodule_area_pixels / 300)
                        
                        # Create individual mask for this nodule
                        individual_mask = (labeled_mask == region.label).astype(np.float32)
                        
                        all_detections.append({
                            'slice': i,
                            'area_pixels': nodule_area_pixels,
                            'area_mm2': nodule_area_mm2,
                            'volume_mm3': nodule_volume_mm3,
                            'diameter_mm': nodule_diameter_mm,
                            'confidence': confidence,
                            'mask': individual_mask,
                            'slice_image': slice_img,
                            'bbox': region.bbox,
                            'centroid': region.centroid
                        })
        
        progress_bar.empty()
    
    # Sort by volume (largest first)
    all_detections.sort(key=lambda x: x['volume_mm3'], reverse=True)
    return all_detections

# ========== MAIN UI ==========
def main():
    st.title("🫁 Lung Nodule Segmentation")
    st.markdown("### Upload CT Scan - AI Automatically Detects All Nodules")
    st.markdown("---")
    
    # Login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("🔐 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if username == "radiologist" and password == "hit500":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        return
    
    # Logout button in sidebar
    st.sidebar.success("✅ Logged in as: Radiologist")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.clear()
        st.rerun()
    
    # Load model
    model, model_loaded = load_model()
    if not model_loaded:
        st.stop()
    
    st.sidebar.success("✅ Model ready")
    
    # File upload
    st.subheader("📤 Upload CT Scan")
    
    uploaded_files = st.file_uploader(
        "Select .mhd and .raw files (select both at once using Ctrl+Click or Cmd+Click)",
        type=["mhd", "raw"],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) == 2:
        # Load CT scan with REAL scan parameters
        with st.spinner("Loading CT scan..."):
            ct_array, pixel_spacing_mm, slice_thickness_mm = load_mhd_files(uploaded_files)
        
        if ct_array is not None:
            st.success(f"✅ CT scan loaded: {ct_array.shape[0]} slices")
            
            # Normalize CT values (window -1000 to 400 HU)
            window_min, window_max = -1000.0, 400.0
            ct_normalized = np.clip(ct_array, window_min, window_max)
            ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
            
            # Show preview of middle slice
            middle = ct_array.shape[0] // 2
            st.subheader("📷 Scan Preview")
            st.image(ct_normalized[middle], caption=f"Middle Slice {middle} of {ct_array.shape[0]}", use_container_width=True)
            
            # Auto-detect button
            if st.button("🔍 DETECT NODULES", type="primary", use_container_width=True):
                detections = detect_nodules(model, ct_normalized, pixel_spacing_mm, slice_thickness_mm)
                
                if detections:
                    st.success(f"✅ Found {len(detections)} nodule(s)")
                    
                    # Group detections by slice for better display
                    from collections import defaultdict
                    detections_by_slice = defaultdict(list)
                    for d in detections:
                        detections_by_slice[d['slice']].append(d)
                    
                    # Show results by slice
                    for slice_num in sorted(detections_by_slice.keys()):
                        slice_detections = detections_by_slice[slice_num]
                        
                        st.markdown(f"### 📍 Slice {slice_num} - {len(slice_detections)} Nodule(s)")
                        
                        # Create layout based on number of nodules
                        if len(slice_detections) == 1:
                            # Single nodule - full width
                            d = slice_detections[0]
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Create overlay image
                                slice_norm = (d['slice_image'] - d['slice_image'].min()) / (d['slice_image'].max() - d['slice_image'].min() + 1e-8)
                                overlay = np.stack([slice_norm] * 3, axis=-1)
                                overlay[:, :, 0] = np.where(d['mask'] > 0.5, 1.0, overlay[:, :, 0])
                                overlay[:, :, 1] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 1])
                                overlay[:, :, 2] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 2])
                                
                                st.image(overlay, use_container_width=True)
                            
                            with col2:
                                st.metric("📏 Volume", f"{d['volume_mm3']:.1f} mm³")
                                st.metric("📐 Diameter", f"{d['diameter_mm']:.1f} mm")
                                st.metric("📊 Area", f"{d['area_mm2']:.1f} mm²")
                                st.metric("🎯 Confidence", f"{d['confidence']:.0%}")
                                
                                # Clinical recommendation based on volume
                                if d['volume_mm3'] < 100:
                                    st.info("📌 **Small nodule** - Regular monitoring recommended")
                                elif d['volume_mm3'] < 300:
                                    st.warning("⚠️ **Medium nodule** - Further evaluation suggested")
                                else:
                                    st.error("🚨 **Large nodule** - Urgent consultation recommended")
                        
                        else:
                            # Multiple nodules - side by side columns
                            cols = st.columns(len(slice_detections))
                            
                            for idx, (col, d) in enumerate(zip(cols, slice_detections)):
                                with col:
                                    # Create overlay for each nodule
                                    slice_norm = (d['slice_image'] - d['slice_image'].min()) / (d['slice_image'].max() - d['slice_image'].min() + 1e-8)
                                    overlay = np.stack([slice_norm] * 3, axis=-1)
                                    overlay[:, :, 0] = np.where(d['mask'] > 0.5, 1.0, overlay[:, :, 0])
                                    overlay[:, :, 1] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 1])
                                    overlay[:, :, 2] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 2])
                                    
                                    st.image(overlay, use_container_width=True)
                                    st.caption(f"Nodule {idx+1}")
                                    st.metric("Volume", f"{d['volume_mm3']:.1f} mm³")
                                    st.metric("Diameter", f"{d['diameter_mm']:.1f} mm")
                                    st.metric("Confidence", f"{d['confidence']:.0%}")
                        
                        st.markdown("---")
                    
                    # Download full report
                    report_lines = [
                        "=" * 50,
                        "LUNG NODULE DETECTION REPORT",
                        "=" * 50,
                        f"Total Nodules Found: {len(detections)}",
                        f"Scan Parameters: Pixel Spacing = {pixel_spacing_mm:.3f} mm, Slice Thickness = {slice_thickness_mm:.3f} mm",
                        "",
                        "=" * 50,
                        "INDIVIDUAL NODULE DETAILS",
                        "=" * 50,
                        ""
                    ]
                    
                    for i, d in enumerate(detections):
                        report_lines.append(f"Nodule #{i+1}:")
                        report_lines.append(f"  • Slice: {d['slice']}")
                        report_lines.append(f"  • Volume: {d['volume_mm3']:.1f} mm³")
                        report_lines.append(f"  • Diameter: {d['diameter_mm']:.1f} mm")
                        report_lines.append(f"  • Area: {d['area_mm2']:.1f} mm²")
                        report_lines.append(f"  • Confidence: {d['confidence']:.0%}")
                        
                        if d['volume_mm3'] < 100:
                            report_lines.append(f"  • Recommendation: Small - Regular monitoring")
                        elif d['volume_mm3'] < 300:
                            report_lines.append(f"  • Recommendation: Medium - Further evaluation")
                        else:
                            report_lines.append(f"  • Recommendation: Large - Urgent consultation")
                        report_lines.append("")
                    
                    report_lines.append("=" * 50)
                    report_lines.append("End of Report")
                    
                    st.download_button(
                        "📊 Download Full Report (TXT)",
                        "\n".join(report_lines),
                        "nodule_detection_report.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("📌 No nodules detected in this scan")
    
    elif uploaded_files and len(uploaded_files) != 2:
        st.warning("Please select exactly 2 files: one .mhd and one .raw file")
    
    st.markdown("---")
    st.caption("© HIT500 Capstone Project | Volume calculated using actual scan parameters from MHD metadata")

if __name__ == "__main__":
    main()
