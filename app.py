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
st.set_page_config(page_title="Lung Nodule Segmentation - Attention U-Net", page_icon="🫁", layout="wide")

# ========== GOOGLE DRIVE SETUP ==========
# REPLACE THESE WITH YOUR NEW MODEL'S INFO
GOOGLE_DRIVE_FILE_ID = "1m0AE-Co5NTloIKV56-M-zF4IOoMReeZn"  # Replace with your new model's file ID
MODEL_FILENAME = "best_model.pth"  # Replace with your new model's filename

def download_model_from_drive():
    try:
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("Downloading model from Google Drive..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_FILENAME, quiet=False)
        return MODEL_FILENAME
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        return None

# ========== ATTENTION U-NET ARCHITECTURE ==========
class AttentionBlock(nn.Module):
    """Attention gate for Attention U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

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
    def __init__(self, in_channels, out_channels, bilinear=False):
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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    """Attention U-Net - for your new trained model"""
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder with Attention
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.att1 = AttentionBlock(F_g=512 // factor, F_l=512, F_int=256)
        
        self.up2 = Up(512, 256 // factor, bilinear)
        self.att2 = AttentionBlock(F_g=256 // factor, F_l=256, F_int=128)
        
        self.up3 = Up(256, 128 // factor, bilinear)
        self.att3 = AttentionBlock(F_g=128 // factor, F_l=128, F_int=64)
        
        self.up4 = Up(128, 64, bilinear)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with Attention
        d4 = self.up1(x5, x4)
        d4 = self.att1(g=d4, x=x4)
        
        d3 = self.up2(d4, x3)
        d3 = self.att2(g=d3, x=x3)
        
        d2 = self.up3(d3, x2)
        d2 = self.att3(g=d2, x=x2)
        
        d1 = self.up4(d2, x1)
        d1 = self.att4(g=d1, x=x1)
        
        logits = self.outc(d1)
        return logits

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    try:
        model_path = download_model_from_drive()
        if model_path is None:
            return None, False
        
        # Use AttentionUNet for your new model
        model = AttentionUNet(n_channels=1, n_classes=1, bilinear=False)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'Unknown')
            best_dice = checkpoint.get('best_dice', 'Unknown')
        else:
            state_dict = checkpoint
            epoch = 'Unknown'
            best_dice = 'Unknown'
        
        # Handle DataParallel wrapper if present
        if state_dict and 'module.' in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        st.sidebar.success(f"✅ Attention U-Net loaded!")
        st.sidebar.info(f"Model Stats:\n"
                       f"• Parameters: {total_params/1e6:.1f}M\n"
                       f"• Checkpoint: Epoch {epoch}\n"
                       f"• Best Dice: {best_dice}\n"
                       f"• Architecture: Attention U-Net")
        
        return model, True
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# ========== MHD FILE LOADING ==========
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

# ========== NODULE DETECTION ==========
def detect_nodules(model, ct_volume, pixel_spacing_mm, slice_thickness_mm, min_area=20):
    detections = []
    
    with st.spinner("🔍 Analyzing CT scan with Attention U-Net..."):
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
                    nodule_area_pixels = region.area
                    
                    if nodule_area_pixels >= min_area:
                        nodule_area_mm2 = nodule_area_pixels * (pixel_spacing_mm ** 2)
                        nodule_volume_mm3 = nodule_area_mm2 * slice_thickness_mm
                        nodule_diameter_mm = 2 * np.sqrt(nodule_area_mm2 / np.pi)
                        confidence = min(0.95, nodule_area_pixels / 300)
                        
                        individual_mask = (labeled_mask == region.label).astype(np.float32)
                        
                        detections.append({
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
    
    detections.sort(key=lambda x: x['volume_mm3'], reverse=True)
    return detections

# ========== MAIN UI ==========
def main():
    st.title("🫁 Lung Nodule Segmentation Tool")
    st.markdown("### Attention U-Net - AI-Powered Detection")
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
    
    st.sidebar.success("✅ Logged in as: Radiologist")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.clear()
        st.rerun()
    
    # Load model
    model, model_loaded = load_model()
    if not model_loaded:
        st.stop()
    
    # File upload
    st.subheader("📤 Upload CT Scan")
    
    uploaded_files = st.file_uploader(
        "Select .mhd and .raw files (select both at once using Ctrl+Click or Cmd+Click)",
        type=["mhd", "raw"],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) == 2:
        with st.spinner("Loading CT scan..."):
            ct_array, pixel_spacing_mm, slice_thickness_mm = load_mhd_files(uploaded_files)
        
        if ct_array is not None:
            st.success(f"✅ CT scan loaded: {ct_array.shape[0]} slices")
            
            # Normalize CT values
            window_min, window_max = -1000.0, 400.0
            ct_normalized = np.clip(ct_array, window_min, window_max)
            ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
            
            # Show preview
            middle = ct_array.shape[0] // 2
            st.subheader("📷 Scan Preview")
            st.image(ct_normalized[middle], caption=f"Middle Slice {middle} of {ct_array.shape[0]}", use_container_width=True)
            
            # Auto-detect button
            if st.button("🔍 DETECT NODULES", type="primary", use_container_width=True):
                detections = detect_nodules(model, ct_normalized, pixel_spacing_mm, slice_thickness_mm)
                
                if detections:
                    st.success(f"✅ Found {len(detections)} nodule(s)")
                    
                    # Group by slice
                    from collections import defaultdict
                    detections_by_slice = defaultdict(list)
                    for d in detections:
                        detections_by_slice[d['slice']].append(d)
                    
                    # Show results
                    for slice_num in sorted(detections_by_slice.keys()):
                        slice_detections = detections_by_slice[slice_num]
                        st.markdown(f"### 📍 Slice {slice_num} - {len(slice_detections)} Nodule(s)")
                        
                        if len(slice_detections) == 1:
                            d = slice_detections[0]
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                slice_norm = (d['slice_image'] - d['slice_image'].min()) / (d['slice_image'].max() - d['slice_image'].min() + 1e-8)
                                overlay = np.stack([slice_norm] * 3, axis=-1)
                                overlay[:, :, 0] = np.where(d['mask'] > 0.5, 1.0, overlay[:, :, 0])
                                overlay[:, :, 1] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 1])
                                overlay[:, :, 2] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 2])
                                st.image(overlay, use_container_width=True)
                            
                            with col2:
                                st.metric("Volume", f"{d['volume_mm3']:.1f} mm³")
                                st.metric("Diameter", f"{d['diameter_mm']:.1f} mm")
                                st.metric("Confidence", f"{d['confidence']:.0%}")
                                
                                if d['volume_mm3'] < 100:
                                    st.info("📌 Small - Monitor")
                                elif d['volume_mm3'] < 300:
                                    st.warning("⚠️ Medium - Follow up")
                                else:
                                    st.error("🚨 Large - Urgent")
                        else:
                            cols = st.columns(len(slice_detections))
                            for idx, (col, d) in enumerate(zip(cols, slice_detections)):
                                with col:
                                    slice_norm = (d['slice_image'] - d['slice_image'].min()) / (d['slice_image'].max() - d['slice_image'].min() + 1e-8)
                                    overlay = np.stack([slice_norm] * 3, axis=-1)
                                    overlay[:, :, 0] = np.where(d['mask'] > 0.5, 1.0, overlay[:, :, 0])
                                    overlay[:, :, 1] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 1])
                                    overlay[:, :, 2] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 2])
                                    st.image(overlay, use_container_width=True)
                                    st.caption(f"Nodule {idx+1}")
                                    st.metric("Volume", f"{d['volume_mm3']:.1f} mm³")
                                    st.metric("Diameter", f"{d['diameter_mm']:.1f} mm")
                        
                        st.markdown("---")
                    
                    # Download report
                    report_lines = [f"Attention U-Net Detection Report", f"Total Nodules: {len(detections)}", ""]
                    for i, d in enumerate(detections):
                        report_lines.append(f"Nodule {i+1}: Slice {d['slice']}, Volume: {d['volume_mm3']:.1f}mm³, Diameter: {d['diameter_mm']:.1f}mm, Confidence: {d['confidence']:.0%}")
                    
                    st.download_button("📊 Download Report", "\n".join(report_lines), "nodule_report.txt")
                else:
                    st.info("No nodules detected")
    
    st.markdown("---")
    st.caption("© HIT500 Capstone Project | Architecture: Attention U-Net")

if __name__ == "__main__":
    main()
