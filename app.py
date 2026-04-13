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

# ========== GOOGLE DRIVE SETUP ==========
GOOGLE_DRIVE_FILE_ID = "1FdIozNEVbIPUsjcdReAfmbgN3Nisx9yQ"  # Replace with your actual FILE ID
MODEL_FILENAME = "checkpoint_epoch_90.pth"

def download_model_from_drive():
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_FILENAME, quiet=False)
            st.success("Model downloaded successfully!")
    return MODEL_FILENAME

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Lung Nodule Segmentation", page_icon="🫁", layout="wide")

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
    model_path = download_model_from_drive()
    model = UNet()
    
    try:
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
            from collections import OrderedDict
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
        st.error(f"Error loading model: {e}")
        return model, False

# ========== MHD FILE LOADING WITH MULTI-UPLOAD ==========
def load_mhd_files_from_uploads(uploaded_files):
    """
    Load MHD and RAW files from multiple file uploads
    """
    mhd_file = None
    raw_file = None
    
    # Separate the files
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
    
    # Check if base names match
    mhd_basename = os.path.splitext(mhd_file.name)[0]
    raw_basename = os.path.splitext(raw_file.name)[0]
    
    if mhd_basename != raw_basename:
        st.warning(f"⚠️ Base names don't match: '{mhd_basename}' vs '{raw_basename}'. They should be the same for proper loading.")
    
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save files with their original names in the same directory
            mhd_path = os.path.join(tmpdir, mhd_file.name)
            raw_path = os.path.join(tmpdir, raw_file.name)
            
            # Write the files
            with open(mhd_path, 'wb') as f:
                f.write(mhd_file.getvalue())
            
            with open(raw_path, 'wb') as f:
                f.write(raw_file.getvalue())
            
            # Update the MHD file to point to the correct raw file name
            with open(mhd_path, 'r') as f:
                mhd_content = f.read()
            
            # Look for ElementDataFile line and update it
            import re
            mhd_content = re.sub(
                r'ElementDataFile\s*=\s*.*',
                f'ElementDataFile = {raw_file.name}',
                mhd_content
            )
            
            # Write back the updated MHD file
            with open(mhd_path, 'w') as f:
                f.write(mhd_content)
            
            # Now read with SimpleITK
            itk_image = sitk.ReadImage(mhd_path)
            
            # Get the array
            ct_array = sitk.GetArrayFromImage(itk_image)
            
            # Get metadata
            spacing = itk_image.GetSpacing()
            origin = itk_image.GetOrigin()
            
            # Reorder to (z, y, x) to match array
            spacing = (spacing[2], spacing[1], spacing[0])
            origin = (origin[2], origin[1], origin[0])
            
            return ct_array, spacing, origin
            
    except Exception as e:
        st.error(f"Error loading MHD/RAW files: {str(e)}")
        return None, None, None

# ========== CT PROCESSING FUNCTIONS ==========
def process_ct_scan(ct_array):
    """Process CT array and return normalized version"""
    window_min, window_max = -1000.0, 400.0
    ct_normalized = np.clip(ct_array, window_min, window_max)
    ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
    return ct_normalized

def segment_nodule_from_array(model, ct_slice):
    """Segment nodule from a CT slice array"""
    # Resize to 512x512
    img_resized = resize(ct_slice, (512, 512), preserve_range=True)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(img_resized).unsqueeze(0).unsqueeze(0)
    
    # Segment
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)
        mask = (probs > 0.5).float().squeeze().numpy()
    
    # Resize back to original dimensions
    return resize(mask, ct_slice.shape[:2], order=0, preserve_range=True)

def calculate_volume(mask, pixel_spacing_mm=0.7, slice_thickness_mm=1.25):
    """Calculate nodule volume in mm³"""
    pixel_area_mm2 = pixel_spacing_mm ** 2
    area_pixels = np.sum(mask)
    return area_pixels * pixel_area_mm2 * slice_thickness_mm

# ========== MAIN UI ==========
st.title("🫁 Lung Nodule Segmentation Tool")
st.markdown("### Memory Efficient CNN-Based Segmentation for CT Scans")
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
else:
    st.sidebar.success("✅ Logged in as: Radiologist")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    # Model architecture details
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧠 Model Architecture**")
    st.sidebar.markdown("- Memory Efficient U-Net")
    st.sidebar.markdown("- 3.4M parameters")
    st.sidebar.markdown("- 32 → 64 → 128 → 256 → 256 channels")
    st.sidebar.markdown("- Input: 512×512")
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.stop()
    
    # File upload section
    st.subheader("📤 Upload CT Scan")
    
    upload_type = st.radio(
        "Select upload format:",
        ["Single Image (PNG/JPG)", "MHD + RAW Files (Select both at once)", "ZIP Archive (MHD+RAW)"],
        horizontal=True
    )
    
    ct_data = None
    
    if upload_type == "Single Image (PNG/JPG)":
        uploaded = st.file_uploader("Choose CT image", type=["png", "jpg", "jpeg"])
        
        if uploaded:
            image = Image.open(uploaded)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            if image.mode != 'L':
                image = image.convert('L')
            original_array = np.array(image)
            st.image(original_array, caption="Uploaded CT Image", use_container_width=True)
            ct_data = original_array
    
    elif upload_type == "MHD + RAW Files (Select both at once)":
        st.info("📌 Select both the .mhd and .raw files together (Ctrl+Click or Shift+Click to select multiple files)")
        
        uploaded_files = st.file_uploader(
            "Choose .mhd and .raw files",
            type=["mhd", "raw"],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(uploaded_files) >= 2:
            with st.spinner("Loading MHD/RAW files..."):
                ct_array, spacing, origin = load_mhd_files_from_uploads(uploaded_files)
                
                if ct_array is not None:
                    st.success("✅ Successfully loaded CT scan!")
                    st.info(f"📊 Scan dimensions: {ct_array.shape[0]} slices × {ct_array.shape[1]} × {ct_array.shape[2]}")
                    
                    # Process and show preview
                    ct_normalized = process_ct_scan(ct_array)
                    
                    # Find best slice (with most variation)
                    slice_std = [np.std(ct_normalized[i]) for i in range(min(ct_normalized.shape[0], 100))]
                    best_slice = np.argmax(slice_std)
                    
                    st.session_state['ct_array'] = ct_normalized
                    st.session_state['ct_array_raw'] = ct_array
                    st.session_state['spacing'] = spacing
                    st.session_state['origin'] = origin
                    st.session_state['num_slices'] = ct_array.shape[0]
                    st.session_state['best_slice'] = best_slice
                    
                    # Show preview
                    st.subheader("📷 Preview Slice")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(ct_normalized[best_slice], cmap='gray')
                    ax.set_title(f"Slice {best_slice} of {ct_array.shape[0]} (Preview)")
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    ct_data = ct_normalized
                else:
                    st.error("Failed to load MHD/RAW files. Make sure you selected both .mhd and .raw files.")
        
        elif uploaded_files and len(uploaded_files) < 2:
            st.warning("Please select both .mhd and .raw files (you need to select 2 files)")
    
    elif upload_type == "ZIP Archive (MHD+RAW)":
        zip_file = st.file_uploader("Choose ZIP file containing .mhd and .raw", type=["zip"])
        
        if zip_file is not None:
            with st.spinner("Extracting and loading ZIP archive..."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Save zip file
                        zip_path = os.path.join(tmpdir, "upload.zip")
                        with open(zip_path, 'wb') as f:
                            f.write(zip_file.getvalue())
                        
                        # Extract zip
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            zf.extractall(tmpdir)
                        
                        # Find .mhd and .raw files
                        mhd_path = None
                        raw_path = None
                        
                        for file in os.listdir(tmpdir):
                            if file.endswith('.mhd'):
                                mhd_path = os.path.join(tmpdir, file)
                            elif file.endswith('.raw'):
                                raw_path = os.path.join(tmpdir, file)
                        
                        if mhd_path and raw_path:
                            # Read the MHD file with SimpleITK
                            itk_image = sitk.ReadImage(mhd_path)
                            ct_array = sitk.GetArrayFromImage(itk_image)
                            
                            # Get metadata
                            spacing = itk_image.GetSpacing()
                            origin = itk_image.GetOrigin()
                            
                            # Reorder to (z, y, x)
                            spacing = (spacing[2], spacing[1], spacing[0])
                            origin = (origin[2], origin[1], origin[0])
                            
                            st.success("✅ Successfully loaded CT scan from ZIP!")
                            st.info(f"📊 Scan dimensions: {ct_array.shape[0]} slices × {ct_array.shape[1]} × {ct_array.shape[2]}")
                            
                            ct_normalized = process_ct_scan(ct_array)
                            
                            slice_std = [np.std(ct_normalized[i]) for i in range(min(ct_normalized.shape[0], 100))]
                            best_slice = np.argmax(slice_std)
                            
                            st.session_state['ct_array'] = ct_normalized
                            st.session_state['ct_array_raw'] = ct_array
                            st.session_state['spacing'] = spacing
                            st.session_state['origin'] = origin
                            st.session_state['num_slices'] = ct_array.shape[0]
                            st.session_state['best_slice'] = best_slice
                            
                            st.subheader("📷 Preview Slice")
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(ct_normalized[best_slice], cmap='gray')
                            ax.set_title(f"Slice {best_slice} of {ct_array.shape[0]}")
                            ax.axis('off')
                            st.pyplot(fig)
                            
                            ct_data = ct_normalized
                        else:
                            st.error("ZIP file must contain both .mhd and .raw files")
                            
                except Exception as e:
                    st.error(f"Error loading ZIP file: {str(e)}")
    
    # Segmentation section for 3D CT data
    if ct_data is not None and len(np.array(ct_data).shape) == 3:
        # 3D CT data - add slice selector
        num_slices = st.session_state.get('num_slices', ct_data.shape[0])
        default_slice = st.session_state.get('best_slice', num_slices // 2)
        
        slice_num = st.slider(
            "Select slice to segment:",
            min_value=0,
            max_value=num_slices - 1,
            value=min(default_slice, num_slices - 1)
        )
        
        current_slice_img = ct_data[slice_num]
        st.image(current_slice_img, caption=f"CT Slice {slice_num} of {num_slices}", use_container_width=True)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            if st.button("🔍 Segment Nodule", type="primary"):
                with st.spinner("Segmenting with AI model..."):
                    mask = segment_nodule_from_array(model, current_slice_img)
                    volume = calculate_volume(mask)
                    st.session_state['mask'] = mask
                    st.session_state['volume'] = volume
                    st.session_state['current_image'] = current_slice_img
                    st.session_state['slice_num'] = slice_num
                st.success("Segmentation complete!")
        
        with col_right:
            st.subheader("📊 Results")
            
            if 'mask' in st.session_state:
                tab1, tab2, tab3 = st.tabs(["🎭 Binary Mask", "🔴 Overlay", "📈 Clinical Metrics"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(st.session_state['mask'], cmap='gray')
                    ax.set_title("White = Nodule, Black = Background")
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    buf = io.BytesIO()
                    plt.figure(figsize=(5,5))
                    plt.imshow(st.session_state['mask'], cmap='gray')
                    plt.axis('off')
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    st.download_button("📥 Download Mask", buf, f"mask_slice_{st.session_state['slice_num']}.png")
                
                with tab2:
                    original = st.session_state['current_image']
                    mask = st.session_state['mask']
                    if original.shape != mask.shape:
                        mask = resize(mask, original.shape, order=0, preserve_range=True)
                    
                    original_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
                    overlay = np.stack([original_norm] * 3, axis=-1)
                    overlay[:, :, 0] = np.where(mask > 0.5, 1.0, overlay[:, :, 0])
                    overlay[:, :, 1] = np.where(mask > 0.5, 0.0, overlay[:, :, 1])
                    overlay[:, :, 2] = np.where(mask > 0.5, 0.0, overlay[:, :, 2])
                    
                    st.image(overlay, caption="🔴 Nodule Highlighted in RED", use_container_width=True)
                
                with tab3:
                    nodule_pixels = np.sum(st.session_state['mask'] > 0.5)
                    total_pixels = st.session_state['mask'].size
                    area_pct = (nodule_pixels / total_pixels) * 100
                    
                    cola, colb, colc = st.columns(3)
                    cola.metric("📏 Volume", f"{st.session_state['volume']:.2f} mm³")
                    colb.metric("📐 Area %", f"{area_pct:.2f}%")
                    colc.metric("🎯 Slice", st.session_state['slice_num'])
                    
                    st.markdown("---")
                    st.markdown("**🏥 Clinical Recommendation**")
                    
                    if area_pct < 1:
                        st.info("📌 **Small nodule** - Regular monitoring recommended. Follow-up in 6-12 months.")
                    elif area_pct < 5:
                        st.warning("⚠️ **Medium nodule** - Further evaluation suggested. Short-term follow-up (3-6 months).")
                    else:
                        st.error("🚨 **Large nodule** - Urgent consultation recommended. Consider biopsy or surgical referral.")
    
    elif ct_data is not None:
        # 2D image data
        col_left, col_right = st.columns(2)
        
        with col_left:
            if st.button("🔍 Segment Nodule", type="primary"):
                with st.spinner("Segmenting with AI model..."):
                    mask = segment_nodule_from_array(model, ct_data)
                    volume = calculate_volume(mask)
                    st.session_state['mask'] = mask
                    st.session_state['volume'] = volume
                    st.session_state['current_image'] = ct_data
                st.success("Segmentation complete!")
        
        with col_right:
            st.subheader("📊 Results")
            
            if 'mask' in st.session_state:
                tab1, tab2, tab3 = st.tabs(["🎭 Binary Mask", "🔴 Overlay", "📈 Clinical Metrics"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(st.session_state['mask'], cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)
                
                with tab2:
                    original = st.session_state['current_image']
                    mask = st.session_state['mask']
                    if original.shape != mask.shape:
                        mask = resize(mask, original.shape, order=0, preserve_range=True)
                    
                    original_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
                    overlay = np.stack([original_norm] * 3, axis=-1)
                    overlay[:, :, 0] = np.where(mask > 0.5, 1.0, overlay[:, :, 0])
                    overlay[:, :, 1] = np.where(mask > 0.5, 0.0, overlay[:, :, 1])
                    overlay[:, :, 2] = np.where(mask > 0.5, 0.0, overlay[:, :, 2])
                    
                    st.image(overlay, use_container_width=True)
                
                with tab3:
                    area_pct = (np.sum(st.session_state['mask'] > 0.5) / st.session_state['mask'].size) * 100
                    st.metric("Estimated Volume", f"{st.session_state['volume']:.2f} mm³")
                    st.metric("Relative Area", f"{area_pct:.2f}%")
    
    st.markdown("---")
    st.caption("© 2026 HIT500 Capstone Project | Model: Memory Efficient U-Net (3.4M params) | Trained on LUNA16")
