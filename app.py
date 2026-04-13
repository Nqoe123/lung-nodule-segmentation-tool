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
import SimpleITK as sitk
import tempfile
import zipfile

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

# ========== MEMORY EFFICIENT U-NET (EXACT ARCHITECTURE FROM TRAINING) ==========
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
    """Memory Efficient U-Net - EXACT architecture from training code"""
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Memory efficient - reduced channels (EXACT from training)
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
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict from checkpoint
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
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()
        
        # Calculate and display model info
        total_params = sum(p.numel() for p in model.parameters())
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info(f"Model Stats:\n"
                       f"• Parameters: {total_params/1e6:.1f}M\n"
                       f"• Checkpoint: Epoch {epoch}\n"
                       f"• Best Dice: {best_dice}")
        
        return model, True
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("""
        **Troubleshooting:**
        1. Make sure you uploaded 'checkpoint_epoch_90.pth' to Google Drive
        2. Update the GOOGLE_DRIVE_FILE_ID with your actual file ID
        3. The file should be from your training output
        
        **To get the FILE_ID:**
        - Upload your model to Google Drive
        - Right-click → "Get link"
        - Copy the file ID from the URL (the long string between /d/ and /view)
        """)
        return model, False

# ========== MHD FILE PROCESSING FUNCTIONS ==========
def load_mhd_file(mhd_file, raw_file=None):
    """
    Load MHD file with its associated RAW file.
    If raw_file is None, SimpleITK will look for the .raw file with the same base name.
    """
    try:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mhd') as tmp_mhd:
            tmp_mhd.write(mhd_file.getvalue())
            mhd_path = tmp_mhd.name
        
        # If raw file is provided separately, save it too
        if raw_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.raw') as tmp_raw:
                tmp_raw.write(raw_file.getvalue())
                raw_path = tmp_raw.name
        else:
            raw_path = None
        
        # Read the MHD file (SimpleITK automatically loads the associated .raw file)
        itk_image = sitk.ReadImage(mhd_path)
        
        # Get the array from the image (z, y, x order)
        ct_array = sitk.GetArrayFromImage(itk_image)
        
        # Get metadata
        spacing = itk_image.GetSpacing()  # (x, y, z) spacing in mm
        origin = itk_image.GetOrigin()    # (x, y, z) origin in world coordinates
        
        # Reorder spacing and origin to (z, y, x) to match array
        spacing = (spacing[2], spacing[1], spacing[0])
        origin = (origin[2], origin[1], origin[0])
        
        # Clean up temp files
        os.unlink(mhd_path)
        if raw_path and os.path.exists(raw_path):
            os.unlink(raw_path)
        
        return ct_array, spacing, origin, itk_image
    
    except Exception as e:
        st.error(f"Error loading MHD file: {e}")
        return None, None, None, None

def process_ct_scan(ct_array, spacing, origin, slice_index=None):
    """
    Process a CT scan array and return a normalized slice.
    If slice_index is None, find the slice with the largest area of interest.
    """
    # Normalize CT values (window -1000 to 400 HU)
    window_min, window_max = -1000.0, 400.0
    ct_normalized = np.clip(ct_array, window_min, window_max)
    ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
    
    # If no slice specified, find slice with most variation (likely where nodules are)
    if slice_index is None:
        # Calculate standard deviation for each slice
        slice_std = [np.std(ct_normalized[i]) for i in range(ct_normalized.shape[0])]
        slice_index = np.argmax(slice_std)
    
    # Ensure slice index is within bounds
    slice_index = max(0, min(slice_index, ct_normalized.shape[0] - 1))
    
    # Extract the slice
    ct_slice = ct_normalized[slice_index]
    
    return ct_slice, slice_index

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

def extract_all_slices(ct_array, spacing, origin):
    """Extract and normalize all slices from a CT scan"""
    window_min, window_max = -1000.0, 400.0
    ct_normalized = np.clip(ct_array, window_min, window_max)
    ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
    
    slices = []
    for i in range(ct_normalized.shape[0]):
        slices.append(ct_normalized[i])
    
    return slices

# ========== HELPER FUNCTIONS ==========
def convert_to_grayscale(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    if image.mode != 'L':
        image = image.convert('L')
    return np.array(image)

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
    
    # Model architecture details in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧠 Model Architecture**")
    st.sidebar.markdown("Encoder Path:")
    st.sidebar.markdown("- Input: 1x512x512")
    st.sidebar.markdown("- 32 channels")
    st.sidebar.markdown("- 64 channels")
    st.sidebar.markdown("- 128 channels")
    st.sidebar.markdown("- 256 channels")
    st.sidebar.markdown("- 256 channels")
    st.sidebar.markdown("")
    st.sidebar.markdown("Decoder Path:")
    st.sidebar.markdown("- 256 → 128 → 64 → 32 → 1")
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.stop()
    
    # File upload section
    st.subheader("📤 Upload CT Scan")
    
    # Option to choose upload type
    upload_type = st.radio(
        "Select upload format:",
        ["Single Image (PNG/JPG)", "MHD + RAW Files", "ZIP Archive (MHD+RAW)"],
        horizontal=True
    )
    
    ct_data = None
    original_array = None
    spacing_info = None
    origin_info = None
    
    if upload_type == "Single Image (PNG/JPG)":
        uploaded = st.file_uploader("Choose CT image", type=["png", "jpg", "jpeg"])
        
        if uploaded:
            image = Image.open(uploaded)
            original_array = convert_to_grayscale(image)
            st.image(original_array, caption="Uploaded CT Image", use_container_width=True)
            ct_data = original_array
    
    elif upload_type == "MHD + RAW Files":
        st.info("📌 Upload both the .mhd and .raw files. Make sure they have the same base name.")
        
        col1, col2 = st.columns(2)
        with col1:
            mhd_file = st.file_uploader("Choose .mhd file", type=["mhd"])
        with col2:
            raw_file = st.file_uploader("Choose .raw file", type=["raw"])
        
        if mhd_file is not None:
            if raw_file is not None:
                with st.spinner("Loading MHD file..."):
                    ct_array, spacing, origin, itk_image = load_mhd_file(mhd_file, raw_file)
                    
                    if ct_array is not None:
                        st.success(f"✅ Successfully loaded CT scan!")
                        st.info(f"📊 Scan info:\n"
                               f"• Dimensions: {ct_array.shape[0]} slices × {ct_array.shape[1]} × {ct_array.shape[2]}\n"
                               f"• Spacing: ({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}) mm\n"
                               f"• Origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})")
                        
                        # Process and display a preview slice
                        ct_slice, slice_idx = process_ct_scan(ct_array, spacing, origin)
                        original_array = ct_slice
                        ct_data = ct_array
                        spacing_info = spacing
                        origin_info = origin
                        st.session_state['ct_array'] = ct_array
                        st.session_state['spacing'] = spacing
                        st.session_state['origin'] = origin
                        st.session_state['current_slice'] = slice_idx
                        
                        # Show preview
                        st.subheader("📷 Preview Slice")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(ct_slice, cmap='gray')
                        ax.set_title(f"Slice {slice_idx} of {ct_array.shape[0]}")
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.error("Failed to load MHD file")
            else:
                st.warning("Please upload both .mhd and .raw files")
    
    elif upload_type == "ZIP Archive (MHD+RAW)":
        zip_file = st.file_uploader("Choose ZIP file containing .mhd and .raw", type=["zip"])
        
        if zip_file is not None:
            with st.spinner("Extracting and loading ZIP archive..."):
                try:
                    # Save zip to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        tmp_zip.write(zip_file.getvalue())
                        zip_path = tmp_zip.name
                    
                    # Extract zip
                    extract_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(extract_dir)
                    
                    # Find .mhd and .raw files
                    mhd_file_path = None
                    raw_file_path = None
                    
                    for file in os.listdir(extract_dir):
                        if file.endswith('.mhd'):
                            mhd_file_path = os.path.join(extract_dir, file)
                        elif file.endswith('.raw'):
                            raw_file_path = os.path.join(extract_dir, file)
                    
                    if mhd_file_path and raw_file_path:
                        # Load the MHD file
                        itk_image = sitk.ReadImage(mhd_file_path)
                        ct_array = sitk.GetArrayFromImage(itk_image)
                        spacing = itk_image.GetSpacing()
                        origin = itk_image.GetOrigin()
                        
                        # Reorder to (z, y, x)
                        spacing = (spacing[2], spacing[1], spacing[0])
                        origin = (origin[2], origin[1], origin[0])
                        
                        st.success(f"✅ Successfully loaded CT scan from ZIP!")
                        st.info(f"📊 Scan info:\n"
                               f"• Dimensions: {ct_array.shape[0]} slices × {ct_array.shape[1]} × {ct_array.shape[2]}\n"
                               f"• Spacing: ({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}) mm")
                        
                        # Process preview slice
                        ct_slice, slice_idx = process_ct_scan(ct_array, spacing, origin)
                        original_array = ct_slice
                        ct_data = ct_array
                        spacing_info = spacing
                        origin_info = origin
                        st.session_state['ct_array'] = ct_array
                        st.session_state['spacing'] = spacing
                        st.session_state['origin'] = origin
                        st.session_state['current_slice'] = slice_idx
                        
                        st.subheader("📷 Preview Slice")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(ct_slice, cmap='gray')
                        ax.set_title(f"Slice {slice_idx} of {ct_array.shape[0]}")
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    # Cleanup
                    os.unlink(zip_path)
                    import shutil
                    shutil.rmtree(extract_dir)
                    
                except Exception as e:
                    st.error(f"Error loading ZIP file: {e}")
    
    # Segmentation section
    if ct_data is not None:
        col_left, col_right = st.columns(2)
        
        with col_left:
            # For 3D CT data, add slice selector
            if upload_type != "Single Image (PNG/JPG)" and 'ct_array' in st.session_state:
                ct_array_full = st.session_state['ct_array']
                max_slice = ct_array_full.shape[0] - 1
                
                slice_num = st.slider(
                    "Select slice to segment:",
                    min_value=0,
                    max_value=max_slice,
                    value=st.session_state.get('current_slice', max_slice // 2)
                )
                
                # Process the selected slice
                window_min, window_max = -1000.0, 400.0
                ct_normalized = np.clip(ct_array_full, window_min, window_max)
                ct_normalized = (ct_normalized - window_min) / (window_max - window_min)
                current_slice_img = ct_normalized[slice_num]
                
                st.image(current_slice_img, caption=f"CT Slice {slice_num}", use_container_width=True)
            else:
                current_slice_img = original_array
                slice_num = 0
            
            if st.button("🔍 Segment Nodule", type="primary"):
                with st.spinner("Segmenting with AI model..."):
                    if upload_type == "Single Image (PNG/JPG)":
                        mask = segment_nodule_from_array(model, original_array)
                    else:
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
                    ax.set_title("White = Nodule Detected, Black = Background")
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    buf = io.BytesIO()
                    plt.figure(figsize=(5,5))
                    plt.imshow(st.session_state['mask'], cmap='gray')
                    plt.axis('off')
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    st.download_button("📥 Download Mask", buf, f"nodule_mask_slice_{st.session_state['slice_num']}.png", mime="image/png")
                
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
    
    st.markdown("---")
    st.caption("© 2026 HIT500 Capstone Project | Model: Memory Efficient U-Net | Trained on LUNA16")

# ========== INSTRUCTIONS EXPANDER ==========
with st.expander("ℹ️ How to use this application"):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Login** using credentials: `radiologist` / `hit500`
    2. **Choose upload format**:
       - **Single Image**: Upload PNG/JPG of a CT slice
       - **MHD + RAW**: Upload both files from LUNA16 dataset
       - **ZIP Archive**: Upload a ZIP containing both .mhd and .raw
    3. **For 3D CT scans**: Use the slider to select which slice to analyze
    4. Click **"Segment Nodule"** to run the AI model
    5. Review results in three tabs
    
    ### About MHD/RAW Format
    
    The LUNA16 dataset uses this format:
    - **.mhd** (MetaHeader): Contains metadata like spacing, origin, dimensions
    - **.raw**: Contains the actual pixel data
    
    SimpleITK automatically reads both files when you load the .mhd file.
    
    ### Model Information
    
    - **Architecture**: Memory Efficient U-Net
    - **Parameters**: 3.4 million
    - **Input Size**: 512x512 grayscale
    - **Training Data**: LUNA16 dataset
    - **Performance**: 0.64 Dice score
    
    ### Clinical Use
    
    This tool assists radiologists in automated nodule detection and volume measurement.
    """)
