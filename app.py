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

# ========== GOOGLE DRIVE SETUP ==========
GOOGLE_DRIVE_FILE_ID = "1L9qrtFk12EAOI_h2ru5BzVFAMmsvHJHV"  # Replace with your actual FILE ID
MODEL_FILENAME = "best_unet_model.pth"

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
        return logits  # Return logits, not sigmoid (BCEWithLogitsLoss expects logits)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    model_path = download_model_from_drive()
    model = UNet()
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict (handle different save formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            best_dice = checkpoint.get('best_dice', 'Unknown')
        else:
            state_dict = checkpoint
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
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.info(f"Model Stats:\n"
                       f"• Parameters: {total_params/1e6:.1f}M\n"
                       f"• Best Dice: {best_dice}\n"
                       f"• Architecture: Memory Efficient U-Net")
        
        return model, True
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("""
        **Troubleshooting:**
        1. Make sure you uploaded the correct model file to Google Drive
        2. Update the GOOGLE_DRIVE_FILE_ID with your actual file ID
        3. The model should be saved from the training script as 'best_model.pth'
        
        **To get the FILE_ID:**
        - Upload your model to Google Drive
        - Right-click → "Get link"
        - Copy the file ID from the URL (the long string between /d/ and /view)
        """)
        return model, False

# ========== HELPER FUNCTIONS ==========
def convert_to_grayscale(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    if image.mode != 'L':
        image = image.convert('L')
    return np.array(image)

def segment_nodule(model, image_array):
    """Segment nodule from CT image using the trained model"""
    # Ensure 2D
    if len(image_array.shape) == 3:
        image_array = image_array[:, :, 0]
    
    # Resize to 512x512 (exactly as in training)
    img_resized = resize(image_array, (512, 512), preserve_range=True)
    
    # Normalize using same method as training (window -1000 to 400)
    window_min, window_max = -1000.0, 400.0
    img_normalized = np.clip(img_resized, window_min, window_max)
    img_normalized = (img_normalized - window_min) / (window_max - window_min)
    
    # Convert to tensor (exactly as in training)
    input_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)
    
    # Segment
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)  # Get logits
        probs = torch.sigmoid(outputs)  # Convert to probabilities
        mask = (probs > 0.5).float().squeeze().numpy()
    
    # Resize back to original dimensions
    return resize(mask, image_array.shape[:2], order=0, preserve_range=True)

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
    st.sidebar.markdown("- 32 channels (DoubleConv)")
    st.sidebar.markdown("- 64 channels (Down + DoubleConv)")
    st.sidebar.markdown("- 128 channels (Down + DoubleConv)")
    st.sidebar.markdown("- 256 channels (Down + DoubleConv)")
    st.sidebar.markdown("- 256 channels (Down + DoubleConv)")
    st.sidebar.markdown("")
    st.sidebar.markdown("Decoder Path:")
    st.sidebar.markdown("- 256 channels (Up + DoubleConv)")
    st.sidebar.markdown("- 128 channels (Up + DoubleConv)")
    st.sidebar.markdown("- 64 channels (Up + DoubleConv)")
    st.sidebar.markdown("- 32 channels (Up + DoubleConv)")
    st.sidebar.markdown("- Output: 1x512x512 (logits)")
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.stop()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📤 Upload CT Scan")
        uploaded = st.file_uploader("Choose CT image", type=["png", "jpg", "jpeg", "dcm"])
        
        if uploaded:
            image = Image.open(uploaded)
            original_array = convert_to_grayscale(image)
            
            st.subheader("📷 Original CT Scan")
            st.image(original_array, caption="Original CT Image", use_container_width=True)
            
            if st.button("🔍 Segment Nodule", type="primary"):
                with st.spinner("Segmenting with AI model..."):
                    mask = segment_nodule(model, original_array)
                    volume = calculate_volume(mask)
                    st.session_state['mask'] = mask
                    st.session_state['volume'] = volume
                    st.session_state['original'] = original_array
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
                
                # Download button for mask
                buf = io.BytesIO()
                plt.figure(figsize=(5,5))
                plt.imshow(st.session_state['mask'], cmap='gray')
                plt.axis('off')
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                st.download_button("📥 Download Segmentation Mask", buf, "nodule_mask.png", mime="image/png")
            
            with tab2:
                original = st.session_state['original']
                mask = st.session_state['mask']
                if original.shape != mask.shape:
                    mask = resize(mask, original.shape, order=0, preserve_range=True)
                
                # Normalize original for display
                original_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
                overlay = np.stack([original_norm] * 3, axis=-1)
                overlay[:, :, 0] = np.where(mask > 0.5, 1.0, overlay[:, :, 0])  # Red channel
                overlay[:, :, 1] = np.where(mask > 0.5, 0.0, overlay[:, :, 1])  # Green channel
                overlay[:, :, 2] = np.where(mask > 0.5, 0.0, overlay[:, :, 2])  # Blue channel
                
                st.image(overlay, caption="🔴 Nodule Highlighted in RED", use_container_width=True)
            
            with tab3:
                nodule_pixels = np.sum(st.session_state['mask'] > 0.5)
                total_pixels = st.session_state['mask'].size
                area_pct = (nodule_pixels / total_pixels) * 100
                
                cola, colb, colc = st.columns(3)
                cola.metric("📏 Estimated Volume", f"{st.session_state['volume']:.2f} mm³")
                colb.metric("📐 Relative Area", f"{area_pct:.2f}%")
                colc.metric("🎯 Model Status", "Active")
                
                st.markdown("---")
                st.markdown("**🏥 Clinical Recommendation**")
                
                if area_pct < 1:
                    st.info("📌 **Small nodule (<1% of image)**\n\n"
                           "• Regular monitoring recommended\n"
                           "• Follow-up CT in 6-12 months\n"
                           "• Low risk profile")
                elif area_pct < 5:
                    st.warning("⚠️ **Medium nodule (1-5% of image)**\n\n"
                              "• Further evaluation suggested\n"
                              "• Short-term follow-up (3-6 months)\n"
                              "• Consider additional imaging")
                else:
                    st.error("🚨 **Large nodule (>5% of image)**\n\n"
                            "• Urgent consultation recommended\n"
                            "• Consider biopsy or surgical referral\n"
                            "• High priority follow-up needed")
    
    st.markdown("---")
    st.caption("© 2026 HIT500 Capstone Project | Model: Memory Efficient U-Net (3.4M params) | Trained on LUNA16 dataset")

# ========== INSTRUCTIONS EXPANDER ==========
with st.expander("ℹ️ How to use this application"):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Login** using credentials: `radiologist` / `hit500`
    2. **Upload** a CT scan image (supports PNG, JPG, JPEG)
    3. Click **"Segment Nodule"** to run the AI model
    4. Review results in three tabs:
       - **Binary Mask**: Pure segmentation mask
       - **Overlay**: Nodule highlighted in red on original CT
       - **Clinical Metrics**: Volume calculation and recommendations
    5. **Download** the segmentation mask for your records
    
    ### Model Information
    
    - **Architecture**: Memory Efficient U-Net
    - **Parameters**: 3.4 million (87% smaller than standard U-Net)
    - **Training Data**: LUNA16 dataset (902 CT slices, 601 scans)
    - **Input Size**: 512x512 grayscale
    - **Performance**: 0.64 Dice score on validation set
    
    ### Technical Details
    
    The model uses:
    - Encoder: 32 -> 64 -> 128 -> 256 -> 256 channels
    - Decoder: 256 -> 128 -> 64 -> 32 -> 1 channel
    - Bilinear upsampling for memory efficiency
    - Batch normalization for stable training
    
    ### Clinical Use
    
    This tool is designed to assist radiologists in:
    - Automated nodule detection
    - Volume measurement for growth tracking
    - Standardized reporting
    - Clinical decision support
    """)
