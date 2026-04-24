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
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation
from scipy import ndimage
import tempfile
import SimpleITK as sitk
from collections import OrderedDict
from PIL import Image
import warnings
from datetime import datetime
import zipfile
import hashlib

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

# ========== ROBUST LUNG CT VALIDATION FUNCTIONS ==========

def extract_lung_features(image_array):
    """Extract feature vector from image for lung CT classification"""
    features = {}
    
    try:
        # Ensure 2D grayscale
        if len(image_array.shape) > 2:
            image_array = image_array[:, :, 0]
        
        # Basic statistics
        features['min'] = float(image_array.min())
        features['max'] = float(image_array.max())
        features['mean'] = float(image_array.mean())
        features['std'] = float(image_array.std())
        features['skew'] = float(ndimage.measurements.center_of_mass(image_array)[0])  # Simple skew proxy
        
        # Percentiles for intensity distribution
        features['p5'] = float(np.percentile(image_array, 5))
        features['p25'] = float(np.percentile(image_array, 25))
        features['p50'] = float(np.percentile(image_array, 50))
        features['p75'] = float(np.percentile(image_array, 75))
        features['p95'] = float(np.percentile(image_array, 95))
        
        # Intensity spread (important for CT - wide range)
        features['intensity_range'] = features['max'] - features['min']
        
        # Check for air pockets (low intensity regions typical in lungs)
        low_intensity_mask = image_array < np.percentile(image_array, 30)
        features['low_intensity_ratio'] = float(low_intensity_mask.sum() / image_array.size)
        
        # Edge density (medical images have specific edge characteristics)
        edges = sobel(image_array)
        edge_threshold = np.percentile(edges, 90)
        features['edge_density'] = float((edges > edge_threshold).sum() / image_array.size)
        
        # Texture analysis using local binary patterns proxy (standard deviation of local regions)
        from skimage.util import view_as_blocks
        try:
            # Divide image into blocks
            h, w = image_array.shape
            block_size = min(32, h//4, w//4)
            if block_size >= 4:
                blocks = view_as_blocks(image_array[:h//block_size*block_size, :w//block_size*block_size], 
                                       (block_size, block_size))
                block_std = np.std(blocks, axis=(2,3))
                features['texture_uniformity'] = float(np.std(block_std))
            else:
                features['texture_uniformity'] = 0
        except:
            features['texture_uniformity'] = 0
        
        # Frequency domain analysis (medical images have specific frequency patterns)
        fft = np.fft.fft2(image_array)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        # Ratio of low to high frequencies
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        radius = min(h, w) // 8
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
        low_freq_mask = dist_from_center <= radius
        high_freq_mask = dist_from_center > radius
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        features['freq_ratio'] = float(low_freq_energy / (high_freq_energy + 1e-8))
        
        # Anatomical structure detection (look for bilateral symmetry - lung feature)
        # Split image into left and right halves
        mid = image_array.shape[1] // 2
        left_half = image_array[:, :mid]
        right_half = image_array[:, mid:]
        # Resize to same size for comparison
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        # Flip right half for comparison
        right_half_flipped = np.fliplr(right_half)
        # Compute similarity
        if left_half.shape == right_half_flipped.shape:
            mse = np.mean((left_half - right_half_flipped)**2)
            features['bilateral_symmetry'] = float(1 / (1 + mse))  # Higher = more symmetric
        else:
            features['bilateral_symmetry'] = 0
        
        return features
        
    except Exception as e:
        st.warning(f"Feature extraction warning: {str(e)}")
        return None

def is_lung_ct_slice(image_array):
    """
    Validate if an image is a lung CT slice using strict criteria
    """
    validation_results = {
        'is_lung': False,
        'confidence': 0.0,
        'reasons': [],
        'fail_reasons': []
    }
    
    try:
        # Ensure 2D array
        if len(image_array.shape) > 2:
            image_array = image_array[:, :, 0]
        
        # Extract features
        features = extract_lung_features(image_array)
        if features is None:
            validation_results['fail_reasons'].append("Could not extract image features")
            return validation_results
        
        # CRITERION 1: Intensity range (Lung CT has wide intensity range: air to bone)
        if features['intensity_range'] > 0.6:  # Wide range typical of CT
            validation_results['reasons'].append(f"✓ Wide intensity range: {features['intensity_range']:.2f}")
        else:
            validation_results['fail_reasons'].append(f"✗ Low intensity range: {features['intensity_range']:.2f} (needs >0.6)")
        
        # CRITERION 2: Low intensity regions (air in lungs)
        if 0.15 < features['low_intensity_ratio'] < 0.45:
            validation_results['reasons'].append(f"✓ Air pockets detected: {features['low_intensity_ratio']:.1%}")
        else:
            validation_results['fail_reasons'].append(f"✗ Abnormal air content: {features['low_intensity_ratio']:.1%} (need 15-45%)")
        
        # CRITERION 3: Edge density (Lung CT has moderate edge density)
        if 0.05 < features['edge_density'] < 0.25:
            validation_results['reasons'].append(f"✓ Appropriate edge density: {features['edge_density']:.2f}")
        else:
            validation_results['fail_reasons'].append(f"✗ Unusual edge density: {features['edge_density']:.2f}")
        
        # CRITERION 4: Texture uniformity (Lung CT has specific texture)
        if features['texture_uniformity'] > 0.05:
            validation_results['reasons'].append(f"✓ Natural texture pattern: {features['texture_uniformity']:.3f}")
        else:
            validation_results['fail_reasons'].append(f"✗ Unnatural texture: {features['texture_uniformity']:.3f}")
        
        # CRITERION 5: Frequency characteristics (CT has more low frequencies)
        if features['freq_ratio'] > 1.5:
            validation_results['reasons'].append(f"✓ Medical imaging frequency pattern: {features['freq_ratio']:.1f}")
        else:
            validation_results['fail_reasons'].append(f"✗ Wrong frequency profile: {features['freq_ratio']:.1f}")
        
        # CRITERION 6: Bilateral symmetry (Lungs are roughly symmetric)
        if features['bilateral_symmetry'] > 0.6:
            validation_results['reasons'].append(f"✓ Bilateral symmetry: {features['bilateral_symmetry']:.2f}")
        else:
            validation_results['fail_reasons'].append(f"✗ Poor symmetry: {features['bilateral_symmetry']:.2f}")
        
        # Calculate confidence (weighted by importance of criteria)
        weights = {
            'intensity_range': 0.25,
            'low_intensity_ratio': 0.25,
            'edge_density': 0.15,
            'texture_uniformity': 0.15,
            'freq_ratio': 0.10,
            'bilateral_symmetry': 0.10
        }
        
        passed_score = 0
        total_score = 0
        
        # Check intensity range
        if features['intensity_range'] > 0.6:
            passed_score += weights['intensity_range']
        total_score += weights['intensity_range']
        
        # Check low intensity ratio
        if 0.15 < features['low_intensity_ratio'] < 0.45:
            passed_score += weights['low_intensity_ratio']
        total_score += weights['low_intensity_ratio']
        
        # Check edge density
        if 0.05 < features['edge_density'] < 0.25:
            passed_score += weights['edge_density']
        total_score += weights['edge_density']
        
        # Check texture uniformity
        if features['texture_uniformity'] > 0.05:
            passed_score += weights['texture_uniformity']
        total_score += weights['texture_uniformity']
        
        # Check frequency ratio
        if features['freq_ratio'] > 1.5:
            passed_score += weights['freq_ratio']
        total_score += weights['freq_ratio']
        
        # Check bilateral symmetry
        if features['bilateral_symmetry'] > 0.6:
            passed_score += weights['bilateral_symmetry']
        total_score += weights['bilateral_symmetry']
        
        validation_results['confidence'] = passed_score / total_score if total_score > 0 else 0
        
        # Determine if lung CT (need >60% confidence AND at least 4 passed criteria)
        if validation_results['confidence'] > 0.6 and len(validation_results['reasons']) >= 4:
            validation_results['is_lung'] = True
            validation_results['reasons'].append(f"✓ Lung CT confirmed ({validation_results['confidence']:.1%} confidence)")
        else:
            validation_results['is_lung'] = False
            
        return validation_results
        
    except Exception as e:
        validation_results['fail_reasons'].append(f"Validation error: {str(e)}")
        validation_results['is_lung'] = False
        return validation_results

def validate_full_ct_volume(image_volume):
    """
    Validate full 3D CT volume
    """
    validation_results = {
        'is_lung': False,
        'confidence': 0.0,
        'reasons': [],
        'fail_reasons': []
    }
    
    try:
        # Check dimensions
        num_slices, height, width = image_volume.shape
        
        if not (50 <= num_slices <= 1000):
            validation_results['fail_reasons'].append(f"Invalid slice count: {num_slices} (should be 50-1000)")
            return validation_results
        
        if not (200 <= height <= 1024 and 200 <= width <= 1024):
            validation_results['fail_reasons'].append(f"Invalid dimensions: {height}×{width}")
            return validation_results
        
        validation_results['reasons'].append(f"✓ Volume dimensions: {num_slices} slices, {height}×{width}")
        
        # Sample slices at different positions
        sample_positions = [0, num_slices//4, num_slices//2, 3*num_slices//4, num_slices-1]
        lung_slice_scores = []
        
        for pos in sample_positions:
            slice_img = image_volume[pos]
            # Normalize if needed
            if slice_img.max() > 1.0:
                slice_img = slice_img / 255.0
            
            result = is_lung_ct_slice(slice_img)
            if result['is_lung']:
                lung_slice_scores.append(result['confidence'])
        
        if lung_slice_scores:
            avg_confidence = np.mean(lung_slice_scores)
            validation_results['confidence'] = avg_confidence
            
            if avg_confidence > 0.6 and len(lung_slice_scores) >= 3:
                validation_results['is_lung'] = True
                validation_results['reasons'].append(f"✓ {len(lung_slice_scores)}/{len(sample_positions)} slices validated as lung CT")
            else:
                validation_results['fail_reasons'].append(f"Only {len(lung_slice_scores)}/{len(sample_positions)} slices passed validation")
        else:
            validation_results['fail_reasons'].append("No valid lung CT slices found in volume")
            
    except Exception as e:
        validation_results['fail_reasons'].append(f"Volume validation error: {str(e)}")
    
    return validation_results

# ========== GOOGLE DRIVE SETUP ==========
GOOGLE_DRIVE_FILE_ID = "1PzCv2fJSr7e0QIfPGtLKLOL-9RSLdR2i"
MODEL_FILENAME = "best_model.pth"

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
        
        # Load checkpoint
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
def load_and_preprocess_image(uploaded_file):
    """Load image and ensure it's in correct format"""
    try:
        # Open image
        image = Image.open(uploaded_file)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Ensure 2D array
        if len(image_array.shape) > 2:
            image_array = image_array[:, :, 0]
        
        # Normalize to [0, 1] range if needed
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        return image_array, True
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None, False

def prepare_for_model(image_array):
    """Prepare image for model input with lung window"""
    # Convert to HU range approximation
    # Typical CT: -1000 HU (air) to +400 HU (soft tissue)
    image_hu = image_array * 1400 - 1000
    
    # Clip to lung window
    image_hu = np.clip(image_hu, -1000, 400)
    
    # Normalize back to [0,1]
    normalized = (image_hu + 1000) / 1400
    
    return normalized

def segment_nodule(model, image_array):
    """Run segmentation on a single image"""
    try:
        # Ensure image is 2D
        if len(image_array.shape) > 2:
            image_array = image_array[:, :, 0]
        
        # Prepare for model
        normalized = prepare_for_model(image_array)
        
        # Resize to 512x512
        resized = resize(normalized, (512, 512), preserve_range=True)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)
        
        # Run model
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            mask = (probs > 0.5).float().squeeze().numpy()
        
        # Resize back to original
        mask_original = resize(mask, image_array.shape, order=0, preserve_range=True)
        
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
                    })
        
        return detections, mask_original
    
    except Exception as e:
        st.error(f"Segmentation error: {str(e)}")
        return [], None

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
            type=["png", "jpg", "jpeg"],
            help="Upload a lung CT slice in PNG, JPG, or JPEG format"
        )
        
        if uploaded_file:
            # Load image
            image_array, success = load_and_preprocess_image(uploaded_file)
            
            if success and image_array is not None:
                # Show preview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(image_array, caption="Uploaded Image", use_container_width=True, clamp=True)
                
                # Validate
                with st.spinner("Validating image type..."):
                    validation = is_lung_ct_slice(image_array)
                
                if not validation['is_lung']:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.error("❌ **INVALID IMAGE TYPE - NOT A LUNG CT SCAN**")
                    
                    st.markdown("**Reasons for rejection:**")
                    for reason in validation['fail_reasons']:
                        st.markdown(f"- {reason}")
                    
                    st.markdown("---")
                    st.markdown("**This system only accepts valid lung CT images.**")
                    st.markdown("The image you uploaded does not have the characteristic features of a lung CT scan:")
                    st.markdown("- Appropriate intensity range (air to tissue)")
                    st.markdown("- Air pockets (dark regions representing lungs)")
                    st.markdown("- Bilateral symmetry (two lung fields)")
                    st.markdown("- Medical imaging texture patterns")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return
                
                # Show validation success
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("✅ **Lung CT Verified**")
                for reason in validation['reasons'][:4]:
                    st.markdown(f"- {reason}")
                st.markdown(f"**Confidence: {validation['confidence']:.1%}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Load model
                with st.spinner("Loading AI Model..."):
                    model, model_success = download_and_load_model()
                
                if model_success and st.button("🔍 Run Analysis", type="primary"):
                    with st.spinner("Segmenting lung nodules..."):
                        detections, mask = segment_nodule(model, image_array)
                    
                    if detections:
                        st.markdown(f'<div class="success-box">✅ <strong>{len(detections)} nodule(s) detected</strong></div>', unsafe_allow_html=True)
                        
                        for idx, d in enumerate(detections):
                            with st.expander(f"Nodule {idx+1} - Area: {d['area']:.0f} pixels²", expanded=True):
                                # Create overlay
                                overlay = np.stack([image_array] * 3, axis=-1)
                                overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
                                overlay[:, :, 0] = np.where(d['mask'] > 0.5, 1.0, overlay[:, :, 0])
                                overlay[:, :, 1] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 1])
                                overlay[:, :, 2] = np.where(d['mask'] > 0.5, 0.0, overlay[:, :, 2])
                                
                                col_img, col_info = st.columns(2)
                                with col_img:
                                    st.image(overlay, caption=f"Nodule {idx+1} Highlighted", use_container_width=True)
                                with col_info:
                                    st.metric("Area", f"{d['area']:.0f} pixels²")
                                    st.metric("Centroid (x,y)", f"({d['centroid'][1]:.0f}, {d['centroid'][0]:.0f})")
                                    
                                    if d['area'] < 100:
                                        st.info("📌 **Size:** Small - Routine monitoring recommended")
                                    elif d['area'] < 300:
                                        st.warning("⚠️ **Size:** Medium - Short-term follow-up recommended")
                                    else:
                                        st.error("🚨 **Size:** Large - Urgent consultation recommended")
                    else:
                        st.markdown('<div class="info-box">✓ No nodules detected in this slice</div>', unsafe_allow_html=True)
            else:
                st.error("Failed to load image. Please try another file.")
    
    else:  # Full CT Volume
        st.markdown('<div class="info-box">📌 <strong>Full CT Analysis:</strong> Upload a ZIP file containing MHD and RAW files.</div>', unsafe_allow_html=True)
        
        uploaded_zip = st.file_uploader(
            "Select ZIP file with .mhd and .raw files",
            type=["zip"],
            help="Upload a ZIP containing both the .mhd and .raw files"
        )
        
        if uploaded_zip:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                mhd_file = None
                raw_file = None
                
                for file in os.listdir(tmpdir):
                    if file.endswith('.mhd'):
                        mhd_file = os.path.join(tmpdir, file)
                    elif file.endswith('.raw'):
                        raw_file = os.path.join(tmpdir, file)
                
                if mhd_file and raw_file:
                    st.success("Files loaded successfully")
                    
                    # Load volume
                    with st.spinner("Loading CT volume..."):
                        img = sitk.ReadImage(mhd_file)
                        array = sitk.GetArrayFromImage(img)
                        
                        if array.max() > 1.0:
                            array = array / 255.0
                    
                    # Validate
                    with st.spinner("Validating CT volume..."):
                        validation = validate_full_ct_volume(array)
                    
                    if not validation['is_lung']:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.error("❌ **INVALID - NOT A LUNG CT VOLUME**")
                        for reason in validation['fail_reasons']:
                            st.markdown(f"- {reason}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        return
                    
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("✅ **Lung CT Volume Verified**")
                    for reason in validation['reasons']:
                        st.markdown(f"- {reason}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show info
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Slices", array.shape[0])
                    with col2:
                        st.metric("Dimensions", f"{array.shape[1]}×{array.shape[2]}")
                    with col3:
                        st.metric("Spacing", f"{img.GetSpacing()[0]:.2f}mm")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Preview
                    mid_slice = array[array.shape[0]//2]
                    st.image(mid_slice, caption="Middle Slice Preview", use_container_width=True, clamp=True)
                    
                    if st.button("🔍 Analyze Full Volume", type="primary"):
                        st.info("Processing full volume...")
                        model, model_success = download_and_load_model()
                        
                        if model_success:
                            all_detections = []
                            progress = st.progress(0)
                            
                            for i in range(array.shape[0]):
                                detections, _ = segment_nodule(model, array[i])
                                for d in detections:
                                    all_detections.append({
                                        'slice': i,
                                        'area': d['area'],
                                        'x': d['centroid'][1],
                                        'y': d['centroid'][0]
                                    })
                                progress.progress((i + 1) / array.shape[0])
                            
                            if all_detections:
                                st.success(f"Found {len(all_detections)} nodules across {array.shape[0]} slices")
                                import pandas as pd
                                df = pd.DataFrame(all_detections)
                                st.dataframe(df, use_container_width=True)
                                
                                csv = df.to_csv(index=False)
                                st.download_button("Download Results CSV", csv, "nodule_detections.csv")
                            else:
                                st.info("No nodules detected in this volume")
                else:
                    st.error("ZIP must contain both .mhd and .raw files")
    
    # Footer
    st.markdown("""
    <div class="clinical-footer">
        <p>LungVision AI | Clinical Decision Support System</p>
        <p>Always verify AI results with clinical expertise | HIT500 Capstone Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
