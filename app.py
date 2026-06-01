import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import OrderedDict
import os

st.set_page_config(page_title="LungVision AI", page_icon="", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0b1120; }
    .main > div { padding: 1rem; }
    h1, h2, h3 { color: #f8fafc; }
    .stButton > button { background: #0ea5e9; color: white; border: none; }
</style>
""", unsafe_allow_html=True)

device = torch.device('cpu')

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

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
    def forward(self, x): return self.conv(x)

class MemoryEfficientUNet(nn.Module):
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

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    paths = ["best_model.pth", "/kaggle/working/best_model.pth", "complete_model_with_metadata.pth"]
    model_path = None
    for p in paths:
        if os.path.exists(p):
            model_path = p
            break
    
    if model_path is None:
        st.error("Model file not found. Please upload best_model.pth")
        return None
    
    model = MemoryEfficientUNet(n_channels=1, n_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    if state_dict and len(state_dict) > 0 and 'module.' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def segment_patch(model, patch_img):
    tensor = torch.FloatTensor(patch_img / 255.0).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().numpy()
    
    pred_mask = (prob > 0.5).astype(np.uint8)
    return pred_mask, prob

# ============================================================
# MAIN APP
# ============================================================
st.title("LungVision AI")
st.markdown("Lung Nodule Segmentation")

st.markdown("### Upload CT Patch")
st.info("Upload a 128x128 PNG patch extracted from a CT scan")

uploaded_file = st.file_uploader("Select PNG image", type=["png"])

if uploaded_file is not None:
    model = load_model()
    if model is None:
        st.stop()
    
    img = np.array(Image.open(uploaded_file).convert('L'), dtype=np.float32)
    
    if img.shape != (128, 128):
        st.warning(f"Image size is {img.shape}, expected 128x128. Resizing...")
        from skimage.transform import resize
        img = resize(img, (128, 128), preserve_range=True)
    
    with st.spinner("Analyzing..."):
        pred_mask, prob = segment_patch(model, img)
    
    # Normalize for display
    img_display = img / 255.0
    mask_display = pred_mask.astype(np.float32)
    
    # Create overlay
    overlay = np.stack([img_display, img_display, img_display], axis=-1)
    overlay[pred_mask > 0, 0] = 1.0
    overlay[pred_mask > 0, 1] = 0.2
    overlay[pred_mask > 0, 2] = 0.2
    
    # Display using matplotlib to avoid Streamlit image issues
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title("Original CT Patch")
    axes[0].axis('off')
    
    axes[1].imshow(mask_display, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red = Nodule)")
    axes[2].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    nodule_area = pred_mask.sum()
    if nodule_area > 0:
        st.success(f"Nodule detected! Area: {nodule_area} pixels")
        confidence = prob.max()
        st.metric("Detection Confidence", f"{confidence:.2%}")
    else:
        st.info("No nodule detected in this patch")

st.markdown("---")
st.caption("Upload a 128x128 PNG patch. The model will output a segmentation mask.")
