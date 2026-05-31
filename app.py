import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import zipfile
import os
import tempfile
import shutil

st.set_page_config(page_title="LungVision AI", page_icon="", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    .main > div { padding: 1rem; }
    h1, h2, h3 { color: #f8fafc; }
    .stButton > button { background: #0ea5e9; color: white; border: none; }
    [data-testid="stFileUploader"] > div { background: #111827; border: 2px dashed #1e2d4a; border-radius: 12px; }
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
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "best_model.pth"

@st.cache_resource
def load_model():
    # Try to find model in different locations
    paths_to_try = [
        "best_model.pth",
        "/kaggle/working/best_model.pth",
        "/mount/src/lung-nodule-segmentation-tool/best_model.pth",
        "complete_model_with_metadata.pth"
    ]
    
    model_path = None
    for path in paths_to_try:
        if os.path.exists(path):
            model_path = path
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
    model = model.to(device)
    model.eval()
    return model

def segment_patch(model, patch_img):
    # Patch is already 128x128 from pre-extracted data
    tensor = torch.FloatTensor(patch_img / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    
    pred_mask = (prob > 0.5).astype(np.uint8)
    return pred_mask, prob

st.title("LungVision AI")
st.markdown("Lung Nodule Segmentation on Pre-extracted 128x128 Patches")

# Upload patches zip file
st.markdown("### Upload Test Patches")
st.info("Upload the luna_test_patches.zip file containing pre-extracted 128x128 patches")

uploaded_file = st.file_uploader("Upload ZIP file with patches", type=["zip"])

if uploaded_file is not None:
    model = load_model()
    if model is None:
        st.stop()
    
    # Extract zip
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "patches.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        
        # Find images
        images_dir = os.path.join(tmpdir, "images")
        if not os.path.exists(images_dir):
            st.error("No 'images' folder found in zip")
            st.stop()
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        if len(image_files) == 0:
            st.error("No PNG files found in images folder")
            st.stop()
        
        st.success(f"Found {len(image_files)} test patches")
        
        # Process each patch
        dice_scores = []
        results = []
        
        for i, img_file in enumerate(image_files[:10]):  # Limit to 10 for speed
            img_path = os.path.join(images_dir, img_file)
            patch_img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
            
            # Get corresponding mask if available
            mask_path = os.path.join(tmpdir, "masks", img_file)
            has_gt = os.path.exists(mask_path)
            if has_gt:
                gt_mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255.0
            else:
                gt_mask = None
            
            # Segment
            pred_mask, prob = segment_patch(model, patch_img)
            
            if has_gt:
                intersection = (pred_mask * gt_mask).sum()
                dice = (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
                dice_scores.append(dice)
            else:
                dice = None
            
            results.append({
                'patch': img_file,
                'dice': dice,
                'image': patch_img,
                'pred_mask': pred_mask,
                'gt_mask': gt_mask
            })
        
        # Display results
        st.markdown("### Results")
        
        for res in results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(res['image'], caption=f"Original: {res['patch']}", use_container_width=True, clamp=True)
            
            with col2:
                st.image(res['pred_mask'], caption="Prediction", use_container_width=True, clamp=True)
            
            with col3:
                if res['gt_mask'] is not None:
                    st.image(res['gt_mask'], caption=f"Ground Truth (Dice: {res['dice']:.4f})", use_container_width=True, clamp=True)
                else:
                    st.image(res['pred_mask'], caption="Segmentation Result", use_container_width=True, clamp=True)
            
            st.divider()
        
        if dice_scores:
            st.markdown("### Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Patches", len(results))
            col2.metric("Average Dice", f"{np.mean(dice_scores):.4f}")
            col3.metric("Best Dice", f"{np.max(dice_scores):.4f}")
            col4.metric("Validation Dice", "0.8871")

st.markdown("---")
st.caption("Upload the luna_test_patches.zip file containing 128x128 patches extracted from LUNA16")
