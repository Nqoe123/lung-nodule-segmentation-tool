import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import os

st.set_page_config(page_title="LungVision AI", page_icon="", layout="wide")

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background: #0b1120; }
    .main > div { padding: 1rem; }
    h1, h2, h3 { color: #f8fafc; }
    .stButton > button { background: #0ea5e9; color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem; }
    .stButton > button:hover { background: #0284c7; }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: #111827;
        border-radius: 16px;
        border: 1px solid #1e2d4a;
        text-align: center;
    }
    .login-container h2 { margin-bottom: 1.5rem; }
    .login-container input { width: 100%; padding: 0.5rem; margin-bottom: 1rem; background: #0f172a; border: 1px solid #1e2d4a; border-radius: 8px; color: #f1f5f9; }
    .glass-panel {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::before {
        content: '';
        display: block;
        width: 20px;
        height: 2px;
        background: #38bdf8;
        border-radius: 2px;
    }
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
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown('<div style="height: 20vh;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; padding: 2.5rem 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;"></div>
            <h2 style="margin-bottom: 0.5rem; font-size: 1.8rem;">LungVision AI</h2>
            <p style="color: #94a3b8; margin-bottom: 2rem;">Clinical Nodule Segmentation</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Radiologist ID", placeholder="Enter your ID")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if username == "radiologist" and password == "hit500":
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

# ============================================================
# MAIN APP
# ============================================================
def show_app():
    st.markdown("""
    <div class="glass-panel" style="margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0; font-size: 1.5rem;">LungVision <span style="color:#38bdf8">AI</span></h1>
            <div style="color:#94a3b8; font-size: 0.85rem;">Radiologist: """ + st.session_state.get('username', 'Guest') + """ | System Online</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="glass-panel"><h3>Control Panel</h3></div>', unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.info("Upload a CT image for automated nodule detection")
        st.markdown("---")
        st.caption("Model trained on LUNA16 and LIDC datasets")
        st.caption("Validation Dice Score: 0.8871")
    
    st.markdown('<div class="section-label">CT Image Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select CT patch (PNG image)", type=["png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        model = load_model()
        if model is None:
            st.stop()
        
        img = np.array(Image.open(uploaded_file).convert('L'), dtype=np.float32)
        
        if img.shape != (128, 128):
            from skimage.transform import resize
            img = resize(img, (128, 128), preserve_range=True)
        
        with st.spinner("Analyzing..."):
            pred_mask, _ = segment_patch(model, img)
        
        img_display = img / 255.0
        mask_display = pred_mask.astype(np.float32)
        
        overlay = np.stack([img_display, img_display, img_display], axis=-1)
        overlay[pred_mask > 0, 0] = 1.0
        overlay[pred_mask > 0, 1] = 0.2
        overlay[pred_mask > 0, 2] = 0.2
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title("Original CT", color='#f1f5f9')
        axes[0].axis('off')
        
        axes[1].imshow(mask_display, cmap='gray')
        axes[1].set_title("Segmentation Mask", color='#f1f5f9')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title("Detection Overlay", color='#f1f5f9')
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        nodule_area = pred_mask.sum()
        if nodule_area > 0:
            st.success(f"Nodule detected! Area: {nodule_area} pixels")
        else:
            st.info("No nodule detected")
    
    st.markdown("---")
    st.caption("LungVision AI - Clinical Decision Support System")

# ============================================================
# ENTRY POINT
# ============================================================
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_login()
    else:
        show_app()

if __name__ == "__main__":
    main()
