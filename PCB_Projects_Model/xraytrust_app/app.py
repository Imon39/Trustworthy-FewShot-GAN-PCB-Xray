"""
XrayTrust — PCB X-ray Reconstruction + Trust Score Demo
Streamlit proof-of-concept for the Innovation Fair.

Run locally:
    streamlit run app.py

Deploy: push this folder to GitHub, then deploy via share.streamlit.io
"""

import io
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from trust_utils import calculate_trust_score

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="XrayTrust — PCB X-ray Reconstruction",
    page_icon="🔬",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shot options — must match the folder names you saved
SHOT_OPTIONS = [5, 10, 20, 40, 100, 200]
MODEL_DIR = "models"


# ============================================================
# 2. EXACT GENERATOR ARCHITECTURE (MATCHING YOUR WEIGHTS)
# ============================================================
class Generator(nn.Module):
    """
    Exact Generator Architecture matching notebook keys:
    enc1, enc2, enc3, enc4, bottleneck, dec4, dec3, dec2, dec1, final
    """
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.enc1 = self._block(1, 64, normalize=False)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        # Bottleneck
        self.bottleneck = self._block(512, 512)

        # Decoder
        self.dec4 = self._upblock(512 + 512, 256)
        self.dec3 = self._upblock(256 + 256, 128)
        self.dec2 = self._upblock(128 + 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Final Layer (using Sigmoid)
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch, normalize=True):
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _upblock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([
            F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=False), e4
        ], dim=1))

        d3 = self.dec3(torch.cat([
            F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False), e3
        ], dim=1))

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False), e2
        ], dim=1))

        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False), e1
        ], dim=1))

        return self.final(d1)


# ============================================================
# 3. PHYSICS-INSPIRED CONSISTENCY LOSS
# ============================================================
def physics_loss(generated_hr, original_lr, scale_factor=4):
    b, c, h, w = generated_hr.shape

    reconstructed_lr = torch.nn.functional.interpolate(
        generated_hr,
        size=(h // scale_factor, w // scale_factor),
        mode="bicubic",
        align_corners=False,
    )
    reconstructed_lr = torch.nn.functional.interpolate(
        reconstructed_lr,
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )
    return nn.functional.l1_loss(reconstructed_lr, original_lr)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    img1 = img1.squeeze().cpu().numpy().astype(np.float64)
    img2 = img2.squeeze().cpu().numpy().astype(np.float64)

    mu1, mu2 = img1.mean(), img2.mean()
    sigma1, sigma2 = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    )
    return float(ssim)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


# ============================================================
# 4. MODEL LOADING & PREPROCESSING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_generator(shot: int) -> Generator:
    model_path = os.path.join(MODEL_DIR, f"generator_{shot}shot.pth")
    G = Generator().to(DEVICE)
    
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint not found at: {model_path}")
        st.stop()
        
    state_dict = torch.load(model_path, map_location=DEVICE)
    G.load_state_dict(state_dict, strict=True)
    G.eval()
    return G


def preprocess_image(uploaded_file, size=512) -> torch.Tensor:
    """
    Preprocess PCB X-ray using the SAME percentile normalization
    used during Improved-GAN training.

    Normalization:
        p_low  = 2nd percentile
        p_high = 98th percentile
        normalized = (pixel - p_low) / (p_high - p_low)
        clipped to [0, 1]
    """
    img = Image.open(uploaded_file)

    # Convert image to grayscale
    if img.mode not in ("I", "I;16", "I;16B", "I;16L", "I;16N", "L"):
        img = img.convert("L")

    img_array = np.array(img, dtype=np.float32)

    # 2%-98% percentile normalization
    p_low = np.percentile(img_array, 2)
    p_high = np.percentile(img_array, 98)

    if p_high > p_low:
        normalized = (img_array - p_low) / (p_high - p_low + 1e-8)
    else:
        normalized = np.zeros_like(img_array, dtype=np.float32)

    normalized = np.clip(normalized, 0.0, 1.0)

    # Resize AFTER normalization
    normalized_img = Image.fromarray(
        (normalized * 255.0).astype(np.uint8),
        mode="L"
    )

    normalized_img = normalized_img.resize(
        (size, size),
        Image.Resampling.BICUBIC
    )

    arr = np.array(normalized_img, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze().detach().cpu().numpy()

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)

    return Image.fromarray(arr, mode="L")

# ============================================================
# 5. SIDEBAR CONFIGURATION
# ============================================================
st.sidebar.title("⚙️ Settings")

shot = st.sidebar.selectbox(
    "Model (trained with N shots)",
    SHOT_OPTIONS,
    index=SHOT_OPTIONS.index(40),
    help="Choose which few-shot-trained generator to use for reconstruction.",
)

mode = st.sidebar.radio(
    "Evaluation mode",
    ["No ground truth (deployment)", "With ground truth (research/demo)"],
    help=(
        "Deployment mode: only the physics-inspired consistency signal is used.\n\n"
        "Research mode: upload a reference HR image to also compute PSNR/SSIM."
    ),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Trust score thresholds were empirically calibrated on the observed "
    "5–200 shot experiment range."
)


# ============================================================
# 6. MAIN PAGE
# ============================================================
st.title("🔬 XrayTrust — PCB X-ray Reconstruction")
st.caption(
    "Data-efficient, physics-inspired GAN framework for PCB X-ray image "
    "reconstruction — with a ground-truth-independent trust/quality signal."
)

st.info(
    "This prototype reconstructs a PCB X-ray from a low-quality input and reports "
    "a **physics-inspired consistency signal** as a trust indicator. It is **not** "
    "an automated defect-detection or PASS/FAIL system.",
    icon="ℹ️",
)

col_upload_1, col_upload_2 = st.columns(2)

with col_upload_1:
    lr_file = st.file_uploader(
        "Upload low-quality / low-resolution PCB X-ray",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="lr_upload",
    )
    if lr_file is not None:
        with st.expander("🔍 Debug: raw file info"):
            _dbg_img = Image.open(lr_file)
            st.write(f"Mode: `{_dbg_img.mode}`, Size: {_dbg_img.size}")
            _dbg_arr = np.array(_dbg_img)
            st.write(f"Pixel range: min={_dbg_arr.min()}, max={_dbg_arr.max()}")
            lr_file.seek(0)

hr_file = None
if mode == "With ground truth (research/demo)":
    with col_upload_2:
        hr_file = st.file_uploader(
            "Upload reference / ground-truth HR image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="hr_upload",
        )

run_button = st.button("🚀 Run Reconstruction", type="primary", use_container_width=False)


# ============================================================
# 7. INFERENCE & RESULTS
# ============================================================
if run_button:
    if lr_file is None:
        st.warning("Please upload a low-quality input image first.")
        st.stop()

    model_path = os.path.join(MODEL_DIR, f"generator_{shot}shot.pth")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    with st.spinner("Loading generator and reconstructing..."):
        G = load_generator(shot)
        lr_tensor = preprocess_image(lr_file)

        with torch.no_grad():
            generated_hr = G(lr_tensor)

        phys_err = physics_loss(generated_hr, lr_tensor).item()

        ssim_val, l1_val, psnr_val = None, None, None
        hr_tensor = None

        if mode == "With ground truth (research/demo)" and hr_file is not None:
            hr_tensor = preprocess_image(hr_file)
            ssim_val = calculate_ssim(generated_hr, hr_tensor)
            l1_val = nn.functional.l1_loss(generated_hr, hr_tensor).item()
            psnr_val = calculate_psnr(generated_hr, hr_tensor)

        trust_result = calculate_trust_score(
            ssim_val=ssim_val,
            physics_error=phys_err,
            l1_error=l1_val,
        )

    st.markdown("---")
    st.subheader("Results")

    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.image(tensor_to_image(lr_tensor), caption="Input (Low Quality)", use_column_width=True)
    with img_col2:
        st.image(tensor_to_image(generated_hr), caption="Reconstructed (Generator Output)", use_column_width=True)
    with img_col3:
        if hr_tensor is not None:
            st.image(tensor_to_image(hr_tensor), caption="Ground Truth (Reference)", use_column_width=True)
        else:
            st.info("No ground truth provided — deployment mode.")

    st.markdown("### 📊 Metrics")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Physics Consistency Error", f"{phys_err:.4f}")
    metric_cols[1].metric("SSIM", f"{ssim_val:.4f}" if ssim_val is not None else "N/A (no ground truth)")
    metric_cols[2].metric("PSNR", f"{psnr_val:.2f} dB" if psnr_val is not None else "N/A (no ground truth)")
    metric_cols[3].metric("L1 Error", f"{l1_val:.4f}" if l1_val is not None else "N/A (no ground truth)")

    st.markdown("### 🛡️ Reconstruction Consistency")
    status_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}[trust_result["status"]]

    trust_col1, trust_col2 = st.columns([1, 2])
    with trust_col1:
        st.metric("Consistency Score", f"{trust_result['trust_score']}%")
        st.markdown(f":{status_color}[**{trust_result['status']}**]")
        st.caption(
            "This score measures reconstruction/physics consistency only. "
            "It does not indicate PCB safety, defect absence, or PASS/FAIL."
                   )
    with trust_col2:
        st.write(f"**Recommendation:** {trust_result['recommendation']}")
        st.caption(
            "Ground truth used for scoring: "
            + ("Yes" if trust_result["ground_truth_used"] else "No — physics-only deployment mode")
        )

else:
    st.markdown("Upload an image and click **Run Reconstruction** to see the generator output and trust score.")