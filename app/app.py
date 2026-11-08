import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NeuroSight",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "Control", 1: "Stroke"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# =========================================================
# 2. IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

# =========================================================
# 3. RESNET18 MODEL
# =========================================================

def make_resnet18():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model

# =========================================================
# 4. LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_models():
    ct_model = make_resnet18()
    mri_model = make_resnet18()
    ct_checkpoint = MODEL_DIR / "ct_model.pth"
    mri_checkpoint = MODEL_DIR / "mri_model.pth"
    if not ct_checkpoint.exists():
        raise FileNotFoundError(f"CT checkpoint not found: {ct_checkpoint}")
    if not mri_checkpoint.exists():
        raise FileNotFoundError(f"MRI checkpoint not found: {mri_checkpoint}")
    try:
        ct_state = torch.load(ct_checkpoint, map_location=DEVICE,  weights_only=True)
        mri_state = torch.load(mri_checkpoint, map_location=DEVICE, weights_only=True)
    except TypeError:
        ct_state = torch.load(ct_checkpoint, map_location=DEVICE)
        mri_state = torch.load(mri_checkpoint, map_location=DEVICE)
    ct_model.load_state_dict(ct_state)
    mri_model.load_state_dict(mri_state)
    ct_model.to(DEVICE)
    mri_model.to(DEVICE)
    ct_model.eval()
    mri_model.eval()
    return ct_model, mri_model

# =========================================================
# 5. GRAD-CAM IMPLEMENTATION
# =========================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self.save_activation)
    def save_activation(self, module, inputs, output):
        self.activations = output
        output.register_hook(self.save_gradient)
    def save_gradient(self, gradient):
        self.gradients = gradient
    def generate(self, image, predicted_class):
        self.model.zero_grad(set_to_none=True)
        self.gradients = None
        with torch.enable_grad():
            image = image.clone().detach().requires_grad_(True)
            logits = self.model(image)
            # Explain the predicted class.
            if predicted_class == 1:
                score = logits[:, 0].sum()
            else:
                score = -logits[:, 0].sum()
            score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Unable to generate Grad-CAM.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8))
        return cam
    def close(self):
        self.handle.remove()

# =========================================================
# 6. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):
    image = image.convert("RGB")
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor.to(DEVICE)

# =========================================================
# 7. PREDICTION
# =========================================================

@torch.no_grad()
def predict(model, image_tensor):
    model.eval()
    logits = model(image_tensor)
    probability = torch.sigmoid(logits).item()
    predicted_class = int(probability >= 0.5)
    confidence = (
        probability
        if predicted_class == 1
        else 1 - probability
    )
    return predicted_class, probability, confidence

# =========================================================
# 8. GRAD-CAM VISUALIZATION
# =========================================================

def generate_gradcam(model, image_tensor, predicted_class):
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam.generate(
            image_tensor,
            predicted_class
        )
    finally:
        gradcam.close()
    return heatmap

def denormalize(tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0, 1)

def display_gradcam(image_tensor, heatmap):
    original = denormalize(image_tensor[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(original)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[1].set_title("Grad-CAM Visualization")
    axes[1].axis("off")
    plt.tight_layout()
    return fig

# =========================================================
# 9. APPLICATION HEADER
# =========================================================

st.markdown(
    """
    <div style="text-align: center; padding: 25px 0 15px 0;">
        <h1 style="font-size: 3.5rem; font-weight: 800; letter-spacing: 2px; margin-bottom: 5px;">NeuroSight</h1>
        <h3 style="font-size: 1.25rem; font-weight: 400; margin-top: 0;">Intelligent Brain Imaging & Stroke Analysis</h3>
        <p style="font-size: 0.95rem; margin-top: 15px;">
            An Explainable AI Framework for CT and MRI-Based
            Brain Image Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

# =========================================================
# 10. SIDEBAR
# =========================================================

st.sidebar.title("Configuration")
modality = st.sidebar.selectbox(
    "Select Imaging Modality",
    ["CT", "MRI"]
)
st.sidebar.markdown("---")
st.sidebar.write("Model Architecture: ResNet18")
st.sidebar.write(f"Device: {DEVICE}")
st.sidebar.write("Input Size: 224 × 224")
st.sidebar.markdown("---")
st.sidebar.info(
    "This application is an experimental research "
    "demonstration and is not intended for clinical diagnosis."
)

# =========================================================
# 11. LOAD MODELS
# =========================================================

try:
    ct_model, mri_model = load_models()
    model = ct_model if modality == "CT" else mri_model
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# =========================================================
# 12. IMAGE UPLOAD
# =========================================================

st.subheader("Upload Brain Image")
uploaded_file = st.file_uploader(
    "Upload a CT or MRI image",
    type=["png", "jpg", "jpeg", "bmp"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(
            image,
            caption=f"Uploaded {modality} Image",
            use_container_width=True
        )
    image_tensor = preprocess_image(image)

    # =====================================================
    # 13. PREDICTION
    # =====================================================

    with st.spinner("Analyzing image..."):
        predicted_class, probability, confidence = predict(
            model,
            image_tensor
        )
    prediction = CLASS_NAMES[predicted_class]
    with col2:
        st.subheader("Classification Result")
        if predicted_class == 1:
            st.error("Prediction: Stroke")
        else:
            st.success("Prediction: Control")
        st.metric("Stroke Probability", f"{probability * 100:.2f}%")
        st.metric("Prediction Confidence", f"{confidence * 100:.2f}%")
        st.progress(float(probability))
    st.divider()

    # =====================================================
    # 14. GRAD-CAM
    # =====================================================

    st.subheader("Explainable AI - Grad-CAM")
    with st.spinner("Generating Grad-CAM heatmap..."):
        heatmap = generate_gradcam(model, image_tensor, predicted_class)
        fig = display_gradcam(image_tensor, heatmap)
        st.pyplot(fig)
        plt.close(fig)
    st.caption(
        "Grad-CAM highlights image regions influencing the "
        "model's predicted class. These regions are not "
        "clinically confirmed stroke lesions."
    )
    st.divider()

    # =====================================================
    # 15. PREDICTION SUMMARY
    # =====================================================

    st.subheader("Prediction Summary")
    summary = pd.DataFrame({
        "Parameter": [
            "Imaging Modality",
            "Model Architecture",
            "Predicted Class",
            "Stroke Probability",
            "Prediction Confidence"
        ],
        "Value": [
            modality,
            "ResNet18",
            prediction,
            f"{probability * 100:.2f}%",
            f"{confidence * 100:.2f}%"
        ]
    })
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info(
        "Upload a brain CT or MRI image to begin classification."
    )

# =========================================================
# 16. MODEL PERFORMANCE
# =========================================================

st.divider()
st.subheader("Model Performance")
performance_file = RESULTS_DIR / "performance_comparison.csv"
if performance_file.exists():
    performance = pd.read_csv(performance_file)
    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "Performance comparison CSV was not found."
    )

# =========================================================
# 17. FOOTER
# =========================================================

st.divider()
st.markdown(
    """
    **AI-Based Stroke Detection Using CT and MRI Images**
    Deep Learning Architecture: ResNet18
    Explainability: Grad-CAM
    Frameworks: PyTorch and Streamlit
    Research prototype — not a clinical diagnostic tool.
    """
)