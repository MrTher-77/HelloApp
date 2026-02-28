import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

MODEL_ID = "nateraw/vit-age-classifier"

@st.cache_resource
def load_age_classifier():
    # 关键：low_cpu_mem_usage=False，避免走 meta 初始化路径
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=False,
    )
    # Streamlit 云端多数没 GPU，这里固定 CPU（device=-1）
    return pipeline(
        "image-classification",
        model=model,
        image_processor=processor,
        device=-1,
    )

st.title("Age Classification using ViT")

age_classifier = load_age_classifier()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    preds = age_classifier(img)
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    st.subheader("Predicted Age Range")
    st.write(f"Top-1: {preds[0]['label']}  (score={preds[0]['score']:.4f})")
    st.write(preds)
