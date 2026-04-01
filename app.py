import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

st.set_page_config(
    page_title="Soil Classifier",
    page_icon="🌱",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = keras.models.load_model("best_soil.keras")
    return model

model = load_model()

CLASS_NAMES = [
    'Alluvial Soil', 'Arid Soil', 'Black Soil',
    'Laterite Soil', 'Mountain Soil', 'Red Soil', 'Yellow Soil'
]

CLASS_INFO = {
    'Alluvial Soil'  : "Sol alluvial — Très fertile, idéal pour l'agriculture.",
    'Arid Soil'      : "Sol aride — Pauvre en matière organique, zones désertiques.",
    'Black Soil'     : "Sol noir — Riche en argile, retient bien l'eau.",
    'Laterite Soil'  : "Sol latéritique — Riche en fer, zones tropicales.",
    'Mountain Soil'  : "Sol de montagne — Peu profond, riche en matière organique.",
    'Red Soil'       : "Sol rouge — Riche en fer, bien drainé.",
    'Yellow Soil'    : "Sol jaune — Modérément fertile, zones humides."
}

st.title("🌱 Soil Classifier")
st.markdown("**Identifiez le type de sol à partir d'une photo**")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Uploadez une image de sol",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploadée", use_column_width=True)

    with st.spinner("Analyse en cours..."):
        img = image.resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = keras.applications.efficientnet.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        predictions = model.predict(arr)[0]
        top3_idx = np.argsort(predictions)[::-1][:3]

    st.markdown("---")
    st.subheader("Résultats")

    best_idx = top3_idx[0]
    best_class = CLASS_NAMES[best_idx]
    best_conf = predictions[best_idx] * 100

    st.success(f"**{best_class}** — {best_conf:.1f}% de confiance")
    st.info(CLASS_INFO[best_class])

    st.markdown("#### Top 3 prédictions")
    for idx in top3_idx:
        st.progress(
            float(predictions[idx]),
            text=f"{CLASS_NAMES[idx]} : {predictions[idx]*100:.1f}%"
        )
else:
    st.info("Uploadez une image pour commencer l'analyse")
    st.markdown("""
    **Types de sol détectables :**
    - 🟤 Alluvial Soil
    - 🟡 Arid Soil
    - ⚫ Black Soil
    - 🔴 Laterite Soil
    - 🟢 Mountain Soil
    - 🔴 Red Soil
    - 🟡 Yellow Soil
    """)
