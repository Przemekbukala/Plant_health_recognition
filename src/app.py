import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from PIL import Image

from diseases import DISEASE_CLASSES, parse_disease_label
from wiki_fetcher import fetch_wikipedia_context
from llm_assistant import stream_ollama, list_available_models
from image_classifier import classify_image

st.set_page_config(
    page_title="Plant Health Recognition",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("# Plant Health Recognition")
    models = list_available_models()
    if models:
        selected_model = models[0]
        st.markdown(f"**Ollama model:** `{selected_model}`")
    else:
        st.warning("No Ollama models found.\nRun: `ollama serve` and `ollama pull llama3`")
        selected_model = None

    st.markdown("---")
    model_type = st.radio(
        "CNN model",
        ["pretrained", "custom"],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        "[Kaggle dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)"
    )

st.title("Plant Health Recognition")
st.markdown("---")

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Upload image")
    uploaded_file = st.file_uploader(
        "Choose a plant leaf photo",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
        analyze_btn = st.button(
            "Analyze",
            type="primary",
            use_container_width=True,
            disabled=selected_model is None,
        )
    else:
        st.info("Upload a plant leaf image to get started.")
        analyze_btn = False

with col_output:
    st.subheader("Diagnosis")

    if analyze_btn and uploaded_file is not None:
        with st.spinner("Running CNN classification..."):
            result = classify_image(image, model_type=model_type)

        confidence_pct = result["confidence"] * 100
        margin_pct = result["margin"] * 100

        st.markdown("### Classification result")
        st.markdown(f"**Confidence:** {confidence_pct:.1f}%")
        st.markdown(f"**Margin (top1 − top2):** {margin_pct:.1f}%")
        st.progress(min(result["confidence"], 1.0))

        with st.expander("Top-5 predictions", expanded=False):
            for cls_name, prob in result["all_predictions"]:
                p, d, _ = parse_disease_label(cls_name)
                st.markdown(f"- **{p} — {d}**: {prob*100:.1f}%")

        if not result["recognized"]:
            st.error(
                f"Could not recognize the plant disease.\n\n"
                f"{result['rejection_reason']}\n\n"
                f"The uploaded image likely does not contain a plant leaf "
            )
        else:
            plant = result["plant"]
            disease = result["disease"]
            is_healthy = result["is_healthy"]
            predicted_class = result["predicted_class"]

            status = "Healthy" if is_healthy else disease
            st.success(f"**{plant}** — {status}")
            st.markdown(f"**Detected class:** `{predicted_class}`")

            st.markdown("**LLM Analysis**")
            with st.spinner("Fetching Wikipedia context..."):
                wiki_data = fetch_wikipedia_context(
                    plant, disease, is_healthy, dataset_class=predicted_class
                )

            wiki_excerpt = wiki_data.get("excerpt", "")

            if wiki_data["error"]:
                st.warning(f"Wikipedia: {wiki_data['error']}")
            else:
                with st.expander(f"Wikipedia: {wiki_data['title']}", expanded=False):
                    st.write(wiki_excerpt)
                    st.markdown(f"[Full article]({wiki_data['url']})")

            st.markdown(f"**Model:** `{selected_model}`")
            response_placeholder = st.empty()
            full_response = ""

            try:
                for token in stream_ollama(
                    plant=plant,
                    disease=disease,
                    is_healthy=is_healthy,
                    wiki_excerpt=wiki_excerpt,
                    model=selected_model,
                ):
                    full_response += token
                    response_placeholder.markdown(full_response)
            except Exception as exc:
                st.error(
                    f"**Cannot reach Ollama.** Start the server:\n\n"
                    f"```bash\nollama serve\n```\n\nError: `{exc}`"
                )
    elif not analyze_btn:
        st.info("Upload an image on the left and click **Analyze**.")

with st.expander("All dataset classes"):
    plants_map: dict[str, list[str]] = {}
    for cls in DISEASE_CLASSES:
        p, d, _ = parse_disease_label(cls)
        plants_map.setdefault(p, []).append(d)

    cols = st.columns(3)
    for i, (p, diseases) in enumerate(sorted(plants_map.items())):
        with cols[i % 3]:
            st.markdown(f"**{p}**")
            for d in diseases:
                suffix = " *(healthy)*" if d == "Healthy" else ""
                st.markdown(f"- {d}{suffix}")
