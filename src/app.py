import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from diseases import DISEASE_CLASSES, DISPLAY_NAMES, parse_disease_label
from wiki_fetcher import fetch_wikipedia_context
from llm_assistant import stream_ollama, list_available_models

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
    st.markdown(
        "[Kaggle dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)"
    )

st.title("Plant Health Recognition")
st.markdown("---")

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Select plant / disease")

    display_options = [DISPLAY_NAMES[c] for c in DISEASE_CLASSES]
    chosen_display = st.selectbox("Dataset class", display_options)
    selected_class = DISEASE_CLASSES[display_options.index(chosen_display)]
    plant, disease, is_healthy = parse_disease_label(selected_class)

    status = "Healthy" if is_healthy else disease
    st.markdown(f"**Plant:** {plant}")
    st.markdown(f"**Condition:** {status}")
    st.markdown(f"**Class:** `{selected_class}`")

    analyze_btn = st.button(
        "Run analysis",
        type="primary",
        use_container_width=True,
        disabled=selected_model is None,
    )

with col_output:
    st.subheader("Model response")

    if analyze_btn:
        with st.spinner("Fetching Wikipedia context..."):
            wiki_data = fetch_wikipedia_context(
                plant, disease, is_healthy, dataset_class=selected_class
            )

        wiki_excerpt = wiki_data.get("excerpt", "")

        if wiki_data["error"]:
            st.warning(f"Wikipedia: {wiki_data['error']}")
        else:
            with st.expander(f"Wikipedia: {wiki_data['title']}", expanded=True):
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
    else:
        st.info("Select a class on the left and click **Run analysis**.")

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
