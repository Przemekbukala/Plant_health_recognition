
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def parse_disease_label(label: str) -> tuple[str, str, bool]:
    """
    Parses a dataset class label into (plant_name, disease_name, is_healthy).
    Examples:
        "Apple___Apple_scab"  -> ("Apple", "Apple scab", False)
    """
    parts = label.split("___", 1)
    plant_raw = parts[0]
    disease_raw = parts[1] if len(parts) > 1 else "Unknown"

    plant_name = plant_raw.replace("_", " ").replace("(", "(").replace(",", ",").strip()

    disease_raw_clean = disease_raw.strip("_").strip()
    is_healthy = disease_raw_clean.lower() == "healthy"

    disease_name = (
        "Healthy"
        if is_healthy
        else disease_raw_clean.replace("_", " ").replace("(", "(").strip()
    )

    return plant_name, disease_name, is_healthy


DISPLAY_NAMES = {
    cls: "{} — {}".format(*parse_disease_label(cls)[:2]) for cls in DISEASE_CLASSES
}

# Exact English Wikipedia article titles  per dataset class.
WIKIPEDIA_ARTICLE_TITLE: dict[str, str] = {
    "Apple___Apple_scab": "Apple scab",
    "Apple___Black_rot": "Black rot",
    "Apple___Cedar_apple_rust": "Cedar-apple rust",
    "Apple___healthy": "Apple",
    "Blueberry___healthy": "Blueberry",
    "Cherry_(including_sour)___Powdery_mildew": "Podosphaera clandestina",
    "Cherry_(including_sour)___healthy": "Sour cherry",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray leaf spot",
    "Corn_(maize)___Common_rust_": "Puccinia sorghi",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern corn leaf blight",
    "Corn_(maize)___healthy": "Maize",
    "Grape___Black_rot": "Black rot (grape disease)",
    "Grape___Esca_(Black_Measles)": "Esca (grape disease)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Isariopsis",
    "Grape___healthy": "Grape",
    "Orange___Haunglongbing_(Citrus_greening)": "Citrus greening disease",
    "Peach___Bacterial_spot": "Xanthomonas arboricola",
    "Peach___healthy": "Peach",
    "Pepper,_bell___Bacterial_spot": "Bacterial leaf spot of peppers and tomato",
    "Pepper,_bell___healthy": "Bell pepper",
    "Potato___Early_blight": "Alternaria solani",
    "Potato___Late_blight": "Phytophthora infestans",
    "Potato___healthy": "Potato",
    "Raspberry___healthy": "Raspberry",
    "Soybean___healthy": "Soybean",
    "Squash___Powdery_mildew": "Powdery mildew",
    "Strawberry___Leaf_scorch": "Diplocarpon earlianum",
    "Strawberry___healthy": "Strawberry",
    "Tomato___Bacterial_spot": "Bacterial leaf spot of peppers and tomato",
    "Tomato___Early_blight": "Alternaria solani",
    "Tomato___Late_blight": "Phytophthora infestans",
    "Tomato___Leaf_Mold": "Tomato leaf mold",
    "Tomato___Septoria_leaf_spot": "Septoria lycopersici",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tetranychus urticae",
    "Tomato___Target_Spot": "Corynespora cassiicola",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato yellow leaf curl virus",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus",
    "Tomato___healthy": "Tomato",
}


def get_wikipedia_queries(plant: str, disease: str, is_healthy: bool) -> list[str]:
    """
    Generic search phrases only (custom text input, or falblack if hardcoded title fails).
    """
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = q.strip()
        if not q or q in seen:
            return
        seen.add(q)
        queries.append(q)

    plant_l = plant.lower()
    disease_l = disease.lower()

    if is_healthy:
        add(f"{plant} cultivation")
        add(f"{plant} agriculture")
        add(f"growing {plant_l}")
        add(plant)
        return queries

    add(f"{plant} {disease}")
    add(f"{disease} {plant}")
    add(f"{plant_l} {disease_l}")
    add(f"{disease} of {plant_l}")
    add(f"{plant} {disease} disease")
    add(f"{disease} plant disease")
    add(disease)
    add(f"{plant} diseases")
    return queries


def get_wikipedia_query(plant: str, disease: str, is_healthy: bool) -> str:
    """Single-query API for backwards compatibility."""
    qs = get_wikipedia_queries(plant, disease, is_healthy)
    return qs[0] if qs else ""


