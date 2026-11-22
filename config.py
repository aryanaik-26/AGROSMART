"""
AGROSMART AI - System Configuration and Constants
Central configuration for file paths, disease taxonomies, treatment protocols,
color themes, and machine learning hyperparameters.
"""

from pathlib import Path

# ==============================================================================
# 1. DIRECTORY PATH CONFIGURATIONS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLANTVILLAGE_DIR = DATA_DIR / "plantvillage"

MODELS_DIR = BASE_DIR / "models"
CROP_RECOMMENDATION_MODELS = MODELS_DIR / "crop_recommendation"
YIELD_PREDICTION_MODELS = MODELS_DIR / "yield_prediction"
DISEASE_DETECTION_MODELS = MODELS_DIR / "disease_detection"

REPORTS_DIR = BASE_DIR / "reports"
GRAPHS_DIR = REPORTS_DIR / "graphs"
METRICS_DIR = REPORTS_DIR / "metrics"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated_reports"

RESEARCH_PAPER_DIR = BASE_DIR / "research_paper"

# Ensure all critical folders exist
for folder in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, PLANTVILLAGE_DIR,
    CROP_RECOMMENDATION_MODELS, YIELD_PREDICTION_MODELS, DISEASE_DETECTION_MODELS,
    GRAPHS_DIR, METRICS_DIR, GENERATED_REPORTS_DIR, RESEARCH_PAPER_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. SEED AND HYPERPARAMETERS
# ==============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# ==============================================================================
# 3. COLOR PALETTE (Modern Earth & Emerald Agriculture Theme)
# ==============================================================================
THEME_COLORS = {
    "primary": "#2D6A4F",       # Deep forest green
    "primary_dark": "#1B4332",  # Pine green
    "primary_light": "#52B788", # Mint green
    "accent": "#74C69D",        # Pastel sage
    "background_light": "#D8F3DC",
    "earth_brown": "#7F5539",   # Rich soil
    "earth_sand": "#DDBEA9",    # Sandy loam
    "warning": "#E76F51",       # Terracotta warning
    "healthy": "#2A9D8F",       # Vibrant healthy teal
    "diseased": "#E63946",      # Alert red
}

# ==============================================================================
# 4. PLANTVILLAGE DISEASE TAXONOMY (38 Classes)
# ==============================================================================
PLANT_DISEASE_CLASSES = [
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
    "Tomato___healthy"
]

# Verified Agronomic Information & Preventive Measures
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "pathogen": "Venturia inaequalis (Fungus)",
        "status": "Diseased",
        "symptoms": "Olive-green to black velvety spots on leaves, becoming cracked and distorted. Fruit develops brown scabby lesions.",
        "organic_control": "Apply copper or sulfur-based sprays before buds open. Rake and destroy fallen leaves in autumn.",
        "chemical_control": "Fungicides such as captan, mancozeb, or myclobutanil at early bud break.",
        "prevention": "Prune trees to improve air circulation and sunlight penetration; plant scab-resistant cultivars (e.g., Liberty, Freedom)."
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Black Rot",
        "pathogen": "Botryosphaeria obtusa (Fungus)",
        "status": "Diseased",
        "symptoms": "Circular purple spots on leaves expanding into 'frog-eye' leaf spots; firm brown rot on fruit with concentric rings.",
        "organic_control": "Prune out dead wood, fire blight cankers, and mummified fruit; apply bio-fungicide Bacillus subtilis.",
        "chemical_control": "Captan or thiophanate-methyl applications during pre-bloom to harvest.",
        "prevention": "Maintain tree vigor with balanced fertilizer; avoid bark injury from mechanical implements."
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "pathogen": "Gymnosporangium juniperi-virginianae (Fungus)",
        "status": "Diseased",
        "symptoms": "Bright yellow-orange spots on upper leaf surfaces; tube-like fungal aecia structures on leaf undersides.",
        "organic_control": "Remove nearby eastern red cedar (Juniperus) trees within 1 km where feasible; spray sulfur early.",
        "chemical_control": "Myclobutanil or propiconazole applied at pink bud through petal fall.",
        "prevention": "Plant resistant cultivars (e.g., Enterprise, Redfree); prune infected juniper galls in late winter."
    },
    "Apple___healthy": {
        "crop": "Apple",
        "disease": "None (Healthy Leaf)",
        "pathogen": "N/A",
        "status": "Healthy",
        "symptoms": "Vibrant uniform green foliage, free from fungal lesions, necrotic edges, or chlorosis.",
        "organic_control": "Continue regular nutrient and water management.",
        "chemical_control": "None required.",
        "prevention": "Maintain regular scouting, soil nutrient testing, and seasonal pruning."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "disease": "Gray Leaf Spot",
        "pathogen": "Cercospora zeae-maydis (Fungus)",
        "status": "Diseased",
        "symptoms": "Small, rectangular tan-to-gray lesions restricted by leaf veins; can coalesce causing premature leaf death.",
        "organic_control": "Crop rotation with non-host crops (soybeans, alfalfa); deep residue tillage.",
        "chemical_control": "Strobilurin or triazole fungicides (azoxystrobin, pyraclostrobin) applied at VT/R1 silking stage.",
        "prevention": "Select hybrid corn varieties with moderate to high resistance ratings; manage crop debris."
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "pathogen": "Puccinia sorghi (Fungus)",
        "status": "Diseased",
        "symptoms": "Oval to elongate cinnamon-brown pustules scattered on both upper and lower leaf surfaces.",
        "organic_control": "Plant resistant corn hybrids; encourage beneficial soil microbial activity.",
        "chemical_control": "Fungicide sprays (pyraclostrobin + metconazole) if rust appears before tasseling on susceptible hybrids.",
        "prevention": "Early planting and avoiding excessive nitrogen fertilization."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "pathogen": "Exserohilum turcicum (Fungus)",
        "status": "Diseased",
        "symptoms": "Long, elliptical cigar-shaped grayish-green to tan lesions (2.5 to 15 cm long) on leaves.",
        "organic_control": "Incorporate infected crop residue through deep plowing; 2-year crop rotation.",
        "chemical_control": "Foliar fungicides (mancozeb, azoxystrobin) applied when weather favors disease progression.",
        "prevention": "Plant hybrids with Ht-gene resistance; maintain balanced potassium levels."
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "disease": "None (Healthy Leaf)",
        "pathogen": "N/A",
        "status": "Healthy",
        "symptoms": "Uniform deep green leaf blade with clear central midrib and no necrotic striping or chlorosis.",
        "organic_control": "Continue optimal N-P-K side-dressing and drip/furrow irrigation.",
        "chemical_control": "None required.",
        "prevention": "Routine field scouting during vegetative and reproductive growth phases."
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "status": "Diseased",
        "symptoms": "Dark brown circular spots with concentric target-like rings on older leaves, surrounded by yellow halos.",
        "organic_control": "Copper hydroxide or copper oxychloride; extract of Reynoutria sachalinensis; mulching to prevent soil splash.",
        "chemical_control": "Chlorothalonil, azoxystrobin, or difenoconazole spray on 7-10 day schedule.",
        "prevention": "Ensure adequate nitrogen and potassium; avoid overhead sprinkler irrigation late in the day."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "status": "Diseased",
        "symptoms": "Water-soaked irregular pale green lesions turning dark brown/black; white fungal fuzz on leaf underside in high humidity.",
        "organic_control": "Bordeaux mixture; copper sulfate sprays; immediate destruction of infected cull piles.",
        "chemical_control": "Metalaxyl, cymoxanil, dimethomorph, or mandipropamid applications.",
        "prevention": "Use certified disease-free seed tubers; ensure wide plant spacing for airflow; destroy volunteer potatoes."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "None (Healthy Leaf)",
        "pathogen": "N/A",
        "status": "Healthy",
        "symptoms": "Compound pinnate leaves with rich green hue, robust cuticle, and absence of blight lesions.",
        "organic_control": "Maintain balanced fertigation and hilling.",
        "chemical_control": "None required.",
        "prevention": "Scout fields twice weekly, especially during cool and humid weather."
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "pathogen": "Xanthomonas perforans (Bacterium)",
        "status": "Diseased",
        "symptoms": "Small, dark, greasy, water-soaked spots (1-3 mm) on leaves, halos developing over time; defoliation in severe cases.",
        "organic_control": "Fixed copper mixed with bio-fungicide Bacillus amyloliquefaciens; compost teas.",
        "chemical_control": "Copper-mancozeb combination sprays or streptomycin on transplants before field setting.",
        "prevention": "Hot-water treat seeds (50°C for 25 min); use drip irrigation to keep foliage dry; rotate crops for 2 years."
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "pathogen": "Alternaria linariae (Fungus)",
        "status": "Diseased",
        "symptoms": "Brown to black spots with concentric rings ('target boards') starting on lowest leaves; yellow chlorosis surrounding lesions.",
        "organic_control": "Copper sprays, neem oil extracts, and pruning bottom 12 inches of foliage after fruit sets.",
        "chemical_control": "Chlorothalonil, mancozeb, or boscalid applied when first lower leaves show symptoms.",
        "prevention": "Mulch soil around plants; stake or cage vines; avoid overhead watering."
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "status": "Diseased",
        "symptoms": "Large, dark greasy-looking spots with white velvety fungal growth on undersides; stems turn dark brown and collapse.",
        "organic_control": "Copper-based fungicides applied preventively; remove and bag infected foliage immediately.",
        "chemical_control": "Cyazofamid, fluopicolide, or chlorothalonil preventive applications.",
        "prevention": "Space plants generously; inspect regularly; choose resistant varieties (e.g., Defiant, Mountain Magic)."
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "pathogen": "Passalora fulva (Fungus)",
        "status": "Diseased",
        "symptoms": "Pale greenish-yellow spots on upper leaf surfaces; olive-green to gray velvety mold growth on corresponding undersides.",
        "organic_control": "Improve greenhouse ventilation; sulfur sprays or bio-fungicide Serenade (Bacillus subtilis).",
        "chemical_control": "Chlorothalonil, mancozeb, or copper fungicides.",
        "prevention": "Lower relative humidity below 85% in greenhouses; increase spacing and install circulation fans."
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici (Fungus)",
        "status": "Diseased",
        "symptoms": "Numerous circular spots (2-3 mm) with dark brown borders and gray/white centers containing tiny black pycnidia.",
        "organic_control": "Remove lower infected foliage; mulch thoroughly; copper soap sprays.",
        "chemical_control": "Chlorothalonil, maneb, or azoxystrobin on a 7-14 day schedule.",
        "prevention": "Avoid handling plants when wet; practice 3-year crop rotation; sterilize garden stakes."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Spider Mites Infestation",
        "pathogen": "Tetranychus urticae (Arachnid Pest)",
        "status": "Diseased",
        "symptoms": "Fine yellow stippling or bronzing on upper leaf surfaces; delicate webbing visible on leaf undersides and stem junctions.",
        "organic_control": "Release predatory mites (Phytoseiulus persimilis); spray neem oil, insecticidal soap, or horticultural mineral oil.",
        "chemical_control": "Abamectin, bifenazate, or spiromesifen (rotate modes of action to prevent resistance).",
        "prevention": "Keep plants well-watered to reduce drought stress; avoid excessive nitrogen fertilizer which stimulates mite reproduction."
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "pathogen": "Corynespora cassiicola (Fungus)",
        "status": "Diseased",
        "symptoms": "Small pinpoint brown lesions expanding into circular brown lesions with distinct concentric target rings.",
        "organic_control": "Copper fungicides and bio-fungicides; enhance row spacing.",
        "chemical_control": "Pyraclostrobin, fluopyram, or chlorothalonil sprays.",
        "prevention": "Maintain adequate calcium nutrition; prune suckers to optimize canopy ventilation."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Begomovirus (transmitted by Whitefly Bemisia tabaci)",
        "status": "Diseased",
        "symptoms": "Severe stunting, erect bushy habit, upward curling of leaf margins, prominent interveinal chlorosis, flower drop.",
        "organic_control": "Yellow sticky cards to trap whiteflies; row covers; insecticidal soaps and neem spray.",
        "chemical_control": "Imidacloprid, thiamethoxam, or pyriproxyfen for systemic vector (whitefly) control.",
        "prevention": "Plant TYLCV-resistant hybrids; eliminate weed hosts around fields; use reflective silver mulches."
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "pathogen": "Tobamovirus (Mechanically transmitted)",
        "status": "Diseased",
        "symptoms": "Mottled light and dark green mosaic patterns on leaves, leaf distortion, 'shoestring' narrowing of leaves, internal fruit browning.",
        "organic_control": "Soak seed in 10% trisodium phosphate (TSP); rogue and burn infected plants immediately.",
        "chemical_control": "No chemical virucides exist; disinfect tools in 20% nonfat dry milk or 10% bleach solution.",
        "prevention": "Wash hands with soap and water before handling plants (especially smokers); use resistant varieties (Tm-22 gene)."
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "None (Healthy Leaf)",
        "pathogen": "N/A",
        "status": "Healthy",
        "symptoms": "Lush dark green serrated leaves with crisp margins, sturdy petioles, and active trichome glandular hairs.",
        "organic_control": "Regular vermicompost top-dressing and consistent soil moisture.",
        "chemical_control": "None required.",
        "prevention": "Maintain soil pH 6.0-6.8 and monitor micronutrients (Ca, Mg, Fe)."
    }
}

# Fill defaults for any remaining classes
for cls in PLANT_DISEASE_CLASSES:
    if cls not in DISEASE_INFO:
        parts = cls.split("___")
        crop_name = parts[0].replace("_", " ").title()
        disease_name = parts[1].replace("_", " ").title() if len(parts) > 1 else "Healthy"
        is_healthy = "healthy" in disease_name.lower()
        DISEASE_INFO[cls] = {
            "crop": crop_name,
            "disease": disease_name,
            "pathogen": "Physiological / Biotic pathogen" if not is_healthy else "N/A",
            "status": "Healthy" if is_healthy else "Diseased",
            "symptoms": f"Standard symptomatic profile for {crop_name} {disease_name}.",
            "organic_control": "Apply organic copper or bio-fungicide preventive spray." if not is_healthy else "None required.",
            "chemical_control": "Consult local agricultural extension office for registered active ingredients." if not is_healthy else "None required.",
            "prevention": "Practice strict field sanitation, balanced NPK fertigation, and certified seed selection."
        }

# ==============================================================================
# 5. CROP RECOMMENDATION TAXONOMY (22 Benchmark Crops)
# ==============================================================================
CROP_RECOMMENDATION_CLASSES = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
    "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
    "banana", "mango", "grapes", "watermelon", "muskmelon",
    "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee"
]

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ==============================================================================
# 6. YIELD PREDICTION FEATURES
# ==============================================================================
YIELD_NUMERIC_FEATURES = [
    "Area", "Rainfall", "Fertilizer_N", "Fertilizer_P", "Fertilizer_K",
    "Temperature", "Humidity", "Soil_pH"
]
YIELD_CATEGORICAL_FEATURES = ["Crop", "State", "Season", "Soil_Type"]
YIELD_TARGET = "Yield"  # Tonnes per hectare (t/ha)
