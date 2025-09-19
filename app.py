import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import networkx as nx
import os
from transformers import AutoModel, AutoConfig
from openai import OpenAI
import json  # for black JSON
import gdown

# .....................................
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    # Replace FILE_ID with Google Drive file ID
    file_id = "1bGRLEC2_5GB53E-zEVH1Z4EQdKGA-YGI"
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)


# -------------------------------
# Load Knowledge Graph + Symptom Map
# -------------------------------
# 1. Create Extended Knowledge Graph (KG)
# -------------------------------
G = nx.DiGraph()

# -------------------------------
# Symptoms → Diseases
# -------------------------------
# Apple
G.add_edge("Olive-brown velvety spots on leaves/fruits", "Apple___Apple_scab")
G.add_edge("Dark sunken lesions with concentric rings", "Apple___Black_rot")
G.add_edge("Orange/yellow spots with black centers", "Apple___Cedar_apple_rust")
G.add_edge("No visible disease", "Apple___healthy")

# Blueberry
G.add_edge("No visible disease", "Blueberry___healthy")

# Cherry
G.add_edge("White powdery coating on leaves", "Cherry_(including_sour)___Powdery_mildew")
G.add_edge("No visible disease", "Cherry_(including_sour)___healthy")

# Corn
G.add_edge("Gray/tan lesions with dark borders", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
G.add_edge("Small reddish-brown pustules on leaves", "Corn_(maize)___Common_rust_")
G.add_edge("Cigar-shaped gray-green lesions on leaves", "Corn_(maize)___Northern_Leaf_Blight")
G.add_edge("No visible disease", "Corn_(maize)___healthy")

# Grape
G.add_edge("Circular black spots on leaves/fruits", "Grape___Black_rot")
G.add_edge("Interveinal chlorosis, necrosis (black measles)", "Grape___Esca_(Black_Measles)")
G.add_edge("Irregular brown spots with yellow halo", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)")
G.add_edge("No visible disease", "Grape___healthy")

# Orange
G.add_edge("Yellow shoots, mottled leaves, misshapen fruits", "Orange___Haunglongbing_(Citrus_greening)")

# Peach 
G.add_edge("Small dark water-soaked spots on leaves", "Peach___Bacterial_spot")
G.add_edge("No visible disease", "Peach___healthy")

# Pepper (typo fix: Pepper instead of 'pper,_bell')
G.add_edge("Brown lesions with yellow halo", "Pepper,_bell___Bacterial_spot")
G.add_edge("No visible disease", "Pepper,_bell___healthy")

# Potato
G.add_edge("Dark concentric spots on leaves", "Potato___Early_blight")
G.add_edge("Large irregular brown/black lesions", "Potato___Late_blight")
G.add_edge("No visible disease", "Potato___healthy")

# Raspberry
G.add_edge("No visible disease", "Raspberry___healthy")

# Soybean
G.add_edge("No visible disease", "Soybean___healthy")

# Squash
G.add_edge("White powdery patches on leaves", "Squash___Powdery_mildew")

# Strawberry
G.add_edge("Irregular brown leaf margins, scorching", "Strawberry___Leaf_scorch")
G.add_edge("No visible disease", "Strawberry___healthy")

# Tomato
G.add_edge("Water-soaked brown spots on leaves", "Tomato___Bacterial_spot")
G.add_edge("Concentric rings, target-like spots", "Tomato___Early_blight")
G.add_edge("Large dark blotches with fuzzy growth", "Tomato___Late_blight")
G.add_edge("Yellow patches on upper leaf, fuzzy underside", "Tomato___Leaf_Mold")
G.add_edge("Small circular dark spots with yellow halo", "Tomato___Septoria_leaf_spot")
G.add_edge("White/yellow stippling + webbing", "Tomato___Spider_mites Two-spotted_spider_mite")
G.add_edge("Brown/black target-like spots", "Tomato___Target_Spot")
G.add_edge("Leaf curling + yellow mosaic", "Tomato___Tomato_Yellow_Leaf_Curl_Virus")
G.add_edge("Mosaic mottling on leaves", "Tomato___Tomato_mosaic_virus")
G.add_edge("No visible disease", "Tomato___healthy")

# -------------------------------
# Diseases → Treatments
# -------------------------------
# Apple
G.add_edge("Apple___Apple_scab", "Fungicides (captan, myclobutanil), prune infected leaves")
G.add_edge("Apple___Black_rot", "Remove mummified fruits, use fungicides")
G.add_edge("Apple___Cedar_apple_rust", "Remove nearby junipers, apply fungicides")
G.add_edge("Apple___healthy", "No treatment needed")

# Blueberry
G.add_edge("Blueberry___healthy", "No treatment needed")

# Cherry
G.add_edge("Cherry_(including_sour)___Powdery_mildew", "Fungicides: sulfur, myclobutanil")
G.add_edge("Cherry_(including_sour)___healthy", "No treatment needed")

# Corn
G.add_edge("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Use resistant hybrids, fungicides")
G.add_edge("Corn_(maize)___Common_rust_", "Resistant hybrids, fungicide if severe")
G.add_edge("Corn_(maize)___Northern_Leaf_Blight", "Crop rotation, resistant varieties, fungicides")
G.add_edge("Corn_(maize)___healthy", "No treatment needed")

# Grape
G.add_edge("Grape___Black_rot", "Fungicides (mancozeb, myclobutanil), prune infected vines")
G.add_edge("Grape___Esca_(Black_Measles)", "Remove infected wood, fungicides not very effective")
G.add_edge("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Remove infected leaves, fungicides")
G.add_edge("Grape___healthy", "No treatment needed")

# Orange
G.add_edge("Orange___Haunglongbing_(Citrus_greening)", "No cure, control psyllid vector, use resistant rootstocks")

# Peach
G.add_edge("Peach___Bacterial_spot", "Copper fungicides, resistant varieties")
G.add_edge("Peach___healthy", "No treatment needed")

# Pepper
G.add_edge("Pepper,_bell___Bacterial_spot", "Use copper-based bactericides, resistant cultivars")
G.add_edge("Pepper,_bell___healthy", "No treatment needed")

# Potato
G.add_edge("Potato___Early_blight", "Use fungicides: Mancozeb, Chlorothalonil")
G.add_edge("Potato___Late_blight", "Copper-based fungicides, resistant varieties")
G.add_edge("Potato___healthy", "No treatment needed")

# Raspberry
G.add_edge("Raspberry___healthy", "No treatment needed")

# Soybean
G.add_edge("Soybean___healthy", "No treatment needed")

# Squash
G.add_edge("Squash___Powdery_mildew", "Sulfur-based fungicides, resistant varieties")

# Strawberry
G.add_edge("Strawberry___Leaf_scorch", "Remove infected leaves, apply fungicides")
G.add_edge("Strawberry___healthy", "No treatment needed")

# Tomato
G.add_edge("Tomato___Bacterial_spot", "Copper sprays, avoid overhead irrigation")
G.add_edge("Tomato___Early_blight", "Fungicides: Chlorothalonil, crop rotation")
G.add_edge("Tomato___Late_blight", "Copper fungicides, remove infected plants")
G.add_edge("Tomato___Leaf_Mold", "Fungicides: Chlorothalonil, improve ventilation")
G.add_edge("Tomato___Septoria_leaf_spot", "Apply fungicides, remove infected leaves")
G.add_edge("Tomato___Spider_mites Two-spotted_spider_mite", "Insecticidal soap, neem oil, predatory mites")
G.add_edge("Tomato___Target_Spot", "Fungicides: Chlorothalonil, Mancozeb")
G.add_edge("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "No cure: use resistant varieties, control whiteflies")
G.add_edge("Tomato___Tomato_mosaic_virus", "Remove infected plants, disinfect tools")
G.add_edge("Tomato___healthy", "No treatment needed")

# -------------------------------
# 2. Map Model Predictions → Symptoms
# -------------------------------
symptom_map = {
    "Apple___Apple_scab": "Olive-brown velvety spots on leaves/fruits",
    "Apple___Black_rot": "Dark sunken lesions with concentric rings",
    "Apple___Cedar_apple_rust": "Orange/yellow spots with black centers",
    "Apple___healthy": "No visible disease",

    "Blueberry___healthy": "No visible disease",

    "Cherry_(including_sour)___Powdery_mildew": "White powdery coating on leaves",
    "Cherry_(including_sour)___healthy": "No visible disease",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray/tan lesions with dark borders",
    "Corn_(maize)___Common_rust_": "Small reddish-brown pustules on leaves",
    "Corn_(maize)___Northern_Leaf_Blight": "Cigar-shaped gray-green lesions on leaves",
    "Corn_(maize)___healthy": "No visible disease",

    "Grape___Black_rot": "Circular black spots on leaves/fruits",
    "Grape___Esca_(Black_Measles)": "Interveinal chlorosis, necrosis (black measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Irregular brown spots with yellow halo",
    "Grape___healthy": "No visible disease",

    "Orange___Haunglongbing_(Citrus_greening)": "Yellow shoots, mottled leaves, misshapen fruits",

    "Peach___Bacterial_spot": "Small dark water-soaked spots on leaves",
    "Peach___healthy": "No visible disease",

    "Pepper,_bell___Bacterial_spot": "Brown lesions with yellow halo",
    "Pepper,_bell___healthy": "No visible disease",

    "Potato___Early_blight": "Dark concentric spots on leaves",
    "Potato___Late_blight": "Large irregular brown/black lesions",
    "Potato___healthy": "No visible disease",

    "Raspberry___healthy": "No visible disease",
    "Soybean___healthy": "No visible disease",

    "Squash___Powdery_mildew": "White powdery patches on leaves",

    "Strawberry___Leaf_scorch": "Irregular brown leaf margins, scorching",
    "Strawberry___healthy": "No visible disease",

    "Tomato___Bacterial_spot": "Water-soaked brown spots on leaves",
    "Tomato___Early_blight": "Concentric rings, target-like spots",
    "Tomato___Late_blight": "Large dark blotches with fuzzy growth",
    "Tomato___Leaf_Mold": "Yellow patches on upper leaf, fuzzy underside",
    "Tomato___Septoria_leaf_spot": "Small circular dark spots with yellow halo",
    "Tomato___Spider_mites Two-spotted_spider_mite": "White/yellow stippling + webbing",
    "Tomato___Target_Spot": "Brown/black target-like spots",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Leaf curling + yellow mosaic",
    "Tomato___Tomato_mosaic_virus": "Mosaic mottling on leaves",
    "Tomato___healthy": "No visible disease"
}

# -------------------------------
# Model Setup
# -------------------------------
class DinoClassifier(nn.Module):
    def __init__(self, base_model, num_classes, hidden_size):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = self.base(x)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
hf_token = os.environ.get("HF_TOKEN")  # Token from environment variable

config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)
base_model = AutoModel.from_pretrained(model_name, config=config, use_auth_token=hf_token)


num_classes = 38
model = DinoClassifier(base_model, num_classes, config.hidden_size)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["classes"]
model.eval()

# -------------------------------
# Image Preprocessing
# -------------------------------
val_test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])

# -------------------------------
# OpenAI API Setup
# -------------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------------------
# Prediction Pipeline (fast)
# -------------------------------
def predict_pipeline(image: Image.Image, get_explanation: bool = True):
    # Preprocess
    x = val_test_tfms(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(1).item()
        pred_class = class_names[pred_idx]

    # KG Lookup
    symptom = symptom_map.get(pred_class, "Unknown")
    treatment = list(G.neighbors(pred_class)) if pred_class in G else ["No treatment found"]

    # JSON summary (black text)
    result_json = {
        "Predicted Disease": pred_class,
        "Symptom": symptom,
        "Treatment": ", ".join(treatment),
    }

    # GPT explanation (optional, can be skipped to speed up)
    if get_explanation:
        prompt = f"""
        You are an agriculture expert.
        Disease: {pred_class}
        Symptom: {symptom}
        Treatment: {', '.join(treatment)}

        Provide a **detailed explanation** in Markdown with sections:
        1. What causes it  
        2. How to control it  
        3. Prevention methods
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            llm_text = response.choices[0].message.content
        except Exception as e:
            llm_text = f"**LLM Error:** {str(e)}"
    else:
        llm_text = "GPT explanation skipped for speed."

    return json.dumps(result_json, indent=2), llm_text

# -------------------------------
# Gradio App 
# -------------------------------
demo = gr.Interface(
    fn=predict_pipeline,
    inputs=[gr.Image(type="pil")],
    outputs=[
        gr.Textbox(label="📊 Prediction Summary", lines=10, interactive=False),
        gr.Markdown(label="📖 Detailed Explanation")
    ],
    title="🌱 Plant Disease Detection with KG + GPT-4o-mini",
    description="Upload a plant leaf image to detect disease, get symptoms, treatment, and optionally an expert explanation."
)

if __name__ == "__main__":
    demo.launch()



