from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans
from skimage import color

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# PAPER SHADES (RGB)
# -------------------------------
PAPER_SHADES = {
    "Bright White": [255, 255, 255],
    "Soft White": [245, 245, 240],
    "Cream": [255, 253, 208],
    "Ivory": [255, 255, 240],
    "Light Gray": [211, 211, 211],
    "Warm White": [255, 244, 229],
    "Cool White": [240, 248, 255],
    "Off White": [250, 249, 246]
}

# -------------------------------
# RGB → LAB
# -------------------------------
def rgb_to_lab(rgb):
    rgb_array = np.array([[rgb]]) / 255.0
    return color.rgb2lab(rgb_array)[0][0]

# Precompute LAB shades
LAB_SHADES = {
    name: rgb_to_lab(rgb)
    for name, rgb in PAPER_SHADES.items()
}

# -------------------------------
# Distance (LAB space)
# -------------------------------
def calculate_distance(c1, c2):
    return np.linalg.norm(c1 - c2)

# -------------------------------
# Root
# -------------------------------
@app.get("/")
def home():
    return {"message": "Paper Color Analyzer running 🚀"}

# -------------------------------
# Analyze API
# -------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize (faster processing)
    image = image.resize((200, 200))

    # Convert to array
    image_array = np.array(image)
    pixels = image_array.reshape((-1, 3))

    # Remove very dark pixels (noise)
    pixels = pixels[np.mean(pixels, axis=1) > 40]

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Count pixels
    counts = np.bincount(labels)
    total = len(labels)
    percentages = (counts / total) * 100

    # Combine info
    color_info = []
    for i in range(len(colors)):
        color_info.append({
            "rgb": colors[i].tolist(),
            "percentage": round(percentages[i], 2)
        })

    # Sort by dominance
    color_info = sorted(color_info, key=lambda x: x["percentage"], reverse=True)

    dominant_rgb = color_info[0]["rgb"]
    dominant_lab = rgb_to_lab(dominant_rgb)

    # Find closest shade
    closest_shade = None
    min_distance = float("inf")

    for shade, lab_value in LAB_SHADES.items():
        dist = calculate_distance(dominant_lab, lab_value)
        if dist < min_distance:
            min_distance = dist
            closest_shade = shade

    # Confidence score
    confidence = round(1 / (1 + min_distance), 3)

    return {
        "dominant_color": dominant_rgb,
        "all_colors": color_info,
        "closest_shade": closest_shade,
        "confidence": confidence
    }
