from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ===================
# Config
# ===================
MODEL_PATH = "food_classifier.h5"
CSV_PATH = "nutrition_info.csv"
DATASET_PATH = "dataset/train"
UPLOAD_FOLDER = "uploads"
FOOD_WEIGHT_CSV = "food_dataset_train.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and data
print("🔹 Loading model and data...")
labels = sorted(os.listdir(DATASET_PATH))
model = tf.keras.models.load_model(MODEL_PATH)
nutrition_df = pd.read_csv(CSV_PATH)
food_weight_df = pd.read_csv(FOOD_WEIGHT_CSV)
print("✅ Model and CSVs loaded successfully.")

# Flask App
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ===================
# Helper: Food Area Ratio
# ===================
def get_food_area_ratio(img_path):
    """Detects largest contour area ratio in image (approx size reference)."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 30, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img_area = img.shape[0] * img.shape[1]
        area_ratio = (w * h) / img_area
        return round(area_ratio, 3)
    except Exception as e:
        print(f"⚠️ Error computing area ratio: {e}")
        return None


# ===================
# Prediction Function
# ===================
def predict_food(img_path):
    """Predict food type and estimate weight + nutrition."""
    # Preprocess for model
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    predicted_food = labels[class_idx]
    print(f"🍽 Predicted: {predicted_food}")

    # Get base weight
    weight_info = food_weight_df[food_weight_df["Food"].str.lower() == predicted_food.lower()]
    estimated_weight_g = None

    if not weight_info.empty:
        base_weight = float(weight_info.iloc[0]["Weight(g)"])
        area_ratio = get_food_area_ratio(img_path) or 0.4  # default fallback
        # Scale weight moderately (avoid extremes)
        estimated_weight_g = round(base_weight * (0.6 + area_ratio * 1.2), 2)
    else:
        estimated_weight_g = 100.0  # fallback default

    # Get nutrition info
    nutrition_info = nutrition_df[nutrition_df["food"].str.lower() == predicted_food.lower()]
    nutrition = {}

    if not nutrition_info.empty:
        nutrition = nutrition_info.iloc[0].to_dict()

    # ✅ Ensure consistent key name for frontend
    nutrition["predicted_weight_g"] = estimated_weight_g

    return predicted_food, nutrition


# ===================
# Routes
# ===================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        food_name, nutrition = predict_food(filepath)
        response = {
            "food_name": food_name,
            "nutrition": nutrition
        }
        print(f"✅ Response: {response}")
        return jsonify(response)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)