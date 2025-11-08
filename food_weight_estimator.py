import cv2
import os
import numpy as np
import pandas as pd

# ==============================================
# CONFIG
# ==============================================
TRAIN_DIR = "dataset/train"
OUTPUT_CSV = "food_dataset_train.csv"

# Approximate base weights per food type (grams for full image)
BASE_WEIGHTS = {
    "burger": 250,
    "cake": 120,
    "ice_cream": 100,
    "pizza": 180,
    "salad": 150,
    "samosa": 80,
    "sandwich": 200,
    "steak": 300
}

# ==============================================
# Function: Estimate Food Weight
# ==============================================
def estimate_weight(food_name, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)

    # Find contours (possible food boundaries)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None

    # Get the largest contour (most likely the food)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area = w * h

    # Calculate image area and bounding box ratio
    img_area = img.shape[0] * img.shape[1]
    area_ratio = area / img_area

    # Estimate weight based on area ratio and food base weight
    base_weight = BASE_WEIGHTS.get(food_name.lower(), 150)
    estimated_weight = round(base_weight * area_ratio * 2, 2)  # *2 to scale visible portion

    # Draw bounding box for visualization
    boxed = img.copy()
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save output image with box
    output_dir = "dataset_with_boxes"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, boxed)

    return area_ratio, estimated_weight, out_path


# ==============================================
# MAIN
# ==============================================
def process_dataset():
    results = []
    for food_class in os.listdir(TRAIN_DIR):
        food_path = os.path.join(TRAIN_DIR, food_class)
        if not os.path.isdir(food_path):
            continue

        for img_name in os.listdir(food_path):
            img_path = os.path.join(food_path, img_name)
            area_ratio, weight, _ = estimate_weight(food_class, img_path)

            if weight:
                results.append({
                    "Food": food_class,
                    "Image": img_name,
                    "AreaRatio": area_ratio,
                    "Weight(g)": weight
                })

    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Food weight dataset created: {OUTPUT_CSV} ({len(results)} entries)")


if __name__ == "__main__":
    process_dataset()