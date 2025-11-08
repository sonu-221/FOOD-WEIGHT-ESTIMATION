import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import argparse
import pandas as pd  # <-- Added for CSV reading

# =====================
# 1. Paths
# =====================
train_dir = "dataset/train"
test_dir = "dataset/test"
model_path = "food_classifier.h5"
nutrition_csv = "nutrition_info.csv"  # <-- CSV file with nutrition data

# =====================
# 2. Data Preprocessing
# =====================
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# The class labels based on folder names
labels = list(train_data.class_indices.keys())

# =====================
# 3. Build Model
# =====================
def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze base model

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =====================
# 4. Train Model
# =====================
def train_model():
    model = build_model(len(labels))
    print("\n🚀 Training model...")
    model.fit(train_data, validation_data=test_data, epochs=8)
    model.save(model_path)
    print(f"\n✅ Model saved as {model_path}")

# =====================
# 5. Predict Food with Nutrition Info
# =====================
def predict_food(img_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("⚠️ Model not found! Please train the model first.")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    predicted_food = labels[class_idx]

    # Load nutrition info from CSV
    if not os.path.exists(nutrition_csv):
        raise FileNotFoundError(f"⚠️ Nutrition file '{nutrition_csv}' not found.")

    nutrition_df = pd.read_csv(nutrition_csv)

    # Match predicted food
    nutrition_info = nutrition_df[nutrition_df['food_name'] == predicted_food]
    if nutrition_info.empty:
        return predicted_food, None

    return predicted_food, nutrition_info.to_dict(orient="records")[0]

# =====================
# 6. CLI Interface
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Path to image for prediction")
    args = parser.parse_args()

    if args.train:
        train_model()

    if args.predict:
        food_name, nutrition = predict_food(args.predict)
        print(f"\n🍽 Predicted Food: {food_name}")
        if nutrition:
            print("📊 Nutritional Info:")
            for key, value in nutrition.items():
                if key != "food_name":
                    print(f"   {key.capitalize()}: {value}")
        else:
            print("⚠️ No nutritional data available for this food.")