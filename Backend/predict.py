from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.layers.TFSMLayer("model", call_endpoint='serving_default')

# Load the labels
class_names = [line.strip() for line in open("model/labels.txt", "r").readlines()]

# Create the array for one image
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load the image
image_path = "../test_images/AnnualCrop_54.jpg"
image = Image.open(image_path).convert("RGB")

# Resize and crop to 224x224
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Convert image to array
image_array = np.asarray(image)

# Normalize to [-1, 1]
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Run prediction
pred = model(tf.convert_to_tensor(data, dtype=tf.float32))

# Extract predictions (automatically handles unknown key name)
predictions = list(pred.values())[0].numpy()[0]

# Get predicted class and confidence
index = np.argmax(predictions)
class_name = class_names[index]
confidence_score = predictions[index]

# Print results
print(f"Class: {class_name}")
print(f"Confidence Score: {confidence_score:.2f}")

