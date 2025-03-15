import pickle
import cv2
import numpy as np

# Load the trained gender prediction model
model_filename = "aux-gpred.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Preprocess the input image for the gender prediction model.
    - Converts image to grayscale
    - Resizes to target size
    - Normalizes pixel values
    
    :param image_path: Path to the input image
    :param target_size: Target image size (width, height)
    :return: Processed image as a NumPy array
    """
    image = cv2.imread(image_path)  # Read image
    if image is None:
        raise ValueError("Error: Unable to read image. Check file path.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, target_size)  # Resize to match model input
    image = image / 255.0  # Normalize pixel values (0 to 1)
    
    return image.flatten().reshape(1, -1)  # Flatten and reshape for model

def predict_gender(image_path):
    """
    Predicts gender based on an uploaded image.
    
    :param image_path: Path to the input image
    :return: Predicted gender (Male/Female)
    """
    processed_image = preprocess_image(image_path)  # Preprocess image
    prediction = model.predict(processed_image)  # Make prediction
    
    return "Male" if prediction[0] == 1 else "Female"

# Example usage
image_path = "5.png"  # Change this to your image path
predicted_gender = predict_gender(image_path)
print("Predicted Gender:", predicted_gender)
