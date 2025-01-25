import os
import subprocess

import torch
from flask import Flask, jsonify, request
from transformers import BertForSequenceClassification, BertTokenizer

# Initialize Flask app
app = Flask(__name__)

# Function to download the model if it doesn't exist
def download_model():
    model_path = "./new_model/model.safetensors"
    
    # Check if the model already exists to avoid re-downloading
    if not os.path.exists(model_path):
        print("Model not found. Downloading...")
        
        # URL of the file from GitHub LFS (update this with your actual link)
        model_url = "https://github.com/colton456p/DiaBERT/raw/main/new_model/model.safetensors"

        # Download the model file
        try:
            subprocess.run(["wget", model_url, "-O", model_path], check=True)
            print("Model downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model: {e}")
            exit(1)

# Download model if not present
download_model()

# Load the fine-tuned model and tokenizer
model_directory = "./new_model"  # Directory containing the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_directory)
tokenizer = BertTokenizer.from_pretrained(model_directory)

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Label mapping
label_mapping = {1: "false", 0: "real"}

# Define a route for the root URL
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the BERT Fine-tuned Model API!",
        "instructions": "Use the /predict endpoint with a POST request to classify text."
    })

# Define route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON request
        data = request.get_json()
        text = data.get("text", None)
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        # Get predicted label and format probabilities
        predicted_label_index = probabilities.argmax()
        predicted_label = label_mapping[predicted_label_index]
        probabilities_percentage = (probabilities * 100).round(2).tolist()

        # Prepare response
        response = {
            "text": text,
            "predicted_label": predicted_label,  # Predicted label in human-readable form
            "probabilities": {
                "false": probabilities_percentage[1],
                "real": probabilities_percentage[0]
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
