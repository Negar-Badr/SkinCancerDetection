from flask import Flask, request, jsonify
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import os

#curl -X POST -F "file=@/Path/to_image" http://127.0.0.1:5000/predict 

app = Flask(__name__)

import os

# Base project directory (you can dynamically fetch this if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the model relative to the project directory
MODEL_PATH = os.path.join(BASE_DIR, "model", "vgg16_finetuned.pth")


# Define and load the model
model = models.vgg16()

# Customize the classifier from the one we did 
num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2) 
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Map to appropriate device
model.to(device)                                                    # Send the model to the appropriate device
model.eval()                                                        # Set to evaluation mode

# Create a folder for uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match your model input size
    transforms.ToTensor()
])
@app.route('/predict', methods=['POST'])
def predict():

    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400 
    
    # Get the file from the request
    file = request.files['file']  

    try:
        # Save the uploaded file to the server
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  # Define the path to save the file
        file.save(file_path)  # Save the file to the defined path

        # Open the saved image file and preprocess it
        # RGB is the most commonly used format
        image = Image.open(file_path).convert('RGB')     
        
        # Apply preprocessing (resize, convert to tensor) and add batch dimension
        image = transform(image).unsqueeze(0).to(device)  

        # Run the image through the model to get predictions
        with torch.no_grad(): 
            output = model(image) 
            _, predicted = torch.max(output, 1)  # Get the predicted class index (highest score)
            result = predicted.item() 

        # Clean up the uploaded file after prediction
        os.remove(file_path) 

        # Return the prediction result as JSON
        return jsonify({"result": int(result)})

    except Exception as e:
        # Handle any exceptions 
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # I put the Flask app in debug mode to test, we can change to false
    app.run(debug=True)
