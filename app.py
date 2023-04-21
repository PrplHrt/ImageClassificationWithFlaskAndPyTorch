# Import the necessary libraries
from flask import Flask, render_template, request
from torchvision import models, transforms
import torch
from torchvision.models import ResNet18_Weights
from PIL import Image
import json

def get_class_name(label):
    with open('labels.json') as f:
        labels = json.load(f)
    return labels[str(label)]


# Define the Flask application
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('home.html')

# Define the image processing route
@app.route('/process', methods=['POST'])
def process():
    # Load the PyTorch model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Get the uploaded image file
    file = request.files['image']
    # Open the image file and apply the transformation
    img = Image.open(file.stream)
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)


    model.eval()
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output.data, 1)
        label = int(predicted.item())
        label = get_class_name(label)

    # Render the results template with the predicted class label
    return render_template('results.html', label=label)

# Define the main function
if __name__ == '__main__':
    app.run(debug=True)
