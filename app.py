# Import the necessary libraries
from flask import Flask, render_template, request
from torchvision import models, transforms
import torch
from torchvision.models import ResNet18_Weights
from PIL import Image, ImageDraw, ImageFont
import json
import pytesseract
import cv2
import numpy as np

def get_class_name(label):
    with open('labels.json') as f:
        labels = json.load(f)
    return labels[str(label)]


# Define the Flask application
app = Flask(__name__)

# Define the home page route
@app.route('/')
def menu():
    return render_template('menu.html')

@app.route('/ocr_menu')
def ocr_menu():
    return render_template('ocr.html')

@app.route('/ocr', methods=['POST'])
def ocr():
     # Check if file was uploaded
    if 'file' not in request.files:
        return 'No image uploaded'
    
    # Read image file and perform OCR
    file = request.files['file']
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    
    # Draw bounding boxes around recognized text
    draw = ImageDraw.Draw(image)
    boxes = pytesseract.image_to_boxes(image)
    print(boxes)
    font = ImageFont.truetype("static/MSMINCHO.TTF", size=30)
    for b in boxes.splitlines():
        b = b.split(' ')
        draw.rectangle(((int(b[1]), image.height - int(b[2])),
                       (int(b[3]), image.height - int(b[4]))),
                       outline='green', width=3)
        draw.text((int(b[1]), image.height - int(b[2]) - 15), b[0], fill=(255,0,0), font=font, stroke_width=2)
    
    # Save the modified image to a temporary file
    temp_file = 'static/temp.jpg'
    image.save(temp_file)
    
    # Render the results page with the recognized text and image
    return render_template('ocr_results.html', ocr_text=text, image=temp_file)

@app.route('/home')
def home():
    return render_template('home.html')

# Define the image processing route
@app.route('/classify', methods=['POST'])
def classify():
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

