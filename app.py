import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import imutils
import random
from flask import Flask, render_template, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Helper function to process the image and detect license plate
def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter for noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Find the contour that corresponds to the license plate
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        # Masking for the area where the number is located
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # Get coordinates to crop the image
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # Use EasyOCR to read the text from the cropped image
        result = reader.readtext(cropped_image)

        # Extract the text
        if result:
            text = result[0][-2]
            return text, img, location
        else:
            return "No text detected", img, location
    else:
        return "License plate not found", img, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save the uploaded image to a temporary location
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    # Process the image to detect the license plate
    text, processed_image, location = process_image(file_path)

    # Convert the processed image to a format that can be displayed in HTML
    processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    plt.imshow(processed_image_bgr)
    plt.axis('off')
    plt.tight_layout()
    processed_image_path = os.path.join(upload_folder, 'processed_image.png')
    plt.savefig(processed_image_path)
    plt.close()

    return jsonify({
        "text": text,
        "processed_image_url": processed_image_path
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

