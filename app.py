from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_gaussian_blur(image, sigma=1):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def apply_box_blur(image, ksize=5):
    return cv2.boxFilter(image, -1, (ksize, ksize))

def apply_median_filter(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def rgb_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read the image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing

        return render_template('process.html', filename=filename)

@app.route('/process/<filename>', methods=['POST'])
def process_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing

    action = request.form.get('action')
    processed_image = None

    if action == 'gaussian':
        processed_image = apply_gaussian_blur(image, sigma=2)
    elif action == 'box':
        processed_image = apply_box_blur(image, ksize=5)
    elif action == 'median':
        processed_image = apply_median_filter(image, ksize=5)
    elif action == 'bilateral':
        processed_image = apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
    elif action == 'gray':
        processed_image = rgb_to_gray(image)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for saving
    elif action == 'hsv':
        processed_image = rgb_to_hsv(image)
    elif action == 'ycbcr':
        processed_image = rgb_to_ycbcr(image)
    
    # Save processed image
    processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{action}_{filename}')
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    cv2.imwrite(processed_file_path, processed_image)

    return render_template('result.html', filename=f'{action}_{filename}')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
