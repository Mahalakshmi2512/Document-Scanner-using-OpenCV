from flask import Flask, request, send_from_directory, jsonify
import os
import cv2
import numpy as np

app = Flask(__name__)

# Define the upload and processed folders and ensure they exist
upload_folder = 'uploads'
processed_folder = 'processed'
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['PROCESSED_FOLDER'] = processed_folder

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    # Check if the filename is empty
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Construct the file path for saving the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    print(f"Uploading file to: {file_path}")  # Debug print

    try:
        # Save the uploaded file
        file.save(file_path)
        print(f"File saved: {os.path.exists(file_path)}")  # Check if the file was saved

        # Process the image and get the paths of the processed images
        final_image_paths = process_image(file_path)

        return jsonify({
            'message': 'File uploaded and processed successfully.',
            'final_images': final_image_paths  # Return list of filenames for all processed steps
        }), 200

    except Exception as e:
        return jsonify({'message': f'Error saving file: {str(e)}'}), 500

@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    try:
        # Serve the requested image from the processed folder
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'message': 'File not found'}), 404

def process_image(file_path):
    # Load the original image
    original_image = cv2.imread(file_path)

    # Step 1: Convert to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image_path = save_image(gray_image, 'gray_image.jpg')

    # Step 2: Equalize histogram for better contrast
    equalized_image = cv2.equalizeHist(gray_image)
    equalized_image_path = save_image(equalized_image, 'equalized_image.jpg')

    # Step 3: Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    blurred_image_path = save_image(blurred_image, 'blurred_image.jpg')

    # Step 4: Apply Adaptive Thresholding
    adaptive_threshold_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    adaptive_threshold_image_path = save_image(adaptive_threshold_image, 'adaptive_threshold_image.jpg')

    # Step 5: Enhance the image quality using the custom function
    enhanced_image = enhance_image_quality(original_image)
    enhanced_image_path = save_image(enhanced_image, 'enhanced_image.jpg')

    # Step 6: Save the final scanned image as 'result.jpg'
    final_image_path = save_image(enhanced_image, 'result.jpg')

    # Return the paths of all processed images, including the final 'result.jpg'
    return [
        gray_image_path.split('/')[-1], 
        equalized_image_path.split('/')[-1], 
        blurred_image_path.split('/')[-1], 
        adaptive_threshold_image_path.split('/')[-1], 
        enhanced_image_path.split('/')[-1], 
        final_image_path.split('/')[-1]  # Final result image
    ]

def enhance_image_quality(image):
    # Enhance image quality using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L-channel with a and b channels
    enhanced_lab_image = cv2.merge((cl, a_channel, b_channel))

    # Convert LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    return enhanced_image

def save_image(image, filename):
    # Construct the file path for saving the processed image
    result_image_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    # Save the processed image
    cv2.imwrite(result_image_path, image)
    print(f"Processed image saved: {result_image_path}")

    return result_image_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


    