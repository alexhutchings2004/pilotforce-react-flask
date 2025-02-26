import os
import uuid
import time
from datetime import datetime
import boto3
from flask import Flask, jsonify
from ultralytics import YOLO
import threading
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Initialize S3 client
s3 = boto3.client('s3')
CORS(app)  # Apply CORS to the entire app

# Define your S3 bucket name
bucket_name = 'drone-images-bucket'

# Path to your model
model_path = 'best.pt'  # Ensure this path is correct

# Load YOLOv8 model
model = YOLO(model_path)

# Directory for downloading images and saving predictions
download_dir = 'downloads/uploads'
predictions_dir = 'downloads/predictions'

# Ensure necessary directories exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

def download_and_infer(image_key):
    # Construct the local file path for downloading
    local_image_path = os.path.join(download_dir, image_key.split('/')[-1])

    print(f"Attempting to download image: s3://{bucket_name}/{image_key}")
    s3.download_file(bucket_name, image_key, local_image_path)
    print(f"Downloaded image to: {local_image_path}")

    # Perform inference on the image
    results = model(local_image_path)  # Run YOLOv8 inference

    # Extract the original image name without extension
    original_image_name = os.path.splitext(os.path.basename(local_image_path))[0]
    
    # Generate a unique identifier based on UUID and timestamp
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save prediction images with unique filenames
    for idx, result in enumerate(results):
        # Construct the filename with the unique identifier
        unique_filename = f"{original_image_name}_predicted_{timestamp}_{unique_id}_{idx}.jpg"
        unique_filepath = os.path.join(predictions_dir, unique_filename)
        
        # Save the predicted image
        result.save(filename=unique_filepath)  # Save each result with a unique filename
        print(f"Saved predicted image: {unique_filepath}")

    # Upload predicted images to the 'predictions' folder in the bucket
    for filename in os.listdir(predictions_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(predictions_dir, filename)
            s3.upload_file(file_path, bucket_name, f'predictions/{filename}')  # Upload to S3

    print(f"Uploaded predictions to S3: s3://{bucket_name}/predictions/")

def monitor_s3():
    # Monitor the 'uploads' folder for new files
    print("Monitoring S3 bucket for new uploads...")
    processed_files = set()  # Keep track of processed files
    while True:
        # List objects in the 'uploads' folder
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='uploads/')
        if 'Contents' in response:
            for obj in response['Contents']:
                image_key = obj['Key']
                if not image_key.endswith('/') and image_key not in processed_files:  # Skip directories and already processed images
                    print(f"Found new image: {image_key}")
                    download_and_infer(image_key)
                    processed_files.add(image_key)  # Mark the image as processed
        time.sleep(5)  # Check every 5 seconds for new files

@app.route('/')
def index():
    return jsonify({"message": "Flask API is running."})  # Serve a simple message

@app.route('/api/predictions', methods=['GET'])
def show_predictions():
    # List all images in the 'predictions' folder from S3
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='predictions/')
    images = []

    # Fetch images from 'predictions' folder
    if 'Contents' in response:
        for obj in response['Contents']:
            # Generate pre-signed URL for each image in the 'predictions' folder
            image_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': obj['Key']},
                ExpiresIn=3600  # URL expires in 1 hour
            )
            images.append(image_url)

    return jsonify({"predictions": images})

if __name__ == '__main__':
    # Start S3 monitoring in a separate thread
    monitoring_thread = threading.Thread(target=monitor_s3)
    monitoring_thread.start()

    # Start Flask application
    app.run(debug=True, port=5001)
