import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from transformers import pipeline
from PIL import Image, ImageDraw
import uuid

app = Flask(__name__)

# Set up the upload folder and configure allowed extensions
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# Load the Hugging Face object detection model
detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the request contains a file
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file or an unsupported file
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        # Save the uploaded file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform object detection
        image = Image.open(filepath)
        detections = detector(image)

        # Draw bounding boxes and labels on the image
        draw = ImageDraw.Draw(image)
        detected_objects = []  # List to store detected objects and scores
        for obj in detections:
            label = obj.get('label', 'Unknown')
            score = obj.get('score', 0.0)  # Set a default score of 0.0 if missing
            box = obj['box']
            
            # Append to the detected_objects list for display
            detected_objects.append({'label': label, 'score': score})
            
            # Draw rectangle
            draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="red", width=2)
            # Add label text
            draw.text((box['xmin'], box['ymin'] - 10), f"{label} ({score:.2f})", fill="red")

        # Save the processed image with bounding boxes
        output_filename = f"detected_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        image.save(output_path)

        # Redirect to result page with filename and detections
        return render_template("result.html", filename=output_filename, objects=detected_objects)

    return render_template("index.html")


@app.route("/uploads/<filename>")
def show_image(filename):
    # Get the list of detected objects from the query parameters
    objects = request.args.getlist('objects')
    return render_template("result.html", filename=filename, objects=objects)

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
