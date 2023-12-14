# import required modules
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
from keras.models import load_model
import jsonify
from flask import jsonify, make_response
from datetime import datetime
import uuid


# Model to Use
predicted_results = []
model_path = './models/Multiclass-Recycool.h5'
loaded_model = load_model(model_path)

def processed_image(image_path):
    file = request.files['file']
    filepath = f'static/temp/{file.filename}'
    file.save(filepath) # save to directory
    # Read and preprocess the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_normalized = img_resized / 255.0 

    # Perform inference
    prediction = loaded_model.predict(np.expand_dims(img_normalized, axis=0))

    # Get the predicted class index and confidence score
    predicted_class_index = np.argmax(prediction)
    confidence_score = np.max(prediction)

    # Define class names (replace with your class names)
    class_names = ['Kaca', 'Kardus', 'Kertas', 'Makanan', 'Plastik ']

    # Get the class name based on the predicted index
    predicted_class_name = class_names[predicted_class_index]

    # Overlay bounding box and text on the image
    h, w, _ = img.shape
    ymin, xmin, ymax, xmax = 50, 50, 200, 200
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    label = f'{predicted_class_name}: {confidence_score:.2f}'
    cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with the bounding box and label
    cv2.imshow('Detected Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Prepare response JSON
    response = {
        'predicted_class': predicted_class_name,
        'confidence_score': float(confidence_score),
        'image_path': f"{request.url_root}{filepath}"
    }

    return jsonify(response)



# create flask app
app = Flask(__name__)

# Function to validate UUID
def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


#Building API
@app.route('/')
def index():
    return render_template('index.html', message='Welcome to API.')

# Post Method
@app.route('/recycool', methods=['POST'])
def predict():
    try:
        # Error Handling
        if 'file' not in request.files:
            error_response = {
                'status': 'ERROR',
                'message': 'No file request'
            }
            return make_response(jsonify(error_response), 400)

        file = request.files['file']
        if file.filename == '':
            error_response = {
                'status': 'ERROR',
                'message': 'No selected file'
            }
            return make_response(jsonify(error_response), 400)

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            error_response = {
                'status': 'ERROR',
                'message': 'Invalid file format. Please upload an image (PNG, JPG, JPEG)'
            }
            return make_response(jsonify(error_response), 400)
        
        
        # Processing Logic
        file = request.files['file']
        filepath = f'static/temp/{file.filename}'
        file.save(filepath)

        # Process
        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_normalized = img_resized / 255.0
        prediction = loaded_model.predict(np.expand_dims(img_normalized, axis=0))

        predicted_class_index = np.argmax(prediction)
        confidence_score = np.max(prediction)

        class_names = ['Kaca', 'Kardus', 'Kertas', 'Makanan', 'Plastik']

        predicted_class_name = class_names[predicted_class_index]
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate a unique ID
        unique_id = str(uuid.uuid4())

        # Prepare response JSON
        prediction_result = {
            'ID': unique_id,
            'predicted_class': predicted_class_name,
            'confidence_score': float(confidence_score),
            'image': f"{request.url_root}{filepath}",
            'insertedAt': timestamp
        }

        predicted_results.append(prediction_result)
        
        response = {
            'status': 'success',
            'message': 'Image upload successfully'
        }

        return make_response(jsonify(response), 201)
    
    except Exception as e:
        error_response = {
            'status': 'ERROR 400',
            'message': 'Failed to upload image'
        }
        return make_response(jsonify(error_response), 400)

# Get Method
@app.route('/recycool', methods=['GET'])
def get_result():
    if predicted_results:
        response = {
            'status': 'Success',
            'data': predicted_results
        }
        return make_response(jsonify(response), 200)
    else:
        error_response = {
            'status': 'ERROR 404',
            'message': 'No data available'
        }
        return make_response(jsonify(error_response), 404)

# Get Method by ID
@app.route('/recycool/<string:unique_id>', methods=['GET'])
def get_prediction(unique_id):
    if not is_valid_uuid(unique_id):
        error_response = {
            'status': 'ERROR 400',
            'message': 'Invalid ID format'
        }
        return make_response(jsonify(error_response), 400)

    found = False
    for prediction in predicted_results:
        if prediction['ID'] == unique_id:
            found = True
            response = {
                'status': 'Success',
                'data': prediction
            }
            return make_response(jsonify(response), 200)
    
    if not found:
        error_response = {
            'status': 'ERROR 404',
            'message': 'Not found for the given ID'
        }
        return make_response(jsonify(error_response), 404)


# Delete Method
@app.route('/recycool/<string:unique_id>', methods=['DELETE'])
def delete_prediction(unique_id):
    global predicted_results

    initial_length = len(predicted_results)
    predicted_results = [prediction for prediction in predicted_results if prediction['ID'] != unique_id]
    final_length = len(predicted_results)

    if final_length < initial_length:
        response = {
            'status': 'Success',
            'message': 'Deleted successfully'
        }
        return make_response(jsonify(response), 200)
    else:
        error_response = {
            'status': 'ERROR 404',
            'message': 'Not found for the given ID'
        }
        return make_response(jsonify(error_response), 404)


# Run flask server
if __name__ == '__main__':
    app.run(debug=True, port=8000)