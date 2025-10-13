# Flask web framework
from flask import Flask, jsonify, request

# Deep learning
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Data handling
import numpy as np
import io
import base64

app = Flask(__name__)

# initialized in main
all_models = {}

TASK_IDS = [1, 2, 3, 4, 5]
DATE_STRING = '10_10_25'
DEBUG = True

@app.route('/classify', methods=['POST'])
def classify_image():
    '''
    Classify surgical image using pre-trained ResNet models.

    Input: {
        'image': base64 encoded image,
        'task_id': integer
    }

    Output: {
        'complete': boolean
        'confidence': float (?)
    }
    '''

    # Validate input
    error_msg, status_code = validate_request(request.json)
    if error_msg:
        return jsonify({'error': error_msg}), status_code
    

    # Extract task_id and make sure it is a valid task. Then set 'model' to corresponding ResNet model
    task_id = request.json['task_id']
    if not task_id in TASK_IDS:
        return jsonify({'error': f'task_id={task_id} not valid.'}), 400
    model = all_models[task_id]

    # Extract the image data and transform to tensor readable by model
    image_b64 = request.json['image']
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
    input_tensor = preprocess(image).unsqueeze(0)

    # Make prediction and return result
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))
        pred = (output > 0.5).float()

    complete = True if pred == 0 else False
    
    return jsonify({
        'complete': complete
    })

        
def validate_request(data):
    '''
    If data is properly formatted, this returns None, None. Otherwise an exception string + error code are returned.

    This is just to ensure that the body of the POST request is valid.
    '''

    if not data:
        return 'Request must be JSON', 400
    
    if 'task_id' not in data:
        return 'task_id required', 400
    
    if 'image' not in data:
        return 'image required', 400
    
    return None, None

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def initialize_models():
    weights_root = f'./model-weights/{DATE_STRING}'

    for id in TASK_IDS:
        # Create template model with modified output
        model = models.resnet101()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)

        model.load_state_dict(torch.load(f'{weights_root}/task_{id}_final_weights_{DATE_STRING}.pth'))
        model.eval()
        all_models[id] = model
        print(f'Model for task {id} initialized successfully.')


if __name__ == '__main__':
    initialize_models()
    app.run(debug=DEBUG, host='0.0.0.0', port=8000)
