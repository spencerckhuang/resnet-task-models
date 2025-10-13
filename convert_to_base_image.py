import base64

# Script to encode image
with open('./sample_images/task1_complete/tissue_1_1_retract_home_in_20250416-181823-240199_frame000141_left.jpg', 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    print(encoded_string)