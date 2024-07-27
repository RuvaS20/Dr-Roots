import os
import json
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="dr_roots_model.tflite")
interpreter.allocate_tensors()

# Load the class mapping
with open('class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

# Load the plant information
with open('plant_data.json', 'r', encoding='utf-8') as f:
    plant_info = json.load(f)

def predict_image(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

    # Set the input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class and confidence
    predicted_class = np.argmax(output_data)
    confidence = output_data[0][predicted_class]

    return predicted_class, confidence

def get_plant_info(plant_name, info_type):
    for plant in plant_info:
        if plant['Common Name'].lower() == plant_name.lower() or plant['Shona Name'].lower() == plant_name.lower():
            return plant.get(info_type, "Information not available")
    return "Plant not found in database"

@app.route('/')
def home():
    return "Hello, Flask is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()

    if request.values.get('NumMedia') != '0':
        # Handle image
        image_url = request.values.get('MediaUrl0')
        if image_url is None:
            msg.body("Sorry, I couldn't find the image you sent. Please try sending it again.")
        else:
            try:
                image_data = requests.get(image_url).content
                image = Image.open(io.BytesIO(image_data))
                
                predicted_class, confidence = predict_image(image)
                
                if confidence > 0.5:
                    plant_name = class_mapping[str(predicted_class)]
                    msg.body(f"I predict this is {plant_name} with {confidence*100:.2f}% confidence.")
                else:
                    msg.body("I'm not confident enough to identify this plant. Please try another image.")
            except requests.exceptions.RequestException as e:
                msg.body("Sorry, I had trouble downloading the image. Please try sending it again.")
            except Exception as e:
                msg.body("Sorry, there was an error processing your image. Please try again.")

    elif incoming_msg.startswith('uses:'):
        plant_name = incoming_msg.split(':', 1)[1].strip()
        info = get_plant_info(plant_name, 'Reported Medicinal Uses')
        msg.body(f"Medicinal uses of {plant_name}: {info}")

    elif incoming_msg.startswith('description:'):
        plant_name = incoming_msg.split(':', 1)[1].strip()
        info = get_plant_info(plant_name, 'Physical Description')
        msg.body(f"Description of {plant_name}: {info}")

    elif incoming_msg.startswith('safety:'):
        plant_name = incoming_msg.split(':', 1)[1].strip()
        info = get_plant_info(plant_name, 'Safety Precautions')
        msg.body(f"Safety precautions for {plant_name}: {info}")

    elif incoming_msg.startswith('preparation:'):
        plant_name = incoming_msg.split(':', 1)[1].strip()
        info = get_plant_info(plant_name, 'Preparation Methods & Parts Used')
        msg.body(f"Preparation methods for {plant_name}: {info}")

    else:
        msg.body("Welcome to Dr. Roots! You can:\n"
                 "1. Send an image for plant identification\n"
                 "2. Ask about a plant using these commands:\n"
                 "   - uses: [plant name]\n"
                 "   - description: [plant name]\n"
                 "   - safety: [plant name]\n"
                 "   - preparation: [plant name]")

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
