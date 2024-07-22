import os
import json
import numpy as np
from flask import Flask, request, jsonify
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
from PIL import Image
import io

from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class mapping
with open('class_mapping.json', 'r') as f:
    idx_to_class = json.load(f)

# Plant information dictionary
plant_info = {
    "Catharanthus roseus": {
        "shona_name": "Chirindamatongo",
        "uses": "Used to treat diabetes, high blood pressure, and certain cancers."
    },
    "Psidium guajava": {
        "shona_name": "Mugwavha",
        "uses": "Used to treat diarrhea, dysentery, and digestive problems."
    },
    "Zingiber officinale": {
        "shona_name": "Tsangamidzi",
        "uses": "Used to treat nausea, colds, and arthritis."
    },
    "Citrus limon": {
        "shona_name": "Mulemoni",
        "uses": "Used for vitamin C, improving digestion, and boosting immunity."
    },
    "Mangifera indica": {
        "shona_name": "Mumango",
        "uses": "Used to treat digestive issues, boost immunity, and improve skin health."
    },
    "Moringa oleifera": {
        "shona_name": "Moringa",
        "uses": "Used as a nutritional supplement and to treat inflammation."
    },
    "Aloe barbadensis": {
        "shona_name": "Gavakava",
        "uses": "Used for skin conditions, burns, and digestive health."
    }
}

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()

    # Check if it's the first message (you might need to implement a way to track this)
    if incoming_msg == 'hi' or incoming_msg == 'hello':
        welcome_message = (
            "Welcome to Doctor Roots! 🌿\n\n"
            "I'm here to help you identify medicinal plants and provide information about their uses. "
            "Simply send me a clear photo of a plant, and I'll do my best to identify it.\n\n"
            "Disclaimer: This bot is for educational purposes only. Always consult with a qualified healthcare "
            "professional before using any plant for medicinal purposes. Do not rely solely on this information "
            "for medical advice."
        )
        msg.body(welcome_message)

    elif request.values.get('NumMedia') != '0':
        try:
            image_url = request.values.get('MediaUrl0')
            logging.info(f"Received image URL: {image_url}")

            image_path = 'temp_image.jpg'
            
            # Download and save the image
            import requests
            image_data = requests.get(image_url).content
            with open(image_path, 'wb') as handler:
                handler.write(image_data)
            logging.info("Image saved successfully")

            # Preprocess and predict
            img_array = preprocess_image(image_path)
            logging.info("Image preprocessed successfully")

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            logging.info("Prediction made successfully")

            predicted_class = idx_to_class[str(np.argmax(prediction[0]))]
            logging.info(f"Predicted class: {predicted_class}")

            # Get plant information
            plant_data = plant_info.get(predicted_class, {})
            shona_name = plant_data.get('shona_name', 'Unknown')
            uses = plant_data.get('uses', 'Information not available.')

            # Construct response
            response = f"The plant appears to be {predicted_class}.\n"
            response += f"Shona name: {shona_name}\n"
            response += f"Medicinal uses: {uses}\n\n"
            response += "Would you like to know more about this plant?"

            msg.body(response)
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            msg.body("Sorry, there was an error processing your image. Please try again.")
            
    else:
        msg.body("Please send an image of a medicinal plant or type 'hi' for information.")

    return str(resp)

if __name__ == '__main__':
    app.run()
