import os
import json
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Load the model and class mapping
model = load_model('medicinal_plant_model.h5')
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
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()

    if request.values.get('NumMedia') != '0':
        image_url = request.values.get('MediaUrl0')
        image_path = 'temp_image.jpg'
        
        # Download and save the image
        import requests
        image_data = requests.get(image_url).content
        with open(image_path, 'wb') as handler:
            handler.write(image_data)

        # Preprocess and predict
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array)
        predicted_class = idx_to_class[str(np.argmax(prediction[0]))]

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
    else:
        msg.body("Please send an image of a medicinal plant.")

    return str(resp)

if __name__ == '__main__':
    app.run()
