# Import necessary libraries and modules
import os
import json
from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
import io
import requests
from dotenv import load_dotenv
import imghdr

# Load environment variables from a .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Load the TFLite model for plant identification
interpreter = tf.lite.Interpreter(model_path="dr_roots_model.tflite")
interpreter.allocate_tensors()

# Load the class mapping from a JSON file
with open('class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

# Load plant information from a JSON file
with open('plant_data.json', 'r', encoding='utf-8') as f:
    plant_info = json.load(f)

# Get Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize the Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Dictionary to track user states for conversation flow
user_states = {}

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

def get_plant_info(plant_name):
    for plant in plant_info:
        if plant['Scientific Name'] == plant_name:
            # Format citations on a new line each
            citations = "\n".join(plant['Citations'])

            info = (
                f"🌿 *Plant Profile: {plant['Common Name']}* 🌿\n"
                f"- Scientific Name: {plant['Scientific Name']}\n"
                f"- Shona Name: {plant['Shona Name']}\n"
                f"\n🍃 What it looks like: \n{plant['Physical Description']}\n"
                f"\n💊 Reported Medicinal Uses: \n{plant['Reported Medicinal Uses']}\n"
                f"\n🧪 How it's prepared & used: \n{plant['Preparation Methods & Parts Used']}\n"
                f"\n🌱 Conservation Status (on the IUCN Red List): {plant['IUCN Red List of Threatened Species']}\n"
                f"\n📚 Want to learn more? Check out these papers: \n{citations}"
            )
            return info
    return "Plant not found in database"

@app.route('/')
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Doctor Roots: Indigenous Medicinal Plant Identifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1, h2 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 10px;
        }
        ul {
            list-style-type: none;
            padding-left: 0;
        }
        li {
            margin-bottom: 10px;
        }
        .emoji {
            font-size: 1.2em;
        }
        .contact {
            background-color: #e7f2f8;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>🌿 Doctor Roots: Indigenous Medicinal Plant Identifier</h1>

    <h2>Project Overview</h2>
    <p>Doctor Roots is an AI-powered WhatsApp bot designed to accurately identify indigenous African medicinal plants and provide safe usage information. This project aims to bridge the gap between traditional herbal medicine and modern technology, ensuring safe and effective use of medicinal plants while promoting conservation. 🌍💚</p>

    <h2>Objective</h2>
    <p>Develop an AI model to identify 7 common African medicinal plants with high accuracy, and integrate this model into a WhatsApp bot for easy access by users. 🤖📲</p>

    <h2>Features</h2>
    <ul>
        <li><span class="emoji">🌿</span> AI-powered plant identification from images</li>
        <li><span class="emoji">📱</span> WhatsApp integration for user-friendly access</li>
        <li><span class="emoji">📝</span> Information on safe usage of identified medicinal plants</li>
        <li><span class="emoji">🌱</span> Support for 7 common African medicinal plant species</li>
    </ul>

    <h2>Target Plants</h2>
    <ol>
        <li><strong>Catharanthus roseus (L.) G. Don</strong> - Chirindamatongo</li>
        <li><strong>Psidium guajava L.</strong> - Mugwavha</li>
        <li><strong>Zingiber officinale Roscoe</strong> - Tsangamidzi</li>
        <li><strong>Citrus limon (L.) Burm. f.</strong> - Mulemoni</li>
        <li><strong>Mangifera indica L.</strong> - Mumango</li>
        <li><strong>Moringa oleifera Lour</strong> - Moringa</li>
        <li><strong>Aloe barbadensis</strong> - Gavakava</li>
    </ol>

    <h2>Technology Stack</h2>
    <ul>
        <li><span class="emoji">🐍</span> Python</li>
        <li><span class="emoji">🔍</span> TensorFlow / Keras</li>
        <li><span class="emoji">☁️</span> Google Colab (for model training)</li>
        <li><span class="emoji">📞</span> Twilio API (for WhatsApp integration)</li>
        <li><span class="emoji">🌐</span> Google Cloud Platform (for deployment)</li>
    </ul>

    <h2>Twilio Sandbox Instructions</h2>
    <p>To test the WhatsApp bot, invite your friends to the Twilio Sandbox by sending a message from your device to:</p>
    <p><strong>📱 WhatsApp: +1 415 523 8886</strong></p>
    <p>with the code: <code>join silly-degree.</code></p>

    <h2>Usage</h2>
    <p>Once deployed, users can interact with the Doctor Roots bot via WhatsApp by sending images of plant leaves. The bot will respond with the plant identification and safe usage information. 📸🌿</p>

    <div class="contact">
        <h2>Contact</h2>
        <p>Ruvarashe Sadya</p>
        <p>📧 <a href="mailto:ruvarashe.sadya@gmail.com">ruvarashe.sadya@gmail.com</a></p>
        <p>🐙 <a href="https://github.com/RuvaS20/Dr-Roots/tree/main">GitHub: Dr-Roots</a></p>
    </div>
</body>
</html>
    """

@app.route("/webhook", methods=["POST"])
def webhook():
    # Retrieve the incoming message text and sender's phone number from the request
    incoming_msg = request.values.get('Body', '').lower().strip()
    from_number = request.values.get('From')
    resp = MessagingResponse()  # Create a new Twilio messaging response
    msg = resp.message()  # Create a new message in the response

    # Retrieve the number of media files sent with the message
    num_media = int(request.values.get('NumMedia', 0))

    # Initialize user state if this is the first interaction with the number
    if from_number not in user_states:
        user_states[from_number] = {'state': 'menu'}

    # If the message is 'menu' or 'start over', reset the user state to the menu
    if incoming_msg in ['menu', 'start over']:
        user_states[from_number]['state'] = 'menu'
        msg.body("🌿 *Welcome to Doctor Roots!* 🌿 \n\nI'm your friendly medicinal plant bot.\n\n📸*Send me a clear photo of a plant - I'll try to identify it and share fun facts about it!*📸\n\nOr choose one of these options:\n1️⃣ Learn more about other plants\n2️⃣ Contact the developer\n\n🚨Important Disclaimer🚨\nThe information disseminated here is for educational purposes only and should not be taken as medical advice.")
        return str(resp)  # Return the response to the user
    elif incoming_msg in ['exit', 'end']:
        # If the message is 'exit' or 'end', remove the user from the state tracking and thank them
        user_states.pop(from_number, None)
        msg.body("Thank you for trying out Doctor Roots! If you have any feedback or questions, feel free to reach out. Have a great day!")
        return str(resp)

    # If the user is in the menu state, send the main menu message and change state to default
    if user_states[from_number]['state'] == 'menu':
        msg.body("🌿 *Welcome to Doctor Roots!* 🌿 \n\nI'm your friendly medicinal plant bot.\n\n📸*Send me a clear photo of a plant - I'll try to identify it and share fun facts about it!*📸\n\nOr choose one of these options:\n1️⃣ Learn more about other plants\n2️⃣ Contact the developer\n\n🚨Important Disclaimer🚨\nThe information disseminated here is for educational purposes only and should not be taken as medical advice.")
        user_states[from_number]['state'] = 'default'

    # If there are media files in the message, process them
    elif num_media > 0:

        # If the state is menu, ask for a text message first
        if user_states[from_number]['state'] == 'menu':
            msg.body("Please send a text message first.")
            return str(resp)

        # Retrieve the URL of the first media file
        media_url = request.values.get('MediaUrl0')
        
        if media_url:
            try:
                # Inform the user that the prediction process has started
                msg.body("Please wait a few seconds while I process the image...")
                # Send the response to the user immediately
                response = str(resp)
                
                # Reset the response to send the prediction result
                resp = MessagingResponse()
                msg = resp.message()

                # Download the image from the URL
                response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
                
                if response.status_code == 200:
                    image_data = response.content
                    content_type = response.headers.get('Content-Type')

                    # Check if the content is an image
                    if 'image' in content_type:
                        image_format = imghdr.what(None, h=image_data)

                        if image_format:
                            # Open the image and make a prediction
                            image = Image.open(io.BytesIO(image_data))
                            predicted_class, confidence = predict_image(image)
                            
                            if confidence >= 0.7:
                                # Get the predicted plant name and information
                                plant_name = class_mapping[str(predicted_class)]
                                info = get_plant_info(plant_name)

                                msg.body(f"*Leaf it to me! 🔍 I'm {confidence*100:.1f}% confident this is {plant_name}!* 🌿\n\n{info} \n\nYou can type 'Menu' to start over or 'Exit' to end the conversation.")

                            else:
                                # Low confidence in prediction
                                msg.body("I'm not confident enough to identify this plant. Please try another image. \n\nYou can type 'Menu' to start over or 'Exit' to end the conversation.")

                        else:
                            # Image format not recognized
                            msg.body("Sorry, the image format is not recognized. Please try a different image.")
                    else:
                        # URL does not point to an image
                        msg.body("The URL does not point to a valid image. Please try sending an image.")
                else:
                    # Failed to download the image
                    msg.body(f"Failed to download image. HTTP status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                # Error occurred during image download
                msg.body(f"Sorry, I had trouble downloading the image. Error: {str(e)}")
            except PIL.UnidentifiedImageError:
                # Image format not supported
                msg.body("Sorry, the image format is not supported. Please try a different image.")
            except tf.errors.InvalidArgumentError as e:
                # TensorFlow error during image processing
                msg.body(f"There was an error processing the image with TensorFlow. Error: {str(e)}")
            except Exception as e:
                # Unexpected error occurred
                msg.body(f"Sorry, there was an unexpected error processing your image. Error: {str(e)}")
                print(f"Unexpected error: {str(e)}")
                
        else:
            # No image found in the message
            msg.body("Sorry, I couldn't find the image you sent. Please try sending it again.")
    
    else:
        if user_states[from_number]['state'] == 'default':
            if incoming_msg == '1':
                # List of plants to learn about
                plant_list = "\n".join([
                    "1️⃣ Madagascar Periwinkle",
                    "2️⃣ Guava",
                    "3️⃣ Ginger",
                    "4️⃣ Lemon",
                    "5️⃣ Mango",
                    "6️⃣ Moringa",
                    "7️⃣ Aloe vera"
                ])
                msg.body(f"🌿 *Eeny, meeny, miny, grow!* 🌿\n\nWhich lucky plant will you get to know?\n\n{plant_list} \n\nYou can type 'Menu' to start over or 'Exit' to end the conversation.")
                user_states[from_number]['state'] = 'selecting_plant'
            elif incoming_msg == '2':
                # Contact developer information
                msg.body("This project was created by Ruva, a passionate CS student, with the aim of helping Africa where 80% of people use traditional medicinal plants (per UN data). There's a critical lack of reliable, accessible tools for accurate plant identification. \n\nWant to contribute to the knowledge base? Reach out using the following: \n👩‍💻 GitHub:https://github.com/RuvaS20 \n📧 Email: ruvarashe.sadya@gmail.com")
            else:
                # Default message for unrecognized input
                msg.body("🌿 *Welcome to Doctor Roots!* 🌿 \n\nI'm your friendly medicinal plant bot.\n\n📸*Send me a clear photo of a plant - I'll try to identify it and share fun facts about it!*📸\n\nOr choose one of these options:\n1️⃣ Learn more about other plants\n2️⃣ Contact the developer\n\n🚨Important Disclaimer🚨 \nThe information disseminated here is for educational purposes only and should not be taken as medical advice.")
        
        elif user_states[from_number]['state'] == 'selecting_plant':
            if incoming_msg.isdigit() and 1 <= int(incoming_msg) <= 7:
                # Retrieve and display information about the selected plant
                plant_index = int(incoming_msg) - 1
                plant_name = [
                    "Catharanthus roseus",
                    "Psidium guajava",
                    "Zingiber officinale Roscoe",
                    "Citrus limon",
                    "Mangifera indica",
                    "Moringa oleifera Lour",
                    "Aloe barbadensis"
                ][plant_index]
                info = get_plant_info(plant_name)
                msg.body(f"{info}")
                user_states[from_number]['state'] = 'default'
            else:
                # Handle invalid plant selection
                msg.body("Invalid selection. Please select a number from the list of plants. Or type 'Menu' to start over or 'Exit' to end the conversation.")
                user_states[from_number]['state'] = 'selecting_plant'

    return str(resp)  # Return the response to be sent back to the user

if __name__ == '__main__':
    app.run(debug=True, port=5000)
