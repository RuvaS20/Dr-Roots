import os
import json
from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
from dotenv import load_dotenv
import imghdr

load_dotenv()

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

# Get Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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
            # format citations on a new line each
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
    return "Hello, Flask is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get('Body', '').lower().strip()
    from_number = request.values.get('From')
    resp = MessagingResponse()
    msg = resp.message()

    num_media = int(request.values.get('NumMedia', 0))

    if from_number not in user_states:
        user_states[from_number] = {'state': 'initial'}

    if incoming_msg in ['menu', 'start over']:
        user_states[from_number]['state'] = 'initial'
        msg.body("🌿 *Welcome to Doctor Roots!* 🌿 \n\nI'm your friendly medicinal plant bot.\n\n📸*Send me a clear photo of a plant - I'll try to identify it and share fun facts about it!*📸\n\nOr choose one of these options:\n1️⃣ Learn more about other plants\n2️⃣ Contact the developer\n\n🚨Important Disclaimer🚨\nThe information disseminated here is for educational purposes only and should not be taken as medical advice.")
        return str(resp)
    elif incoming_msg in ['exit', 'end']:
        user_states.pop(from_number, None)
        msg.body("Thank you for trying out Doctor Roots! If you have any feedback or questions, feel free to reach out. Have a great day!")
        return str(resp)

    if user_states[from_number]['state'] == 'initial':
        msg.body("🌿 *Welcome to Doctor Roots!* 🌿 \n\nI'm your friendly medicinal plant bot.\n\n📸*Send me a clear photo of a plant - I'll try to identify it and share fun facts about it!*📸\n\nOr choose one of these options:\n1️⃣ Learn more about other plants\n2️⃣ Contact the developer\n\n🚨Important Disclaimer🚨\nThe information disseminated here is for educational purposes only and should not be taken as medical advice.")
        user_states[from_number]['state'] = 'default'
    elif num_media > 0:
        if user_states[from_number]['state'] == 'initial':
            msg.body("Please send a text message first.")
            return str(resp)

        media_url = request.values.get('MediaUrl0')
        
        if media_url:
            try:
                response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
                
                if response.status_code == 200:
                    image_data = response.content
                    content_type = response.headers.get('Content-Type')

                    if 'image' in content_type:
                        image_format = imghdr.what(None, h=image_data)

                        if image_format:
                            image = Image.open(io.BytesIO(image_data))
                            predicted_class, confidence = predict_image(image)
                            
                            if confidence >= 0.7:
                                
                                plant_name = class_mapping[str(predicted_class)]
                                
                                info = get_plant_info(plant_name)

                                msg.body(f"*Leaf it to me! 🔍 I'm {confidence*100:.2f}% confident this is {plant_name}!* 🌿\n\n{info} \n\nYou can type 'Menu' to start over or 'Exit' to end the conversation.")

                            else:
                                msg.body("I'm not confident enough to identify this plant. Please try another image. \n\nYou can type 'Menu' to start over or 'Exit' to end the conversation.")

                        else:
                            msg.body("Sorry, the image format is not recognized. Please try a different image.")
                    else:
                        msg.body("The URL does not point to a valid image. Please try sending an image.")
                else:
                    msg.body(f"Failed to download image. HTTP status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                msg.body(f"Sorry, I had trouble downloading the image. Error: {str(e)}")
            except PIL.UnidentifiedImageError:
                msg.body("Sorry, the image format is not supported. Please try a different image.")
            except tf.errors.InvalidArgumentError as e:
                msg.body(f"There was an error processing the image with TensorFlow. Error: {str(e)}")
            except Exception as e:
                msg.body(f"Sorry, there was an unexpected error processing your image. Error: {str(e)}")
                print(f"Unexpected error: {str(e)}")
                
        else:
            msg.body("Sorry, I couldn't find the image you sent. Please try sending it again.")
    
    else:
        if user_states[from_number]['state'] == 'default':
            if incoming_msg == '1':
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
                msg.body("This project was created by Ruva, a passionate CS student, with the aim of helping Africa where 80% of people use traditional medicinal plants (per UN data). There's a critical lack of reliable, accessible tools for accurate plant identification. \n\nWant to contribute to the knowledge base? Reach out using the following: \n👩‍💻 GitHub:https://github.com/RuvaS20 \n📧 Email: ruvarashe.sadya@gmail.com")
            else:
                msg.body("🌿 *Welcome to Doctor Roots!* 🌿 \n\nI'm your friendly medicinal plant bot.\n\n📸*Send me a clear photo of a plant - I'll try to identify it and share fun facts about it!*📸\n\nOr choose one of these options:\n1️⃣ Learn more about other plants\n2️⃣ Contact the developer\n\n🚨Important Disclaimer🚨 \nThe information disseminated here is for educational purposes only and should not be taken as medical advice.")
        
        elif user_states[from_number]['state'] == 'selecting_plant':
            if incoming_msg.isdigit() and 1 <= int(incoming_msg) <= 7:
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
                msg.body("Invalid selection. Please select a number from the list of plants. Or type 'Menu' to start over or 'Exit' to end the conversation.")
                user_states[from_number]['state'] = 'selecting_plant'

    return str(resp)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
