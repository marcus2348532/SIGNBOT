import streamlit as st
import pickle
import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import mediapipe as mp
import pyttsx3
import threading
import speech_recognition as sr
import base64
from deepface import DeepFace
import time

# **Set up Streamlit layout**
st.set_page_config(layout="wide")

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function to set background
set_background("background.jpg")  # Change to your actual image file name

# Load sign language recognition model
model = tf.keras.models.load_model("signlang_lstm_model_improved.h5")

# Load label encoder and scaler
with open("label_encoder.p", "rb") as f:
    label_encoder = pickle.load(f)

with open("scaler.p", "rb") as f:
    scaler = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Google Gemini API
genai.configure(api_key="AIzaSyCe8pgNYKgUsw5LeNGtxGjxAfI3DjxGZEU")

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def text_to_speech(text):
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()

def get_response(input_text, emotion):
    """Generate chatbot response based on detected emotion."""
    prompt = ( f"The user seems to be {emotion}. Respond in a friendly and engaging way. "
    f"The input is translated from Indian Sign Language (ISL), so the structure may be different from standard English. "
    f"First, rephrase it into proper English if necessary, while keeping the original meaning. "
    f"Then, provide a clear and helpful response based on the corrected version."
    f"User input: {input_text}")
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text

def recognize_emotion_from_face(frame):
    """Detects emotion from the user's face every 10 seconds."""
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result:
            return result[0]['dominant_emotion']
    except Exception:
        return "neutral"
    return "neutral"

# **UI Setup**
st.title("🤖 Signbot")

# **Input Selection**
st.markdown("### Choose your input method:")
option = st.radio("", ["Text Input", "Voice Input", "Sign Language"], index=None)

if option == "Text Input":
    user_input = st.text_input("Type your message:")
    if st.button("Send"):
        if user_input:
            chatbot_response = get_response(user_input, "neutral")
            st.markdown(f"**💬 Chatbot:** {chatbot_response}")
            text_to_speech(chatbot_response)

elif option == "Voice Input":
    if st.button("Start Listening"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("🎤 Listening... Speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for background noise
            try:
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)  # Increase timeout
                speech_text = recognizer.recognize_google(audio)
                st.write(f"🗣 Recognized Speech: {speech_text}")
                chatbot_response = get_response(speech_text, "neutral")
                st.markdown(f"**💬 Chatbot:** {chatbot_response}")
                text_to_speech(chatbot_response)
            except sr.WaitTimeoutError:
                st.write("⏳ Timeout: No speech detected, please try again.")
            except sr.UnknownValueError:
                st.write("❌ Could not understand your speech.")
            except sr.RequestError:
                st.write("⚠️ Could not connect to the recognition service.")

elif option == "Sign Language":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("## Webcam Feed")
        
        # **Emotion Display Below "Webcam Feed" Heading**
        emotion_placeholder = st.markdown("**😊 Current Emotion:** Neutral")

        stframe = st.empty()

    with col2:
        st.markdown("## Chatbot Responses")
        chat_area = st.empty()

    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        
        sentence = []
        last_word = None
        no_sign_counter = 0
        last_emotion_check = time.time()
        detected_emotion = "neutral"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            detected_word = None

            # **Emotion recognition every 10 seconds**
            if time.time() - last_emotion_check > 5:
                detected_emotion = recognize_emotion_from_face(frame_rgb)
                emotion_placeholder.markdown(f"**😊 Current Emotion:** {detected_emotion}")
                last_emotion_check = time.time()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

                    if landmarks.shape[0] == 42:
                        landmarks = landmarks.reshape(1, -1)  
                        landmarks = scaler.transform(landmarks)
                        landmarks = landmarks.reshape(1, 1, 42)

                        prediction = model.predict(landmarks)
                        predicted_label_index = np.argmax(prediction)
                        detected_word = label_encoder.inverse_transform([predicted_label_index])[0]

                        if detected_word != last_word and detected_word not in sentence:
                            sentence.append(detected_word)
                            last_word = detected_word
                            chat_area.markdown(f"**✋ Recognized Sign:** {detected_word}")

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            else:
                no_sign_counter += 1

                if no_sign_counter > 20 and sentence:
                    final_sentence = " ".join(sentence)
                    chat_area.markdown(f"**📝 Formed Sentence:** {final_sentence}")

                    # **Generate response using both text and emotion**
                    chatbot_response = get_response(final_sentence, detected_emotion)
                    chat_area.markdown(f"**💬 Chatbot:** {chatbot_response}")
                    text_to_speech(chatbot_response)

                    sentence = []
                    last_word = None
                    no_sign_counter = 0

            stframe.image(frame, channels="BGR")

        cap.release()
