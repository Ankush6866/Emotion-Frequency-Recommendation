import cv2
import numpy as np
from keras.models import load_model
import joblib
import random
import time
import random
import webbrowser  # Add this at the top of your file


# Load pre-trained emotion recognition model
emotion_model = load_model('Face_Emotion.keras')

# Load frequency recommendation model
xgb_model = joblib.load('xgb_model.joblib')

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to calculate frequency based on emotion and other factors


# Function to calculate frequency based on emotion and other factors
def calculate_frequency(emotion, gender, time_of_day, location, weather, activity, feedback, age, mood_intensity, sleep_quality):
    # Emotion-based frequency range (with decimal values)
    frequency_range = {
        'angry': (80.0, 150.0),
        'disgust': (100.0, 200.0),
        'fear': (120.0, 250.0),
        'happy': (250.0, 400.0),
        'neutral': (100.0, 200.0),
        'sad': (80.0, 150.0),
        'surprise': (300.0, 500.0)
    }

    # Get the frequency range for the detected emotion
    min_freq, max_freq = frequency_range.get(emotion, (0, 0))

    # Simulate a realistic frequency value with decimal precision
    frequency = round(random.uniform(min_freq, max_freq), 2)  # Generate with decimal precision

    # Print the calculated frequency based on emotion
    print(f"Emotion Detected: {emotion}")
    print(f"Suggested Frequency: {frequency} Hz")

    return frequency

# Prompt for user inputs (before opening camera)
gender = input("Enter gender (Male/Female): ")
time_of_day = input("Enter time of day (Morning/Evening/Night): ")
location = input("Enter location (Outdoors/Public Space/Work): ")
weather = input("Enter weather (Rainy/Snowy/Sunny): ")
activity = input("Enter activity (Relaxing/Socializing/Working): ")

# Get numeric inputs
feedback = float(input("Enter feedback (0-10): "))
age = int(input("Enter age: "))
mood_intensity = float(input("Enter mood intensity (1-10): "))
sleep_quality = float(input("Enter sleep quality (1-10): "))

# Start the camera
cap = cv2.VideoCapture(0)

# Give the camera a moment to adjust
time.sleep(2)

# Start emotion detection for 8-10 seconds
start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load face cascade (haarcascade file)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Detect the first face (assuming only one face)
        x, y, w, h = faces[0]

        # Crop the face region
        face_region = gray[y:y+h, x:x+w]
        face_region_resized = cv2.resize(face_region, (48, 48))

        # Normalize and reshape the face region for model input
        face_region_normalized = face_region_resized / 255.0
        face_region_reshaped = np.reshape(face_region_normalized, (1, 48, 48, 1))

        # Predict emotion
        emotion_probabilities = emotion_model.predict(face_region_reshaped)
        predicted_class = np.argmax(emotion_probabilities, axis=1)
        predicted_emotion = emotion_labels[predicted_class[0]]

        # Display the emotion on the camera screen (for real-time feedback)
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Stop after 8-10 seconds
        if time.time() - start_time >= 15:
            break

    # Display the camera feed
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# After emotion detection, calculate and display frequency
frequency = calculate_frequency(predicted_emotion, gender, time_of_day, location, weather, activity, feedback, age, mood_intensity, sleep_quality)

print(f"\nFinal Frequency Recommendation: {round(frequency, 2)} Hz")
# After calculating the frequency
print(f"Playing the frequency: {frequency} Hz")
url = f"https://onlinetonegenerator.com/?freq={frequency}"
webbrowser.open(url)

