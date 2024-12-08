import cv2
import numpy as np
from keras.models import load_model
import joblib
import time
import webbrowser


# Load pre-trained emotion recognition model
emotion_model = load_model('Face_Emotion.keras')

# Load frequency recommendation model (XGBoost)
xgb_model = joblib.load('xgb_model.joblib')
# Get the feature names from the XGBoost model
feature_names = xgb_model.get_booster().feature_names

# Print the feature names
#print(f"Feature names expected by the model: {feature_names}")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Function to calculate frequency based on emotion and other factors using the XGBoost model
def calculate_frequency(emotion, gender, time_of_day, location, weather, activity, feedback, age, mood_intensity, sleep_quality):
    # Create a feature vector using the emotion and other inputs
    # Map emotion to a numeric value (one-hot encoding or direct mapping)
    emotion_mapping = {label: idx for idx, label in enumerate(emotion_labels)}
    emotion_feature = emotion_mapping.get(emotion, -1)  # Use -1 if emotion is not found (fallback)

    # Prepare the input features for the frequency prediction model
    features = np.array([
        mood_intensity, sleep_quality, feedback, age,
        emotion == 'disgust', emotion == 'fear', emotion == 'happy', emotion == 'neutral', emotion == 'sad', emotion == 'surprise',  # One-hot encoding for emotion
        gender == 'Male',  # One-hot encoding for gender
        time_of_day == 'Evening', time_of_day == 'Morning', time_of_day == 'Night',  # One-hot encoding for time of day
        location == 'Outdoors', location == 'Public Space', location == 'Work',  # One-hot encoding for location
        weather == 'Rainy', weather == 'Snowy', weather == 'Sunny',  # One-hot encoding for weather
        activity == 'Relaxing', activity == 'Socializing', activity == 'Working'  # One-hot encoding for activity
    ])

    features = features.astype(float).reshape(1, -1)  # Reshape for XGBoost model

    # Predict frequency using the XGBoost model
    predicted_frequency = xgb_model.predict(features)

    # Return the predicted frequency
    return predicted_frequency[0]


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
print(f"\nDetected Emotion: {predicted_emotion}")
print(f"\nFinal Frequency Recommendation: {round(frequency, 2)} Hz")

# After calculating the frequency
print(f"Playing the frequency: {frequency} Hz")
url = f"https://onlinetonegenerator.com/?freq={frequency}"
webbrowser.open(url)
