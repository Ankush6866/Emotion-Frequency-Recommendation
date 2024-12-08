
---

# Real-Time Emotion-Based Frequency Recommendation System
This project is a real-time system that detects emotions from facial expressions using a live camera feed and predicts a personalized frequency recommendation based on various factors, including the detected emotion, gender, time, location, weather, activity, feedback, age, mood intensity, and sleep quality. The recommended frequency is then played on an online tone generator for practical use.

## Overview
This project combines two machine learning models:
1. **Emotion Recognition Model**: Detects emotions in real-time using the camera feed.
2. **Frequency Recommendation Model**: Recommends a frequency based on detected emotion and additional user inputs.

The integration of these models is implemented in the `emotion_frequency.py` script, which allows users to input details, capture real-time emotion, and receive a frequency recommendation played on the [Online Tone Generator](https://onlinetonegenerator.com).

---

## Datasets

1. **CNN Model Dataset**:
   - The dataset for the CNN model is stored in `archive.zip`.
   - It contains two folders: `train` and `test` which can be used for training and testing the model.

2. **Frequency Prediction Model Dataset**:
   - The `Emotion_frequency_dataset.csv` file is used for the second model, which is for emotion-based frequency prediction.
   - This dataset includes features like emotion, gender, age, weather, and feedback  used for predicting the frequency for various combinations of inputs.

## Files in the Project
### 1. **Models and Related Files**
- **Emotion Recognition Model**
  - `Face_Emotion.keras`: Pre-trained Keras model for emotion detection.
  - `facialemotionmodel.json`: Model architecture in JSON format.
  - `facialemotionmodel.weights.h5`: Model weights for emotion recognition.

- **Frequency Recommendation Model**
  - `xgb_model.joblib`: Pre-trained XGBoost model for frequency prediction.

### 2. **Integration Script**
- **`emotion_frequency.py`**: Combines the two models, captures user inputs, detects emotion via the camera, predicts the frequency using both inputs, and plays the tone on the online tone generator.

---

## How the System Works
### Step 1: **Create the Models**
1. Train or acquire:
   - A real-time **Emotion Recognition Model** capable of detecting one of the following emotions:
     - `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
   - A **Frequency Recommendation Model** trained with the following features:
     - Mood Intensity
     - Sleep Quality
     - Feedback
     - Age
     - One-hot encoded variables for:
       - Emotion (`angry`,`disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`)
       - Gender (`Male`,`Female`)
       - Time of Day (`Morning`, `Evening`, `Night`)
       - Location (`Outdoors`, `Public Space`, `Work`)
       - Weather (`Rainy`, `Snowy`, `Sunny`)
       - Activity (`Relaxing`, `Socializing`, `Working`)

2. Save the models and their weights in appropriate files:
   - For the emotion recognition model: `Face_Emotion.keras`, `facialemotionmodel.json`, and `facialemotionmodel.weights.h5`.
   - For the frequency recommendation model: `xgb_model.joblib`.

---

### Step 2: **Combine Models in `emotion_frequency.py`**
1. The script integrates both models and enables:
   - Real-time emotion detection using the camera feed.
   - Frequency recommendation based on the detected emotion and user inputs.

2. Inputs required from the user include:
   - **Gender**: Male or Female.
   - **Time of Day**: Morning, Evening, or Night.
   - **Location**: Outdoors, Public Space, or Work.
   - **Weather**: Rainy, Snowy, or Sunny.
   - **Activity**: Relaxing, Socializing, or Working.
   - **Feedback**: A numeric rating (0–10).
   - **Age**: User's age.
   - **Mood Intensity**: A numeric scale (1–10).
   - **Sleep Quality**: A numeric scale (1–10).

3. Once the inputs are gathered, the camera is activated to detect emotion in real-time.

4. Both inputs (user-provided and detected emotion) are combined into a feature vector and passed to the frequency recommendation model.

5. The predicted frequency is displayed and played on the [Online Tone Generator](https://onlinetonegenerator.com).

---

## Running the Script
### Prerequisites
- Install required Python packages:
  ```bash
  pip install numpy opencv-python keras joblib xgboost
  ```
- Ensure the model files are in the same directory as `emotion_frequency.py`.

### Steps to Run
1. Open a terminal and navigate to the directory containing `emotion_frequency.py`.
2. Run the script:
   ```bash
   python emotion_frequency.py
   ```
3. Follow the prompts:
   - Input gender, time of day, location, weather, activity, feedback, age, mood intensity, and sleep quality.
   - The camera will activate for real-time emotion detection.
4. After 8-10 seconds, the system will:
   - Display the detected emotion and the recommended frequency.
   - Automatically play the frequency on the [Online Tone Generator](https://onlinetonegenerator.com).

---

## Example Workflow
1. **Input**:
   - Gender: Male
   - Time of Day: Morning
   - Location: Work
   - Weather: Sunny
   - Activity: Working
   - Feedback: 7
   - Age: 25
   - Mood Intensity: 8
   - Sleep Quality: 6

2. **Emotion Detection**:
   - Camera detects the emotion as "happy".

3. **Frequency Recommendation**:
   - System predicts a frequency of `364.0 Hz`.

4. **Output**:
   - Detected Emotion: `happy`
   - Final Frequency Recommendation: `364.0 Hz`
   - Frequency automatically opens on the online tone generator in the browser.

---
