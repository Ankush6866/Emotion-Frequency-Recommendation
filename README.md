

# Facial Emotion Detection and Frequency Prediction Model

This project involves the development and training of two machine learning models. The first model detects emotions from facial expressions, while the second model predicts a frequency based on various factors such as emotion, gender, time, location, weather, activity, feedback, age, mood intensity, and sleep quality.
## Datasets

1. **CNN Model Dataset**:
   - The dataset for the CNN model is stored in `archive.zip`.
   - It contains two folders: `train` and `test` which can be used for training and testing the model.

2. **Frequency Prediction Model Dataset**:
   - The `Emotion_frequency_dataset.csv` file is used for the second model, which is for emotion-based frequency prediction.
   - This dataset includes features like emotion, gender, age, weather, and feedback  used for predicting the frequency for various combinations of inputs.


## Key Features:
1. **Facial Emotion Detection**: The model classifies facial expressions into 7 distinct emotions: 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', and 'surprise'.
2. **Frequency Prediction**: The model predicts a frequency (value) based on the detected emotion and other contextual inputs.

## Project Setup:

### Prerequisites:
1. **Libraries**: 
   - `keras` for building the deep learning model
   - `xgboost` for training the frequency prediction model
   - `pandas`, `numpy` for data manipulation
   - `matplotlib` for visualizing results
   - `tqdm` for progress bars
   - `sklearn` for preprocessing and splitting the dataset

2. **Environment**: Google Colab or local Python environment with necessary packages installed.



### Installation:
Install the required packages using pip:
```bash
pip install keras xgboost pandas numpy matplotlib scikit-learn tqdm keras-preprocessing
```

### Steps to Run:

1. **Prepare the Dataset**: 
   - The training images are located in `images/train` and testing images in `images/test`.
   - The dataset for frequency prediction is in `emotion_frequency_dataset.csv`.

2. **Train the Emotion Detection Model**:
   - This model uses Convolutional Neural Networks (CNNs) to detect facial emotions from grayscale images of faces.
   - The model is trained using images, labels (7 emotions), and saved as `Face_Emotion.keras` and the architecture is saved as `facialemotionmodel.json`.

3. **Train the Frequency Prediction Model**:
   - The dataset is preprocessed with one-hot encoding for categorical columns.
   - An XGBoost regressor model is used to predict the frequency based on input features.
   - The trained frequency prediction model is saved in `emotion_frequency_model.pkl`.

4. **Model Evaluation**:
   - The emotion detection model is evaluated based on classification accuracy.
   - The frequency prediction model is evaluated using Mean Squared Error (MSE) and R² score.

### Usage:
- **Facial Emotion Detection**: 
   - You can use the model to predict the emotion from a given image using the `ef(image)` function.
   
   Example:
   ```python
   image_path = 'path_to_image.jpg'
   pred = model.predict(ef(image_path))
   print(f'Model Prediction: {pred_label}')
   ```

- **Frequency Prediction**:
   - The frequency prediction model uses various input features (emotion, mood intensity, etc.) and returns the predicted frequency.

   Example:
   ```python
   prediction = frequency_model.predict(input_features)
   ```

### Saving and Loading the Models:
- **Emotion Detection Model**:
   - The model is saved in `.keras` format and architecture in `.json` format.
   - Weights are saved in `.h5` format.
  
   To load the model:
   ```python
   model = load_model('path_to_model/Face_Emotion.keras')
   ```

- **Frequency Prediction Model**:
   - The trained XGBoost model is saved as `emotion_frequency_model.pkl`.

   To load the model:
   ```python
   import pickle
   with open('emotion_frequency_model.pkl', 'rb') as f:
       frequency_model = pickle.load(f)
   ```

### Evaluation:
- The model performance for emotion detection is measured using accuracy.
- The frequency prediction model's performance is evaluated using MSE and R² score.

### Example Output:

```python
Original image: 'angry'
Predicted emotion: 'angry'

Frequency prediction: 125.0
```

### Visualizations:
- Feature importance for the frequency prediction model can be visualized using bar charts.

## License:
This project is licensed under the MIT License.

---
