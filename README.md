
# Breast Cancer Classification Model

This project implements a **Breast Cancer Classification Model** using a simple Neural Network. The model is designed to predict whether breast cancer is **Malignant** or **Benign** based on various cell nuclei features.

---

## Model Architecture

The Neural Network was built using **TensorFlow** and **Keras**, with the following architecture:

```python
from tensorflow import keras

# Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),         # Input layer (30 features)
    keras.layers.Dense(20, activation='relu'),       # Hidden layer with 20 neurons
    keras.layers.Dense(2, activation='sigmoid')      # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Model Summary
- **Input Layer**: Flattens the input data (expects 30 features).
- **Hidden Layer**: A Dense layer with **20 neurons** and **ReLU** activation function.
- **Output Layer**: A Dense layer with **2 neurons** and **Sigmoid** activation function for binary classification.

---

## Dataset Overview

The dataset contains measurements of cell nuclei from breast cancer biopsies. The goal is to classify the cancer type into:
- **0**: Benign
- **1**: Malignant

### Example of Dataset Structure:
| Feature 1   | Feature 2   | ... | Feature 30 | Diagnosis |
|-------------|-------------|-----|------------|-----------|
| 17.99       | 10.38       | ... | 0.1189     | M         |
| 13.34       | 15.46       | ... | 0.0874     | B         |

**Note**: The diagnosis was label-encoded to `0` for Benign and `1` for Malignant.

---

## Model Training and Evaluation

The model was trained using **80%** of the data and validated on the remaining **20%**. Below are the performance metrics:

### Training Results:
- **Training Accuracy**: `97.68%`
- **Training Loss**: `0.0988`
- **Validation Accuracy**: `97.83%`
- **Validation Loss**: `0.0940`

### Test Results:
- **Test Accuracy**: `97.70%`
- **Test Loss**: `0.1020`

---

## How to Use the Model

Here’s an example of how to use the model for making predictions:

```python
import numpy as np

# Sample input data (30 features)
input_data = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,
              0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,
              0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,
              0.6656,0.7119,0.2654,0.4601,0.1189]

# Convert input to numpy array and reshape
data_as_array = np.asarray(input_data).reshape(1, -1)

# Make prediction
prediction = model.predict(data_as_array)
predicted_class = np.argmax(prediction)

# Output prediction result
if predicted_class == 1:
    print('The breast cancer is Malignant')
else:
    print('The breast cancer is Benign')
```

---

## Conclusion

This **Breast Cancer Classification Model** is a simple yet effective tool for identifying cancer types using cell nuclei features. With a high accuracy rate, it provides reliable predictions that can assist in early diagnosis and treatment planning.

Feel free to explore, modify, and optimize the model to improve its performance!
