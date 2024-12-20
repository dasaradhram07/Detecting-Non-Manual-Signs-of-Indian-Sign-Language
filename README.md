# Non-Manual Sign Language Detection

This project aims to detect non-manual features of Indian Sign Language using **MediaPipe Holistic** and **LSTM (Long Short-Term Memory) neural networks**. The system captures facial expressions, head movements, and body posture, converting them into text for seamless communication.

## What are Non Manual Signs ? and How are they different from Manual signs ?

![image](https://github.com/user-attachments/assets/7cf6ed74-73cf-4919-ad0b-b1f67eb5c283)

## Table of Contents
1. [Install and Import Dependencies](#1-install-and-import-dependencies)
2. [Key Points Using MP Holistic](#2-key-points-using-mp-holistic)
3. [Extract Keypoint Values](#3-extract-keypoint-values)
4. [Setup Folders for Collections](#4-setup-folders-for-collections)
5. [Collect Key Point Values for Training and Testing](#5-collect-key-point-values-for-training-and-testing)
6. [Preprocess Data and Create Labels and Features](#6-preprocess-data-and-create-labels-and-features)
7. [Build and Train LSTM Neural Network](#7-build-and-train-lstm-neural-network)
8. [Make Predictions](#8-make-predictions)
9. [Save Weights](#9-save-weights)
10. [Evaluation Using Confusion Matrix and Accuracy](#10-evaluation-using-confusion-matrix-and-accuracy)
11. [Testing in Real Time](#11-testing-in-real-time)
12. [ Acknowledgements](#12-acknowledgements)
   


---

## 1. INSTALL AND IMPORT DEPENDENCIES

To get started with the project, you will need to install several Python libraries and dependencies. Below is a list of the required dependencies:

- **TensorFlow**: for training the neural network.
- **Keras**: for building the LSTM model.
- **OpenCV**: for capturing video and processing images.
- **MediaPipe**: for extracting keypoints related to body, face, and hands.
- **NumPy**: for data manipulation and processing.
- **Pandas**: for data organization.
- **Matplotlib**: for visualizing the results.

### Installation Command:
```bash
pip install tensorflow keras opencv-python mediapipe numpy pandas matplotlib
```
## 2. KEY POINTS USING MP HOLISTIC

**MediaPipe Holistic** is an efficient tool for extracting keypoints from the body, face, and hands, which are useful in recognizing non-manual features in sign language gestures. The following keypoint categories are captured using `mp.solutions.holistic`:

- **Face landmarks**: Captures facial features such as eye movement, lip shape, and expressions.
- **Hand landmarks**: Detects hand gestures, which are important in sign language.
- **Pose landmarks**: Identifies body movements, including head posture and arm positions.

The extraction is performed frame by frame, and the resulting keypoints are fed into the machine learning model.

Example:
```python
import mediapipe as mp

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Process a frame
results = holistic.process(frame)
```

## 3. EXTRACT KEYPOINT VALUES

After processing the video frames with MediaPipe Holistic, you can extract the keypoints as x, y, and z coordinates for each detected landmark. These values are crucial for training the model.
Steps:
- Capture a frame from the webcam or video feed.
- Apply MediaPipe Holistic's process() method to extract keypoints.
- Extract the x, y, and z values for each landmark in the pose_landmarks, face_landmarks, and hand_landmarks attributes.

Example :
```python
if results.pose_landmarks:
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
```
Store these values in a structured format (e.g., NumPy arrays) for further processing.

## 4. SETUP FOLDERS FOR COLLECTIONS
To organize the data efficiently, set up folders for storing keypoint data. As we are using labelled data , we are taking input of 30 videos for each sign and each video consists of 30 frames , the respective key point data is saved in the form of numpy arrays , the folders are named with their respective sign .

Folder structure:
```bash 
/project_directory
    /MP_DATA
        /YES
        /NO
        /NAMASTHE
        /SHRUGGING
```

Further these data is splitted into training and testing 
Make sure that your dataset is organized and consistent before training.

- **/NAMASTHE**: Store keypoint data and labels of sign Namasthe.
- **/YES**: Store keypoint data and labels of sign yes.
- **/NO**: Store keypoint data and labels of sign no.
- **/SHRUGGING**: Store keypoint data and labels of sign Shrugging.

## 5. COLLECT KEY POINT VALUES FOR TRAINING AND TESTING 
The next step is to collect keypoint data for both training and testing the model. This involves capturing keypoints for various gestures, storing them in a structured format (.npy) .

Ensuring  collected data is sufficient for each gesture or sign language expression you want the model to recognize.
Make sure the data is evenly distributed to prevent bias.

**Action and Sequence Loops**: 
- The code iterates over a list of predefined actions (actions) and for each action, it processes a specific number of sequences (no_sequences). Each sequence contains a set number of frames (sequence_length).

**Exporting Keypoints:**
- After detecting the landmarks in each frame, the extract_keypoints() function is called to extract the keypoints (coordinates of the landmarks).
- These keypoints are saved to disk in a specified directory (DATA_PATH), where each keypoint set is saved as a .npy file.

## 6. PREPROCESS DATA AND CREATE LABELS AND FEATURES
Once you have collected the keypoints, preprocessing is necessary to prepare the data for training:

1. **Normalization:** Normalize the x, y, and z values of the keypoints to a consistent scale.
2. **Feature Extraction:** Extract features from the keypoints (such as relative distances or angles) that will be used as input for the model.
3. **Label Encoding: **Convert gesture labels into numeric values for training the model (e.g., using one-hot encoding or integer encoding).

Example :
```python
from sklearn.preprocessing import LabelEncoder

# Normalize keypoints
features = np.array(features) / np.max(features)

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
```
## 7. BUILD AND TRAIN LSTM NEURAL NETWORK
Now that the data is prepared, build the LSTM neural network to process the temporal sequences of keypoints. LSTM layers are ideal for sequence-based data such as sign language gestures.

The LSTM model can be built as follows:

1. **Input layer:** Input shape matches the dimensions of the keypoint data.
2. **LSTM layers:** These layers process the sequence of keypoints over time.
3. **Dense layers:** Output the predictions for gesture classification.
4. **Batch Normalization layer:** To improve the training of deep neural networks by normalizing the input of each layer
LSTM MODEL :
``` Python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam

model = Sequential()
model.add(Input(shape=(30, 1662)))

# LSTM layers with dropout and batch normalization
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(BatchNormalization())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

# Compile model with a lower learning rate
optimizer = Adam(learning_rate=0.00005)  # Further reduce learning rate for fine-tuning
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```
## 8. MAKE PREDICTIONS 
Once the model is trained, use it to make predictions on new input data. The input data will consist of real-time captured keypoints, which are processed through the trained model.

Example :
```Python
# Make predictions on new data
predictions = model.predict(X_test)

# Get the predicted class label
predicted_label = np.argmax(predictions)
```
You can use these predictions to display the corresponding gesture label.

## 9. SAVE WEIGHTS
After training the model, save its weights to disk so that you can load the model later without retraining.

Example:
``` Python 
# Save the model weights
model.save_weights('action.keras')
```
Later, you can load the saved weights:
```Python
# Load the saved weights
model.load_weights('action.keras')
```
## 10. EVALUATION USING CONFUSION MATRIX AND ACCURACY
To evaluate the model's performance, calculate accuracy and generate a confusion matrix. The confusion matrix will help you visualize how well the model is performing on each class.

Example:
```python
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score

ytrue =np.argmax(y_train,axis=1).tolist()
yhat = np.argmax(yhat , axis=1).tolist()

accuracy_score(ytrue,yhat)
```
## 11. TESTING IN REAL TIME
The final step is testing the model in real time. You can capture frames from a webcam, process them to extract keypoints, and use the trained model to classify the gestures.

Example:
```Python 
# 1.New Detection variables
sequence = []
sentence = []
threshold = 0.4

cap = cv2.VideoCapture(0)
# set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic :
    while cap.isOpened():
        #read feed
        ret, frame = cap.read()

        #make detection
        image,results = mediapipe_detection(frame,holistic)
        print(results)
        
        #draw landmarks
        draw_styled_landmarks(image,results)
        
        #2.Prediction logic
        keypoints = extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:30]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence ,axis =0))[0]
            print(actions[np.argmax(res)])
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:30]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        #3. Viz logic
            if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # show to screen
        cv2.imshow('OpenCV Feed', image)

        #break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```
## 12. Acknowledgements
- **TensorFlow:** https://www.tensorflow.org/
- **Keras:** https://keras.io/
- **MediaPipe:** https://google.github.io/mediapipe/
- **OpenCV:** https://opencv.org/
- **NumPy:** https://numpy.org/
- **Pandas:** https://pandas.pydata.org/
- **Matplotlib:** https://matplotlib.org/


