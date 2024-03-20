import cv2
import mediapipe as mp
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pyttsx3

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

def load_labels(file_path):
    labels = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            label = row[0]  
            labels.append(label)
    return labels

model = keras.models.load_model("sign_language_model.h5")

label_encoder = LabelEncoder()
train_labels = load_labels("LetterVectorsNormalised.csv")
label_encoder.fit(train_labels)

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

engine = pyttsx3.init()

def normalizer(vectorAxis):
    normalized = []
    ax_range = max(vectorAxis) - min(vectorAxis)
    for i in range(0, len(vectorAxis)):
        normalized.append((vectorAxis[i] - min(vectorAxis)) / ax_range)
    return normalized

while True:
    data, image = cap.read()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mphands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # create hand vector
            vector, vectorX, vectorY, vectorZ = [], [], [], []

            for landmark in hand_landmarks.landmark:
                vectorX.append(landmark.x)
                vectorY.append(landmark.y)
                vectorZ.append(landmark.z)

            distances = []

            vector.extend(normalizer(vectorX))
            vector.extend(normalizer(vectorY))
            vector.extend(normalizer(vectorZ))

            prediction = model.predict(np.array([vector]))

            predicted_labels = label_encoder.inverse_transform(np.argsort(prediction[0])[::-1][:3])
            predicted_probabilities = np.sort(prediction[0])[::-1][:3] * 100  # Convert to percentage

            cv2.putText(image, (predicted_labels[0].capitalize()), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, (f"{predicted_labels[0]}: {predicted_probabilities[0]:.2f}%\n"), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, (f"{predicted_labels[1]}: {predicted_probabilities[1]:.2f}%\n"), (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, (f"{predicted_labels[2]}: {predicted_probabilities[2]:.2f}%\n"), (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Read out predicted letter
            engine.say(predicted_labels[0])
            engine.runAndWait()

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
