import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

def load_dataset(file_path):
    data = []
    labels = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            label = row[0]  
            vector = [float(val) for val in row[1:]]  
            data.append(vector)
            labels.append(label)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return np.array(data), np.array(labels)

data, labels = load_dataset("LetterVectorsNormalised.csv")

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Input(shape=(len(train_data[0]),)),  
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),  
    keras.layers.Dense(64, activation="relu"),   
    keras.layers.Dense(26, activation="softmax")  # Output layer with 26 units for each letter
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_data, train_labels, epochs=100, validation_split=0.15)  

test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")

model.save("sign_language_model.h5")
