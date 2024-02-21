import cv2
import mediapipe as mp
import math
import csv

file = open("LetterVectorsNormalised.csv", "a")
writer = csv.writer(file)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

def normiliser(vectorAxis):
    normalised = []
    axrange = max(vectorAxis)-min(vectorAxis)
    for i in range(0, len(vectorAxis)):
        normalised.append((vectorAxis[i]-min(vectorAxis))/axrange)
    return normalised


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
            vector, vectorX, vectorY, vectorZ= ["v"], [], [], []

            for landmark in hand_landmarks.landmark:
                vectorX.append(landmark.x)
                vectorY.append(landmark.y)
                vectorZ.append(landmark.z)
            #normalise vector
            
            vector.extend(normiliser(vectorX))
            vector.extend(normiliser(vectorY))
            vector.extend(normiliser(vectorZ))
            
            #print(vector)
            
    cv2.imshow('MediaPipe Hands', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        writer.writerow(vector)
        print (f"Vector saved- {vector[0]}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
file.close()