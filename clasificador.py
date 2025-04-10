"""
Este módulo permite analizar la postura de una persona en una imagen utilizando MediaPipe y comparar sus coordenadas
con la base de datos de posturas previamente registradas para clasificarla.
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

bd = pd.read_csv('resultado_posturas.csv')

estadisticas_bd = bd.groupby('Estado').agg(
    Nose_Y_Mean=('Nose_Y', 'mean'), Nose_Y_SD=('Nose_Y', 'std'),
    Left_Shoulder_Y_Mean=('Left_Shoulder_Y', 'mean'), Left_Shoulder_Y_SD=('Left_Shoulder_Y', 'std'),
    Right_Shoulder_Y_Mean=('Right_Shoulder_Y', 'mean'), Right_Shoulder_Y_SD=('Right_Shoulder_Y', 'std')
).reset_index()

mp_pose = mp.solutions.pose

def analizar_pose(frame):
    """
    Analiza un frame (imagen) y clasifica la postura de la persona detectada.

    Parámetros:
    - frame (numpy.ndarray): Imagen en formato BGR (de OpenCV).

    Retorna:
    - str: Estado o postura predicha.
    """
    with mp_pose.Pose() as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            min_dist = float('inf')
            estado_predicho = "Desconocido"

            for _, row in estadisticas_bd.iterrows():
                dist = np.sqrt(
                    (nose_y - row['Nose_Y_Mean']) ** 2 +
                    (left_shoulder_y - row['Left_Shoulder_Y_Mean']) ** 2 +
                    (right_shoulder_y - row['Right_Shoulder_Y_Mean']) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    estado_predicho = row['Estado']

            return estado_predicho

        return "No Detectado"
