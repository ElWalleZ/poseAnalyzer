"""
Este script detecta personas usando YOLOv8 y analiza su postura
con MediaPipe Pose. El análisis se hace de forma paralela
para cada persona detectada, respetando un límite máximo de personas.
"""

import threading
import time
import cv2
from ultralytics import YOLO
from clasificador import analizar_pose

modelo = YOLO('yolov8n.pt')

MAX_PERSONAS = 5 #Número de personas a analizar simultáneamente
cap = cv2.VideoCapture(0)

personas_info = {}
analisis_en_progreso = {}

frame_count = 0

FRAMES_ENTRE_ANALISIS = 10 # Cada cuántos frames analizar la pose


def hilo_analisis_pose(persona_id, recorte):
    """
    Función que ejecuta el análisis de pose en un hilo independiente.

    Parámetros:
    persona_id (int): ID de la persona detectada por YOLO.
    recorte (numpy.ndarray): Imagen recortada de la persona.

    Este hilo analiza la postura y actualiza el estado en personas_info.
    """
    estado = analizar_pose(recorte)
    personas_info[persona_id]["estado"] = estado
    analisis_en_progreso[persona_id] = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    analizar_este_frame = (frame_count % FRAMES_ENTRE_ANALISIS == 0)

    resultados = modelo.track(frame, persist=True, classes=[0], verbose=False)[0]
    personas_detectadas = 0
    personas_actuales = set()

    for box in resultados.boxes:
        if personas_detectadas >= MAX_PERSONAS:
            break

        if box.id is None:
            continue

        persona_id = int(box.id.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        recorte = frame[y1:y2, x1:x2].copy()
        personas_actuales.add(persona_id)

        if persona_id not in personas_info:
            personas_info[persona_id] = {"bbox": (x1, y1, x2, y2), "estado": "Analizando..."}

        personas_info[persona_id]["bbox"] = (x1, y1, x2, y2)

        if analizar_este_frame and analisis_en_progreso.get(persona_id) is None:
            analisis_en_progreso[persona_id] = True
            threading.Thread(target=hilo_analisis_pose, args=(persona_id, recorte)).start()

        personas_detectadas += 1

    for persona_id in list(personas_info.keys()): # Eliminar personas que ya no están
        if persona_id not in personas_actuales:
            personas_info.pop(persona_id)
            analisis_en_progreso.pop(persona_id, None)

    for persona_id, info in personas_info.items():
        x1, y1, x2, y2 = info["bbox"]
        estado = info["estado"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, estado, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Pose Analyzer YOLO - Mejorado', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
