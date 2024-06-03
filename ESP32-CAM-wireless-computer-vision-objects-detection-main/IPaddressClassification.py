import cv2 
import numpy as np
import csv

# Ruta del archivo CSV para guardar los recuentos
csv_file = 'object_counts.csv'

winName = 'Video Stream'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cap = cv2.VideoCapture(0)  # 0 para la primera cámara conectada, puedes cambiarlo si tienes varias cámaras

# Abre el archivo CSV en modo de escritura
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Count'])  # Escribir la primera fila con los encabezados

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:
            # Contadores para diferentes clases de objetos
            object_counts = {}

            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1]
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                cv2.putText(img, className, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                # Actualizar los contadores de objetos detectados
                if className not in object_counts:
                    object_counts[className] = 1
                else:
                    object_counts[className] += 1

            # Escribir los contadores de objetos detectados en el archivo CSV
            for key, value in object_counts.items():
                writer.writerow([key, value])

            # Mostrar los contadores de objetos detectados en el video
            for idx, (key, value) in enumerate(object_counts.items()):
                cv2.putText(img, f"{key}: {value}", (10, 50 + 30 * idx), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(winName, img)

        # Espera a que se presione ESC para salir del bucle
        key = cv2.waitKey(1)
        if key == 27:
            break

# Cierra el archivo CSV
file.close()
cap.release()
cv2.destroyAllWindows()