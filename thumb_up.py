# Importa las bibliotecas necesarias
import cv2
import mediapipe as mp

# Inicializa el módulo de detección de manos de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializa la cámara para capturar el video (el parámetro 0 indica la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Bucle principal para la captura de video
while True:
    # Captura un fotograma del video
    success, image = cap.read()

    # Si la captura fue exitosa
    if not success:
        continue

    # Voltea la imagen horizontalmente para evitar el efecto espejo
    image = cv2.flip(image, 1)

    # Convierte la imagen de BGR a RGB (necesario para MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesa la imagen para detectar las manos
    results = hands.process(image_rgb)

    # Verifica si se detectaron múltiples manos en la imagen
    if results.multi_hand_landmarks:
        # Itera sobre todas las manos detectadas
        for landmarks in results.multi_hand_landmarks:
            # Obtiene las coordenadas de los extremos de los dedos
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Define un margen para mejorar la precisión de la detección
            margin = 0.03  # Ajusta este valor según sea necesario

            # Detección del pulgar arriba
            if thumb_tip.y < index_finger_tip.y - margin and thumb_tip.y < middle_finger_tip.y - margin:
                cv2.putText(image, "Tienes el pulgar arriba :)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Detección del dedo medio arriba
            elif middle_finger_tip.y < thumb_tip.y - margin and middle_finger_tip.y < index_finger_tip.y - margin:
                cv2.putText(image, "Tienes el dedo medio arriba >:(", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Detección del índice arriba y el meñique abajo
            elif index_finger_tip.y < pinky_tip.y - margin:
                cv2.putText(image, "Rock", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Si no se cumple ninguna condición, muestra un punto
            else:
                cv2.putText(image, ".", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Muestra la imagen con las detecciones de gestos de mano
    cv2.imshow("Hand Gestures", image)

    # Espera la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos de la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
