
import cv2
import mediapipe as mp

# Inicializamos MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Encendemos la cámara
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    while True:
        # Leemos un fotograma
        data, image = cap.read()

        if not data:
            print("No se pudo leer la cámara")
            break

        # Efecto espejo
        image = cv2.flip(image, 1)

        # Convertimos BGR a RGB para MediaPipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detectamos las manos
        results = hands.process(rgb)

        # Dibujamos los puntos de las manos
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

        # Mostramos la imagen
        cv2.imshow("Mi primer Hand Tracker", image)

        # Presiona Q para cerrar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
