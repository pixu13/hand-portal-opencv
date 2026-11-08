import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,min_detection_confidence=0.6,min_tracking_confidence=0.5) as hands:

    while True:
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

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            height, width = image.shape[:2]

            def get_point(hand, landmark_id):
                landmark = hand.landmark[landmark_id]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                return (x, y)

            left_hand,right_hand= sorted(results.multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)

            vertex_hands= np.array([get_point(left_hand, 4),get_point(left_hand, 8),get_point(right_hand, 4),get_point(right_hand, 8)],dtype=np.int32)

            #thermal effect
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thermal_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)


            #binary mask
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [vertex_hands], 255)

            #we keep the filter only inside the polygon
            portal_region = cv2.bitwise_and(thermal_image, thermal_image, mask=mask)

            #keep the original image outside the polygon
            inverse_mask = cv2.bitwise_not(mask)
            background_region = cv2.bitwise_and(image, image, mask=inverse_mask)


            #combine the two regions
            image = cv2.add(portal_region, background_region)

            print("Image:", image.shape)
            print("Thermal:", thermal_image.shape)
            print("Mask:", mask.shape)


            cv2.polylines(image,[vertex_hands],isClosed=True,color=(255, 0, 255),thickness=3)

            for vertex in vertex_hands:
                cv2.circle(image,vertex,radius=7,color=(0, 255, 0),thickness=-1)

            

        # Dibujamos los puntos de las manos
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand,mp_hands.HAND_CONNECTIONS)

                finger_tips = [4, 8, 12, 16, 20]
                finger_mcps = [2, 5, 9, 13, 17]

        # Mostramos la imagen
        cv2.imshow("Mi primer Hand Tracker", image)

        # Presiona esc  para cerrar
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()