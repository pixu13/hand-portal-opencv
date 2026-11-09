import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)


color_maps = [
    ("JET", cv2.COLORMAP_JET),
    ("HOT", cv2.COLORMAP_HOT),
    ("COOL", cv2.COLORMAP_COOL),
    ("BONE", cv2.COLORMAP_BONE),
    ("OCEAN", cv2.COLORMAP_OCEAN),
    ("SUMMER", cv2.COLORMAP_SUMMER),
    ("WINTER", cv2.COLORMAP_WINTER),
    ("RAINBOW", cv2.COLORMAP_RAINBOW),
    ("PINK", cv2.COLORMAP_PINK),
    ("HSV", cv2.COLORMAP_HSV)
]

current_color_map_index=0
num_planes=1

filter_touch=False
plane_touch=False


with mp_hands.Hands(
    max_num_hands=2,min_detection_confidence=0.6,min_tracking_confidence=0.5) as hands:

    while True:
        data, image = cap.read()

        if not data:
            print("error")
            break

        # mirrored image
        image = cv2.flip(image, 1)

        # bgr to rgb
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # hand detection
        results = hands.process(rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            height, width = image.shape[:2]

            def relative_finger_distance(hand, point1_id, point2_id):
                point1=np.array(get_point(hand, point1_id))
                point2=np.array(get_point(hand, point2_id))

                wrist = np.array(get_point(hand, 0))
                middle_mcp = np.array(get_point(hand, 9))

                finger_distance = np.linalg.norm(point1 - point2)
                hand_size= np.linalg.norm(wrist - middle_mcp)

                return finger_distance / max(hand_size, 1) 

            def get_point(hand, landmark_id):
                landmark = hand.landmark[landmark_id]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                return (x, y)

            left_hand,right_hand= sorted(results.multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)

            #measure the finger distance 
            filter_distance= min(relative_finger_distance(hand,4,20) for hand in [left_hand,right_hand])


            #measure the geasture of the hands
            plane_distance= min(relative_finger_distance(hand,4,12) for hand in [left_hand,right_hand])

            touch_threshold=0.35
            release_threshold=0.55

            #gesture 1: pinky fingers and thumbs are touching to change the color map
            if not filter_touch and filter_distance<touch_threshold:
                current_color_map_index=(current_color_map_index+1)%len(color_maps)
                filter_touch=True

            elif filter_touch and filter_distance>release_threshold:
                filter_touch=False


            #gesture 2: index fingers and thumbs are touching to change the number of planes
            if not plane_touch and plane_distance<touch_threshold:
                num_planes=2 if num_planes==1 else 1
                plane_touch=True

            elif plane_touch and plane_distance>release_threshold:
                plane_touch=False


            vertex_hands= np.array([get_point(left_hand, 4),get_point(left_hand, 8),get_point(right_hand, 8),get_point(right_hand, 4)],dtype=np.int32)
            polygons = [vertex_hands]

            #add a second plane between the index and middle fingers if the number of planes is 2
            if num_planes==2:
                second_plane= np.array([get_point(left_hand, 8),get_point(left_hand, 12),get_point(right_hand, 12),get_point(right_hand, 8)],dtype=np.int32)
                polygons.append(second_plane)



            #apply current color map to the image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thermal_image = cv2.applyColorMap(gray_image, color_maps[current_color_map_index][1])



            #binary mask
            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in polygons:
                cv2.fillPoly(mask, [polygon], 255)

            #we keep the filter only inside the polygon
            portal_region = cv2.bitwise_and(thermal_image, thermal_image, mask=mask)

            #keep the original image outside the polygon
            inverse_mask = cv2.bitwise_not(mask)
            background_region = cv2.bitwise_and(image, image, mask=inverse_mask)


            #combine the two regions
            image = cv2.add(portal_region, background_region)


            for polygon in polygons:
                cv2.polylines(image, [polygon], isClosed=True, color=(255, 255, 255), thickness=1)

                for vertex in polygon:
                    cv2.circle(image, tuple(vertex), radius=7, color=(255, 255, 255), thickness=-1)
            

        # hand landmarks
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand,mp_hands.HAND_CONNECTIONS)

                finger_tips = [4, 8, 12, 16, 20]
                finger_mcps = [2, 5, 9, 13, 17]

        # show the image
        cv2.imshow("portal", image)

        # esc to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()