import mediapipe as mp
import cv2
import time




mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
cap.set(3, 1280) 
cap.set(4, 720)  



# Load your meme images


MEMES = {
    "PEACE": cv2.imread('C:/Users/nihal/OneDrive/Desktop/RAJKUMAR.jpg'),
    "THUMBS_UP": cv2.imread('C:/Users/nihal/OneDrive/Desktop/SRK.jpg'),
    "OPEN_PALM": cv2.imread('C:/Users/nihal/OneDrive/Desktop/arjun.jpg')
}


for gesture, img in MEMES.items():
    if img is None:
        print(f"Error: Could not load meme for {gesture}. Check the file path.")
        # Create a black placeholder image if a meme is missing
        MEMES[gesture] = cv2.resize(cv2.imread('C:/Users/nihal/OneDrive/Desktop/black.jpg'), (400, 400))



meme_display_time = 0
cooldown = 3 



while True:
    success, frame = cap.read()
    if not success:
        break

    
    frame = cv2.flip(frame, 1)
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_detected = len(faces) > 0
    
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # If a face is detected, proceed to detect hands
    if face_detected:
        # Process the frame to find hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                
                
                # Get landmark coordinates
                landmarks = hand_landmarks.landmark
                
                # Tip IDs for fingers
                tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
                
                # Create a list to store which fingers are up
                fingers_up = []

                
                if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

                
                for id in range(1, 5):
                    if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)
                
                
                total_fingers = fingers_up.count(1)
                
                current_gesture = "UNKNOWN"
                
                
                if total_fingers == 2 and fingers_up[1] and fingers_up[2]:
                    current_gesture = "PEACE"
                elif total_fingers == 1 and fingers_up[0]:
                    current_gesture = "THUMBS_UP"
                elif total_fingers == 5:
                    current_gesture = "OPEN_PALM"
                
                
                if current_gesture in MEMES and (time.time() - meme_display_time > cooldown):
                    meme_image = MEMES[current_gesture]
                    cv2.imshow("Meme", meme_image)
                    meme_display_time = time.time() 

                
                cv2.putText(frame, current_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    
    cv2.imshow("Gesture Meme Controller", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()