import cv2
import mediapipe as mp
import pyautogui

mp_maos = mp.solutions.hands
maos = mp_maos.Hands()

mp_desenhado = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

fator_deslocamento = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = maos.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            if hand_landmarks.landmark[mp_maos.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_maos.HandLandmark.INDEX_FINGER_MCP].x:
   
                x, y = int(hand_landmarks.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                pyautogui.moveTo(x * fator_deslocamento, y * fator_deslocamento)
            
            mp_desenhado.draw_landmarks(frame, hand_landmarks, mp_maos.HAND_CONNECTIONS)  # Alteramos 'mp_drawing' para 'mp_desenhado'
    
    cv2.imshow("Hand Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
