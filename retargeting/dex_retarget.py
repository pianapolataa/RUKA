import cv2
import mediapipe as mp
import numpy as np
import pybullet as p
import pybullet_data
import time
from dex_retargeting.retargeting_config import RetargetingConfig
from retargeting.dex_retarget_controller_mp import DexRukav2Handler
from ruka_hand.control.hand import Hand
from ruka_hand.utils.trajectory import move_to_pos

controller = DexRukav2Handler()

def main():
    # 3. SETUP MEDIAPIPE
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(1)
    first_pos = True

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            # p.stepSimulation()
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_world_landmarks:
                hand_landmarks = results.multi_hand_world_landmarks[0]
                
                # Raw Points (21, 3)
                points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                command = controller.get_command(points)
                command = np.array([
                    2815.        , 2952.39456761, 1713.66489977, 1756.        ,
                    2857.        , 1776.45128655, 2195.        , 1776.        ,
                    1100.81154701, 1343.81727454, 1207.        , 1402.96441607,
                    3080.25904964, 2357.        , 1650.08004243, 1654.45926902
                ])

                print(np.array(command))
                curr_pos = controller.hand.read_pos()
                if first_pos == True:
                    move_to_pos(curr_pos, command, controller.hand, traj_len=35)
                    first_pos = False
                else: move_to_pos(curr_pos, command, controller.hand, traj_len=10)

            cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            time.sleep(1./60.)

    p.disconnect()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()