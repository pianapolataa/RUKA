import cv2
import mediapipe as mp
import numpy as np

def angle_between(v1, v2):
    # Compute angle in radians
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)

image_path = "img.jpg" 
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Could not load image from {image_path}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("No hand detected.")
    else:
        annotated = image_bgr.copy()
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            if handedness != "Left":  # only process right hand
                continue

            print(f"\nDetected Right hand:")

            # Draw landmarks
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            lm = hand_landmarks.landmark
            points = np.array([[l.x, l.y, l.z] for l in lm])

            wrist = points[0]
            index_mcp = points[5]
            pinky_mcp = points[17]
            middle_mcp = points[9]
            ring_mcp = points[13]
            thumb_cmc = points[1]
            thumb_mcp = points[2] 
            thumb_ip  = points[3]
            thumb_tip = points[4] 
            palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
            palm_normal = palm_normal / np.linalg.norm(palm_normal)
            horiz = index_mcp - pinky_mcp

            wrist_axis = (ring_mcp + middle_mcp) / 2 - wrist
            wrist_axis = wrist_axis / np.linalg.norm(wrist_axis)
            v1 = wrist_axis
            v2 = thumb_ip - thumb_cmc
            v1_proj = v1 - np.dot(v1, palm_normal) * palm_normal
            v2_proj = v2 - np.dot(v2, palm_normal) * palm_normal
            thumb_ip_flex = 90 - np.degrees(angle_between(v1_proj, v2_proj))

            v1 = index_mcp - pinky_mcp
            v1_proj = v1 - np.dot(v1, wrist_axis) * wrist_axis
            v2_proj = v2 - np.dot(v2, wrist_axis) * wrist_axis
            print(np.degrees(angle_between(v1, v2)))
            thumb_mcp_flex = np.degrees(angle_between(v1_proj, v2_proj))

            v1 = thumb_ip - thumb_mcp
            v2 = thumb_tip - thumb_ip 
            thumb_dip_flex = np.degrees(angle_between(v1, v2))

            angles = {
                "thumb": {
                    "mcp": thumb_mcp_flex,
                    "ip": thumb_ip_flex,
                    "dip": thumb_dip_flex
                }
            }
            finger_joints = {
                "index": [5,6,7,8],
                "middle": [9,10,11,12],
                "ring": [13,14,15,16],
                "pinky": [17,18,19,20]
            }

            for finger, idxs in finger_joints.items():
                finger_flexion = []
                mcp = points[idxs[0]]
                pip = points[idxs[1]]          
                v1 = points[10] - points[9]
                v2 = pip - mcp
                v1_proj = v1 - np.dot(v1, horiz) * horiz
                v2_proj = v2 - np.dot(v2, horiz) * horiz
                finger_flexion.append(np.degrees(angle_between(v1_proj, v2_proj)))

                v1_proj = v1 - np.dot(v1, palm_normal) * palm_normal
                v2_proj = v2 - np.dot(v2, palm_normal) * palm_normal
                sideways_angle = np.degrees(angle_between(v1_proj, v2_proj))

                v1 = points[idxs[1]] - points[idxs[0]]
                v2 = points[idxs[2]] - points[idxs[1]]
                finger_flexion.append(np.degrees(angle_between(v1, v2)))

                angles[finger] = {"flexion": finger_flexion, "sideways_mcp": sideways_angle}

            # Wrist angles
            yaw = 0
            pitch = 0
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            angles["wrist"] = {"yaw": yaw_deg, "pitch": pitch_deg}

            motor_positions = np.zeros(16)
            motor_positions[12] = angles["thumb"]["mcp"]
            motor_positions[13] = angles["thumb"]["ip"]
            motor_positions[11] = angles["thumb"]["dip"]
            motor_positions[7] = angles["index"]["sideways_mcp"]
            motor_positions[8] = angles["index"]["flexion"][0]
            motor_positions[6] = angles["index"]["flexion"][1]
            motor_positions[10] = angles["middle"]["flexion"][0]
            motor_positions[9]  = angles["middle"]["flexion"][1]
            motor_positions[3] = angles["ring"]["sideways_mcp"]
            motor_positions[4] = angles["ring"]["flexion"][0] 
            motor_positions[5] = angles["ring"]["flexion"][1]
            motor_positions[1] = angles["pinky"]["sideways_mcp"]
            motor_positions[0] = angles["pinky"]["flexion"][0]
            motor_positions[2] = angles["pinky"]["flexion"][1]
            motor_positions[14] = angles["wrist"]["yaw"]
            motor_positions[15] = angles["wrist"]["pitch"]

            print(motor_positions)

        cv2.imshow("Right Hand Landmarks", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
