import cv2
import mediapipe as mp
import numpy as np


def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)


class HandReader:
    def __init__(self, cam_index=1):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.initial_wrist_axis = None
        self.initial_palm_normal = None
        self.initial_horiz = None
        self.frame_cnt = 0

    def get_motor_positions(self):
        self.frame_cnt += 1
        if (self.frame_cnt % 5 != 0):
            return None, None
        
        # get one frame, return motor_positions or None if no hand
        success, frame = self.cap.read()
        if not success:
            return None, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        motor_positions = np.zeros(16)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                if handedness != "Right":  # only right hand
                    continue

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
                horiz = index_mcp - ring_mcp 
                horiz = horiz / np.linalg.norm(horiz)
                palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
                palm_normal = palm_normal / np.linalg.norm(palm_normal)

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
                    v1_proj = v1 - np.dot(v1, palm_normal) * palm_normal
                    v2_proj = v2 - np.dot(v2, palm_normal) * palm_normal
                    sideways_angle = np.degrees(angle_between(v1_proj, v2_proj))

                    v1 = mcp - wrist
                    finger_flexion.append(np.degrees(angle_between(v1, v2)))

                    v1 = points[idxs[1]] - points[idxs[0]]
                    v2 = points[idxs[2]] - points[idxs[1]]
                    finger_flexion.append(np.degrees(angle_between(v1, v2)))

                    angles[finger] = {"flexion": finger_flexion, "sideways_mcp": sideways_angle}

                # Save initial baseline if not set yet
                if self.initial_wrist_axis is None:
                    self.initial_wrist_axis = wrist_axis
                    self.initial_palm_normal = palm_normal
                    self.initial_horiz = horiz
                    
                # Wrist angles
                v1 = wrist_axis
                v2 = self.initial_wrist_axis
                v1_proj = v1 - np.dot(v1, self.initial_palm_normal) * self.initial_palm_normal
                v2_proj = v2 - np.dot(v2, self.initial_palm_normal) * self.initial_palm_normal
                yaw = np.degrees(angle_between(v1_proj, v2_proj))
                if np.dot(np.cross(v2_proj, v1_proj), self.initial_palm_normal) > 0: # wrist axis -> pinky, angle going left
                    yaw = -yaw
                
                v1_proj = v1 - np.dot(v1, self.initial_horiz) * self.initial_horiz
                v2_proj = v2 - np.dot(v2, self.initial_horiz) * self.initial_horiz
                pitch = np.degrees(angle_between(v1_proj, v2_proj))
                angles["wrist"] = {"yaw": yaw, "pitch": pitch}

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


                return motor_positions, frame

        return None, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
