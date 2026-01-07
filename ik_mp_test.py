# import cv2
# import mediapipe as mp
# import numpy as np
# from ik_controller import IkRUKAv2Handler

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(1)
# ik_handler = IkRUKAv2Handler() 

# with mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# ) as hands:

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame = cv2.flip(frame, 1)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 points = []
#                 for lm in hand_landmarks.landmark:
#                     points.append([lm.x, lm.y, lm.z])
#                 points = np.array(points, dtype=float)



#                 angles = ik_handler.points_to_joint_angles(points)
#                 # print("Joint angles (deg):", np.round(angles, 2))
#                 print(angles[12], angles[13], angles[11])

#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         cv2.imshow("MediaPipe Hands", frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
#             break

# cap.release()
# cv2.destroyAllWindows()

# """
#     FINGER_NAMES_TO_MOTOR_IDS = {
#         "Thumb": [12, 13, 11],
#         "Index": [7, 8, 6],
#         "Middle": [10, 9],
#         "Ring": [3, 4, 5],
#         "Pinky": [1, 0, 2],
#         "Wrist": [14, 15]
#     }
# """

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mpl_toolkits.mplot3d import Axes3D
from ik_controller import IkRUKAv2Handler # Assuming this path is correct

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
connections = mp_hands.HAND_CONNECTIONS # Define connections for plotting

cap = cv2.VideoCapture(1)
ik_handler = IkRUKAv2Handler() 

# --- 1. SETUP NON-BLOCKING 3D PLOT ---
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("MediaPipe Raw Landmark Coordinates (3D)")
ax.set_xlabel("X (Normalized)")
ax.set_ylabel("Y (Normalized)")
ax.set_zlabel("Z (Relative Depth)")

# Set fixed limits based on normalized MP coordinates for stability
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-0.5, 0.5]) # Normalized Z is usually around 0, sometimes negative for depth.
ax.invert_zaxis() # Invert Z-axis if you want a more intuitive depth visualization

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])
# --- END PLOT SETUP ---


with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # --- 2. CLEAR AND RE-DRAW PLOT ---
        ax.clear()
        ax.set_title("MediaPipe Raw Landmark Coordinates (3D)")
        ax.set_xlabel("X (Normalized)")
        ax.set_ylabel("Y (Normalized)")
        ax.set_zlabel("Z (Relative Depth)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([-0.5, 0.5])
        ax.invert_zaxis()
        ax.set_box_aspect([1, 1, 1])
        # --- END CLEAR/RE-DRAW ---


        if results.multi_hand_landmarks:
            for hand_world in results.multi_hand_world_landmarks:
                points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_world.landmark],
                    dtype=float
                )

                # wrist = points[0]
                # index_mcp = points[5]
                # middle_mcp = points[9]
                # pinky_mcp = points[17]
                # points = points - wrist

                # palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
                # palm_normal = palm_normal / np.linalg.norm(palm_normal)
                # x_axis = index_mcp - wrist
                # y_axis = np.cross(palm_normal, x_axis)
                # x_axis = x_axis / np.linalg.norm(x_axis)
                # y_axis = y_axis / np.linalg.norm(y_axis)
                # points = points - points[0]
                # R_hand = np.stack([x_axis, y_axis, palm_normal], axis=1)  # columns = hand axes
                # R_inv = R_hand.T  # because we want world → hand frame
                # points = points @ R_inv 


                # # --- 3. PLOT RAW POINTS ---

                # axis_length = 0.1 
                # # Plot X-axis (Red)
                # ax.plot([0, axis_length * x_axis[0]], 
                #         [0, axis_length * x_axis[1]], 
                #         [0, axis_length * x_axis[2]], 
                #         color='r', linewidth=3, label='X-axis')
                # # Plot Y-axis (Green)
                # ax.plot([0, axis_length * y_axis[0]], 
                #         [0, axis_length * y_axis[1]], 
                #         [0, axis_length * y_axis[2]], 
                #         color='g', linewidth=3, label='Y-axis')
                # # Plot Z-axis (Blue - Palm Normal)
                # ax.plot([0, axis_length * palm_normal[0]], 
                #         [0, axis_length * palm_normal[1]], 
                #         [0, axis_length * palm_normal[2]], 
                #         color='b', linewidth=3, label='Z-axis (Normal)')
                
                # # Plot the 21 points
                # x, y, z = points[:, 0], points[:, 1], points[:, 2]
                # ax.scatter(x, y, z, c='r', marker='o', s=15)
                
                # # Plot the connections (lines)
                # for connection in connections:
                #     start_idx, end_idx = connection
                #     ax.plot(
                #         [x[start_idx], x[end_idx]],
                #         [y[start_idx], y[end_idx]],
                #         [z[start_idx], z[end_idx]],
                #         c='gray', linestyle='-', linewidth=1
                #     )
                
                # # # --- END PLOT RAW POINTS ---

                angles = ik_handler.points_to_joint_angles(points)
                print("Joint angles (deg):", np.round(angles, 2))
                # print(angles[12], angles[13], angles[11])

                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- 4. UPDATE PLOT ---
        plt.draw()
        plt.pause(0.001) # Small pause to allow Matplotlib to update

        cv2.imshow("MediaPipe Hands", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
# --- 5. CLEAN UP PLOT ---
plt.ioff()
plt.close(fig)
# --- END CLEAN UP ---