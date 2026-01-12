
# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
# import mujoco
# import time
# from ikpy import chain
# from mpl_toolkits.mplot3d import Axes3D

# R_debug = np.array([
#     [-0.35242119,  0.92712930, -0.12739924],
#     [-0.77335109, -0.36517949, -0.51823935],
#     [-0.52699848, -0.08411419,  0.84569345],
# ])
# finger_chains = {
#     "thumb":  [1, 2, 3, 4],
#     "index":  [5, 6, 7, 8],
#     "middle": [9, 10, 11, 12],
#     "ring":   [13, 14, 15, 16],
#     "pinky":  [17, 18, 19, 20],
# }

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# connections = mp_hands.HAND_CONNECTIONS # Define connections for plotting

# cap = cv2.VideoCapture(1)

# # --- 1. SETUP NON-BLOCKING 3D PLOT ---
# plt.ion()  # Turn on interactive mode
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("MediaPipe Raw Landmark Coordinates (3D)")
# ax.set_xlabel("X (Normalized)")
# ax.set_ylabel("Y (Normalized)")
# ax.set_zlabel("Z (Relative Depth)")

# # Set fixed limits based on normalized MP coordinates for stability
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-0.5, 0.5]) # Normalized Z is usually around 0, sometimes negative for depth.
# ax.invert_zaxis() # Invert Z-axis if you want a more intuitive depth visualization

# # Set equal aspect ratio
# ax.set_box_aspect([1, 1, 1])
# # --- END PLOT SETUP ---

# def get_start_angles(chain):
#         q = []
#         for link in chain.links:
#             if link.bounds is not None: 
#                 if link.name in ['index_splay', 'ring_splay', 'pinky_splay']:
#                     q.append((link.bounds[0] + link.bounds[1]) / 2)  # middle bound
#                 else: q.append(link.bounds[0])  # lower bound
#             else: q.append(0.0)  # fixed joint
#         return q
    
# def apply_offset(fk_matrix, offset): 
#     rot = fk_matrix[:3, :3] 
#     pos = fk_matrix[:3, 3] 
#     world_offset = rot @ np.array(offset) 
#     return pos + world_offset

# tip_offsets = {
#     "thumb":  [-0.017, 0.0025, -0.005],
#     "index":  [-0.022, 0, -0.004],
#     "middle": [-0.02, 0.002, -0.004],
#     "ring":   [-0.02, 0.002, -0.004],
#     "pinky":  [-0.017, 0.002, -0.004]
# }
# mcp_offsets = {
#     "wrist": [0, 0, -0.015],
#     "index": [0, 0, -0.008],
#     "pinky": [0, 0, 0.006]
# }
# chain_names = {
#     "index":  ['index_splay', 'index_mcp', 'index_pip', 'index_dip'],
#     "middle": ['mid_mcp', 'mid_pip', 'mid_dip'],
#     "ring":   ['ring_splay', 'ring_mcp', 'ring_pip', 'ring_dip'],
#     "pinky":  ['pinky_splay', 'pinky_mcp', 'pinky_pip', 'pinky_dip'],
#     "thumb":  ['thumb_cmc', 'thumb_mcp', 'thumb_ip']
# }
# chains = {}
# for finger, links in chain_names.items():
#     base_elements = ['backhand', links[0]]  # common base + finger knuckle
#     chains[finger] = chain.Chain.from_urdf_file(
#         "assets/robot.urdf",
#         base_elements=base_elements,
#         name=finger,
#         last_link_vector=tip_offsets[finger],
#     )
# q = get_start_angles(chains['index'])
# fk = chains['index'].forward_kinematics(q, full_kinematics=True)
# wrist_pos = apply_offset(fk[0], mcp_offsets["wrist"])
# index_mcp_pos = apply_offset(fk[1], mcp_offsets["index"])

# q = get_start_angles(chains['pinky'])
# fk = chains['pinky'].forward_kinematics(q, full_kinematics=True)
# pinky_mcp_pos = apply_offset(fk[1], mcp_offsets["pinky"])
# hand_width = np.linalg.norm(index_mcp_pos - wrist_pos)

# # Hand frame axes
# init_x_axis = index_mcp_pos - wrist_pos 
# init_z_axis = np.cross(index_mcp_pos - wrist_pos, pinky_mcp_pos - wrist_pos) 
# init_y_axis = np.cross(init_z_axis, init_x_axis)
# init_x_axis = init_x_axis / np.linalg.norm(init_x_axis)
# init_y_axis = init_y_axis / np.linalg.norm(init_y_axis)
# init_z_axis = init_z_axis / np.linalg.norm(init_z_axis)
# R_robot = np.stack([init_x_axis, init_y_axis, init_z_axis], axis=1)
# print(R_robot)

# points_target = []
# for finger in tip_offsets.keys():
#     q = get_start_angles(chains[finger])
#     fk = chains[finger].forward_kinematics(q, full_kinematics=True)
#     points_target.append(fk[-1][:3, 3])


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
#             for hand_world in results.multi_hand_world_landmarks:
#                 points = np.array(
#                     [[lm.x, lm.y, lm.z] for lm in hand_world.landmark],
#                     dtype=float
#                 )

#         # points_target = points @ R_debug
#         # wrist_t = points_target[0]
#         # index_mcp_t = points_target[5]
#         # pinky_mcp_t = points_target[17]
#         # z_t = np.cross(index_mcp_t - wrist_t, pinky_mcp_t - wrist_t)
#         # z_t = z_t / np.linalg.norm(z_t)
#         # x_t = index_mcp_t - wrist_t
#         # y_t = np.cross(z_t, x_t)
#         # x_t = x_t / np.linalg.norm(x_t)
#         # y_t = y_t / np.linalg.norm(y_t)
#         # R_robot = np.stack([x_t, y_t, z_t], axis=1) 

#         points = points - points[0]
#         wrist = points[0]
#         index_mcp = points[5]
#         pinky_mcp = points[17]

#         palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
#         palm_normal = palm_normal / np.linalg.norm(palm_normal)
#         x_axis = index_mcp - wrist
#         y_axis = np.cross(palm_normal, x_axis)
#         x_axis = x_axis / np.linalg.norm(x_axis)
#         y_axis = y_axis / np.linalg.norm(y_axis)
#         R_hand = np.stack([x_axis, y_axis, palm_normal], axis=1)  # columns = hand axes
#         width = np.linalg.norm(index_mcp - wrist)

#         rel_hand_frame = (hand_width / width) * (points @ R_hand  @ R_robot.T) + wrist_pos
#         ax.cla()

#         # ---- plot only the 5 fingertip points ----
#         finger_names = ["thumb", "index", "middle", "ring", "pinky"]
#         finger_colors = ["red", "green", "blue", "purple", "orange"]

#         for i, pt in enumerate(points_target):
#             ax.scatter(
#                 pt[0], pt[1], pt[2],
#                 c=finger_colors[i],
#                 s=80,
#                 marker="o",
#                 label=f"{finger_names[i]}_target"
#             )

#         # ---- plot rel_hand_frame ----
#         ax.scatter(
#             rel_hand_frame[:, 0],
#             rel_hand_frame[:, 1],
#             rel_hand_frame[:, 2],
#             c="red",
#             s=40,
#             marker="^",
#             label="rel_hand_frame"
#         )

#         for finger, idxs in finger_chains.items():
#             for i in range(len(idxs) - 1):
#                 a, b = idxs[i], idxs[i + 1]
#                 ax.plot(
#                     [rel_hand_frame[a, 0], rel_hand_frame[b, 0]],
#                     [rel_hand_frame[a, 1], rel_hand_frame[b, 1]],
#                     [rel_hand_frame[a, 2], rel_hand_frame[b, 2]],
#                     color="red",
#                     linewidth=2,
#                     linestyle="--"
#                 )

#         for idx in [1, 5, 9, 13, 17]:
#             ax.plot(
#                 [rel_hand_frame[0, 0], rel_hand_frame[idx, 0]],
#                 [rel_hand_frame[0, 1], rel_hand_frame[idx, 1]],
#                 [rel_hand_frame[0, 2], rel_hand_frame[idx, 2]],
#                 color="red",
#                 alpha=0.5,
#                 linestyle="--"
#             )


#         # ---- wrist markers ----
#         ax.scatter(*wrist_pos, c="black", s=120, marker="s", label="wrist_target")
#         ax.scatter(*rel_hand_frame[0], c="green", s=120, marker="s", label="wrist_rel")

#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         ax.set_title("points_target vs rel_hand_frame")
#         ax.set_box_aspect([1, 1, 1])
#         ax.legend()

#         # plt.pause(10000)   # <-- THIS is the key

#         cv2.imshow("MediaPipe Hands", frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
#             break

# cap.release()
# cv2.destroyAllWindows()
# # --- 5. CLEAN UP PLOT ---
# plt.ioff()
# plt.close(fig)
# # --- END CLEAN UP ---


import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import pybullet_data
import time
import pybullet as p
from ikpy import chain
from mpl_toolkits.mplot3d import Axes3D

finger_chains = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
connections = mp_hands.HAND_CONNECTIONS # Define connections for plotting

cap = cv2.VideoCapture(1)

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

def apply_offset(pos, orn, offset):
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    world_offset = rot_matrix.dot(offset)
    return pos + world_offset

client_id = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0) # No gravity so hand doesn't droop
robot_id = p.loadURDF("/Users/sissi/Downloads/RUKA/assets/robot.urdf", useFixedBase=True)

joint_name_to_index = {}
link_name_to_index = {}
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode()
    link_name = info[12].decode()
    joint_name_to_index[joint_name] = i
    link_name_to_index[link_name] = i
    print(f"Joint {i}: {joint_name} -> Link: {link_name}")

mcp_offsets = {
    "wrist": [0, 0, -0.015],
    "index": [0, 0, -0.008],
    "pinky": [0, 0, 0.006]
}

state = p.getLinkState(robot_id, 1)
link_pos = state[4]
link_orn = state[5]
wrist_pos = apply_offset(link_pos, link_orn, mcp_offsets["wrist"])
state = p.getLinkState(robot_id, 2)
link_pos = state[4]
link_orn = state[5]
index_mcp_pos = apply_offset(link_pos, link_orn, mcp_offsets["index"])
state = p.getLinkState(robot_id, 13)
link_pos = state[4]
link_orn = state[5]
pinky_mcp_pos = apply_offset(link_pos, link_orn, mcp_offsets["pinky"])
hand_width = (0 * np.linalg.norm(pinky_mcp_pos - wrist_pos) + 2.5 * np.linalg.norm(index_mcp_pos - wrist_pos)) / 3

init_x_axis = index_mcp_pos - wrist_pos 
init_z_axis = np.cross(index_mcp_pos - wrist_pos, pinky_mcp_pos - wrist_pos) 
init_y_axis = np.cross(init_z_axis, init_x_axis)
init_x_axis = init_x_axis / np.linalg.norm(init_x_axis)
init_y_axis = init_y_axis / np.linalg.norm(init_y_axis)
init_z_axis = init_z_axis / np.linalg.norm(init_z_axis)
R_robot = np.stack([init_x_axis, init_y_axis, init_z_axis], axis=1)

points_target = []
for name in ["thumb_actual_tip", "index_actual_tip", "middle_actual_tip", "ring_actual_tip", "pinky_actual_tip"]:
    state = p.getLinkState(robot_id, link_name_to_index[name])
    link_pos = state[4]
    link_orn = state[5]
    points_target.append(apply_offset(link_pos, link_orn, [0, 0, 0]))

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_world in results.multi_hand_world_landmarks:
                points = np.array([[lm.x, lm.y, lm.z] for lm in hand_world.landmark], dtype=float)

        points = points - points[0]
        wrist = points[0]
        index_mcp = points[5]
        pinky_mcp = points[17]

        palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
        palm_normal = palm_normal / np.linalg.norm(palm_normal)
        x_axis = index_mcp - wrist
        y_axis = np.cross(palm_normal, x_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        R_hand = np.stack([x_axis, y_axis, palm_normal], axis=1)  # columns = hand axes
        width = (2.5 * np.linalg.norm(index_mcp - wrist) + 0 * np.linalg.norm(pinky_mcp - wrist)) / 3

        rel_hand_frame = (hand_width / width) * (points @ R_hand  @ R_robot.T) + wrist_pos

        ax.cla()

        # ---- plot only the 5 fingertip points ----
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        finger_colors = ["red", "green", "blue", "purple", "orange"]

        for i, pt in enumerate(points_target):
            ax.scatter(
                pt[0], pt[1], pt[2],
                c=finger_colors[i],
                s=80,
                marker="o",
                label=f"{finger_names[i]}_target"
            )

        # ---- plot rel_hand_frame ----
        ax.scatter(
            rel_hand_frame[:, 0],
            rel_hand_frame[:, 1],
            rel_hand_frame[:, 2],
            c="red",
            s=40,
            marker="^",
            label="rel_hand_frame"
        )

        for finger, idxs in finger_chains.items():
            for i in range(len(idxs) - 1):
                a, b = idxs[i], idxs[i + 1]
                ax.plot(
                    [rel_hand_frame[a, 0], rel_hand_frame[b, 0]],
                    [rel_hand_frame[a, 1], rel_hand_frame[b, 1]],
                    [rel_hand_frame[a, 2], rel_hand_frame[b, 2]],
                    color="red",
                    linewidth=2,
                    linestyle="--"
                )

        for idx in [1, 5, 9, 13, 17]:
            ax.plot(
                [rel_hand_frame[0, 0], rel_hand_frame[idx, 0]],
                [rel_hand_frame[0, 1], rel_hand_frame[idx, 1]],
                [rel_hand_frame[0, 2], rel_hand_frame[idx, 2]],
                color="red",
                alpha=0.5,
                linestyle="--"
            )


        # ---- wrist markers ----
        ax.scatter(*wrist_pos, c="black", s=120, marker="s", label="wrist_target")
        ax.scatter(*rel_hand_frame[0], c="green", s=120, marker="s", label="wrist_rel")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("points_target vs rel_hand_frame")
        ax.set_box_aspect([1, 1, 1])
        ax.legend()

        cv2.imshow("MediaPipe Hands", frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close(fig)