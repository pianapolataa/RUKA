import numpy as np
import pybullet as p
import pybullet_data
import time
from dex_retargeting.retargeting_config import RetargetingConfig
from ruka_hand.control.hand import Hand
from ruka_hand.utils.trajectory import move_to_pos
from get_video_reading import HandReader

def apply_offset(pos, orn, offset):
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    world_offset = rot_matrix.dot(offset)
    return pos + world_offset

# 1. SETUP PYBULLET
client_id = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0) 
robot_id = p.loadURDF("/Users/sissi/Downloads/RUKA/assets/robot.urdf", useFixedBase=True)

# Map Joints
joint_map = {}
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    if info[2] != p.JOINT_FIXED:
        joint_map[info[1].decode("utf-8")] = i

mcp_offsets = {
    "wrist": [0, 0, -0.015],
    "index": [0, 0, -0.008],
    "pinky": [0, 0, 0.006]
}

# Calculate Initial Frames and Hand Scaling
state = p.getLinkState(robot_id, 1)
wrist_pos = apply_offset(state[4], state[5], mcp_offsets["wrist"])

state = p.getLinkState(robot_id, 2)
index_mcp_pos = apply_offset(state[4], state[5], mcp_offsets["index"])

state = p.getLinkState(robot_id, 13)
pinky_mcp_pos = apply_offset(state[4], state[5], mcp_offsets["pinky"])

state = p.getLinkState(robot_id, 7)
mid_mcp_pos = apply_offset(state[4], state[5], [0, 0, 0])

hand_width = np.linalg.norm(mid_mcp_pos - wrist_pos)

init_x_axis = index_mcp_pos - wrist_pos 
init_z_axis = np.cross(index_mcp_pos - wrist_pos, pinky_mcp_pos - wrist_pos) 
init_y_axis = np.cross(init_z_axis, init_x_axis)
init_x_axis /= np.linalg.norm(init_x_axis)
init_y_axis /= np.linalg.norm(init_y_axis)
init_z_axis /= np.linalg.norm(init_z_axis)
R_robot = np.stack([init_x_axis, init_y_axis, init_z_axis], axis=1)

handler = HandReader()
handler_joint_order = [
    "pinky_mcp", "pinky_splay", "pinky_pip",
    "ring_splay", "ring_mcp", "ring_pip",
    "index_pip", "index_splay", "index_mcp",
    "mid_pip", "mid_mcp",
    "thumb_ip", "thumb_cmc", "thumb_mcp",
    "wrist_yaw", "wrist_pitch"
]
lower_limit_map = {}
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode("utf-8")
    lower_limit = info[8]  # Index 8 is the 'lower' limit in PyBullet
    lower_limit_map[joint_name] = lower_limit

def replay_motion(trajectory_path="mp_trajectory.npy"):
    try:
        data = np.load(trajectory_path)
        print(f"Loaded {data.shape[0]} frames from {trajectory_path}")
    except FileNotFoundError:
        print("Error: Trajectory file not found.")
        return

    try:
        for i in range(data.shape[0]):
            points = data[i]
            angles = handler.get_motor_positions(points)
            q_rad = np.radians(angles)
            # 2. Map values to names and ADD LOWER BOUNDS
            angle_dict = {}
            for name, val in zip(handler_joint_order, q_rad):
                if name == "wrist_yaw" or name == "wrist_pitch": 
                    angle_dict[name] = val
                    continue
                offset = lower_limit_map.get(name, 0)
                angle_dict[name] = val + offset

            # 3. Define the PIP -> DIP mirroring logic
            # These are the joints that should copy their parent PIP's value
            mirror_map = {
                "index_dip": angle_dict.get("index_pip", 0),
                "middle_dip": angle_dict.get("middle_pip", 0),
                "ring_dip": angle_dict.get("ring_pip", 0),
                "pinky_dip": angle_dict.get("pinky_pip", 0)
            }
            print(q_rad)
            
            for joint_name, val in angle_dict.items():
                if joint_name in joint_map:
                    p.resetJointState(robot_id, joint_map[joint_name], val)
            for dip_name, mirrored_val in mirror_map.items():
                if dip_name in joint_map:
                    p.resetJointState(robot_id, joint_map[dip_name], mirrored_val)

            p.stepSimulation()
            time.sleep(0.1)
        

        print("Replay finished.")
        p.disconnect()

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")

if __name__ == "__main__":
    replay_motion()
            