import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
from mpl_toolkits.mplot3d import Axes3D

# --- 1. HARDCODED INPUT DATA (5 fingers x 5 joints x 3 coords) ---
static_data = np.array([
    [[-0.00, -0.04,  0.04], [-0.01, -0.01,  0.08], [-0.01,  0.02,  0.10], [-0.01,  0.05,  0.12], [-0.01,  0.07,  0.14]], # Thumb
    [[-0.00, -0.04,  0.04], [ 0.04,  0.01,  0.13], [ 0.04,  0.05,  0.12], [ 0.04,  0.07,  0.11], [ 0.03,  0.09,  0.09]], # Index
    [[-0.00, -0.04,  0.04], [ 0.06,  0.00,  0.12], [ 0.06,  0.05,  0.11], [ 0.05,  0.07,  0.10], [ 0.04,  0.09,  0.08]], # Middle
    [[-0.00, -0.04,  0.04], [ 0.07,  0.00,  0.10], [ 0.07,  0.04,  0.09], [ 0.06,  0.07,  0.08], [ 0.04,  0.08,  0.07]], # Ring
    [[-0.00, -0.04,  0.04], [ 0.07,  0.00,  0.07], [ 0.08,  0.03,  0.08], [ 0.07,  0.05,  0.07], [ 0.06,  0.07,  0.07]]  # Pinky
])

def apply_offset(pos, orn, offset):
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    world_offset = rot_matrix.dot(offset)
    return pos + world_offset

def main():
    # --- 2. SETUP PYBULLET FOR URDF POINTS ---
    # We use DIRECT mode to avoid launching the GUI window just for math
    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load your robot
    try:
        robot_id = p.loadURDF("/Users/sissi/Downloads/RUKA/assets/robot.urdf", useFixedBase=True)
    except:
        print("Error: Could not load robot.urdf. Check the path.")
        return

    # Map Joint/Link Names
    link_name_to_index = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        link_name = info[12].decode()
        link_name_to_index[link_name] = i

    # --- 3. GET ROBOT BASIS & TARGETS ---
    mcp_offsets = {"wrist": [0, 0, -0.015], "index": [0, 0, -0.008], "pinky": [0, 0, 0.006]}

    # Get Wrist & MCPs for Basis
    state = p.getLinkState(robot_id, 1); wrist_pos = apply_offset(state[4], state[5], mcp_offsets["wrist"])
    state = p.getLinkState(robot_id, 2); index_mcp_pos = apply_offset(state[4], state[5], mcp_offsets["index"])
    state = p.getLinkState(robot_id, 13); pinky_mcp_pos = apply_offset(state[4], state[5], mcp_offsets["pinky"])
    
    # Calculate Robot Hand Width (Index to Pinky)
    robot_width = np.linalg.norm(index_mcp_pos - pinky_mcp_pos) # Simplified width

    # Calculate Robot Basis Vectors
    # Note: Using standard "X is Sideways, Y is Forward" convention logic
    init_x_axis = index_mcp_pos - wrist_pos 
    init_z_axis = np.cross(index_mcp_pos - wrist_pos, pinky_mcp_pos - wrist_pos) 
    init_y_axis = np.cross(init_z_axis, init_x_axis)
    
    # Normalize
    init_x_axis /= np.linalg.norm(init_x_axis)
    init_y_axis /= np.linalg.norm(init_y_axis)
    init_z_axis /= np.linalg.norm(init_z_axis)
    R_robot = np.stack([init_x_axis, init_y_axis, init_z_axis], axis=1)

    # Get Robot Fingertip Targets
    points_target = []
    target_names = ["thumb_actual_tip", "index_actual_tip", "middle_actual_tip", "ring_actual_tip", "pinky_actual_tip"]
    for name in target_names:
        if name in link_name_to_index:
            state = p.getLinkState(robot_id, link_name_to_index[name])
            points_target.append(apply_offset(state[4], state[5], [0, 0, 0]))
        else:
            print(f"Warning: Link {name} not found in URDF")
            points_target.append([0,0,0])

    rel_hand_frame = static_data.reshape(5, 5, 3)

    # --- 5. PLOTTING ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Robot Fingertip Targets (Large Dots)
    finger_colors = ["red", "green", "blue", "purple", "orange"]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    
    for i, pt in enumerate(points_target):
        ax.scatter(pt[0], pt[1], pt[2], c=finger_colors[i], s=100, marker="o", label=f"{finger_names[i]} Target")

    # Plot Transformed Hand (Lines & Small Triangles)
    for i in range(5):
        chain = rel_hand_frame[i]
        # Plot joints
        ax.scatter(chain[:,0], chain[:,1], chain[:,2], c=finger_colors[i], s=20, marker="^")
        # Plot lines
        ax.plot(chain[:,0], chain[:,1], chain[:,2], c=finger_colors[i], linestyle='--')

    # Plot Wrists
    ax.scatter(*wrist_pos, c="black", s=150, marker="X", label="Robot Wrist")
    
    ax.set_title("Hardcoded Points Retargeted to URDF Space")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    
    # Force Aspect Ratio
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

    plt.show()
    p.disconnect()

if __name__ == "__main__":
    main()