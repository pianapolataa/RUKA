import numpy as np
from ikpy import chain
from ruka_hand.control.hand import Hand
from ruka_hand.utils.trajectory import move_to_pos
from dex_retargeting.retargeting_config import RetargetingConfig


# Min/max angles same as before
min_deg = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -25, 0], dtype=float)
max_deg = np.array([90, 40, 85, 15, 90, 85, 70, 20, 90, 80, 90, 90, 145, 90, 25, 60], dtype=float)

def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)

class IkRUKAv2Handler:
    def __init__(self, urdf_path="assets/robot.urdf", hand_type="right"):
        # self.hand_type = hand_type
        # self.hand = Hand(hand_type)
        self.initial_wrist_axis = None
        self.initial_palm_normal = None
        self.initial_horiz = None
        self.set_up_ikpy(urdf_path)
    
    def get_start_angles(self, chain):
        q = []
        for link in chain.links:
            if link.bounds is not None: 
                if np.isinf(link.bounds[0]) or np.isinf(link.bounds[1]): q.append(0.0)
                elif link.name in ['index_splay', 'ring_splay', 'pinky_splay']:
                    q.append((link.bounds[0] + link.bounds[1]) / 2)  # middle bound
                else: q.append(link.bounds[0])  # lower bound
            else: q.append(0.0)  # fixed joint
        return q
    
    def apply_offset(self, fk_matrix, offset): 
        rot = fk_matrix[:3, :3] 
        pos = fk_matrix[:3, 3] 
        world_offset = rot @ np.array(offset) 
        return pos + world_offset

    def set_up_ikpy(self, urdf_path):
        self.tip_offsets = {
            "thumb":  [-0.017, 0.0025, -0.005],
            "index":  [-0.022, 0, -0.004],
            "middle": [-0.02, 0.002, -0.004],
            "ring":   [-0.02, 0.002, -0.004],
            "pinky":  [-0.017, 0.002, -0.004]
        }
        self.mcp_offsets = {
            "wrist": [0, 0, -0.015],
            "index": [0, 0, -0.008],
            "pinky": [0, 0, 0.006]
        }
        self.chain_names = {
            "index":  ['index_splay', 'index_mcp', 'index_pip', 'index_dip'],
            "middle": ['mid_mcp', 'mid_pip', 'mid_dip'],
            "ring":   ['ring_splay', 'ring_mcp', 'ring_pip', 'ring_dip'],
            "pinky":  ['pinky_splay', 'pinky_mcp', 'pinky_pip', 'pinky_dip'],
            "thumb":  ['thumb_cmc', 'thumb_mcp', 'thumb_ip']
        }
        self.chains = {}
        for finger, links in self.chain_names.items():
            base_elements = ['backhand', links[0]]  # common base + finger knuckle
            self.chains[finger] = chain.Chain.from_urdf_file(
                urdf_path,
                base_elements=base_elements,
                name=finger,
                last_link_vector=self.tip_offsets[finger],
            )
        q = self.get_start_angles(self.chains['index'])
        fk = self.chains['index'].forward_kinematics(q, full_kinematics=True)
        self.wrist_pos = self.apply_offset(fk[0], self.mcp_offsets["wrist"])
        self.index_mcp_pos = self.apply_offset(fk[1], self.mcp_offsets["index"])

        q = self.get_start_angles(self.chains['pinky'])
        fk = self.chains['pinky'].forward_kinematics(q, full_kinematics=True)
        self.pinky_mcp_pos = self.apply_offset(fk[1], self.mcp_offsets["pinky"])
        self.hand_width = np.linalg.norm(self.index_mcp_pos - self.wrist_pos)

        # Hand frame axes
        self.init_x_axis = self.index_mcp_pos - self.wrist_pos 
        self.init_z_axis = np.cross(self.index_mcp_pos - self.wrist_pos, self.pinky_mcp_pos - self.wrist_pos) 
        self.init_y_axis = np.cross(self.init_z_axis, self.init_x_axis)
        self.init_x_axis = self.init_x_axis / np.linalg.norm(self.init_x_axis)
        self.init_y_axis = self.init_y_axis / np.linalg.norm(self.init_y_axis)
        self.init_z_axis = self.init_z_axis / np.linalg.norm(self.init_z_axis)
        self.R_robot = np.stack([self.init_x_axis, self.init_y_axis, self.init_z_axis], axis=1)

        for finger in self.tip_offsets.keys():
            q = self.get_start_angles(self.chains[finger])
            fk = self.chains[finger].forward_kinematics(q, full_kinematics=True)


    def get_wrist_angles(self, horiz, wrist_axis, palm_normal):
        if self.initial_wrist_axis is None:
            self.initial_wrist_axis = wrist_axis
            self.initial_palm_normal = palm_normal
            self.initial_horiz = horiz
        
        v1 = wrist_axis
        v2 = self.initial_wrist_axis
        v1_proj = v1 - np.dot(v1, self.initial_palm_normal) * self.initial_palm_normal
        v2_proj = v2 - np.dot(v2, self.initial_palm_normal) * self.initial_palm_normal
        yaw = angle_between(v1_proj, v2_proj)
        if np.dot(np.cross(v2_proj, v1_proj), self.initial_palm_normal) > 0: 
            yaw = -yaw
        
        v1_proj = v1 - np.dot(v1, self.initial_horiz) * self.initial_horiz
        v2_proj = v2 - np.dot(v2, self.initial_horiz) * self.initial_horiz
        pitch = angle_between(v1_proj, v2_proj)
        if np.dot(np.cross(v2_proj, v1_proj), self.initial_horiz) > 0:
            pitch = 0
        return yaw, pitch
    
    def to_robot_frame(self, positions, x_axis, y_axis, palm_normal, norm_len):
        R_hand = np.stack([x_axis, y_axis, palm_normal], axis=1)  # columns = hand axes
        scale_factor = self.hand_width / norm_len
        rel_hand_frame = scale_factor * (positions @ R_hand  @ self.R_robot.T) + self.wrist_pos
        # print("Scaled and rotated to ikpy:", rel_hand_frame)
        # print(self.hand_width, norm_len)
        return rel_hand_frame

    def get_ik_finger_angles(self, target_points):
        q_all = []
        for finger, target in zip(self.chain_names.keys(), target_points):
            ch = self.chains[finger]
            q_seed = self.get_start_angles(ch)
            q_sol = ch.inverse_kinematics(
                target,
                initial_position=q_seed
            )
            q_finger = q_sol[1:len(ch.links)]
            q_all.extend(q_finger)
        return np.array(q_all)

    def points_to_joint_angles(self, points):
        angles = np.zeros(16)
        # wrist = points[0][0]
        # index_mcp = points[1][1]
        # middle_mcp = points[2][1]
        # pinky_mcp = points[4][1]
        # thumb_tip = points[0][4]
        # index_tip = points[1][4]
        # middle_tip = points[2][4]
        # ring_tip = points[3][4]
        # pinky_tip = points[4][4]
        points = points - points[0]
        wrist = points[0]
        index_mcp = points[5]
        middle_mcp = points[9]
        pinky_mcp = points[17]
        thumb_tip = points[4]
        index_tip = points[8] 
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        horiz = index_mcp - pinky_mcp
        horiz = horiz / np.linalg.norm(horiz)
        wrist_axis = middle_mcp - wrist
        wrist_axis = wrist_axis / np.linalg.norm(wrist_axis)
        palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
        palm_normal = palm_normal / np.linalg.norm(palm_normal)

        angles[14], angles[15] = self.get_wrist_angles(horiz, wrist_axis, palm_normal)

        x_axis = index_mcp - wrist
        y_axis = np.cross(palm_normal, x_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        norm_len = np.linalg.norm(index_mcp - wrist)
        transformed_fingertips = self.to_robot_frame([index_tip, middle_tip, ring_tip, pinky_tip, thumb_tip], x_axis, y_axis, palm_normal, norm_len)
        finger_deg = self.get_ik_finger_angles(transformed_fingertips)
        
        angles[7] = finger_deg[2]
        angles[8] = finger_deg[3]
        angles[6] = finger_deg[4]
        angles[10] = finger_deg[6]
        angles[9] = finger_deg[7]
        angles[3] = finger_deg[9]
        angles[4] = finger_deg[10]
        angles[5] = finger_deg[11]
        angles[1] = finger_deg[13]
        angles[0] = finger_deg[14]
        angles[2] = finger_deg[15]
        angles[12] = finger_deg[17]
        angles[13] = finger_deg[18]
        angles[11] = finger_deg[19]
        return np.degrees(angles)

    def compute_motor_pos(self, test_pos):
        test_pos = np.array(test_pos, dtype=float)
        clamped = np.clip(test_pos, min_deg, max_deg)
        normed = clamped / (max_deg - min_deg)
        positions = normed * (self.hand.curled_bound - self.hand.tensioned_pos) + self.hand.tensioned_pos
        positions[1] = 2285 + normed[1] * abs(self.hand.curled_bound[1] - self.hand.tensioned_pos[1])
        positions[3] = 2070 - normed[3] * abs(self.hand.curled_bound[3] - self.hand.tensioned_pos[3])
        positions[7] = 2125 + normed[7] * abs(self.hand.curled_bound[7] - self.hand.tensioned_pos[7])
        positions[14] = 1990 + normed[14] * abs(self.hand.curled_bound[14] - self.hand.tensioned_pos[14])
        return positions

    def get_command(self, points_24):
        joint_angles = self.points_to_joint_angles(points_24)
        motor_positions = self.compute_motor_pos(joint_angles)
        return motor_positions

    def reset(self):
        motor_positions = self.compute_motor_pos(np.zeros(16))
        curr_pos = self.hand.read_pos()
        move_to_pos(curr_pos=curr_pos, des_pos=motor_positions, hand=self.hand, traj_len=35)

    def close(self):
        self.hand.close()

IkRUKAv2Handler()

"""
    FINGER_NAMES_TO_MOTOR_IDS = {
        "Thumb": [12, 13, 11],
        "Index": [7, 8, 6],
        "Middle": [10, 9],
        "Ring": [3, 4, 5],
        "Pinky": [1, 0, 2],
        "Wrist": [14, 15]
    }
"""